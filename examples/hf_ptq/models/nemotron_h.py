# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MTP construction and quantization calibration hooks for Nemotron-H checkpoints."""

from __future__ import annotations

import copy
import json
import sys
import weakref
from pathlib import Path

_NATIVE_MTP_PATCHED = False
_PATCHED_OMNI_CLASSES = set()
_MTP_MODELS = weakref.WeakSet()
_MTP_FORWARD_MODELS = weakref.WeakSet()


def _normalize_mtp_block_types(block_types, mixer_types):
    """Normalize checkpoint attention names to the installed Transformers registry."""
    aliases = {"attention": "full_attention", "full_attention": "attention"}
    normalized = []
    for block_type in block_types:
        if block_type in mixer_types:
            normalized.append(block_type)
        elif aliases.get(block_type) in mixer_types:
            normalized.append(aliases[block_type])
        else:
            raise ValueError(
                f"Unsupported NemotronH MTP block type {block_type!r}; "
                f"available mixer types: {sorted(mixer_types)}"
            )
    return normalized


def _has_nemotron_h_mtp(checkpoint_path: str) -> bool:
    """Return whether a local checkpoint contains the flattened NemotronH MTP tail."""
    try:
        weight_map = json.loads(
            (Path(checkpoint_path) / "model.safetensors.index.json").read_text()
        ).get("weight_map", {})
    except (OSError, ValueError):
        return False
    return (
        "language_model.mtp.layers.0.eh_proj.weight" in weight_map
        and "language_model.mtp.layers.1.final_layernorm.weight" in weight_map
    )


def prepare_for_loading(checkpoint_path: str, trust_remote_code: bool) -> bool:
    """Attach native NemotronH MTP blocks before ``from_pretrained`` loads weights."""
    if not _has_nemotron_h_mtp(checkpoint_path):
        return False

    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        MIXER_TYPES,
        NemotronHBlock,
        NemotronHForCausalLM,
        NemotronHRMSNorm,
    )

    get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3",
        checkpoint_path,
        trust_remote_code=trust_remote_code,
    )

    global _NATIVE_MTP_PATCHED

    patched_modules = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.endswith(".modeling_nemotron_h_omni"):
            continue
        omni_class = getattr(module, "NemotronH_Omni_Reasoning_V3", None)
        if omni_class is None or omni_class in _PATCHED_OMNI_CLASSES:
            continue

        class NemotronHMTP(torch.nn.Module):
            """MTP fusion around native ``NemotronHBlock`` attention and MoE layers."""

            def __init__(self, config):
                super().__init__()
                block_config = copy.deepcopy(config)
                # Some Omni remote-code revisions drop this field while materializing ``llm_config``.
                # The detected tensor layout is the canonical attention + MoE NemotronH MTP tail.
                block_types = list(
                    getattr(config, "mtp_layers_block_type", None) or ("attention", "moe")
                )
                block_config.layers_block_type = _normalize_mtp_block_types(
                    block_types, MIXER_TYPES
                )
                self.layers = torch.nn.ModuleList(
                    [
                        NemotronHBlock(block_config, layer_idx)
                        for layer_idx in range(len(block_types))
                    ]
                )

                # These tensors have no equivalent in a normal NemotronH decoder block. Attach them to
                # native blocks so their state-dict paths remain ``mtp.layers.*``.
                first_layer, last_layer = self.layers[0], self.layers[-1]
                first_layer.enorm = NemotronHRMSNorm(
                    config.hidden_size, eps=config.layer_norm_epsilon
                )
                first_layer.hnorm = NemotronHRMSNorm(
                    config.hidden_size, eps=config.layer_norm_epsilon
                )
                first_layer.eh_proj = torch.nn.Linear(
                    config.hidden_size * 2, config.hidden_size, bias=False
                )
                last_layer.final_layernorm = NemotronHRMSNorm(
                    config.hidden_size, eps=config.layer_norm_epsilon
                )

            def forward(self, hidden_states, decoder_input, attention_mask=None, position_ids=None):
                from transformers.masking_utils import create_causal_mask

                first_layer, *remaining_layers = self.layers
                decoder_input = torch.cat(
                    (decoder_input[:, 1:, :], torch.zeros_like(decoder_input[:, :1, :])), dim=1
                )
                mtp_hidden = first_layer.eh_proj(
                    torch.cat(
                        (first_layer.enorm(decoder_input), first_layer.hnorm(hidden_states)), dim=-1
                    )
                )
                attention_mask = create_causal_mask(
                    config=first_layer.config,
                    inputs_embeds=mtp_hidden,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=position_ids,
                )
                mtp_hidden = first_layer(
                    mtp_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                for layer in remaining_layers:
                    mtp_hidden = layer(mtp_hidden, use_cache=False)
                return self.layers[-1].final_layernorm(mtp_hidden)

        if not _NATIVE_MTP_PATCHED:
            original_post_init = NemotronHForCausalLM.post_init

            def patched_post_init(self, *args, **kwargs):
                if not hasattr(self, "mtp"):
                    self.mtp = NemotronHMTP(self.config)
                    _MTP_MODELS.add(self)
                return original_post_init(self, *args, **kwargs)

            NemotronHForCausalLM.post_init = patched_post_init
            _NATIVE_MTP_PATCHED = True

        original_init = omni_class.__init__

        def patched_init(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            if not hasattr(self.language_model, "mtp"):
                raise RuntimeError("NemotronH MTP was not constructed during model initialization")

        omni_class.__init__ = patched_init
        _PATCHED_OMNI_CLASSES.add(omni_class)
        patched_modules.append(module_name)

    if not patched_modules:
        raise RuntimeError("Could not find the Nemotron-H Omni remote module to patch")
    print(f"Installed NemotronH MTP constructor patch in: {patched_modules}", flush=True)
    return True


def mtp_loaded_during_model_load(model) -> bool:
    """Return whether ``model`` contains an MTP tail constructed by this adapter."""
    language_model = getattr(model, "language_model", model)
    return language_model in _MTP_MODELS


def _mtp_has_enabled_input_quantizer(mtp) -> bool:
    """Return whether recipe application enabled an MTP activation quantizer."""
    return any(
        name.endswith("_input_quantizer") and getattr(module, "is_enabled", False)
        for name, module in mtp.named_modules()
    )


def prepare_for_calibration(full_model) -> bool:
    """Install an MTP forward for recipes with calibrated MTP activation quantizers."""
    language_model = getattr(full_model, "language_model", None)
    if language_model is None:
        return False

    mtp = getattr(language_model, "mtp", None)
    if mtp is None:
        return False
    if language_model in _MTP_FORWARD_MODELS:
        return True

    original_forward = language_model.forward

    def forward_with_mtp(*args, **kwargs):
        captured = []
        handle = language_model.model.norm_f.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0])
        )
        try:
            outputs = original_forward(*args, **kwargs)
        finally:
            handle.remove()
        # Weight-only quantizers calibrate from their parameters. The MTP tail needs a forward only
        # when the recipe enables an input quantizer that must observe activations.
        if captured and _mtp_has_enabled_input_quantizer(mtp):
            decoder_input = kwargs.get("inputs_embeds")
            if decoder_input is None and kwargs.get("input_ids") is not None:
                decoder_input = language_model.model.embeddings(kwargs["input_ids"])
            if decoder_input is not None:
                mtp(
                    captured[-1],
                    decoder_input,
                    attention_mask=kwargs.get("attention_mask"),
                    position_ids=kwargs.get("position_ids"),
                )
        return outputs

    language_model.forward = forward_with_mtp
    _MTP_FORWARD_MODELS.add(language_model)
    print("Installed NemotronH MTP calibration forward", flush=True)
    return True
