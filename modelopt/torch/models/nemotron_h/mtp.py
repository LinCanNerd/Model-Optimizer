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

"""Checkpoint loading and calibration support for the Nemotron-H MTP tail."""

from __future__ import annotations

import copy
import inspect
import weakref
from contextlib import contextmanager
from functools import wraps

import torch
from safetensors import SafetensorError
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import create_causal_mask

from modelopt.torch.quantization.model_calib import _needs_activation_forward_for_max_calib
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map

__all__ = ["mtp_loaded_during_model_load", "prepare_for_calibration", "prepare_for_loading"]

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
        weight_map = indexed_weight_map(checkpoint_path)
    except (OSError, ValueError, SafetensorError):
        return False
    return (
        "language_model.mtp.layers.0.eh_proj.weight" in weight_map
        and "language_model.mtp.layers.1.final_layernorm.weight" in weight_map
    )


class _NemotronHMTP(torch.nn.Module):
    """MTP fusion around native attention and MoE blocks with checkpoint-compatible names."""

    def __init__(self, config):
        super().__init__()
        # Native Nemotron-H is optional: older Transformers can still load unrelated models.
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            MIXER_TYPES,
            NemotronHBlock,
            NemotronHRMSNorm,
        )

        block_config = copy.deepcopy(config)
        # Some remote-code revisions drop this field while materializing ``llm_config``.
        # The detected tensor layout is the canonical attention + MoE MTP tail.
        block_types = list(getattr(config, "mtp_layers_block_type", None) or ("attention", "moe"))
        block_config.layers_block_type = _normalize_mtp_block_types(block_types, MIXER_TYPES)
        self.layers = torch.nn.ModuleList(
            [NemotronHBlock(block_config, layer_idx) for layer_idx in range(len(block_types))]
        )
        first_layer, last_layer = self.layers[0], self.layers[-1]
        first_layer.enorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        first_layer.hnorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        first_layer.eh_proj = torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        last_layer.final_layernorm = NemotronHRMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def forward(self, hidden_states, decoder_input, attention_mask=None, position_ids=None):
        first_layer, *remaining_layers = self.layers
        decoder_input = torch.cat(
            (decoder_input[:, 1:, :], torch.zeros_like(decoder_input[:, :1, :])), dim=1
        )
        mtp_hidden = first_layer.eh_proj(
            torch.cat((first_layer.enorm(decoder_input), first_layer.hnorm(hidden_states)), dim=-1)
        )
        attention_mask = create_causal_mask(
            config=first_layer.config,
            inputs_embeds=mtp_hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        mtp_hidden = first_layer(
            mtp_hidden, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        )
        for layer in remaining_layers:
            mtp_hidden = layer(mtp_hidden, use_cache=False)
        return self.layers[-1].final_layernorm(mtp_hidden)


@contextmanager
def prepare_for_loading(checkpoint_path: str, trust_remote_code: bool):
    """Scope MTP construction to a checkpoint load, before its weights are placed."""
    if not _has_nemotron_h_mtp(checkpoint_path):
        yield
        return
    if not trust_remote_code:
        raise ValueError("Loading Nemotron-H MTP remote code requires trust_remote_code=True")

    omni_class = get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3", checkpoint_path
    )
    original_init = omni_class.__init__

    @wraps(original_init)
    def init_with_mtp(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        language_model = self.language_model
        if not hasattr(language_model, "mtp"):
            language_model.mtp = _NemotronHMTP(language_model.config)
        _MTP_MODELS.add(language_model)

    omni_class.__init__ = init_with_mtp
    try:
        yield
    finally:
        omni_class.__init__ = original_init


def mtp_loaded_during_model_load(model) -> bool:
    """Return whether ``model`` contains an MTP tail constructed by this adapter."""
    language_model = getattr(model, "language_model", model)
    return language_model in _MTP_MODELS


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
    forward_signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def forward_with_mtp(*args, **kwargs):
        if not _needs_activation_forward_for_max_calib(mtp):
            return original_forward(*args, **kwargs)
        captured = []
        handle = language_model.model.norm_f.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0])
        )
        try:
            outputs = original_forward(*args, **kwargs)
        finally:
            handle.remove()
        if captured:
            arguments = forward_signature.bind(*args, **kwargs).arguments
            decoder_input = arguments.get("inputs_embeds")
            if decoder_input is None and arguments.get("input_ids") is not None:
                decoder_input = language_model.model.embeddings(arguments["input_ids"])
            if decoder_input is not None:
                mtp(
                    captured[-1],
                    decoder_input,
                    attention_mask=arguments.get("attention_mask"),
                    position_ids=arguments.get("position_ids"),
                )
        return outputs

    language_model.forward = forward_with_mtp
    _MTP_FORWARD_MODELS.add(language_model)
    print("Installed NemotronH MTP calibration forward", flush=True)
    return True
