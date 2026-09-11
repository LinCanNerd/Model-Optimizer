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

"""DP=EP and MTP support for ``nemotron_h``.

Two pieces, applied by monkey-patching stock transformers:
  1. register a custom EP style ``nemotron_experts_ep`` (``NemotronHExpertsEP``) — nemotron_h's gate
     returns logits only (routing happens inside the MoE block), so ``ep_router`` (which remaps on the
     router output) can't be used; instead remap top-k indices to local at the EXPERTS input;
  2. set ``base_model_ep_plan`` (experts under ``mixer.experts``, non-gated up/down_proj).
The export reads nemotron_h's ``n_routed_experts`` (handled generically in moe_utils).

Version floor: transformers >= 5.6 (the shared EP framework + ``MoeTensorParalellExperts`` /
``all_reduce_forward`` this plugin subclasses/uses; enforced by ``moe.load_and_prepare_ep``). No higher
floor -- nemotron_h's native fused experts are present from 5.6.
"""

from __future__ import annotations

import copy
import json
import sys
import weakref
from pathlib import Path

BASE_MODEL_EP_PLAN = {
    "layers.*.mixer.experts.up_proj": "grouped_gemm",
    "layers.*.mixer.experts.down_proj": "grouped_gemm",
    "layers.*.mixer.experts": "nemotron_experts_ep",
    "mtp.layers.*.mixer.experts.up_proj": "grouped_gemm",
    "mtp.layers.*.mixer.experts.down_proj": "grouped_gemm",
    "mtp.layers.*.mixer.experts": "nemotron_experts_ep",
}

_REGISTERED = False
_NATIVE_MTP_PATCHED = False
_PATCHED_OMNI_CLASSES = set()
_MTP_MODELS = weakref.WeakSet()
_MTP_FORWARD_MODELS = weakref.WeakSet()


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
                block_config.layers_block_type = [
                    "full_attention" if block_type == "attention" else block_type
                    for block_type in block_types
                ]
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


def _register_style():
    """Register NemotronHExpertsEP as the ``nemotron_experts_ep`` parallel style (idempotent).
    Defined lazily so this module imports without transformers/torch present."""
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        MoeTensorParalellExperts,
        all_reduce_forward,
    )

    class NemotronHExpertsEP(MoeTensorParalellExperts):
        """Expert parallelism for nemotron_h's block-routed MoE.

        ``ep_router`` (RouterParallel) cannot be used: it remaps indices on the *router output*, but
        nemotron_h's gate returns logits only — the group-limited top-k routing happens inside the MoE
        block, so the gate never produces ``(scores, indices)`` for RouterParallel to remap. Instead we
        do the EP index-remap at the EXPERTS input: tokens routed to this rank's local experts get a
        local index (``global - ep_rank*local``); non-local tokens are sent to local index 0 with
        weight 0, so the experts' ``index_add`` makes them a no-op. ``grouped_gemm`` on up/down_proj
        sets ``mod.num_experts`` to the local count; the inherited ``all_reduce_forward`` sums the
        partial expert outputs across the EP group."""

        def _prepare_input_fn(self, mod, inputs, device_mesh):
            hidden, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
            local = mod.num_experts  # local count after grouped_gemm shards dim 0
            start = device_mesh.get_local_rank() * local
            is_local = (top_k_index >= start) & (top_k_index < start + local)
            local_index = torch.where(is_local, top_k_index - start, torch.zeros_like(top_k_index))
            local_weights = torch.where(is_local, top_k_weights, torch.zeros_like(top_k_weights))
            return (hidden, local_index, local_weights)

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return all_reduce_forward(outputs, device_mesh)

    ALL_PARALLEL_STYLES.register("nemotron_experts_ep", NemotronHExpertsEP())
    _REGISTERED = True


def apply(hf_config, **_):
    """Register the nemotron_experts_ep style + set nemotron_h's base_model_ep_plan (if absent).

    Standalone ``nemotron_h``: the plan goes on the config itself. For the Omni VLM
    (``nemotron_h_omni``) the MoE experts live in the NESTED ``llm_config`` sub-model (a native
    ``NemotronHForCausalLM``), so the plan must be attached THERE, not on the top VLM config.
    transformers' ``post_init`` aggregates each sub-model's ``_ep_plan`` up to the root WITH its
    module-name prefix, so a plan on ``llm_config`` surfaces at the VLM root as
    ``language_model.<base>.layers.*.mixer.experts`` -- matching the real param paths, so the
    group-injecting ``torch_a2a_experts`` ``_prepare_input_fn`` is applied to the nested experts. A
    plan on the TOP config never reaches them (=> "torch_a2a requires the EP process_group"). Mirrors
    ``kimi_k25`` attaching its plan to ``text_config``; ``_ep_config`` then finds it via its
    ``text_config`` / ``llm_config`` recursion.
    """
    _register_style()
    # Route the plan onto the nested nemotron_h sub-config for VLMs (Omni); else the config itself.
    target = hf_config
    for _attr in ("text_config", "llm_config"):
        sub = getattr(hf_config, _attr, None)
        if sub is not None and getattr(sub, "model_type", None) == "nemotron_h":
            target = sub
            break
    if not getattr(target, "base_model_ep_plan", None):
        target.base_model_ep_plan = dict(BASE_MODEL_EP_PLAN)
    return hf_config
