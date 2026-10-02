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

import copy
import json
import warnings
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_activation_scaling_factor, postprocess_state_dict
from modelopt.torch.export.unified_export_hf import _process_quantized_modules
from modelopt.torch.models import hf
from modelopt.torch.models.nemotron_h import mtp as adapter
from modelopt.torch.models.nemotron_h.mtp import _normalize_mtp_block_types
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    _needs_activation_forward_for_max_calib,
    max_calibrate,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize(
    ("block_types", "mixer_types", "expected"),
    [
        (["attention", "moe"], {"attention", "mamba", "moe"}, ["attention", "moe"]),
        (
            ["full_attention", "moe"],
            {"attention", "mamba", "moe"},
            ["attention", "moe"],
        ),
        (
            ["attention", "moe"],
            {"full_attention", "mamba", "moe"},
            ["full_attention", "moe"],
        ),
    ],
)
def test_normalize_mtp_block_types(block_types, mixer_types, expected):
    assert _normalize_mtp_block_types(block_types, mixer_types) == expected


def test_normalize_mtp_block_types_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported NemotronH MTP block type 'unknown'"):
        _normalize_mtp_block_types(["unknown"], {"attention", "mamba", "moe"})


@pytest.mark.parametrize("layout", ["sharded", "single", "missing", "invalid"])
def test_checkpoint_detection_and_consent(tmp_path, monkeypatch, layout):
    tensors = {
        "language_model.mtp.layers.0.eh_proj.weight": torch.ones(1),
        "language_model.mtp.layers.1.final_layernorm.weight": torch.ones(1),
    }
    if layout == "sharded":
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")})
        )
    elif layout == "single":
        save_file(tensors, tmp_path / "model.safetensors")
    elif layout == "invalid":
        (tmp_path / "model.safetensors").write_bytes(b"not safetensors")
    loader = Mock(side_effect=AssertionError("Untrusted remote code executed"))
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", loader)
    detected = layout in ("sharded", "single")
    assert adapter._get_mtp_layout(tmp_path) == (("language_model.", 2) if detected else None)
    if detected:
        with (
            pytest.raises(ValueError, match="trust_remote_code=True"),
            adapter.prepare_for_loading(tmp_path, False),
        ):
            pytest.fail("Consent was not enforced")
    else:
        with adapter.prepare_for_loading(tmp_path, False):
            pass
    loader.assert_not_called()


@pytest.mark.parametrize("config_key", [None, "llm_config", "text_config"])
@pytest.mark.parametrize("declared", [False, True])
def test_missing_mtp_weights_warn_without_constructing(tmp_path, monkeypatch, config_key, declared):
    config = {"num_nextn_predict_layers": int(declared)}
    if config_key:
        config = {config_key: config}
    (tmp_path / "config.json").write_text(json.dumps(config))
    loader = Mock(side_effect=AssertionError("No remote code is needed without MTP weights"))
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", loader)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with hf.prepare_model_for_loading("nemotron_h", tmp_path, False):
            pass
    assert len(caught) == int(declared)
    if declared:
        assert "config declares MTP but no MTP tensors were found" in str(caught[0].message)
    loader.assert_not_called()


@pytest.mark.parametrize(
    "keys",
    [
        ["mtp.layers.0.eh_proj.weight"],
        ["mtp.layers.0.eh_proj.weight", "mtp.layers.3.final_layernorm.weight"],
        ["mtp.layers.0.eh_proj.weight", "language_model.mtp.layers.1.final_layernorm.weight"],
        ["decoder.mtp.layers.0.eh_proj.weight", "decoder.mtp.layers.1.final_layernorm.weight"],
    ],
)
def test_unsupported_mtp_layout_rejects_silent_passthrough(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "model-00001.safetensors")})
    )
    with (
        pytest.raises(ValueError, match="Unsupported Nemotron-H MTP tensor layout"),
        adapter.prepare_for_loading(tmp_path, False),
    ):
        pytest.fail("Unsupported MTP tensors would be silently passed through")


@pytest.fixture(params=[("wrapped", 2), ("wrapped", 4), ("plain", 2), ("plain", 4)])
def tiny_checkpoint(tmp_path, monkeypatch, request):
    native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    layout, num_blocks = request.param
    config = native.NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        layers_block_type=["attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        use_mamba_kernels=False,
        attn_implementation="eager",
        num_nextn_predict_layers=num_blocks // 2,
        mtp_layers_block_type=["attention", "moe"] * (num_blocks // 2),
    )

    class TinyOmni(PreTrainedModel):
        config_class = native.NemotronHConfig
        base_model_prefix = "language_model"

        def __init__(self, config):
            super().__init__(config)
            self.language_model = native.NemotronHForCausalLM(config)
            self.post_init()

        def forward(self, *args, **kwargs):
            return self.language_model(*args, **kwargs)

    cls = TinyOmni if layout == "wrapped" else native.NemotronHForCausalLM
    model = cls(config).eval()
    language_model = getattr(model, "language_model", model)
    language_model.mtp = adapter._NemotronHMTP(config)
    # Fused expert parameters are allocated with empty(); a real checkpoint supplies their values.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    # Write the source layout directly; older HF save_pretrained rewrites the language prefix.
    config.save_pretrained(tmp_path)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", Mock(return_value=TinyOmni))
    return cls, model, tmp_path


@pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_omni"])
def test_loading_places_mtp_weights_and_restores_constructor(
    tiny_checkpoint, monkeypatch, model_type
):
    """Both checkpoint model types load exact MTP weights without leaking constructor patches."""
    cls, source, checkpoint = tiny_checkpoint
    original_init = cls.__init__
    loader = cls if hasattr(source, "language_model") else AutoModelForCausalLM
    for _ in range(2):
        with hf.prepare_model_for_loading(
            model_type, checkpoint, trust_remote_code=hasattr(source, "language_model")
        ):
            loaded, info = loader.from_pretrained(checkpoint, output_loading_info=True)
        assert cls.__init__ is original_init
        assert not info["missing_keys"] and not info["unexpected_keys"]
        assert hf.mtp_loaded_during_model_load(loaded)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
        fresh = cls(source.config)
        assert not hasattr(getattr(fresh, "language_model", fresh), "mtp")

    with (
        pytest.raises(RuntimeError, match="load failure"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        raise RuntimeError("load failure")
    assert cls.__init__ is original_init
    if not hasattr(source, "language_model"):
        adapter.get_class_from_dynamic_module.assert_not_called()
        return

    class OtherOmni(cls):
        def __init__(self, config):
            original_init(self, config)
            self.other_constructor_called = True

    other_init = OtherOmni.__init__
    with adapter.prepare_for_loading(checkpoint, True):
        assert not hasattr(type(source.language_model)(source.config), "mtp")
        monkeypatch.setattr(adapter, "get_class_from_dynamic_module", Mock(return_value=OtherOmni))
        with adapter.prepare_for_loading(checkpoint, True):
            assert OtherOmni(source.config).other_constructor_called
            assert hasattr(cls(source.config).language_model, "mtp")
        assert OtherOmni.__init__ is other_init
    assert cls.__init__ is original_init
    with (
        pytest.raises(RuntimeError, match="load failure"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        raise RuntimeError("load failure")
    assert OtherOmni.__init__ is other_init


def test_remote_mtp_class_requires_consent(tiny_checkpoint, monkeypatch):
    cls, source, checkpoint = tiny_checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_local.NemotronHWithMTP"}
    (checkpoint / "config.json").write_text(json.dumps(config))
    loader = Mock(return_value=cls)
    monkeypatch.setattr(adapter, "get_class_from_dynamic_module", loader)
    original_init = cls.__init__
    with (
        pytest.raises(ValueError, match="trust_remote_code=True"),
        adapter.prepare_for_loading(checkpoint, False),
    ):
        pytest.fail("Remote MTP class was loaded without consent")
    loader.assert_not_called()
    with adapter.prepare_for_loading(checkpoint, True):
        loaded = cls.from_pretrained(checkpoint)
    assert cls.__init__ is original_init
    loader.assert_called_once_with("modeling_local.NemotronHWithMTP", checkpoint)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_mtp_block_count_mismatch_restores_constructor(tiny_checkpoint):
    cls, source, checkpoint = tiny_checkpoint
    config = copy.deepcopy(source.config)
    config.mtp_layers_block_type = ["attention"]
    original_init = cls.__init__
    with (
        pytest.raises(ValueError, match="MTP block count does not match"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        cls(config)
    assert cls.__init__ is original_init


def test_mtp_loading_context_warns_if_constructor_is_bypassed(tiny_checkpoint):
    cls, _, checkpoint = tiny_checkpoint
    original_init = cls.__init__
    with (
        pytest.warns(UserWarning, match="MTP weights may remain unquantized"),
        adapter.prepare_for_loading(checkpoint, True),
    ):
        pass
    assert cls.__init__ is original_init


def test_mtp_later_attention_blocks_remain_causal(tiny_checkpoint):
    _, model, _ = tiny_checkpoint
    mtp = getattr(model, "language_model", model).mtp.eval()
    hidden = torch.randn(1, 4, model.config.hidden_size)
    embeddings = torch.randn_like(hidden)
    with torch.no_grad():
        expected = mtp(hidden, embeddings)
        hidden[:, -1] += 10
        embeddings[:, -1] += 10
        actual = mtp(hidden, embeddings)
    # MTP uses next-token embeddings, so only the last two positions may change.
    torch.testing.assert_close(actual[:, :-2], expected[:, :-2], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "attributes", "needs_forward"),
    [
        ("input_quantizer", {}, True),
        ("up_proj_input_quantizer", {}, True),
        ("k_bmm_quantizer", {}, True),
        ("v_bmm_quantizer", {}, True),
        ("output_quantizer", {}, True),
        ("weight_quantizer", {}, False),
        ("weight_quantizer.0", {}, False),
        ("up_proj_weight_quantizers.0", {}, False),
        ("input_quantizer", {"enable": False}, False),
        ("k_bmm_quantizer", {"constant_amax": 448.0}, False),
        ("input_quantizer", {"type": "dynamic"}, False),
    ],
)
def test_activation_forward_gate(name, attributes, needs_forward):
    mtp = torch.nn.Module()
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), **attributes))
    if "." in name:
        mtp.add_module(name.split(".")[0], torch.nn.ModuleList([quantizer]))
    else:
        mtp.add_module(name, quantizer)
    assert _needs_activation_forward_for_max_calib(mtp) == needs_forward


@pytest.mark.parametrize(
    "activation", ["projection", "shared_projection", "kv", "output", "weight_only", "cast"]
)
def test_calibration_forward_and_export(tiny_checkpoint, activation):
    cls, _, checkpoint = tiny_checkpoint
    with adapter.prepare_for_loading(checkpoint, True):
        model = cls.from_pretrained(checkpoint).eval()
    language_model = getattr(model, "language_model", model)
    prefix = "language_model." if language_model is not model else ""
    projection_name = (
        prefix
        + "mtp."
        + (
            "layers.1.mixer.shared_experts.up_proj"
            if activation == "shared_projection"
            else "layers.0.eh_proj"
        )
    )
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["quant_cfg"].append({"quantizer_name": "*", "enable": False})
    cfg["quant_cfg"].append(
        {"quantizer_name": f"{projection_name}.weight_quantizer", "enable": True}
    )
    if activation.endswith("projection"):
        cfg["quant_cfg"].append(
            {"quantizer_name": f"{projection_name}.input_quantizer", "enable": True}
        )
    extra_names = []
    if activation in ("kv", "output", "cast"):
        extra_names = (
            [f"{prefix}mtp.layers.0.mixer.{kind}_bmm_quantizer" for kind in ("k", "v")]
            if activation != "output"
            else [f"{projection_name}.output_quantizer"]
        )
        attributes = {"num_bits": (4, 3), "axis": None}
        if activation == "cast":
            attributes["constant_amax"] = 448.0
        for name in extra_names:
            cfg["quant_cfg"].append({"quantizer_name": name, "cfg": attributes, "enable": True})
    mtq.quantize(model, {**cfg, "algorithm": None})
    mtp = language_model.mtp
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
    original_forward = language_model.forward
    hf.prepare_model_for_calibration(model)
    wrapped_forward = language_model.forward
    assert wrapped_forward is not original_forward
    hf.prepare_model_for_calibration(model)
    assert language_model.forward is wrapped_forward
    calls = []
    mtp.register_forward_hook(lambda *_: calls.append(True))

    outputs = []
    max_calibrate(
        model, lambda calibrated: outputs.append(calibrated(inputs, use_cache=False).logits)
    )
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    assert calls == ([] if activation in ("weight_only", "cast") else [True])
    assert not language_model.model.norm_f._forward_pre_hooks
    for name in extra_names:
        quantizer = model.get_submodule(name)
        assert quantizer.amax is not None and torch.isfinite(quantizer.amax).all()
        assert quantizer.amax.max() > 0
    if activation.endswith("projection"):
        projection = model.get_submodule(projection_name)
        assert projection.input_quantizer.amax.max() > 0
        expected_scale = get_activation_scaling_factor(projection).squeeze().clone()
        _process_quantized_modules(model, torch.bfloat16)
        exported = postprocess_state_dict(model.state_dict(), maxbound=448, quantization=None)
        torch.testing.assert_close(exported[f"{projection_name}.input_scale"], expected_scale)


def test_lifecycle_dispatch_ignores_unsupported_models():
    """Unrelated model families need neither checkpoint inspection nor auxiliary forwards."""
    with hf.prepare_model_for_loading("unsupported", "unused-checkpoint", False):
        pass
    model = torch.nn.Linear(2, 2)
    hf.prepare_model_for_calibration(model)
    assert not hf.mtp_loaded_during_model_load(model)


@pytest.mark.parametrize("failure", ["base", "mtp"])
def test_calibration_removes_capture_hook_on_error(tiny_checkpoint, failure):
    _, model, _ = tiny_checkpoint
    language_model = getattr(model, "language_model", model)
    language_model.mtp.input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if failure == "mtp":
        language_model.mtp.layers[0].eh_proj = torch.nn.Linear(1, 32)
    assert adapter.prepare_for_calibration(model)
    with pytest.raises((ValueError, RuntimeError), match=r"exactly one|cannot be multiplied"):
        model(input_ids=None if failure == "base" else torch.tensor([[1, 2]]), use_cache=False)
    assert not language_model.model.norm_f._forward_pre_hooks
