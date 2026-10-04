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

"""Combined-weight quantization, adapter training, and ModelOpt checkpoint round trips."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def _model():
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))


def _optimizer(model):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)


def _step(model, teacher, optimizer, inputs):
    optimizer.zero_grad()
    with torch.no_grad():
        target = teacher(inputs).softmax(dim=-1)
    loss = F.kl_div(model(inputs).log_softmax(dim=-1), target, reduction="batchmean")
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_quant_lora_training_restore_and_merge():
    torch.manual_seed(42)
    teacher = _model()
    student = copy.deepcopy(teacher)
    inputs = torch.randn(4, 16)
    mtq.quantize(student, mtq.INT8_DEFAULT_CFG, lambda m: m(inputs))
    baseline = student(inputs).detach()
    backbone = {name: p.detach().clone() for name, p in student.named_parameters()}
    mtq.enable_quant_lora(student, {"rank": 4, "alpha": 8})
    torch.testing.assert_close(student(inputs), baseline, rtol=0, atol=0)
    assert all(p.requires_grad == ("lora_" in name) for name, p in student.named_parameters())
    optimizer = _optimizer(student)
    for _ in range(2):
        _step(student, teacher, optimizer, inputs)
    assert student[0].lora_A.grad.count_nonzero() > 0
    assert student[0].lora_B.count_nonzero() > 0
    for name, value in backbone.items():
        torch.testing.assert_close(dict(student.named_parameters())[name], value, rtol=0, atol=0)

    restored = mto.restore_from_modelopt_state(_model(), mto.modelopt_state(student))
    restored_optimizer = _optimizer(restored)
    restored.load_state_dict(student.state_dict())
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert all(p.requires_grad == ("lora_" in name) for name, p in restored.named_parameters())
    for model, opt in ((student, optimizer), (restored, restored_optimizer)):
        _step(model, teacher, opt, inputs)
    torch.testing.assert_close(restored(inputs), student(inputs), rtol=0, atol=0)

    # A nonzero update must be quantized jointly, rather than added after quantization.
    layer = restored[0]
    with torch.no_grad():
        layer.lora_B.normal_()
    weight = layer._parameters["weight"]
    combined = weight + layer.lora_scale * layer.lora_B @ layer.lora_A
    expected = layer.output_quantizer(
        F.linear(layer.input_quantizer(inputs), layer.weight_quantizer(combined), layer.bias)
    )
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    separate = F.linear(layer.input_quantizer(inputs), layer.weight_quantizer(weight), layer.bias)
    separate += F.linear(layer.input_quantizer(inputs), combined - weight)
    assert not torch.allclose(layer(inputs), separate)

    output = restored(inputs).detach()
    mtq.merge_quant_lora(restored)
    assert not any("lora_" in name for name, _ in restored.named_parameters())
    torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)
    checkpoint = io.BytesIO()
    mto.save(restored, checkpoint)
    checkpoint.seek(0)
    merged_restored = mto.restore(_model(), checkpoint)
    torch.testing.assert_close(merged_restored(inputs), output, rtol=0, atol=0)
    assert not any("lora_" in name for name, _ in merged_restored.named_parameters())


@pytest.mark.parametrize("targets", [["0"], ["2"]])
def test_quant_lora_targets(targets):
    student = _model()
    inputs = torch.randn(2, 16)
    mtq.quantize(student, mtq.INT8_DEFAULT_CFG, lambda m: m(inputs))
    mtq.enable_quant_lora(student, {"target_modules": targets})
    assert [name for name, m in student.named_modules() if hasattr(m, "lora_A")] == targets
    before = {name: p.requires_grad for name, p in student.named_parameters()}
    with pytest.raises(ValueError, match="already enabled"):
        mtq.enable_quant_lora(student, {"target_modules": targets})
    assert before == {name: p.requires_grad for name, p in student.named_parameters()}


def test_quant_lora_rejects_unquantized_model():
    with pytest.raises(ValueError, match="No supported fake-quantized"):
        mtq.enable_quant_lora(_model())
