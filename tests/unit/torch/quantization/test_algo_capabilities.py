# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the per-algorithm capability declarations."""

from typing import Literal

import pytest
from pydantic import ValidationError

from modelopt.torch.quantization.algo_cfg import (
    ACTS,
    WEIGHT,
    WRITABLE_TOKENS,
    AlgoCapabilities,
    capabilities_for,
)
from modelopt.torch.quantization.config import LocalHessianCalibConfig, QuantizeAlgorithmConfig
from modelopt.torch.quantization.mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
    _writes_weights,
)


def _known_algorithms():
    names = getattr(CalibrateModeRegistry, "_name2descriptor", {})
    return sorted(
        n.removesuffix("_calibrate")
        for n in names
        if n.endswith("_calibrate") and not n.startswith("_")
    )


def test_every_registered_algorithm_overrides_the_conservative_default():
    # The base class supplies a pessimistic default, so "not None" is vacuous. What matters is
    # that a newly added descriptor which forgets to declare fails here, rather than quietly
    # running with `may_write=WRITABLE_TOKENS` and a weight write-back on every layer.
    base = BaseCalibrateModeDescriptor._capabilities
    for algo in _known_algorithms():
        caps = capabilities_for(algo)
        assert caps is not None, algo
        assert caps != base, f"{algo} still carries the pessimistic default"


def test_calib_mutates_weights_is_derived_per_algorithm():
    # The behavioural core of the PR: what each algorithm dispatches with.
    assert _writes_weights("mse", {}) is False
    assert _writes_weights("max", {}) is False
    assert _writes_weights("gptq", {}) is True
    assert _writes_weights("awq_lite", {}) is True
    # An unrecognized method is assumed to write weights, matching the config-time validator.
    assert _writes_weights("not_an_algorithm", {}) is True
    assert _writes_weights(None, {}) is True


def test_a_custom_algorithm_inherits_conservative_capabilities():
    class _CustomConfig(QuantizeAlgorithmConfig):
        method: Literal["my_custom_algo"] = "my_custom_algo"

    @CalibrateModeRegistry.register_mode
    class _CustomDescriptor(BaseCalibrateModeDescriptor):
        _calib_func = None

        @property
        def config_class(self):
            return _CustomConfig

    try:
        caps = capabilities_for("my_custom_algo")
        assert caps is not None, "a registered algorithm must have capabilities"
        assert caps.may_write == WRITABLE_TOKENS, "assume it writes everything"
        assert not caps.scopable, "assume it cannot be scoped"
    finally:
        CalibrateModeRegistry.remove_mode("my_custom_algo_calibrate")


def test_calib_mutates_weights_false_is_rejected_for_every_weight_writing_algorithm():
    checked = []
    for algo in _known_algorithms():
        if WEIGHT not in capabilities_for(algo).may_write:
            continue
        config_class = CalibrateModeRegistry[
            BaseCalibrateModeDescriptor._get_mode_name(algo)
        ].config_class
        with pytest.raises(ValidationError, match="mutates layer weights in-place"):
            config_class(layerwise={"enable": True, "calib_mutates_weights": False})
        checked.append(algo)

    assert checked, "no weight-writing algorithm found -- the check would pass vacuously"


@pytest.mark.parametrize(
    ("algo", "key"),
    [("lsq", "scale_algorithm"), ("nvfp4_act_headroom", "weight_scale_algorithm")],
)
def test_a_delegating_algorithm_inherits_its_sub_algorithms_capabilities(algo, key):
    # `local_hessian` reads activations; the delegating algorithm must say so on its behalf.
    # (`nvfp4_act_headroom` needs activations for its own work, so it declares ACTS either way.)
    assert ACTS in capabilities_for(algo, {key: {"method": "local_hessian"}}).requires


def test_lsq_only_reads_activations_when_its_sub_algorithm_does():
    assert ACTS not in capabilities_for("lsq", {"scale_algorithm": {"method": "max"}}).requires
    assert (
        ACTS in capabilities_for("lsq", {"scale_algorithm": {"method": "local_hessian"}}).requires
    )


def test_a_sub_algorithm_given_as_a_config_object_folds_the_same_as_a_dict():
    # `method` is a defaulted field, so `model_dump(exclude_unset=True)` drops it and the
    # sub-algorithm becomes unidentifiable -- taking the unknown-algorithm fallback and
    # silently discarding the fold. The two spellings must agree.
    as_object = capabilities_for("lsq", {"scale_algorithm": LocalHessianCalibConfig()})
    as_dict = capabilities_for("lsq", {"scale_algorithm": {"method": "local_hessian"}})
    assert as_object == as_dict
    assert ACTS in as_object.requires, "sub-algorithm's reads were dropped"
    assert WEIGHT not in as_object.may_write, "fell through to the unknown-algorithm fallback"


@pytest.mark.parametrize("sub", [{"method": "not_an_algorithm"}, {"method": "local_hessian"}])
def test_a_delegating_algorithm_is_no_more_scopable_than_its_sub_algorithm(sub):
    # `scopable=False` is the restrictive value, so it has to travel: a delegating algorithm
    # cannot be scoped more finely than the algorithm it runs first. No shipped pair exercises
    # this yet -- both delegating algorithms are already unscopable and every reachable
    # sub-algorithm is scopable -- so the fold is called directly with a scopable outer.
    outer = AlgoCapabilities(
        writes_whole_module=False, refines="weight", may_write=frozenset(), scopable=True
    )
    folded = BaseCalibrateModeDescriptor._with_sub_algorithm(outer, sub, frozenset())

    if sub["method"] == "not_an_algorithm":
        assert not folded.scopable, "an unknown sub-algorithm must not stay scopable"
        assert folded.writes_whole_module, "an unknown sub-algorithm must be whole-module"
    else:
        # local_hessian is scopable but writes whole modules.
        assert folded.scopable
        assert folded.writes_whole_module, "sub-algorithm's whole-module write was dropped"


def test_a_delegating_algorithm_falls_back_to_the_conservative_upper_bound():
    caps = capabilities_for("lsq", {"scale_algorithm": {"method": "not_an_algorithm"}})
    assert caps.may_write >= WRITABLE_TOKENS
