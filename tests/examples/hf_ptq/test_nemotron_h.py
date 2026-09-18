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

import pytest

from examples.hf_ptq.models.nemotron_h import _normalize_mtp_block_types


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
