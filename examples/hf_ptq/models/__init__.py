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

"""Model-specific lifecycle hooks for the Hugging Face PTQ example."""

import importlib

# model_type -> module name under this package
_PLUGINS = {
    "nemotron_h": "nemotron_h",
    "nemotron_h_omni": "nemotron_h",
}


def _get_plugin(model_type):
    """Return the optional ModelOpt support plugin for a Hugging Face model type."""
    name = _PLUGINS.get(model_type)
    return None if name is None else importlib.import_module(f".{name}", __package__)


def prepare_model_for_loading(model_type, checkpoint_path: str, trust_remote_code: bool) -> None:
    """Run the optional model-specific pre-``from_pretrained`` lifecycle hook."""
    plugin = _get_plugin(model_type)
    hook = getattr(plugin, "prepare_for_loading", None) if plugin is not None else None
    if hook is not None:
        hook(checkpoint_path, trust_remote_code)


def prepare_model_for_calibration(model) -> None:
    """Run the optional model-specific hook that augments the default calibration forward."""
    plugin = _get_plugin(getattr(getattr(model, "config", None), "model_type", None))
    hook = getattr(plugin, "prepare_for_calibration", None) if plugin is not None else None
    if hook is not None:
        hook(model)


def mtp_loaded_during_model_load(model) -> bool:
    """Whether the model plugin loaded native MTP tensors during ``from_pretrained``."""
    plugin = _get_plugin(getattr(getattr(model, "config", None), "model_type", None))
    hook = getattr(plugin, "mtp_loaded_during_model_load", None) if plugin is not None else None
    return bool(hook(model)) if hook is not None else False
