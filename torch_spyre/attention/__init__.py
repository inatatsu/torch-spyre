# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Spyre Flash Attention Backend

This module provides a flash attention implementation for Spyre devices
that can be activated via torch.nn.attention.activate_flash_attention("fa_spyre").
"""

from .flash_attention import register_spyre_flash_attention

__all__ = ["register_spyre_flash_attention"]

# Made with Bob
