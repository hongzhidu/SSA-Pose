# Copyright 2023 The HuggingFace Team. All rights reserved.
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
#
# --------
#
# Modified 2024 by the Tripo AI and Stability AI Team.
#
# Copyright (c) 2024 Tripo AI & Stability AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...utils import BaseModule
from .basic_transformer_block import BasicTransformerBlock


class Transformer1D(nn.Module):
    def __init__(self, ):
        super(Transformer1D, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_dim = 16


        inner_dim = self.num_attention_heads * self.attention_head_dim

        linear_cls = nn.Linear

        # 2. Define input layers
        self.in_channels = 128

        self.norm = torch.nn.GroupNorm(
            num_groups=8,
            num_channels=128,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = linear_cls(128, inner_dim)

        # 3. Define transformers blocks
        self.cross_attention = BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=0.0,
                    cross_attention_dim=256,
                    activation_fn="geglu",
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type="layer_norm",
                    norm_elementwise_affine=True,
                )

        self.self_attention = BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=0.0,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type="layer_norm",
                    norm_elementwise_affine=True,
                )

        # 4. Define output layers
        self.out_channels = 128

        self.proj_out = linear_cls(inner_dim, 128)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):


        # 1. Input
        batch, _, seq_len = hidden_states.shape

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, seq_len, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        hidden_states = self.cross_attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        hidden_states1 = self.self_attention(
                hidden_states,
                encoder_hidden_states=None,
                encoder_attention_mask=encoder_attention_mask,
            )


        return hidden_states.permute(0, 2, 1).contiguous(), hidden_states1.permute(0, 2, 1).contiguous()
