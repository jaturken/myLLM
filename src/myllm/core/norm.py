# Copyright 2023-2026 Sebastian Raschka, licensed under the Apache License 2.0 (see LICENSE.txt)
# Source: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
#   - Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - Code repository: https://github.com/rasbt/LLMs-from-scratch
# Licensed under the Apache License, Version 2.0; see the LICENSE.txt file in this directory.

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
