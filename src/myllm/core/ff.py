# Copyright 2023-2026 Sebastian Raschka, licensed under the Apache License 2.0 (see LICENSE.txt)
# Source: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
#   - Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - Code repository: https://github.com/rasbt/LLMs-from-scratch
# Licensed under the Apache License, Version 2.0; see the LICENSE.txt file in this directory.

from torch import nn

from .activation import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
