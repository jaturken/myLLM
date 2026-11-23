# Copyright 2023-2026 Sebastian Raschka, licensed under the Apache License 2.0 (see LICENSE.txt)
# Source: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
#   - Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - Code repository: https://github.com/rasbt/LLMs-from-scratch
# Licensed under the Apache License, Version 2.0; see the LICENSE.txt file in this directory.

from torch import nn

from .ff import FeedForward
from .mha import MultiHeadAttention
from .norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
