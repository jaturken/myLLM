# Copyright 2023-2026 Sebastian Raschka, licensed under the Apache License 2.0 (see LICENSE.txt)
# Source: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
#   - Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - Code repository: https://github.com/rasbt/LLMs-from-scratch
# Licensed under the Apache License, Version 2.0; see the LICENSE.txt file in this directory.

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}
