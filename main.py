# Copyright 2023-2026 Sebastian Raschka, licensed under the Apache License 2.0 (see LICENSE.txt)
# Source: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
#   - Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - Code repository: https://github.com/rasbt/LLMs-from-scratch
# Licensed under the Apache License, Version 2.0; see the LICENSE.txt file in this directory.

# This file can be run as a standalone script.

import tiktoken
import torch

from myllm.constants import GPT_CONFIG_124M
from myllm.core.gpt import GPTModel
from myllm.generate_text import generate_text_simple


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
