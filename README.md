# myLLM

A from-scratch GPT-style large language model in PyTorch. The goal of this project is to reproduce and play around with the logic of LLMs, following Sebastian Raschka's book [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch).

The code is a packaged, modular version of the book's GPT-2 (124M) implementation. It is a learning project, not a production library.

## Status

Implemented so far:

- GPT-2 124M architecture (`GPTModel`), assembled from the building blocks below
- Multi-head causal self-attention
- Layer normalization, GELU activation and feed-forward network
- Transformer block with pre-norm and residual (shortcut) connections
- Greedy text generation (`generate_text_simple`)

Not implemented yet: training loop, loss computation, sampling strategies (temperature, top-k), loading pretrained weights and fine-tuning. The model is currently **untrained** (randomly initialized), so generated text is gibberish. That is expected.

## Requirements

- Python 3.14 (see [.python-version](.python-version))
- [uv](https://docs.astral.sh/uv/) for dependency management

Runtime dependencies: `torch`, `tiktoken`, `tqdm`, `numpy`. The dev group adds `pytest`, `jupyterlab`, `matplotlib`, `pandas` and `tensorflow` (the latter is used in the book for loading the original OpenAI GPT-2 weights).

Supported platforms: Linux, Windows and Apple Silicon macOS. Intel macOS is not supported because its torch builds have no Python 3.14 wheels.

## Getting started

```bash
# Install dependencies (including the dev group)
uv sync

# Run the demo: encode a prompt, generate 10 tokens, decode the result
uv run main.py

# Run the tests
uv run pytest
```

The demo feeds the prompt `"Hello, I am"` through an untrained `GPTModel` and prints the input tokens, the output tokens and the decoded text.

## Project layout

```
.
├── main.py                    # Demo script: tokenize, generate, decode
├── src/myllm/
│   ├── constants.py           # GPT_CONFIG_124M model configuration
│   ├── generate_text.py       # generate_text_simple (greedy decoding)
│   └── core/
│       ├── gpt.py             # GPTModel: embeddings + transformer blocks + output head
│       ├── transformer.py     # TransformerBlock
│       ├── mha.py             # MultiHeadAttention (causal)
│       ├── ff.py              # FeedForward
│       ├── activation.py      # GELU
│       └── norm.py            # LayerNorm
└── tests/                     # pytest tests
```

## Model configuration

`GPT_CONFIG_124M` in [src/myllm/constants.py](src/myllm/constants.py):

| Parameter        | Value  | Meaning                           |
| ---------------- | ------ | --------------------------------- |
| `vocab_size`     | 50257  | GPT-2 BPE vocabulary size         |
| `context_length` | 1024   | Maximum number of input tokens    |
| `emb_dim`        | 768    | Embedding dimension               |
| `n_heads`        | 12     | Attention heads per block         |
| `n_layers`       | 12     | Number of transformer blocks      |
| `drop_rate`      | 0.1    | Dropout rate                      |
| `qkv_bias`       | False  | Bias in query/key/value projections |

## Usage

```python
import tiktoken
import torch

from myllm.constants import GPT_CONFIG_124M
from myllm.core.gpt import GPTModel
from myllm.generate_text import generate_text_simple

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # disable dropout

tokenizer = tiktoken.get_encoding("gpt2")
idx = torch.tensor(tokenizer.encode("Hello, I am")).unsqueeze(0)

out = generate_text_simple(
    model=model,
    idx=idx,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print(tokenizer.decode(out.squeeze(0).tolist()))
```

## Architecture overview

```
token ids ─► token embedding ─┐
                              ├─► + ─► dropout ─► N × TransformerBlock ─► LayerNorm ─► Linear ─► logits
position ids ─► pos embedding ┘

TransformerBlock:
  x ─► LayerNorm ─► MultiHeadAttention ─► dropout ─► + x
    ─► LayerNorm ─► FeedForward (Linear ► GELU ► Linear) ─► dropout ─► + x
```

## Attribution and license

This project is based on the code and ideas from Sebastian Raschka's book *Build a Large Language Model (From Scratch)*:

- Book: https://www.manning.com/books/build-a-large-language-model-from-scratch
- Code repository: https://github.com/rasbt/LLMs-from-scratch

Copyright 2023-2026 Sebastian Raschka. The original code is licensed under the Apache License 2.0, and this project is distributed under the same license; see [LICENSE.txt](LICENSE.txt). The attribution header is kept in every source file.
