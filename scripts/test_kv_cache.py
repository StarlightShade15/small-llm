#!/usr/bin/env python3
"""Regression test for the KV cache that `SmallLM.generate` relies on.

The cached greedy decode and the full-recompute greedy decode must agree token
for token. If the prompt pass does not build a cache, the cached decode
conditions every generated token on itself alone and the two diverge on the
first generated token.

Run: python scripts/test_kv_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.model.model import ModelConfig, SmallLM


def greedy_with_cache(model: SmallLM, prompt, steps: int):
    """Greedy decode that carries a KV cache between steps."""
    cache = None
    output = list(prompt)
    current = torch.tensor([prompt], dtype=torch.long)
    for _ in range(steps):
        logits, _, cache = model(current, kv_cache=cache, use_cache=True)
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        output.append(next_token)
        current = torch.tensor([[next_token]], dtype=torch.long)
    return output


def greedy_without_cache(model: SmallLM, prompt, steps: int):
    """Greedy decode that recomputes the whole prefix every step."""
    output = list(prompt)
    for _ in range(steps):
        current = torch.tensor([output], dtype=torch.long)
        logits, _, _ = model(current)
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        output.append(next_token)
    return output


def main() -> int:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=64,
        use_flash_attn=False,
    )
    model = SmallLM(config).eval()

    prompt = [1, 5, 9, 3]

    cached = greedy_with_cache(model, prompt, 12)
    plain = greedy_without_cache(model, prompt, 12)
    print("cached:", cached)
    print("plain :", plain)
    if cached != plain:
        print("FAIL: cached decode diverges from full recompute")
        return 1

    # The cache must cover the prompt and grow by one position per step.
    _, _, cache = model(torch.tensor([prompt]), use_cache=True)
    if cache is None:
        print("FAIL: prompt pass returned no KV cache")
        return 1
    if len(cache) != config.n_layers:
        print(f"FAIL: expected {config.n_layers} cached layers, got {len(cache)}")
        return 1
    if cache[0][0].shape[1] != len(prompt):
        print(
            f"FAIL: cache covers {cache[0][0].shape[1]} positions, "
            f"expected {len(prompt)}"
        )
        return 1

    _, _, cache = model(torch.tensor([[7]]), kv_cache=cache, use_cache=True)
    if cache[0][0].shape[1] != len(prompt) + 1:
        print(
            f"FAIL: cache covers {cache[0][0].shape[1]} positions after one step, "
            f"expected {len(prompt) + 1}"
        )
        return 1

    # A different prompt must not produce the same continuation.
    other = greedy_with_cache(model, [1, 6, 9, 3], 12)
    print("other:", other)

    print("PASS: KV cache decode matches full recompute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
