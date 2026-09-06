"""Verify that an embedded MLX residual is numerically identical to its adapter."""

from __future__ import annotations

import gc
import json
from pathlib import Path


def run(base: str, adapter: str, merged: str) -> int:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    expected = _last_token_logits(mx, np, load, Path(base), Path(adapter))
    actual = _last_token_logits(mx, np, load, Path(merged), None)
    difference = np.abs(expected - actual)
    maximum = float(difference.max(initial=0.0))
    exact = bool(np.array_equal(expected, actual))
    print(
        json.dumps(
            {
                "exact": exact,
                "logits": int(expected.size),
                "max_absolute_difference": maximum,
            },
            sort_keys=True,
        )
    )
    if not exact:
        raise RuntimeError(
            "embedded MLX residual differs from base-plus-adapter "
            f"(maximum absolute logit difference: {maximum})"
        )
    return 0


def _last_token_logits(mx, np, load, model_path: Path, adapter_path: Path | None):
    model, tokenizer = load(
        str(model_path.resolve()),
        adapter_path=str(adapter_path.resolve()) if adapter_path else None,
    )
    tokens = list(tokenizer.encode("Hello"))
    if not tokens:
        raise RuntimeError("tokenizer returned no tokens for the fusion verification prompt")
    logits = model(mx.array([tokens]))[0, -1].astype(mx.float32)
    mx.eval(logits)
    snapshot = np.array(logits, copy=True)
    del logits, model, tokenizer
    mx.clear_cache()
    gc.collect()
    return snapshot
