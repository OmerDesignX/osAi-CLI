"""Generate current-policy answers behind osAi's network socket guard."""

from __future__ import annotations

import json
from pathlib import Path


def run(config_path: str | Path) -> int:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from .io import atomic_json

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model, tokenizer = load(
        config["model"],
        adapter_path=config["adapter"],
        trust_remote_code=False,
    )
    model.eval()
    results = []
    for request in config["requests"]:
        mx.random.seed(int(request["seed"]))
        prompt = _format_prompt(tokenizer, request["prompt"])
        response = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=int(config["max_tokens"]),
            sampler=make_sampler(
                temp=float(config["temperature"]),
                top_p=float(config["top_p"]),
            ),
            verbose=False,
        )
        results.append(
            {
                "source_index": int(request["source_index"]),
                "sample_index": int(request["sample_index"]),
                "seed": int(request["seed"]),
                "response": str(response).strip(),
            }
        )
        print(
            "rollout_source="
            f"{request['source_index']} sample={request['sample_index']} "
            f"characters={len(str(response).strip())}",
            flush=True,
        )
    atomic_json(
        config["output"],
        {"schema_version": 1, "local_only": True, "results": results},
    )
    return 0


def _format_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (AttributeError, TypeError, ValueError):
        return f"user: {prompt}\nassistant:"
