"""Isolated, offline MLX-VLM LoRA runner used by the public backend."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _rows(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("multimodal dataset rows must be objects")
                result.append(value)
    return result


def _config_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        return config
    return dict(vars(config))


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _apply_lora(model: Any, config: dict[str, Any]) -> None:
    import mlx.nn as nn
    from mlx_vlm.trainer.lora import LoRaLayer
    from mlx_vlm.trainer.utils import freeze_model, set_module_by_name

    language = model.language_model
    freeze_model(model)
    candidates: list[tuple[str, Any, int]] = []
    targets = tuple(config["target_modules"])
    for name, module in language.named_modules():
        layer = _layer_index(name)
        if layer is None or not any(name.endswith(target) for target in targets):
            continue
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            candidates.append((name, module, layer))
    if not candidates:
        raise ValueError(
            "none of the requested LoRA projections exist in this VLM language model: "
            + ", ".join(targets)
        )
    selected_layers = sorted({layer for _, _, layer in candidates})[-config["num_layers"] :]
    for name, module, layer in candidates:
        if layer in selected_layers:
            set_module_by_name(
                language,
                name,
                LoRaLayer(
                    module,
                    config["rank"],
                    config["scale"] * config["rank"],
                    config["dropout"],
                ),
            )
    model.config.lora = {
        "rank": config["rank"],
        "alpha": config["scale"] * config["rank"],
        "dropout": config["dropout"],
        "keys": list(targets),
        "num_layers": len(selected_layers),
    }


def _assistant_id(processor: Any, configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    tokenizer = getattr(processor, "tokenizer", processor)
    for name in ("assistant_token_id", "assistant_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and value >= 0:
            return value
    for token in getattr(tokenizer, "additional_special_tokens", ()) or ():
        if "assistant" not in str(token).casefold():
            continue
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError(
        "this processor does not expose a distinct assistant token; pass "
        "--assistant-token-id or use --no-mask-prompt"
    )


def _optimizer(name: str, learning_rate: float):
    import mlx.optimizers as optim

    if name == "sgd":
        return optim.SGD(learning_rate=learning_rate)
    if name == "adam":
        return optim.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return optim.AdamW(learning_rate=learning_rate)
    raise ValueError(f"unsupported MLX-VLM optimizer: {name}")


def run(config_path: str | Path) -> int:
    import mlx.core as mx
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, evaluate, train
    from mlx_vlm.trainer.utils import not_supported_for_training, print_trainable_parameters
    from mlx_vlm.utils import load

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    mx.random.seed(config["seed"])
    model, processor = load(
        config["model"], processor_config={"trust_remote_code": False}
    )
    model_config = _config_dict(model)
    model_type = model_config.get("model_type")
    if model_type in not_supported_for_training:
        raise ValueError(f"{model_type} audio/vision backprop is not supported by MLX-VLM")
    resize = config.get("image_resize_shape")
    training = VisionDataset(
        _rows(Path(config["data"]) / "train.jsonl"),
        model_config,
        processor,
        image_resize_shape=resize,
    )
    if not len(training):
        raise ValueError("multimodal training dataset has no rows")
    first = training[0]
    first_length = int(first["input_ids"].shape[-1])
    if first_length > config["max_seq_length"]:
        raise ValueError(
            f"--max-seq-length {config['max_seq_length']} is too small for the first "
            f"media example ({first_length} tokens); increase the context limit or "
            "reduce the image/video resolution or frame count"
        )
    valid_path = Path(config["data"]) / "valid.jsonl"
    test_path = Path(config["data"]) / "test.jsonl"
    validation = (
        VisionDataset(_rows(valid_path), model_config, processor, image_resize_shape=resize)
        if valid_path.is_file()
        else None
    )
    test = (
        VisionDataset(_rows(test_path), model_config, processor, image_resize_shape=resize)
        if test_path.is_file()
        else None
    )
    _apply_lora(model, config)
    print_trainable_parameters(model)
    assistant_id = (
        _assistant_id(processor, config.get("assistant_token_id"))
        if config["mask_prompt"]
        else 0
    )
    args = TrainingArgs(
        batch_size=config["batch_size"],
        iters=config["iters"],
        val_batches=config["val_batches"],
        steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"],
        steps_per_save=config["save_every"],
        max_seq_length=config["max_seq_length"],
        adapter_file=str(Path(config["adapter_path"]) / "adapters.safetensors"),
        grad_checkpoint=config["grad_checkpoint"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["grad_accumulation_steps"],
    )
    train(
        model,
        _optimizer(config["optimizer"], config["learning_rate"]),
        training,
        validation,
        args,
        train_on_completions=config["mask_prompt"],
        assistant_id=assistant_id,
    )
    if test is not None:
        loss = evaluate(
            model,
            test,
            config["batch_size"],
            config["val_batches"],
            config["max_seq_length"],
            train_on_completions=config["mask_prompt"],
            assistant_id=assistant_id,
        )
        print(f"Test loss {loss:.8f}", flush=True)
    return 0


def verify(config_path: str | Path) -> int:
    import mlx.core as mx
    from mlx_vlm.utils import load

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model, _ = load(config["model"], processor_config={"trust_remote_code": False})
    _apply_lora(model, config)
    model.load_weights(
        str(Path(config["model"]) / "osai_adapter" / "adapters.safetensors"),
        strict=False,
    )
    mx.eval(model.trainable_parameters())
    print("VLM fusion adapter loaded", flush=True)
    return 0
