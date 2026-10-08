"""Fuse mlx-vlm's LoRaLayer weights into a standalone quantized VLM."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def fuse(model_path: Path, adapter_path: Path, output: Path) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_vlm.trainer.lora import LoRaLayer
    from mlx_vlm.utils import load, save_weights

    model, _processor = load(str(model_path), adapter_path=str(adapter_path))
    fused = []
    for name, layer in model.named_modules():
        if not isinstance(layer, LoRaLayer):
            continue
        linear = layer.original_layer
        quantized = isinstance(linear, nn.QuantizedLinear)
        weight = (
            mx.dequantize(
                linear.weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
            if quantized
            else linear.weight
        )
        out_features, in_features = weight.shape
        has_bias = "bias" in linear
        replacement = nn.Linear(in_features, out_features, bias=has_bias)
        delta = (layer.A @ layer.B).T * layer.scale
        replacement.weight = weight + delta.astype(weight.dtype)
        if has_bias:
            replacement.bias = linear.bias
        if quantized:
            replacement = nn.QuantizedLinear.from_linear(
                replacement, linear.group_size, linear.bits, mode=linear.mode
            )
        fused.append((name, replacement))
    if not fused:
        raise ValueError("No mlx-vlm LoRA layers were found to fuse")
    model.update_modules(tree_unflatten(fused))
    output.mkdir(parents=True)
    for source in model_path.rglob("*"):
        if not source.is_file() or ".git" in source.parts:
            continue
        relative = source.relative_to(model_path)
        if (
            relative.parts[0] in {"osai_adapter", "merged"}
            or source.suffix == ".safetensors"
            or source.name == "model.safetensors.index.json"
        ):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    save_weights(output, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fuse(args.model, args.adapter, args.output)


if __name__ == "__main__":
    main()
