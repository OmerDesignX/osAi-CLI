"""Convert MLX LoRA tensors directly to a llama.cpp GGUF adapter.

This deliberately never fuses or dequantizes the base checkpoint. Conversion is
limited to ordinary attention/MLP linear projections whose layouts are shared by
MLX LM and llama.cpp. Architecture-specific packed/reordered layers are rejected.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DependencyError, VerificationError
from .formats import ModelInspection
from .paths import llama_cpp_root

_ADAPTER_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[ab])$")
_SAFE_MODULE = re.compile(
    r"^(?:language_model\.)?model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


@dataclass(frozen=True, slots=True)
class GgufAdapterResult:
    path: Path
    architecture: str
    tensor_count: int
    size_bytes: int
    lora_alpha: float


def convert_mlx_adapter(
    adapter_dir: str | Path,
    output: str | Path,
    *,
    base_gguf: ModelInspection,
    dtype: str = "f16",
) -> GgufAdapterResult:
    adapter_root = Path(adapter_dir).expanduser().resolve()
    adapter_path = adapter_root / "adapters.safetensors"
    config_path = adapter_root / "adapter_config.json"
    if not adapter_path.is_file() or not config_path.is_file():
        raise VerificationError(
            "MLX adapter directory needs adapters.safetensors and "
            f"adapter_config.json: {adapter_root}"
        )
    if base_gguf.format.value != "gguf":
        raise VerificationError("GGUF adapter export requires a GGUF base inspection")
    if dtype not in {"f16", "f32"}:
        raise VerificationError("GGUF adapter dtype must be f16 or f32")

    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid MLX adapter config: {exc}") from exc
    lora = adapter_config.get("lora_parameters") or {}
    rank = int(lora.get("rank", 0))
    mlx_scale = float(lora.get("scale", 0))
    if rank < 1 or mlx_scale <= 0:
        raise VerificationError("adapter config has invalid rank or scale")
    # PEFT/llama.cpp applies alpha/rank. MLX LM applies `scale` directly.
    lora_alpha = mlx_scale * rank

    weights = _load_safetensors(adapter_path)
    pairs: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for name, tensor in weights.items():
        match = _ADAPTER_KEY.match(name)
        if not match:
            unexpected.append(name)
            continue
        module = match.group("module")
        if not _SAFE_MODULE.match(module):
            unexpected.append(name)
            continue
        pairs.setdefault(module, {})[match.group("side")] = tensor
    if unexpected:
        raise VerificationError(
            "GGUF export refuses tensors requiring architecture-specific transforms: "
            + ", ".join(sorted(unexpected))
        )
    incomplete = [module for module, values in pairs.items() if set(values) != {"a", "b"}]
    if incomplete or not pairs:
        raise VerificationError(
            "adapter contains missing LoRA A/B pairs: " + ", ".join(sorted(incomplete))
        )

    gguf = _import_gguf()
    architecture = base_gguf.architecture
    if architecture == "unknown":
        raise VerificationError("GGUF base architecture metadata could not be read")
    arch_enum = next(
        (enum for enum, name in gguf.MODEL_ARCH_NAMES.items() if name == architecture), None
    )
    if arch_enum is None:
        raise VerificationError(f"llama.cpp does not expose a tensor map for {architecture}")
    if base_gguf.block_count is None:
        raise VerificationError("GGUF base block count metadata could not be read")
    tensor_map = gguf.get_tensor_name_map(arch_enum, base_gguf.block_count)

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    tensor_count = 0
    try:
        writer = gguf.GGUFWriter(temporary, architecture)
        writer.add_name(destination.stem)
        writer.add_type(gguf.GGUFType.ADAPTER)
        writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
        writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha)
        np = _import_numpy()
        target_dtype = np.float16 if dtype == "f16" else np.float32
        for module, values in sorted(pairs.items()):
            source_name = module.removeprefix("language_model.") + ".weight"
            tensor_name = tensor_map.get_name(source_name, try_suffixes=(".weight",))
            if tensor_name is None:
                raise VerificationError(
                    f"cannot map MLX adapter module {module!r} for GGUF architecture {architecture}"
                )
            mlx_a = np.asarray(values["a"])
            mlx_b = np.asarray(values["b"])
            if mlx_a.ndim != 2 or mlx_b.ndim != 2:
                raise VerificationError(f"LoRA tensors must be matrices: {module}")
            if mlx_a.shape[1] != rank or mlx_b.shape[0] != rank:
                raise VerificationError(
                    f"adapter rank mismatch for {module}: "
                    f"A={mlx_a.shape}, B={mlx_b.shape}, rank={rank}"
                )
            # MLX stores A=(input, rank), B=(rank, output). llama.cpp stores
            # A=(rank, input), B=(output, rank), matching PEFT conventions.
            writer.add_tensor(f"{tensor_name}.lora_a", mlx_a.T.astype(target_dtype))
            writer.add_tensor(f"{tensor_name}.lora_b", mlx_b.T.astype(target_dtype))
            tensor_count += 2
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise

    result = _verify_adapter_file(destination, gguf, architecture, tensor_count, lora_alpha)
    return result


def _verify_adapter_file(path, gguf, architecture, tensor_count, lora_alpha):
    try:
        reader = gguf.GGUFReader(path)
        actual_type = reader.get_field("general.type").contents()
        actual_adapter = reader.get_field("adapter.type").contents()
        actual_arch = reader.get_field("general.architecture").contents()
        actual_alpha = float(reader.get_field("adapter.lora.alpha").contents())
    except (OSError, ValueError, AttributeError) as exc:
        raise VerificationError(f"cannot read generated GGUF adapter {path}: {exc}") from exc
    if actual_type != "adapter" or actual_adapter != "lora":
        raise VerificationError(f"invalid GGUF adapter metadata in {path}")
    if actual_arch != architecture or len(reader.tensors) != tensor_count:
        raise VerificationError(f"GGUF adapter architecture or tensor count mismatch in {path}")
    if abs(actual_alpha - lora_alpha) > 1e-5:
        raise VerificationError(f"GGUF adapter alpha mismatch in {path}")
    return GgufAdapterResult(
        path=path,
        architecture=architecture,
        tensor_count=tensor_count,
        size_bytes=path.stat().st_size,
        lora_alpha=lora_alpha,
    )


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors.numpy import load_file

        return load_file(path)
    except (ImportError, TypeError):
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise DependencyError("adapter conversion requires safetensors or MLX") from exc
        return {name: value for name, value in mx.load(str(path)).items()}


def _import_gguf():
    path = llama_cpp_root() / "gguf-py"
    if not path.is_dir():
        raise DependencyError(f"vendored llama.cpp gguf-py not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        import gguf
    except ImportError as exc:
        raise DependencyError("GGUF adapter export requires numpy and vendored gguf-py") from exc
    return gguf


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyError("GGUF adapter export requires numpy") from exc
    return np
