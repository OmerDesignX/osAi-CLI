"""Strict inspection for the only supported model containers: MLX and GGUF."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ModelFormat
from .errors import DependencyError, ModelFormatError
from .paths import llama_cpp_root

_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
_UNQUANTIZED_GGUF_FILE_TYPES = {0, 1, 32}  # F32, F16, BF16
_FLOAT_GGML_TENSOR_TYPES = {0, 1, 30}  # F32, F16, BF16


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    scheme: str
    bits: int | None = None
    group_size: int | None = None
    mode: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInspection:
    format: ModelFormat
    path: Path
    architecture: str
    quantization: QuantizationSpec
    size_bytes: int
    shards: tuple[Path, ...]
    tensor_count: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    context_length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["format"] = self.format.value
        result["path"] = str(self.path)
        result["shards"] = [str(path) for path in self.shards]
        return result


def inspect_model(path: str | Path, expected: ModelFormat | None = None) -> ModelInspection:
    model_path = Path(path).expanduser().resolve()
    fusion_manifest = model_path / "osai_fusion.json"
    if model_path.is_dir() and fusion_manifest.is_file():
        try:
            fusion_format = json.loads(fusion_manifest.read_text(encoding="utf-8")).get("format")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelFormatError(f"invalid fusion manifest {fusion_manifest}: {exc}") from exc
        if fusion_format == ModelFormat.GGUF.value:
            from .fusion import resolve_gguf_fusion_bundle

            model_path = resolve_gguf_fusion_bundle(model_path).model
        elif fusion_format == ModelFormat.MLX.value:
            from .fusion import resolve_mlx_fusion_adapter

            resolve_mlx_fusion_adapter(model_path)
        else:
            raise ModelFormatError(f"unsupported fusion format in {fusion_manifest}")
    if expected is ModelFormat.GGUF or model_path.suffix.lower() == ".gguf":
        result = inspect_gguf(model_path)
    elif expected is ModelFormat.MLX or model_path.is_dir():
        result = inspect_mlx(model_path)
    else:
        raise ModelFormatError(f"{model_path} is neither a GGUF file nor an MLX model directory")
    if expected is not None and result.format is not expected:
        raise ModelFormatError(f"expected {expected.value}, found {result.format.value}")
    return result


def inspect_mlx(path: Path) -> ModelInspection:
    if not path.is_dir():
        raise ModelFormatError(f"MLX model directory does not exist: {path}")
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ModelFormatError(
            f"MLX directory must contain config.json and model.safetensors.index.json: {path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelFormatError(f"invalid MLX metadata in {path}: {exc}") from exc

    quant = config.get("quantization") or config.get("quantization_config")
    bits = quant.get("bits") if isinstance(quant, dict) else None
    if (
        not isinstance(bits, int)
        or isinstance(bits, bool)
        or not 2 <= bits <= 8
    ):
        raise ModelFormatError(
            "the MLX checkpoint is not weight-quantized; osai refuses full-precision bases"
        )

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelFormatError(f"missing weight_map in {index_path}")
    shard_names = sorted(set(weight_map.values()))
    shards = tuple(path / name for name in shard_names)
    _validate_materialized(shards, label="MLX")

    text_config = config.get("text_config", config)
    architectures = config.get("architectures") or [config.get("model_type", "unknown")]
    expected_size = index.get("metadata", {}).get("total_size")
    actual_size = sum(shard.stat().st_size for shard in shards)
    if isinstance(expected_size, int):
        # Safetensors files add a JSON header and alignment around the tensor
        # payload counted by the index. The files must contain at least that
        # payload, with tightly bounded container overhead.
        maximum_overhead = max(16 * 1024 * 1024, expected_size // 100)
        if not expected_size <= actual_size <= expected_size + maximum_overhead:
            raise ModelFormatError(
                "MLX shard bytes are inconsistent with the index: "
                f"payload={expected_size}, files={actual_size}"
            )

    return ModelInspection(
        format=ModelFormat.MLX,
        path=path,
        architecture=str(architectures[0]),
        quantization=QuantizationSpec(
            scheme="mlx-affine",
            bits=bits,
            group_size=quant.get("group_size"),
            mode=quant.get("mode", "affine"),
        ),
        size_bytes=actual_size,
        shards=shards,
        tensor_count=len(weight_map),
        block_count=_optional_int(text_config.get("num_hidden_layers")),
        embedding_length=_optional_int(text_config.get("hidden_size")),
        context_length=_optional_int(text_config.get("max_position_embeddings")),
    )


def inspect_gguf(path: Path) -> ModelInspection:
    if not path.is_file():
        raise ModelFormatError(f"GGUF file does not exist: {path}")
    shards = discover_gguf_shards(path)
    _validate_materialized(shards, label="GGUF")
    for shard in shards:
        with shard.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ModelFormatError(f"invalid GGUF magic (or unresolved LFS pointer): {shard}")

    architecture = "unknown"
    tensor_count = None
    block_count = None
    embedding_length = None
    context_length = None
    scheme = _quant_scheme_from_name(path.name)
    try:
        reader = _gguf_reader(path)
        _require_quantized_gguf(reader, path)
        architecture = _field(reader, "general.architecture", "unknown")
        tensor_count = _field(reader, "GGUF.tensor_count", None)
        block_count = _field(reader, f"{architecture}.block_count", None)
        embedding_length = _field(reader, f"{architecture}.embedding_length", None)
        context_length = _field(reader, f"{architecture}.context_length", None)
        file_type = _field(reader, "general.file_type", None)
        if scheme == "unknown" and file_type is not None:
            scheme = f"gguf-file-type-{file_type}"
    except DependencyError:
        # Magic, materialization, shard checks, and a quantized filename remain
        # useful in a dependency-light installation. Training installs gguf-py's
        # NumPy dependency and always performs the metadata-level check above.
        if scheme == "unknown":
            raise ModelFormatError(
                f"cannot verify GGUF quantization for {path}; install the 'gguf' extra"
            ) from None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelFormatError(f"invalid GGUF metadata in {path}: {exc}") from exc

    return ModelInspection(
        format=ModelFormat.GGUF,
        path=path,
        architecture=str(architecture),
        quantization=QuantizationSpec(scheme=scheme),
        size_bytes=sum(shard.stat().st_size for shard in shards),
        shards=shards,
        tensor_count=_optional_int(tensor_count),
        block_count=_optional_int(block_count),
        embedding_length=_optional_int(embedding_length),
        context_length=_optional_int(context_length),
    )


def discover_gguf_shards(path: Path) -> tuple[Path, ...]:
    match = _SHARD_RE.match(path.name)
    if not match:
        return (path,)
    count = int(match.group("count"))
    stem = match.group("stem")
    shards = tuple(
        path.with_name(f"{stem}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)
    )
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise ModelFormatError("missing GGUF shard(s): " + ", ".join(missing))
    return shards


def gguf_tensor_types(path: str | Path) -> dict[str, int]:
    """Return the exact GGML type for every tensor in one complete GGUF file."""
    model_path = Path(path).expanduser().resolve()
    try:
        reader = _gguf_reader(model_path)
        return {str(tensor.name): int(tensor.tensor_type) for tensor in reader.tensors}
    except DependencyError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelFormatError(f"cannot read GGUF tensor types from {model_path}: {exc}") from exc


def assert_compatible(mlx: ModelInspection, gguf: ModelInspection) -> None:
    if mlx.format is not ModelFormat.MLX or gguf.format is not ModelFormat.GGUF:
        raise ModelFormatError("compatibility comparison requires one MLX and one GGUF model")
    checks = (
        ("block count", mlx.block_count, gguf.block_count),
        ("embedding length", mlx.embedding_length, gguf.embedding_length),
        ("context length", mlx.context_length, gguf.context_length),
    )
    mismatches = [
        f"{name}: MLX={left}, GGUF={right}"
        for name, left, right in checks
        if left is not None and right is not None and left != right
    ]
    if mismatches:
        raise ModelFormatError(
            "companion MLX and GGUF architectures differ: " + "; ".join(mismatches)
        )
    mlx_name = mlx.architecture.lower().replace("_", "")
    gguf_name = gguf.architecture.lower().replace("_", "")
    if mlx_name != "unknown" and gguf_name != "unknown":
        compatible = ("qwen35" in mlx_name and "qwen35" in gguf_name) or mlx_name == gguf_name
        if not compatible:
            raise ModelFormatError(
                f"companion architectures differ: MLX={mlx.architecture}, GGUF={gguf.architecture}"
            )


def _validate_materialized(paths: tuple[Path, ...], label: str) -> None:
    for path in paths:
        if not path.is_file():
            raise ModelFormatError(f"missing {label} shard: {path}")
        with path.open("rb") as handle:
            prefix = handle.read(len(_LFS_PREFIX))
        if prefix == _LFS_PREFIX:
            raise ModelFormatError(
                f"{path} is a Git LFS pointer, not model data; materialize the weights first"
            )


def _gguf_reader(path: Path):
    gguf_python = llama_cpp_root() / "gguf-py"
    if not gguf_python.is_dir():
        raise DependencyError(f"vendored llama.cpp gguf-py not found at {gguf_python}")
    value = str(gguf_python)
    if value not in sys.path:
        sys.path.insert(0, value)
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise DependencyError(
            "GGUF metadata inspection requires numpy and vendored gguf-py"
        ) from exc
    return GGUFReader(path)


def _field(reader: Any, key: str, default: Any) -> Any:
    field = reader.get_field(key)
    return default if field is None else field.contents()


def _require_quantized_gguf(reader: Any, path: Path) -> None:
    if str(_field(reader, "general.type", "model")) == "adapter":
        return

    file_type = _optional_int(_field(reader, "general.file_type", None))
    if file_type is not None and (file_type & 1023) in _UNQUANTIZED_GGUF_FILE_TYPES:
        raise ModelFormatError(
            f"GGUF base is full precision (file type {file_type}), not quantized: {path}"
        )

    matrix_types = {
        int(tensor.tensor_type)
        for tensor in reader.tensors
        if len(tuple(tensor.shape)) >= 2
    }
    if not matrix_types or matrix_types.issubset(_FLOAT_GGML_TENSOR_TYPES):
        raise ModelFormatError(f"GGUF base has no quantized matrix tensors: {path}")


def _quant_scheme_from_name(name: str) -> str:
    match = re.search(r"-(Q(?:\d|I)[A-Z0-9_]+)(?:-|\.gguf)", name.upper())
    return match.group(1) if match else "unknown"


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
