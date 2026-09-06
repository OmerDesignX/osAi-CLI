"""Validated, dependency-free configuration models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .errors import ConfigurationError


class ModelFormat(str, Enum):
    MLX = "mlx"
    GGUF = "gguf"


DEFAULT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One quantization-preserving LoRA run.

    ``model`` may be an MLX directory or a GGUF first shard. Native GGUF
    backprop keeps that packed base frozen. ``companion_mlx`` is needed only
    when an MLX gradient run will export its learned adapter to GGUF.
    """

    model: Path
    data: Path
    output: Path
    format: ModelFormat | None = None
    companion_mlx: Path | None = None
    iterations: int = 10
    batch_size: int = 1
    max_seq_length: int = 128
    num_layers: int = 1
    rank: int = 4
    scale: float = 8.0
    dropout: float = 0.0
    learning_rate: float = 1e-5
    optimizer: str = "auto"
    gguf_batch_size: int = 8
    gguf_threads: int = 2
    seed: int = 0
    grad_checkpoint: bool = True
    grad_accumulation_steps: int = 1
    mask_prompt: bool = True
    save_every: int = 10
    steps_per_report: int = 1
    steps_per_eval: int = 10
    val_batches: int = 1
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGETS)
    strict_base_hash: bool = False
    merge_model: bool = True
    materialize_base: bool = True
    auto_settings: bool = False
    auto_profile: str | None = None
    memory_budget_bytes: int | None = None
    multi_gpu: str = "auto"
    devices: tuple[str, ...] = ()
    split_mode: str = "layer"
    tensor_split: tuple[float, ...] = ()
    main_gpu: int = 0
    distributed_workers: int = 0

    def __post_init__(self) -> None:
        for name in ("model", "data", "output", "companion_mlx"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if self.format is not None and not isinstance(self.format, ModelFormat):
            try:
                object.__setattr__(self, "format", ModelFormat(self.format))
            except ValueError as exc:
                raise ConfigurationError("format must be exactly 'mlx' or 'gguf'") from exc
        if not isinstance(self.target_modules, tuple):
            object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if not isinstance(self.devices, tuple):
            object.__setattr__(self, "devices", tuple(self.devices))
        if not isinstance(self.tensor_split, tuple):
            object.__setattr__(self, "tensor_split", tuple(self.tensor_split))
        self.validate()

    def validate(self) -> None:
        positive_ints = {
            "iterations": self.iterations,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "gguf_batch_size": self.gguf_batch_size,
            "gguf_threads": self.gguf_threads,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "save_every": self.save_every,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
        }
        invalid = [name for name, value in positive_ints.items() if value < 1]
        if invalid:
            raise ConfigurationError(f"values must be at least 1: {', '.join(invalid)}")
        if self.val_batches < -1:
            raise ConfigurationError("val_batches must be -1 or non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if self.scale <= 0:
            raise ConfigurationError("scale must be positive")
        if self.optimizer not in {"auto", "adam", "adamw", "sgd", "adafactor"}:
            raise ConfigurationError(
                "optimizer must be auto, adam, adamw, sgd, or adafactor"
            )
        for name in ("strict_base_hash", "merge_model", "materialize_base", "auto_settings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be true or false")
        if self.auto_profile not in {None, "compact", "balanced", "performance", "maximum"}:
            raise ConfigurationError("auto_profile is not a recognized hardware profile")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes < 1:
            raise ConfigurationError("memory_budget_bytes must be positive")
        if self.multi_gpu not in {"auto", "on", "off"}:
            raise ConfigurationError("multi_gpu must be auto, on, or off")
        if self.split_mode not in {"none", "layer", "row", "tensor"}:
            raise ConfigurationError("split_mode must be none, layer, row, or tensor")
        if self.main_gpu < 0 or self.distributed_workers < 0:
            raise ConfigurationError("main_gpu and distributed_workers cannot be negative")
        if any(not value.strip() or "," in value for value in self.devices):
            raise ConfigurationError("device names must be non-empty and cannot contain commas")
        if any(value <= 0 for value in self.tensor_split):
            raise ConfigurationError("tensor_split values must be positive")
        if not self.target_modules:
            raise ConfigurationError("target_modules cannot be empty")
        unsafe = [key for key in self.target_modules if not _safe_module_key(key)]
        if unsafe:
            raise ConfigurationError(
                "target module names may contain only letters, digits, underscores, and dots: "
                + ", ".join(unsafe)
            )

    @property
    def effective_format(self) -> ModelFormat:
        if self.format is not None:
            return self.format
        if self.model.is_file() and self.model.suffix.lower() == ".gguf":
            return ModelFormat.GGUF
        if self.model.is_dir() and (self.model / "config.json").exists():
            return ModelFormat.MLX
        raise ConfigurationError(
            f"cannot infer model format from {self.model}; pass format='mlx' or format='gguf'"
        )

    @property
    def training_model(self) -> Path:
        if self.effective_format is ModelFormat.MLX:
            return self.model
        if self.companion_mlx is None:
            raise ConfigurationError(
                "MLX-based GGUF adapter export requires companion_mlx: the matching "
                "quantized MLX checkpoint used to compute gradients"
            )
        return self.companion_mlx

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("model", "data", "output", "companion_mlx"):
            value = result[key]
            result[key] = str(value) if value is not None else None
        result["format"] = self.effective_format.value
        result["target_modules"] = list(self.target_modules)
        result["devices"] = list(self.devices)
        result["tensor_split"] = list(self.tensor_split)
        return result

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        data: str | Path | None = None,
        output: str | Path | None = None,
    ) -> TrainingConfig:
        config_path = Path(path)
        try:
            raw = config_path.read_bytes()
            if config_path.suffix.lower() == ".toml":
                values = tomllib.loads(raw.decode("utf-8"))
                values = values.get("training", values)
            elif config_path.suffix.lower() == ".json":
                values = json.loads(raw)
            else:
                raise ConfigurationError("configuration must be .toml or .json")
        except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from exc

        root = config_path.resolve().parent
        if data is not None:
            values["data"] = Path(data).expanduser().resolve()
        if output is not None:
            values["output"] = Path(output).expanduser().resolve()
        for key in ("model", "data", "output", "companion_mlx"):
            if values.get(key) is not None:
                candidate = Path(values[key]).expanduser()
                values[key] = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (root / candidate).resolve()
                )
        return cls(**values)


def _safe_module_key(value: str) -> bool:
    return bool(value) and all(c.isalnum() or c in "._" for c in value)
