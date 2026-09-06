"""Conservative hardware-aware training profiles."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from .config import DEFAULT_TARGETS
from .errors import ConfigurationError
from .formats import ModelInspection
from .hardware import Engine
from .system import physical_memory_bytes

_GIB = 1024**3
_COMPACT_TARGETS = ("mlp.down_proj",)
_BALANCED_TARGETS = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True, slots=True)
class AutoTrainingSettings:
    """Resolved settings plus the memory assumptions behind them."""

    profile: str
    physical_memory_bytes: int
    memory_budget_bytes: int
    model_size_bytes: int
    batch_size: int
    max_seq_length: int
    num_layers: int
    rank: int
    gguf_batch_size: int
    gguf_threads: int
    target_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result


def select_auto_settings(
    model: ModelInspection,
    *,
    engine: str | Engine,
    memory_bytes: int | None = None,
    cpu_count: int | None = None,
) -> AutoTrainingSettings:
    """Choose the largest conservative preset for the model and host RAM.

    The budget reserves 25% of physical memory for the OS, runtime, mapped
    libraries, and transient buffers. Manual CLI values can override any field
    after this profile is selected.
    """

    selected_engine = engine if isinstance(engine, Engine) else Engine(engine)
    if selected_engine is Engine.AUTO:
        raise ConfigurationError("automatic settings require a resolved training engine")
    total = memory_bytes if memory_bytes is not None else physical_memory_bytes()
    if total is None or total <= 0:
        raise ConfigurationError(
            "automatic settings could not detect physical RAM; use manual training settings"
        )
    budget = int(total * 0.75)
    if model.size_bytes > total * 0.55:
        raise ConfigurationError(
            f"automatic settings reject a {model.size_bytes / _GIB:.2f} GiB quantized model "
            f"on a {total / _GIB:.2f} GiB host; the model alone exceeds the 55% RAM safety "
            "limit, so use a smaller quantized model or a host with more memory"
        )

    headroom = budget - model.size_bytes
    if total <= 10 * _GIB or headroom < 4 * _GIB:
        profile = "compact"
    elif total <= 20 * _GIB or headroom < 8 * _GIB:
        profile = "balanced"
    elif total <= 40 * _GIB or headroom < 16 * _GIB:
        profile = "performance"
    else:
        profile = "maximum"

    profiles = {
        "compact": (1, 64, 1, 2, 8, 2, _COMPACT_TARGETS),
        "balanced": (1, 128, 2, 4, 8, 4, _BALANCED_TARGETS),
        "performance": (2, 256, 4, 8, 16, 8, DEFAULT_TARGETS),
        "maximum": (4, 1024, 8, 16, 32, 16, DEFAULT_TARGETS),
    }
    batch_size, context, layers, rank, gguf_batch, threads, targets = profiles[profile]

    if selected_engine is Engine.LLAMA_CPP:
        backprop_profiles = {
            "compact": (64, 1, 1, 8, _COMPACT_TARGETS),
            "balanced": (128, 1, 2, 8, _COMPACT_TARGETS),
            "performance": (128, 2, 4, 16, _BALANCED_TARGETS),
            "maximum": (256, 4, 8, 16, _BALANCED_TARGETS),
        }
        context, layers, rank, gguf_batch, targets = backprop_profiles[profile]

    if model.context_length is not None:
        context = min(context, model.context_length)
    context = max(32, context)
    if model.block_count is not None:
        layers = min(layers, model.block_count)
    layers = max(1, layers)
    workers = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    threads = min(threads, workers)
    gguf_batch = min(gguf_batch, context)
    while context % gguf_batch:
        gguf_batch -= 1

    return AutoTrainingSettings(
        profile=profile,
        physical_memory_bytes=total,
        memory_budget_bytes=budget,
        model_size_bytes=model.size_bytes,
        batch_size=batch_size,
        max_seq_length=context,
        num_layers=layers,
        rank=rank,
        gguf_batch_size=gguf_batch,
        gguf_threads=threads,
        target_modules=tuple(targets),
    )
