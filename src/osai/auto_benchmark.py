"""Bounded local inference probes for automatic training settings.

An inference pass verifies that the selected model and accelerator can load at
the proposed context and microbatch. The training profile also retains a large
memory reserve because inference does not allocate backward graphs or optimizer
state. A successful probe cannot guarantee that other processes will not consume
memory later.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .auto_settings import AutoTrainingSettings, select_auto_settings
from .errors import ConfigurationError, DependencyError
from .formats import ModelInspection
from .hardware import Accelerator, Engine, select_llama_accelerator
from .multi_gpu import available_llama_devices, llama_device_free_bytes
from .offline import offline_environment
from .paths import llama_binary, llama_runtime_build

_PROFILES = ("maximum", "performance", "balanced", "compact")
_CACHE_AGE_SECONDS = 12 * 60 * 60
_PROBE_TIMEOUT_SECONDS = 180


def _probe_timeout() -> int:
    # A cold model load can exceed 25 seconds even when the tiny pilot is fast.
    # Bound a stuck inference process without rejecting healthy calibrations.
    return _PROBE_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    settings: AutoTrainingSettings
    engine: str
    accelerator: str
    devices: tuple[str, ...]
    elapsed_seconds: float
    cached: bool = False

    def as_dict(self) -> dict:
        return {
            "settings": self.settings.as_dict(),
            "engine": self.engine,
            "accelerator": self.accelerator,
            "devices": list(self.devices),
            "elapsed_seconds": self.elapsed_seconds,
            "cached": self.cached,
        }


def benchmark_auto_settings(
    model: ModelInspection,
    *,
    engine: Engine,
    accelerator: str = "auto",
    multi_gpu: str = "auto",
    devices: tuple[str, ...] = (),
    adapter: Path | None = None,
    force: bool = False,
    required_context: int | None = None,
) -> BenchmarkResult:
    """Find the highest memory-bounded profile that passes local inference.

    The probe never reads a dataset. For data-parallel llama training every
    selected GPU must load the entire model, so each is checked separately.
    """

    if multi_gpu not in {"auto", "on", "off"}:
        raise ConfigurationError("multi_gpu must be auto, on, or off")
    if engine is Engine.LLAMA_CPP:
        selected = select_llama_accelerator(accelerator)
        binary = llama_binary("llama-completion")
        if binary is None:
            raise DependencyError("llama-completion is missing; run `osai build-llama`")
        discovered = available_llama_devices(binary, selected)
        chosen = devices or discovered
        if selected is Accelerator.CPU:
            chosen = ()
        elif multi_gpu == "off":
            chosen = chosen[:1]
        if multi_gpu == "on" and len(chosen) < 2:
            raise ConfigurationError(
                "multi-GPU was requested but fewer than two devices were found"
            )
        if devices and any(device not in discovered for device in devices):
            raise ConfigurationError("a selected GPU is unavailable to llama.cpp")
        if selected is not Accelerator.CPU and not chosen:
            raise ConfigurationError("the selected GPU backend reports no usable devices")
        free_by_device = llama_device_free_bytes(binary, selected)
        free_gpu_bytes = (
            min(free_by_device[device] for device in chosen)
            if chosen and all(device in free_by_device for device in chosen)
            else None
        )
        executable = binary
    elif engine is Engine.MLX:
        selected = Accelerator.CPU if accelerator == "cpu" else Accelerator.AUTO
        chosen = ()
        free_gpu_bytes = None
        executable = Path(sys.executable)
    else:
        raise ConfigurationError("automatic benchmark requires a resolved engine")

    baseline = select_auto_settings(model, engine=engine)
    if required_context is not None and required_context < 32:
        raise ConfigurationError("the required training context must be at least 32 tokens")
    adapter_file = (
        adapter / "adapters.safetensors" if adapter is not None and adapter.is_dir() else adapter
    )
    key_data = {
        "schema": 2,
        "required_context": required_context,
        "model": str(model.path.resolve()),
        "shards": [
            (str(shard), shard.stat().st_size, shard.stat().st_mtime_ns) for shard in model.shards
        ],
        "engine": engine.value,
        "accelerator": selected.value,
        "devices": chosen,
        "free_gpu_gib": free_gpu_bytes // 1024**3 if free_gpu_bytes is not None else None,
        "multi_gpu": multi_gpu,
        "memory": baseline.physical_memory_bytes,
        "cpu": os.cpu_count(),
        "binary": (str(executable), executable.stat().st_mtime_ns),
        "adapter": (
            (str(adapter_file), adapter_file.stat().st_size, adapter_file.stat().st_mtime_ns)
            if adapter_file is not None
            else None
        ),
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    cache = llama_runtime_build().parent / "auto-benchmarks" / f"{digest}.json"
    if not force:
        cached = _read_cache(cache, model, engine, required_context)
        if cached is not None and _gpu_profile_fits(
            cached.profile, model.size_bytes, free_gpu_bytes
        ):
            return BenchmarkResult(cached, engine.value, selected.value, chosen, 0.0, True)

    started = time.monotonic()
    failures: list[str] = []
    for profile in _PROFILES[_PROFILES.index(baseline.profile) :]:
        settings = select_auto_settings(model, engine=engine, profile_limit=profile)
        if required_context is not None:
            settings = _fit_required_context(settings, required_context)
        if not _gpu_profile_fits(settings.profile, model.size_bytes, free_gpu_bytes):
            failures.append(f"{profile}: insufficient free GPU memory for training reserve")
            continue
        try:
            if engine is Engine.LLAMA_CPP:
                for device in chosen or ("",):
                    _probe_llama(executable, model.path, settings, selected, device, adapter)
            else:
                _probe_mlx(model.path, settings, adapter)
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            failures.append(f"{profile}: {exc}")
            continue
        result = BenchmarkResult(
            settings, engine.value, selected.value, chosen, time.monotonic() - started
        )
        temporary = cache.with_name(f"{cache.stem}-{os.getpid()}-{time.time_ns()}.tmp")
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps({"created": time.time(), "settings": settings.as_dict()}),
                encoding="utf-8",
            )
            os.replace(temporary, cache)
        except OSError:
            # A read-only runtime cache must not turn a successful probe into a failure.
            pass
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        return result
    raise ConfigurationError(
        "inference benchmark could not load this model at any safe profile: "
        + "; ".join(failures)[-1000:]
    )


def _fit_required_context(
    settings: AutoTrainingSettings, required_context: int
) -> AutoTrainingSettings:
    """Spend the fitted memory budget on context before batch or adapter size."""
    pressure = max(1.0, required_context / settings.max_seq_length)
    return replace(
        settings,
        max_seq_length=required_context,
        batch_size=max(1, settings.batch_size // math.ceil(pressure)),
        rank=max(2, int(settings.rank / math.sqrt(pressure))),
        num_layers=max(1, int(settings.num_layers / math.sqrt(pressure))),
        gguf_batch_size=max(1, int(settings.gguf_batch_size / math.sqrt(pressure))),
    )


def _gpu_profile_fits(profile: str, model_bytes: int, free_bytes: int | None) -> bool:
    if free_bytes is None:
        return True
    training_reserve = {
        "compact": 2 * 1024**3,
        "balanced": 3 * 1024**3,
        "performance": 6 * 1024**3,
        "maximum": 10 * 1024**3,
    }[profile]
    system_reserve = max(512 * 1024**2, free_bytes // 10)
    return model_bytes + training_reserve + system_reserve <= free_bytes


def _probe_llama(
    binary: Path,
    model: Path,
    settings: AutoTrainingSettings,
    accelerator: Accelerator,
    device: str,
    adapter: Path | None,
) -> None:
    prompt = "A short local hardware benchmark. " * min(64, max(4, settings.max_seq_length // 24))
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        "1",
        "--no-conversation",
        "--single-turn",
        "-c",
        str(settings.max_seq_length),
        "-b",
        str(settings.gguf_batch_size),
        "-ub",
        str(settings.gguf_batch_size),
        "-t",
        str(settings.gguf_threads),
        "--temp",
        "0",
        "-fit",
        "off",
    ]
    if accelerator is Accelerator.CPU:
        command += ["-dev", "none", "-ngl", "0", "--no-op-offload"]
    else:
        command += ["-dev", device, "-ngl", "auto"]
    if adapter is not None:
        command += ["--lora", str(adapter)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_probe_timeout(),
        check=False,
        env=offline_environment(),
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr + "\n" + completed.stdout)[-500:].strip())


def _probe_mlx(model: Path, settings: AutoTrainingSettings, adapter: Path | None) -> None:
    prompt = "A short local hardware benchmark. " * min(64, max(4, settings.max_seq_length // 24))
    command = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--max-tokens",
        "1",
        "--temp",
        "0",
    ]
    if adapter is not None:
        command += ["--adapter-path", str(adapter)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_probe_timeout(),
        check=False,
        env=offline_environment(),
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr + "\n" + completed.stdout)[-500:].strip())


def _read_cache(
    cache: Path, model: ModelInspection, engine: Engine, required_context: int | None
) -> AutoTrainingSettings | None:
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - payload["created"] > _CACHE_AGE_SECONDS:
            return None
        settings = payload["settings"]
        result = select_auto_settings(model, engine=engine, profile_limit=settings["profile"])
        if required_context is not None:
            result = _fit_required_context(result, required_context)
        if asdict(result) != {**settings, "target_modules": tuple(settings["target_modules"])}:
            return None
        return result
    except (OSError, KeyError, TypeError, ValueError):
        return None
