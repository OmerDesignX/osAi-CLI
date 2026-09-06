"""Dependency-light accelerator and execution-engine selection."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


class Engine(str, Enum):
    AUTO = "auto"
    MLX = "mlx"
    LLAMA_CPP = "llama.cpp"


class Accelerator(str, Enum):
    AUTO = "auto"
    METAL = "metal"
    MPS = "mps"
    CUDA = "cuda"
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class HardwareReport:
    os: str
    architecture: str
    metal: bool
    mps: bool
    cuda: bool
    vulkan: bool
    cpu: bool
    recommended_engine: str
    recommended_llama_accelerator: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware() -> HardwareReport:
    system_name = platform.system()
    machine = platform.machine().lower()
    apple_silicon = system_name == "Darwin" and machine in {"arm64", "aarch64"}
    metal = system_name == "Darwin" and macos_version_at_least(12)
    mps = _mps_available()
    cuda = shutil.which("nvcc") is not None
    vulkan = _vulkan_build_available()
    mlx_ready = (
        apple_silicon
        and macos_version_at_least(14)
        and _module_importable("mlx.core")
        and _module_importable("mlx_lm")
    )
    if metal:
        llama_accelerator = Accelerator.METAL
    elif cuda:
        llama_accelerator = Accelerator.CUDA
    elif vulkan:
        llama_accelerator = Accelerator.VULKAN
    else:
        llama_accelerator = Accelerator.CPU
    return HardwareReport(
        os=system_name,
        architecture=platform.machine(),
        metal=metal,
        mps=mps,
        cuda=cuda,
        vulkan=vulkan,
        cpu=True,
        recommended_engine=(Engine.MLX if mlx_ready else Engine.LLAMA_CPP).value,
        recommended_llama_accelerator=llama_accelerator.value,
    )


def select_engine(requested: str | Engine, report: HardwareReport | None = None) -> Engine:
    try:
        engine = requested if isinstance(requested, Engine) else Engine(requested)
    except ValueError as exc:
        raise ConfigurationError("engine must be auto, mlx, or llama.cpp") from exc
    if engine is not Engine.AUTO:
        return engine
    detected = report or detect_hardware()
    return Engine(detected.recommended_engine)


def select_llama_accelerator(
    requested: str | Accelerator, report: HardwareReport | None = None
) -> Accelerator:
    try:
        accelerator = (
            requested if isinstance(requested, Accelerator) else Accelerator(requested)
        )
    except ValueError as exc:
        raise ConfigurationError(
            "accelerator must be auto, metal, mps, cuda, vulkan, or cpu"
        ) from exc
    detected = report or detect_hardware()
    if accelerator is Accelerator.AUTO:
        return Accelerator(detected.recommended_llama_accelerator)
    if accelerator is Accelerator.MPS:
        raise ConfigurationError(
            "MPS is a PyTorch API, not a llama.cpp backend; use Metal on macOS. "
            "MPS availability is still reported for diagnostics."
        )
    available = {
        Accelerator.METAL: detected.metal,
        Accelerator.CUDA: detected.cuda,
        Accelerator.VULKAN: detected.vulkan,
        Accelerator.CPU: True,
    }
    if not available[accelerator]:
        raise ConfigurationError(f"requested accelerator is unavailable: {accelerator.value}")
    return accelerator


def _module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _mps_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _vulkan_build_available() -> bool:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk and Path(sdk).is_dir():
        return True
    return any(shutil.which(command) for command in ("glslc", "glslangValidator", "vulkaninfo"))


def macos_version_at_least(minimum_major: int) -> bool:
    """Return whether this is macOS at or above the requested major version."""

    if platform.system() != "Darwin":
        return False
    version = platform.mac_ver()[0]
    try:
        return int(version.split(".", 1)[0]) >= minimum_major
    except (ValueError, IndexError):
        return False
