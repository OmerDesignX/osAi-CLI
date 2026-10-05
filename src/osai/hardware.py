"""Dependency-light accelerator and execution-engine selection."""

from __future__ import annotations

import ctypes.util
import importlib.util
import json
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
    compiled_llama_accelerator: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware() -> HardwareReport:
    system_name = platform.system()
    machine = platform.machine().lower()
    apple_silicon = system_name == "Darwin" and machine in {"arm64", "aarch64"}
    metal = system_name == "Darwin" and macos_version_at_least(12)
    mps = _mps_available()
    cuda = system_name in {"Windows", "Linux"} and _nvidia_driver_available()
    vulkan = _vulkan_runtime_available()
    compiled = _compiled_llama_accelerator()
    if (system_name == "Darwin" and compiled in {"cuda", "vulkan", "cuda+vulkan"}) or (
        system_name != "Darwin" and compiled == "metal"
    ):
        compiled = None
    mlx_ready = (
        apple_silicon
        and macos_version_at_least(14)
        and _module_importable("mlx.core")
        and _module_importable("mlx_lm")
    )
    compiled_backends = set(compiled.split("+")) if compiled else set()
    if compiled == "cpu":
        llama_accelerator = Accelerator.CPU
    elif "metal" in compiled_backends and metal:
        llama_accelerator = Accelerator.METAL
    elif "cuda" in compiled_backends and cuda:
        llama_accelerator = Accelerator.CUDA
    elif "vulkan" in compiled_backends and vulkan:
        llama_accelerator = Accelerator.VULKAN
    elif compiled is None and metal:
        llama_accelerator = Accelerator.METAL
    elif compiled is None and cuda:
        llama_accelerator = Accelerator.CUDA
    elif compiled is None and vulkan:
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
        compiled_llama_accelerator=compiled,
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
    requested: str | Accelerator,
    report: HardwareReport | None = None,
    *,
    for_build: bool = False,
) -> Accelerator:
    try:
        accelerator = requested if isinstance(requested, Accelerator) else Accelerator(requested)
    except ValueError as exc:
        raise ConfigurationError(
            "accelerator must be auto, metal, mps, cuda, vulkan, or cpu"
        ) from exc
    detected = report or detect_hardware()
    if accelerator is Accelerator.AUTO:
        if for_build:
            for candidate, ready in (
                (Accelerator.METAL, detected.metal),
                (Accelerator.CUDA, detected.cuda and _cuda_build_available()),
                (Accelerator.VULKAN, detected.vulkan and _vulkan_build_available()),
            ):
                if ready:
                    return candidate
            return Accelerator.CPU
        return Accelerator(detected.recommended_llama_accelerator)
    if accelerator is Accelerator.MPS:
        raise ConfigurationError(
            "MPS is a PyTorch API, not a llama.cpp backend; use Metal on macOS. "
            "MPS availability is still reported for diagnostics."
        )
    available = {
        Accelerator.METAL: detected.metal,
        Accelerator.CUDA: (detected.cuda or _cuda_build_available())
        if for_build
        else detected.cuda,
        Accelerator.VULKAN: (detected.vulkan or _vulkan_build_available())
        if for_build
        else detected.vulkan,
        Accelerator.CPU: True,
    }
    if not available[accelerator]:
        raise ConfigurationError(f"requested accelerator is unavailable: {accelerator.value}")
    if (
        not for_build
        and accelerator is not Accelerator.CPU
        and detected.compiled_llama_accelerator is not None
        and accelerator.value not in detected.compiled_llama_accelerator.split("+")
    ):
        raise ConfigurationError(
            f"llama.cpp is built for {detected.compiled_llama_accelerator}, not "
            f"{accelerator.value}; run `osai build-llama --accelerator {accelerator.value}`"
        )
    return accelerator


def _compiled_llama_accelerator() -> str | None:
    from .paths import llama_build_is_current, llama_runtime_build, project_root

    root = project_root()
    candidates = (
        llama_runtime_build() / "OSAI_BUILD.json",
        root / "vendor" / "llama.cpp" / "build" / "OSAI_BUILD.json",
    )
    for path in candidates:
        if not llama_build_is_current(path.parent):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = data.get("llamaAccelerator")
        if value in {"cpu", "metal", "cuda", "vulkan", "cuda+vulkan"}:
            return value
    return None


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
    if sdk:
        compiler = Path(sdk) / ("Bin/glslc.exe" if os.name == "nt" else "bin/glslc")
        if compiler.is_file():
            return True
    # Linux distributions usually install Vulkan headers, loader, and shader
    # compiler into the system prefix rather than defining VULKAN_SDK.
    return os.name != "nt" and shutil.which("glslc") is not None


def _cuda_build_available() -> bool:
    sdk = os.environ.get("CUDA_PATH")
    if shutil.which("nvcc"):
        return True
    compiler = Path(sdk or "") / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    return bool(sdk and compiler.is_file())


def _nvidia_driver_available() -> bool:
    if shutil.which("nvidia-smi") is not None:
        return True
    if os.name == "nt":
        return (
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        ).is_file()
    return False


def _vulkan_runtime_available() -> bool:
    if os.name == "nt":
        return (
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "vulkan-1.dll"
        ).is_file()
    return ctypes.util.find_library("vulkan") is not None


def macos_version_at_least(minimum_major: int) -> bool:
    """Return whether this is macOS at or above the requested major version."""

    if platform.system() != "Darwin":
        return False
    version = platform.mac_ver()[0]
    try:
        return int(version.split(".", 1)[0]) >= minimum_major
    except (ValueError, IndexError):
        return False
