"""Portable host and dependency diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hardware import detect_hardware, macos_version_at_least
from .paths import llama_binary, llama_cpp_root, project_root


@dataclass(frozen=True, slots=True)
class SystemReport:
    os: str
    os_version: str
    architecture: str
    python: str
    physical_memory_bytes: int | None
    platform_supported: bool
    platform_support: str
    mlx_platform_supported: bool
    mlx_importable: bool
    mlx_lm_importable: bool
    cmake: str | None
    llama_cpp_source: str | None
    llama_cli: str | None
    llama_finetune: str | None
    llama_perplexity: str | None
    metal_available: bool
    mps_available: bool
    cuda_available: bool
    vulkan_available: bool
    recommended_engine: str
    recommended_llama_accelerator: str
    local_only: bool
    project_root: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def doctor() -> SystemReport:
    system_name = platform.system()
    machine = platform.machine().lower()
    platform_supported, platform_note = _platform_support(system_name)
    mlx_supported = (
        system_name == "Linux"
        or (
            system_name == "Darwin"
            and machine in {"arm64", "aarch64"}
            and macos_version_at_least(14)
        )
    )
    hardware = detect_hardware()
    return SystemReport(
        os=system_name,
        os_version=platform.platform(),
        architecture=platform.machine(),
        python=platform.python_version(),
        physical_memory_bytes=physical_memory_bytes(),
        platform_supported=platform_supported,
        platform_support=platform_note,
        mlx_platform_supported=mlx_supported,
        mlx_importable=_module_importable("mlx.core"),
        mlx_lm_importable=_module_importable("mlx_lm"),
        cmake=shutil.which("cmake"),
        llama_cpp_source=str(llama_cpp_root()) if llama_cpp_root().is_dir() else None,
        llama_cli=_path_or_none(llama_binary("llama-cli")),
        llama_finetune=_path_or_none(llama_binary("llama-finetune")),
        llama_perplexity=_path_or_none(llama_binary("llama-perplexity")),
        metal_available=hardware.metal,
        mps_available=hardware.mps,
        cuda_available=hardware.cuda,
        vulkan_available=hardware.vulkan,
        recommended_engine=hardware.recommended_engine,
        recommended_llama_accelerator=hardware.recommended_llama_accelerator,
        local_only=True,
        project_root=str(project_root()),
    )


def physical_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(value)
        except (OSError, ValueError, subprocess.SubprocessError):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(pages * page_size)
            except (OSError, ValueError, TypeError):
                return None
    if sys.platform.startswith("linux"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError, TypeError):
            return None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        except (AttributeError, OSError):
            return None
    return None


def _module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _platform_support(system_name: str) -> tuple[bool, str]:
    if system_name == "Darwin":
        if not macos_version_at_least(12):
            return False, "macOS 12 Monterey or newer is required"
        if macos_version_at_least(14):
            return True, "macOS 14+ supports MLX and llama.cpp; macOS 12-13 use llama.cpp"
        return True, "macOS 12-13 are supported through llama.cpp; bundled MLX requires macOS 14+"
    if system_name == "Windows":
        try:
            supported = int(platform.release().split(".", 1)[0]) >= 10
        except (ValueError, IndexError):
            supported = False
        return supported, "Windows 10/11 are supported through llama.cpp"
    if system_name == "Linux":
        return True, "Linux is supported; Debian 12 or Ubuntu 22.04+ is the packaged baseline"
    return False, "supported systems are macOS, Windows, and Linux"
