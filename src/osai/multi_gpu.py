"""Shared multi-device command construction."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .errors import ConfigurationError
from .hardware import Accelerator


class DeviceSettings(Protocol):
    multi_gpu: str
    devices: Sequence[str]
    split_mode: str
    tensor_split: Sequence[float]
    main_gpu: int


def available_llama_devices(
    binary: Path | None, accelerator: Accelerator, *, include_integrated: bool = False
) -> tuple[str, ...]:
    """Resolve native IDs, preferring external cards and optionally including integrated GPUs."""

    if binary is None or accelerator is Accelerator.CPU:
        return ()
    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode:
        return ()
    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return ()
    devices = re.findall(
        rf"^\s*({prefix}\d+):\s*(.+)$",
        result.stdout + "\n" + (getattr(result, "stderr", "") or ""),
        re.M | re.I,
    )
    if accelerator in {Accelerator.VULKAN, Accelerator.METAL}:
        integrated = re.compile(
            r"Radeon\(TM\) Graphics|Intel.*(?:UHD|Iris|HD).*Graphics|Integrated Graphics",
            re.I,
        )
        discrete = [(device, label) for device, label in devices if not integrated.search(label)]
        if discrete:
            external = re.compile(r"\b(?:eGPU|external|removable)\b", re.I)
            discrete.sort(key=lambda item: not bool(external.search(item[1])))
            preferred = discrete + (
                [(device, label) for device, label in devices if integrated.search(label)]
                if include_integrated
                else []
            )
            return tuple(device for device, _label in preferred)
    return tuple(device for device, _label in devices)


def llama_device_free_bytes(binary: Path | None, accelerator: Accelerator) -> dict[str, int]:
    """Read the native backend's current free-memory estimates when available."""

    if binary is None or accelerator is Accelerator.CPU:
        return {}
    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return {}
    output = result.stdout + "\n" + (getattr(result, "stderr", "") or "")
    return {
        device: int(free) * 1024**2
        for device, free in re.findall(
            rf"^\s*({prefix}\d+):[^\n]*?\b(\d+)\s+MiB\s+free\b",
            output,
            re.M | re.I,
        )
    }


def llama_device_arguments(
    accelerator: Accelerator,
    settings: DeviceSettings,
) -> list[str]:
    """Translate validated settings to llama.cpp's native device flags."""

    if settings.multi_gpu not in {"auto", "on", "off"}:
        raise ConfigurationError("multi_gpu must be auto, on, or off")
    if settings.split_mode not in {"none", "layer", "row", "tensor"}:
        raise ConfigurationError("split_mode must be none, layer, row, or tensor")
    if settings.multi_gpu == "on" and settings.split_mode == "none":
        raise ConfigurationError("multi-GPU mode requires layer, row, or tensor splitting")
    if settings.multi_gpu == "on" and settings.devices and len(settings.devices) < 2:
        raise ConfigurationError("multi-GPU mode needs at least two explicit devices")
    if accelerator is Accelerator.CPU:
        return ["-fit", "off", "-dev", "none", "-ngl", "0", "--no-op-offload"]

    arguments = ["-fit", "off", "-ngl", "auto"]
    if settings.devices:
        devices = settings.devices[:1] if settings.multi_gpu == "off" else settings.devices
        arguments.extend(["-dev", ",".join(devices)])
    mode = "none" if settings.multi_gpu == "off" else settings.split_mode
    arguments.extend(["-sm", mode, "-mg", str(settings.main_gpu)])
    if settings.tensor_split and settings.multi_gpu != "off":
        arguments.extend(["-ts", ",".join(format(value, ".8g") for value in settings.tensor_split)])
    return arguments
