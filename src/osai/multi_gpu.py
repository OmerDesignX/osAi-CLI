"""Shared multi-device command construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .errors import ConfigurationError
from .hardware import Accelerator


class DeviceSettings(Protocol):
    multi_gpu: str
    devices: Sequence[str]
    split_mode: str
    tensor_split: Sequence[float]
    main_gpu: int


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
        arguments.extend(
            ["-ts", ",".join(format(value, ".8g") for value in settings.tensor_split)]
        )
    return arguments
