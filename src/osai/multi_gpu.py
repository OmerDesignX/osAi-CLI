"""Shared multi-device command construction."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class LlamaDevicePlan:
    """A stable llama.cpp model-sharding plan for heterogeneous GPUs."""

    devices: tuple[str, ...]
    tensor_split: tuple[float, ...]
    main_gpu: int


def automatic_llama_device_plan(
    devices: Sequence[str], free_by_device: dict[str, int]
) -> LlamaDevicePlan:
    """Weight shards by usable VRAM and keep non-shard buffers on the largest GPU.

    llama.cpp accepts relative ``--tensor-split`` weights.  Passing usable MiB
    keeps asymmetric cards from receiving equal model shares, while selecting
    the roomiest card as ``--main-gpu`` leaves the smaller cards available for
    layer shards.  If a backend cannot report memory, llama.cpp retains its own
    automatic distribution.
    """

    selected = tuple(devices)
    if len(selected) < 2:
        return LlamaDevicePlan(selected, (), 0)
    if not all(
        isinstance(free_by_device.get(device), int) and free_by_device[device] > 0
        for device in selected
    ):
        return LlamaDevicePlan(selected, (), 0)
    available = tuple(free_by_device[device] for device in selected)
    usable = tuple(max(1, value - max(512 * 1024**2, value // 10)) for value in available)
    weights = tuple(round(value / 1024**2, 3) for value in usable)
    return LlamaDevicePlan(selected, weights, max(range(len(available)), key=available.__getitem__))


def available_llama_devices(
    binary: Path | None,
    accelerator: Accelerator,
    *,
    include_integrated: bool = False,
    include_amd_vulkan: bool = False,
) -> tuple[str, ...]:
    """Resolve native IDs, excluding integrated Vulkan adapters for training."""

    if binary is None or accelerator is Accelerator.CPU:
        return ()
    result, output = _llama_device_report(binary, accelerator)
    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return ()
    devices = list(
        dict(
            re.findall(
                rf"^\s*({prefix}\d+):\s*(.+)$",
                output,
                re.M | re.I,
            )
        ).items()
    )
    if accelerator is Accelerator.VULKAN:
        typed = [
            (device, re.search(r"\[type=(dedicated|integrated|other)\]\s*$", label, re.I))
            for device, label in devices
        ]
        if any(kind is None for _device, kind in typed):
            raise ConfigurationError(
                "the Vulkan trainer cannot identify dedicated GPUs; rebuild llama.cpp "
                "with `osai build-llama` before training"
            )
        return tuple(
            device
            for device, kind in typed
            if kind is not None and kind.group(1).casefold() == "dedicated"
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
            return tuple(device for device, _label in discrete)
    selected = tuple(device for device, _label in devices)
    if accelerator is Accelerator.CUDA and include_amd_vulkan:
        selected += _dedicated_amd_vulkan_ids(output)
    return selected


def _llama_device_report(binary: Path, accelerator: Accelerator):
    """Run llama.cpp discovery without hiding an unusable native backend.

    Returning an empty inventory for a GPU build made the desktop app report
    only "no usable devices", which concealed loader failures and Metal access
    problems. CPU discovery remains intentionally quiet, while a selected GPU
    backend now includes the bounded native diagnostic in the error.
    """

    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            # The first Metal process can compile its shader library before it
            # can enumerate devices, especially on Intel Macs with AMD GPUs.
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            f"{accelerator.value} device discovery timed out; restart osAi and try again"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"could not start the {accelerator.value} device probe: {exc}"
        ) from exc
    output = result.stdout + "\n" + (getattr(result, "stderr", "") or "")
    if result.returncode:
        detail = _bounded_device_diagnostic(output)
        raise ConfigurationError(
            f"the {accelerator.value} device probe exited with {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result, output


def _bounded_device_diagnostic(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-6:])[-1200:]


def _dedicated_amd_vulkan_ids(output: str) -> tuple[str, ...]:
    return tuple(
        device
        for device, label in re.findall(r"^\s*(Vulkan\d+):\s*(.+)$", output, re.M | re.I)
        if re.search(r"\b(?:AMD|Radeon)\b", label, re.I)
        and re.search(r"\[type=dedicated\]\s*$", label, re.I)
    )


def llama_device_free_bytes(binary: Path | None, accelerator: Accelerator) -> dict[str, int]:
    """Read the native backend's current free-memory estimates when available."""

    if binary is None or accelerator is Accelerator.CPU:
        return {}
    _result, output = _llama_device_report(binary, accelerator)
    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return {}
    free = {
        device: int(free) * 1024**2
        for device, free in re.findall(
            rf"^\s*({prefix}\d+):[^\n]*?\b(\d+)\s+MiB\s+free\b",
            output,
            re.M | re.I,
        )
    }
    if accelerator is Accelerator.CUDA:
        for device in _dedicated_amd_vulkan_ids(output):
            match = re.search(
                rf"^\s*{re.escape(device)}:[^\n]*?\b(\d+)\s+MiB\s+free\b",
                output,
                re.M | re.I,
            )
            if match:
                free[device] = int(match.group(1)) * 1024**2
    return free


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
    if len(set(settings.devices)) != len(settings.devices):
        raise ConfigurationError("each GPU must appear only once in the device list")
    if any(not math.isfinite(value) or value <= 0 for value in settings.tensor_split):
        raise ConfigurationError("tensor_split values must be finite and positive")
    if (
        settings.multi_gpu != "off"
        and settings.devices
        and settings.tensor_split
        and len(settings.tensor_split) != len(settings.devices)
    ):
        raise ConfigurationError(
            "tensor_split needs one weight for every selected GPU; leave it empty "
            "to distribute the model automatically"
        )
    if accelerator is Accelerator.CPU:
        return ["-fit", "off", "-dev", "none", "-ngl", "0", "--no-op-offload"]

    arguments = ["-fit", "off", "-ngl", "auto"]
    if settings.devices:
        devices = settings.devices[:1] if settings.multi_gpu == "off" else settings.devices
        if not 0 <= settings.main_gpu < len(devices):
            raise ConfigurationError("main_gpu must index the selected GPU list")
        arguments.extend(["-dev", ",".join(devices)])
    # A single explicit Metal device is the normal Apple-silicon topology.
    # Do not ask llama.cpp to split a model when there is nowhere to split it;
    # Intel Macs with two or more physical Metal devices still retain the
    # requested layer split below.
    one_explicit_device = bool(settings.devices) and len(devices) == 1
    mode = "none" if settings.multi_gpu == "off" or one_explicit_device else settings.split_mode
    arguments.extend(["-sm", mode, "-mg", str(settings.main_gpu)])
    if settings.tensor_split and settings.multi_gpu != "off":
        arguments.extend(["-ts", ",".join(format(value, ".8g") for value in settings.tensor_split)])
    return arguments
