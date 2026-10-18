"""Shared multi-device command construction."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class LlamaDeviceInventory:
    """One native snapshot of device identities and memory capacity."""

    devices: tuple[str, ...]
    free_by_device: dict[str, int]
    total_by_device: dict[str, int]
    labels_by_device: dict[str, str] = field(default_factory=dict)


def automatic_llama_device_plan(
    devices: Sequence[str],
    free_by_device: dict[str, int],
    total_by_device: dict[str, int] | None = None,
) -> LlamaDevicePlan:
    """Weight shards by usable VRAM and put shared buffers on the largest GPU.

    llama.cpp accepts relative ``--tensor-split`` weights.  Passing usable MiB
    keeps asymmetric cards from receiving equal model shares. The primary GPU
    is selected by physical VRAM when the backend reports it, independently of
    device numbering or temporary load. If free memory is unavailable, total
    VRAM also provides a stable shard weighting. If neither value is reported,
    llama.cpp retains its own automatic distribution.
    """

    selected = tuple(devices)
    if len(selected) < 2:
        return LlamaDevicePlan(selected, (), 0)
    total_by_device = total_by_device or {}
    complete_free = all(
        isinstance(free_by_device.get(device), int) and free_by_device[device] > 0
        for device in selected
    )
    complete_total = all(
        isinstance(total_by_device.get(device), int) and total_by_device[device] > 0
        for device in selected
    )
    if not complete_free and not complete_total:
        return LlamaDevicePlan(selected, (), 0)
    available_source = free_by_device if complete_free else total_by_device
    primary_source = total_by_device if complete_total else available_source
    available = tuple(available_source[device] for device in selected)
    physical = tuple(primary_source[device] for device in selected)
    usable = tuple(max(1, value - max(512 * 1024**2, value // 10)) for value in available)
    weights = tuple(round(value / 1024**2, 3) for value in usable)
    return LlamaDevicePlan(
        selected,
        weights,
        max(range(len(physical)), key=physical.__getitem__),
    )


def available_llama_devices(
    binary: Path | None,
    accelerator: Accelerator,
    *,
    include_integrated: bool = False,
    include_amd_vulkan: bool = False,
) -> tuple[str, ...]:
    """Resolve native IDs, excluding integrated Vulkan adapters for training."""

    inventory = llama_device_inventory_details(
        binary,
        accelerator,
        include_integrated=include_integrated,
        include_amd_vulkan=include_amd_vulkan,
    )
    return inventory.devices


def llama_device_inventory(
    binary: Path | None,
    accelerator: Accelerator,
    *,
    include_integrated: bool = False,
    include_amd_vulkan: bool = False,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Read device identities and memory from one bounded native probe."""

    inventory = llama_device_inventory_details(
        binary,
        accelerator,
        include_integrated=include_integrated,
        include_amd_vulkan=include_amd_vulkan,
    )
    return inventory.devices, inventory.free_by_device


def llama_device_inventory_details(
    binary: Path | None,
    accelerator: Accelerator,
    *,
    include_integrated: bool = False,
    include_amd_vulkan: bool = False,
) -> LlamaDeviceInventory:
    """Read identities, current free memory and total VRAM in one probe."""

    if binary is None or accelerator is Accelerator.CPU:
        return LlamaDeviceInventory((), {}, {})
    _result, output = _llama_device_report(binary, accelerator)
    devices = _llama_devices_from_report(
        output,
        accelerator,
        include_integrated=include_integrated,
        include_amd_vulkan=include_amd_vulkan,
    )
    free, total = _llama_memory_bytes_from_report(output, accelerator)
    labels = _llama_device_labels_from_report(output, accelerator)
    return LlamaDeviceInventory(
        devices,
        {device: free[device] for device in devices if device in free and free[device] > 0},
        {device: total[device] for device in devices if device in total and total[device] > 0},
        {device: labels[device] for device in devices if device in labels},
    )


def _llama_device_labels_from_report(output: str, accelerator: Accelerator) -> dict[str, str]:
    """Return stable native IDs with human-readable physical device names."""

    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return {}
    labels: dict[str, str] = {}
    for device, raw_label in re.findall(
        rf"^\s*({prefix}\d+):\s*(.+)$",
        output,
        re.M | re.I,
    ):
        label = re.sub(r"\s*\[type=(?:dedicated|integrated|other)\]\s*$", "", raw_label, flags=re.I)
        label = re.sub(
            r"\s*\(\s*\d+\s+MiB\s*,\s*\d+\s+MiB\s+free\s*\)\s*$",
            "",
            label,
            flags=re.I,
        ).strip()
        labels.setdefault(device, label or device)
    return labels


def _llama_devices_from_report(
    output: str,
    accelerator: Accelerator,
    *,
    include_integrated: bool = False,
    include_amd_vulkan: bool = False,
) -> tuple[str, ...]:
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
            timeout=_native_device_probe_timeout(accelerator),
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


def _native_device_probe_timeout(accelerator: Accelerator) -> int:
    """Allow legacy Metal drivers to finish their one-time shader warmup."""

    # Intel Macs with multiple discrete AMD GPUs can spend several minutes in
    # Metal's shader compiler before llama.cpp prints its device inventory.
    # CUDA and Vulkan enumeration should remain fast enough to fail promptly.
    return 15 * 60 if accelerator is Accelerator.METAL else 90


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
    return _llama_free_bytes_from_report(output, accelerator)


def _llama_free_bytes_from_report(output: str, accelerator: Accelerator) -> dict[str, int]:
    free, _total = _llama_memory_bytes_from_report(output, accelerator)
    return free


def _llama_memory_bytes_from_report(
    output: str, accelerator: Accelerator
) -> tuple[dict[str, int], dict[str, int]]:
    prefix = {
        Accelerator.CUDA: "CUDA",
        Accelerator.VULKAN: "Vulkan",
        Accelerator.METAL: "MTL",
    }.get(accelerator)
    if prefix is None:
        return {}, {}
    memory = {
        device: (int(total) * 1024**2, int(free) * 1024**2)
        for device, total, free in re.findall(
            rf"^\s*({prefix}\d+):[^\n]*\(\s*(\d+)\s+MiB\s*,\s*(\d+)\s+MiB\s+free\s*\)",
            output,
            re.M | re.I,
        )
    }
    total = {device: values[0] for device, values in memory.items()}
    free = {device: values[1] for device, values in memory.items()}
    if accelerator is Accelerator.CUDA:
        for device in _dedicated_amd_vulkan_ids(output):
            match = re.search(
                rf"^\s*{re.escape(device)}:[^\n]*\(\s*(\d+)\s+MiB\s*,\s*(\d+)\s+MiB\s+free\s*\)",
                output,
                re.M | re.I,
            )
            if match:
                total[device] = int(match.group(1)) * 1024**2
                free[device] = int(match.group(2)) * 1024**2
    return free, total


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
