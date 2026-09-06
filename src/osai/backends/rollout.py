"""Local inference adapters used by live alignment rollouts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..errors import DependencyError, TrainingError
from ..hardware import Accelerator
from ..io import atomic_json, atomic_text
from ..multi_gpu import DeviceSettings, llama_device_arguments
from ..offline import offline_environment
from ..paths import llama_binary
from ..process import run_logged
from ..rollouts import RolloutRequest, RolloutSettings
from .mlx import MlxBackend


class RolloutDeviceSettings(DeviceSettings, Protocol):
    max_seq_length: int


def generate_mlx_answers(
    backend: MlxBackend,
    model: Path,
    adapter: Path,
    requests: tuple[RolloutRequest, ...],
    settings: RolloutSettings,
    *,
    work: Path,
    log_path: Path,
) -> tuple[str, ...]:
    config_path = work / "mlx_rollouts.json"
    result_path = work / "mlx_rollout_results.json"
    atomic_json(
        config_path,
        {
            "model": str(model),
            "adapter": str(adapter),
            "output": str(result_path),
            "requests": [asdict(item) for item in requests],
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        },
    )
    run_logged(
        [
            str(backend.python),
            "-m",
            "osai._offline_runner",
            "rollout-mlx",
            str(config_path),
        ],
        log_path=log_path,
        env=backend.environment(),
        timeout=max(300.0, 300.0 * len(requests)),
    )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        responses = tuple(item["response"] for item in payload["results"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"MLX rollout runner returned invalid results: {exc}") from exc
    return responses


def generate_gguf_answers(
    model: Path,
    adapter: Path,
    requests: tuple[RolloutRequest, ...],
    settings: RolloutSettings,
    *,
    accelerator: Accelerator,
    devices: RolloutDeviceSettings,
    log_path: Path,
) -> tuple[tuple[str, ...], Accelerator]:
    binary = llama_binary("llama-completion")
    if binary is None:
        raise DependencyError("llama-completion is not built; run `osai build-llama`")
    selected = accelerator
    responses: list[str] = []
    log_sections: list[str] = []
    for index, request in enumerate(requests):
        try:
            response, diagnostics = _generate_one_gguf(
                binary,
                model,
                adapter,
                request,
                settings,
                selected,
                devices,
            )
        except TrainingError:
            if selected is Accelerator.CPU:
                raise
            selected = Accelerator.CPU
            response, diagnostics = _generate_one_gguf(
                binary,
                model,
                adapter,
                request,
                settings,
                selected,
                devices,
            )
        responses.append(response)
        log_sections.append(
            f"request={index} seed={request.seed} accelerator={selected.value}\n"
            f"{diagnostics.strip()}\n"
        )
    atomic_text(log_path, "\n".join(log_sections))
    return tuple(responses), selected


def _generate_one_gguf(
    binary: Path,
    model: Path,
    adapter: Path,
    request: RolloutRequest,
    settings: RolloutSettings,
    accelerator: Accelerator,
    devices: RolloutDeviceSettings,
) -> tuple[str, str]:
    command = [
        str(binary),
        "-m",
        str(model),
        "--lora",
        str(adapter),
        "-p",
        f"user: {request.prompt}\nassistant:",
        "-n",
        str(settings.max_tokens),
        "-c",
        str(max(64, devices.max_seq_length)),
        "--temp",
        str(settings.temperature),
        "--top-p",
        str(settings.top_p),
        "--seed",
        str(request.seed),
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
        "--log-colors",
        "off",
        *llama_device_arguments(accelerator, devices),
    ]
    try:
        completed = subprocess.run(
            command,
            env=offline_environment(),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"GGUF rollout generation failed to start: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise TrainingError(
            f"GGUF rollout exited with status {completed.returncode}: {detail[-1000:]}"
        )
    return completed.stdout.strip(), completed.stderr
