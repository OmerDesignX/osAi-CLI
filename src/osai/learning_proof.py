"""Deterministic, local-only before/after behavioral learning probes."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .backends.mlx import MlxBackend
from .config import ModelFormat
from .errors import ConfigurationError, DependencyError, TrainingError
from .formats import inspect_model
from .hardware import Accelerator, Engine, select_llama_accelerator
from .io import atomic_json
from .offline import offline_environment
from .paths import llama_binary
from .system import doctor


@dataclass(frozen=True, slots=True)
class LearningProofResult:
    engine: str
    model: Path
    adapter: Path
    prompt: str
    expected: str
    before: str
    after: str
    baseline_contained_expected: bool
    adapter_contained_expected: bool
    passed: bool
    accelerator: str
    output: Path

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("model", "adapter", "output"):
            payload[key] = str(payload[key])
        return payload


def prove_learning(
    *,
    engine: str | Engine,
    model: str | Path,
    adapter: str | Path,
    prompt: str,
    expected: str,
    output: str | Path,
    accelerator: str | Accelerator = Accelerator.AUTO,
    python: str | Path | None = None,
    max_tokens: int = 16,
    context: int = 128,
) -> LearningProofResult:
    """Prove a learned response by comparing deterministic base and adapter output."""

    try:
        selected_engine = engine if isinstance(engine, Engine) else Engine(engine)
    except ValueError as exc:
        raise ConfigurationError("learning proof engine must be mlx or llama.cpp") from exc
    if selected_engine is Engine.AUTO:
        raise ConfigurationError("learning proof requires --engine mlx or llama.cpp")
    if not prompt.strip() or not expected.strip():
        raise ConfigurationError("prompt and expected response must be non-empty")
    if max_tokens < 1 or context < 32:
        raise ConfigurationError("max_tokens must be positive and context must be at least 32")

    model_path = Path(model).expanduser().resolve()
    adapter_path = Path(adapter).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not adapter_path.exists():
        raise ConfigurationError(f"adapter does not exist: {adapter_path}")
    for label, protected in (("model", model_path), ("adapter", adapter_path)):
        if output_path == protected or (
            protected.is_dir() and output_path.is_relative_to(protected)
        ):
            raise ConfigurationError(f"proof output must not overwrite or be inside the {label}")

    if selected_engine is Engine.MLX:
        inspect_model(model_path, ModelFormat.MLX)
        if not adapter_path.is_dir():
            raise ConfigurationError("MLX adapter must be its adapter directory")
        before, after = _mlx_outputs(
            model_path, adapter_path, prompt, max_tokens=max_tokens, python=python
        )
        selected_accelerator = Accelerator.METAL
    else:
        model_path = inspect_model(model_path, ModelFormat.GGUF).path
        if not adapter_path.is_file():
            raise ConfigurationError("GGUF adapter must be an adapter.gguf file")
        selected_accelerator = select_llama_accelerator(accelerator)
        before, selected_accelerator = _gguf_output(
            model_path,
            None,
            prompt,
            max_tokens=max_tokens,
            context=context,
            accelerator=selected_accelerator,
        )
        after, selected_accelerator = _gguf_output(
            model_path,
            adapter_path,
            prompt,
            max_tokens=max_tokens,
            context=context,
            accelerator=selected_accelerator,
        )

    normalized_expected = _normalize(expected)
    baseline_match = normalized_expected in _normalize(before)
    adapter_match = normalized_expected in _normalize(after)
    result = LearningProofResult(
        engine=selected_engine.value,
        model=model_path,
        adapter=adapter_path,
        prompt=prompt,
        expected=expected,
        before=before.strip(),
        after=after.strip(),
        baseline_contained_expected=baseline_match,
        adapter_contained_expected=adapter_match,
        passed=adapter_match and not baseline_match,
        accelerator=selected_accelerator.value,
        output=output_path,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_only": True,
        "deterministic": True,
        "result": result.as_dict(),
        "system": doctor().as_dict(),
    }
    atomic_json(output_path, payload)
    return result


def _mlx_outputs(
    model: Path,
    adapter: Path,
    prompt: str,
    *,
    max_tokens: int,
    python: str | Path | None,
) -> tuple[str, str]:
    backend = MlxBackend(python)
    backend.preflight()
    common = [
        str(backend.python),
        "-m",
        "osai._offline_runner",
        "mlx_lm",
        "generate",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        "0",
        "--seed",
        "0",
        "--verbose",
        "false",
        "--chat-template-config",
        '{"enable_thinking":false}',
    ]
    before = _run_command(common, backend.environment())
    adapted = [*common, "--adapter-path", str(adapter)]
    after = _run_command(adapted, backend.environment())
    return before, after


def _gguf_output(
    model: Path,
    adapter: Path | None,
    prompt: str,
    *,
    max_tokens: int,
    context: int,
    accelerator: Accelerator,
) -> tuple[str, Accelerator]:
    binary = llama_binary("llama-completion")
    if binary is None:
        raise DependencyError("llama-completion is not built; run `osai build-llama`")
    command = [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "-c",
        str(context),
        "--temp",
        "0",
        "--seed",
        "0",
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",
        "--log-colors",
        "off",
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    accelerator_args = _llama_accelerator_args(accelerator)
    command.extend(accelerator_args)
    try:
        return _run_command(command, offline_environment()), accelerator
    except TrainingError:
        if accelerator is Accelerator.CPU:
            raise
        cpu_command = command[: -len(accelerator_args)]
        cpu_command.extend(_llama_accelerator_args(Accelerator.CPU))
        return _run_command(cpu_command, offline_environment()), Accelerator.CPU


def _llama_accelerator_args(accelerator: Accelerator) -> list[str]:
    if accelerator is Accelerator.CPU:
        return ["-dev", "none", "-ngl", "0", "-fit", "off", "--no-op-offload"]
    return ["-ngl", "auto"]


def _run_command(command: list[str], environment: dict[str, str]) -> str:
    try:
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"learning probe could not run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise TrainingError(
            f"learning probe exited with status {completed.returncode}: {detail[-1000:]}"
        )
    return completed.stdout


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
