"""Build and execute the vendored llama.cpp deployment backend."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from ..hardware import Accelerator, select_llama_accelerator
from ..offline import offline_environment
from ..paths import llama_binary, llama_cpp_root
from ..process import ProcessResult, run_logged


@dataclass(frozen=True, slots=True)
class LlamaValidationResult:
    binary: Path
    log_file: Path
    elapsed_seconds: float
    accelerator: str


@dataclass(frozen=True, slots=True)
class LlamaBuildResult:
    process: ProcessResult
    accelerator: str
    fallback_from: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return self.process.elapsed_seconds

    @property
    def log_path(self) -> Path:
        return self.process.log_path


def build_llama_cpp(
    *,
    log_path: str | Path,
    jobs: int | None = None,
    accelerator: str | Accelerator = Accelerator.AUTO,
    cpu_fallback: bool = True,
) -> LlamaBuildResult:
    root = llama_cpp_root()
    if not root.is_dir():
        raise DependencyError(f"vendored llama.cpp source not found: {root}")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise DependencyError("CMake is required to build vendored llama.cpp")
    if jobs is not None and jobs < 1:
        raise DependencyError("build jobs must be at least 1")
    build = root / "build"
    destination = Path(log_path)
    destination.unlink(missing_ok=True)
    selected = select_llama_accelerator(accelerator)
    fallback_from = None
    try:
        _configure(cmake, root, build, destination, selected)
    except TrainingError:
        if not cpu_fallback or selected is Accelerator.CPU:
            raise
        fallback_from = selected.value
        selected = Accelerator.CPU
        print(f"osai: {fallback_from} configuration failed; retrying with CPU")
        _configure(cmake, root, build, destination, selected)

    build_command = [
        cmake,
        "--build",
        str(build),
        "--config",
        "Release",
        "--target",
        "llama-completion",
        "llama-finetune",
        "llama-perplexity",
        "llama-quantize",
        "llama-export-lora",
        "llama-gguf-split",
    ]
    if jobs is not None:
        build_command.extend(["--parallel", str(jobs)])
    else:
        build_command.append("--parallel")
    try:
        process = run_logged(build_command, log_path=destination, cwd=root)
    except TrainingError:
        if not cpu_fallback or selected is Accelerator.CPU:
            raise
        fallback_from = selected.value
        selected = Accelerator.CPU
        print(f"osai: {fallback_from} build failed; retrying with CPU")
        _configure(cmake, root, build, destination, selected)
        process = run_logged(build_command, log_path=destination, cwd=root)
    return LlamaBuildResult(process, selected.value, fallback_from)


def _configure(
    cmake: str, root: Path, build: Path, log_path: Path, accelerator: Accelerator
) -> None:
    flags = [
        cmake,
        "-S",
        str(root),
        "-B",
        str(build),
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        # Some macOS toolchains stall during host-specific CPU feature probes.
        # Metal performs model compute, so keep the CPU fallback portable.
        "-DGGML_NATIVE=OFF",
        f"-DGGML_METAL={'ON' if accelerator is Accelerator.METAL else 'OFF'}",
        f"-DGGML_CUDA={'ON' if accelerator is Accelerator.CUDA else 'OFF'}",
        f"-DGGML_VULKAN={'ON' if accelerator is Accelerator.VULKAN else 'OFF'}",
    ]
    run_logged(flags, log_path=log_path, cwd=root)


def validate_adapter(
    base_model: str | Path,
    adapter: str | Path,
    *,
    log_path: str | Path,
    prompt: str = "Reply with only: adapter validation passed",
    tokens: int = 8,
    context: int = 128,
    accelerator: str | Accelerator = Accelerator.AUTO,
) -> LlamaValidationResult:
    if tokens < 1 or context < 1:
        raise ConfigurationError("validation tokens and context must be at least 1")
    binary = llama_binary("llama-cli")
    if binary is None:
        raise DependencyError(
            "llama-completion (or the legacy llama-cli) is not built; "
            "run `osai build-llama` first"
        )
    command = [
        str(binary),
        "-m",
        str(Path(base_model).resolve()),
        "--lora",
        str(Path(adapter).resolve()),
        "-p",
        prompt,
        "-n",
        str(tokens),
        "-c",
        str(context),
        "--temp",
        "0",
    ]
    selected = select_llama_accelerator(accelerator)
    if selected is Accelerator.CPU:
        command.extend(["-dev", "none", "-ngl", "0", "-fit", "off", "--no-op-offload"])
    else:
        command.extend(["-ngl", "auto"])
    fallback_from = None
    destination = Path(log_path)
    destination.unlink(missing_ok=True)
    successful_offset = 0
    try:
        result = run_logged(command, log_path=destination, env=offline_environment())
    except TrainingError:
        if selected is Accelerator.CPU:
            raise
        fallback_from = selected.value
        selected = Accelerator.CPU
        command = command[:-2]
        command.extend(["-dev", "none", "-ngl", "0", "-fit", "off", "--no-op-offload"])
        print(f"osai: {fallback_from} execution failed; retrying with CPU")
        successful_offset = destination.stat().st_size if destination.exists() else 0
        result = run_logged(command, log_path=destination, env=offline_environment())
    with result.log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(successful_offset)
        log_text = handle.read()
    lowered = log_text.lower()
    if "failed to load lora adapter" in lowered or "error loading model" in lowered:
        raise VerificationError(f"llama.cpp rejected the model or adapter; see {result.log_path}")
    if fallback_from:
        with result.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[osai] accelerator_fallback={fallback_from}->cpu\n")
    return LlamaValidationResult(binary, result.log_path, result.elapsed_seconds, selected.value)
