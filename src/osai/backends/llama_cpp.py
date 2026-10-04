"""Build and execute the vendored llama.cpp deployment backend."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError, DependencyError, TrainingError, VerificationError
from ..hardware import (
    Accelerator,
    _vulkan_build_available,
    detect_hardware,
    select_llama_accelerator,
)
from ..io import atomic_json
from ..offline import offline_environment
from ..paths import llama_binary, llama_cpp_root, llama_runtime_build
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
    build_dir: Path | None = None,
    also_vulkan: bool = False,
) -> LlamaBuildResult:
    root = llama_cpp_root()
    if not root.is_dir():
        raise DependencyError(f"vendored llama.cpp source not found: {root}")
    cmake = _cmake_executable()
    if cmake is None:
        raise DependencyError("CMake is required to build vendored llama.cpp")
    if jobs is not None and jobs < 1:
        raise DependencyError("build jobs must be at least 1")
    build = build_dir or llama_runtime_build()
    build.mkdir(parents=True, exist_ok=True)
    _repair_stale_ninja_cache(build)
    (build / "OSAI_BUILD.json").unlink(missing_ok=True)
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    requested = select_llama_accelerator(accelerator, for_build=True)
    automatic = Accelerator(accelerator) is Accelerator.AUTO
    candidates = [(requested, also_vulkan and requested is Accelerator.CUDA)]
    if candidates[0][1]:
        candidates.append((Accelerator.CUDA, False))
    if automatic and requested is Accelerator.CUDA and _vulkan_build_available():
        candidates.append((Accelerator.VULKAN, False))
    if automatic and cpu_fallback and requested is not Accelerator.CPU:
        candidates.append((Accelerator.CPU, False))

    build_command = [
        cmake,
        "--build",
        str(build),
        "--config",
        "Release",
        "--target",
        "llama-completion",
        "llama-tokenize",
        "llama-finetune",
        "llama-perplexity",
        "llama-quantize",
        "llama-export-lora",
        "llama-gguf-split",
    ]
    build_command.extend(["--parallel", str(jobs or min(2, os.cpu_count() or 1))])
    for index, (selected, with_vulkan) in enumerate(candidates):
        try:
            _configure(
                cmake,
                root,
                build,
                destination,
                selected,
                with_vulkan,
            )
            process = run_logged(build_command, log_path=destination, cwd=root)
            break
        except TrainingError:
            if index == len(candidates) - 1:
                raise
            print(
                f"osai: {selected.value}{'+vulkan' if with_vulkan else ''} build failed; "
                f"retrying with {candidates[index + 1][0].value}"
                f"{'+vulkan' if candidates[index + 1][1] else ''}"
            )
    if os.name == "nt" and selected is Accelerator.CUDA:
        _copy_cuda_runtime_dlls(build)
    atomic_json(
        build / "OSAI_BUILD.json",
        {
            "schemaVersion": 1,
            "llamaAccelerator": (
                "cuda+vulkan" if selected is Accelerator.CUDA and with_vulkan else selected.value
            ),
        },
    )
    return LlamaBuildResult(
        process,
        selected.value,
        requested.value if selected is not requested else None,
    )


def _copy_cuda_runtime_dlls(build: Path) -> None:
    toolkit = os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    if not toolkit and nvcc:
        toolkit = str(Path(nvcc).resolve().parent.parent)
    if not toolkit:
        raise DependencyError("CUDA_PATH is required to package the CUDA runtime DLLs")
    source = Path(toolkit) / "bin"
    destination = build / "bin"
    for prefix in ("cudart64", "cublas64", "cublasLt64"):
        matches = tuple(source.glob(f"{prefix}_*.dll"))
        if len(matches) != 1:
            raise DependencyError(f"expected one {prefix} runtime DLL in {source}")
        shutil.copy2(matches[0], destination / matches[0].name)


def ensure_runtime_accelerator(requested: str | Accelerator) -> None:
    """Build a native GPU backend once when the host has the required SDK.

    A locally built backend takes precedence for subsequent runs. First-run
    setup normally creates this build; a later SDK install can upgrade it.
    """

    choice = requested if isinstance(requested, Accelerator) else Accelerator(requested)
    if choice is Accelerator.CPU:
        return
    report = detect_hardware()
    candidate = choice
    if choice is Accelerator.AUTO:
        candidate = Accelerator.CPU
        if report.metal and platform.system() == "Darwin":
            candidate = Accelerator.METAL
        elif report.cuda and (shutil.which("nvcc") or os.environ.get("CUDA_PATH")):
            candidate = Accelerator.CUDA
        elif report.vulkan and _vulkan_build_available():
            candidate = Accelerator.VULKAN
        if candidate is Accelerator.CPU:
            if any(
                backend in (report.compiled_llama_accelerator or "").split("+")
                for backend in ("metal", "cuda", "vulkan")
            ):
                return
            if report.metal or report.cuda or report.vulkan:
                raise DependencyError(
                    "a GPU was detected, but no supported GPU llama.cpp backend can be built; "
                    "install the platform build tools or explicitly select --accelerator cpu"
                )
            return
    if candidate.value in (report.compiled_llama_accelerator or "").split("+"):
        return
    if candidate is Accelerator.CUDA and not (
        shutil.which("nvcc")
        or (
            os.environ.get("CUDA_PATH")
            and (
                Path(os.environ["CUDA_PATH"]) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            ).is_file()
        )
    ):
        raise DependencyError("CUDA training requires the CUDA Toolkit (nvcc) to build llama.cpp")
    if candidate is Accelerator.VULKAN and not _vulkan_build_available():
        raise DependencyError(
            "Vulkan training requires the Vulkan SDK or a system glslc compiler to build llama.cpp"
        )
    if _cmake_executable() is None:
        if choice is Accelerator.AUTO:
            packaged = (report.compiled_llama_accelerator or "").split("+")
            if any(backend in packaged for backend in ("metal", "cuda", "vulkan")):
                print(
                    "osai: CMake is unavailable; using the packaged GPU trainer",
                    flush=True,
                )
                return
        raise DependencyError("CMake is required to build a GPU llama.cpp backend")
    print(f"osai: building llama.cpp {candidate.value} backend for this host")
    try:
        build_llama_cpp(
            log_path=llama_runtime_build() / "osai-runtime-build.log",
            jobs=min(4, os.cpu_count() or 1),
            accelerator=candidate,
            cpu_fallback=False,
            build_dir=llama_runtime_build(),
            also_vulkan=(
                candidate is Accelerator.CUDA and report.vulkan and _vulkan_build_available()
            ),
        )
    except (TrainingError, DependencyError, OSError) as exc:
        if choice is not Accelerator.AUTO:
            raise
        available = (detect_hardware().compiled_llama_accelerator or "").split("+")
        if not any(backend in available for backend in ("metal", "cuda", "vulkan")):
            raise DependencyError(
                "the GPU llama.cpp build failed and no GPU trainer is available"
            ) from exc
        print(f"osai: {candidate.value} build failed; continuing with the available backend")


def _cmake_executable() -> str | None:
    """Find CMake from PATH or the Python environment running this CLI."""

    configured = os.environ.get("OSAI_CMAKE")
    if configured and Path(configured).is_file():
        return configured
    candidate = shutil.which("cmake")
    if candidate:
        return candidate
    sibling = Path(sys.executable).parent / ("cmake.exe" if os.name == "nt" else "cmake")
    return str(sibling) if sibling.is_file() else None


def _repair_stale_ninja_cache(build: Path) -> None:
    """A cached Vulkan shader build may refer to a removed app build tool."""

    cache = (
        build
        / "ggml"
        / "src"
        / "ggml-vulkan"
        / "vulkan-shaders-gen-prefix"
        / "src"
        / "vulkan-shaders-gen-build"
        / "CMakeCache.txt"
    )
    if not cache.is_file():
        return
    contents = cache.read_text(encoding="utf-8", errors="replace")
    for line in contents.splitlines():
        if not line.startswith("CMAKE_MAKE_PROGRAM:"):
            continue
        key, _, configured = line.partition("=")
        if Path(configured).is_file() or shutil.which(configured):
            return
        ninja = shutil.which("ninja")
        if not ninja:
            sibling = Path(sys.executable).parent / ("ninja.exe" if os.name == "nt" else "ninja")
            ninja = str(sibling) if sibling.is_file() else None
        if ninja:
            pending = cache.with_name("CMakeCache.txt.pending")
            pending.write_text(
                contents.replace(line, f"{key}={Path(ninja).as_posix()}", 1),
                encoding="utf-8",
            )
            os.replace(pending, cache)
            print("osai: repaired stale Ninja path in Vulkan build cache", flush=True)
        return


def _configure(
    cmake: str,
    root: Path,
    build: Path,
    log_path: Path,
    accelerator: Accelerator,
    also_vulkan: bool = False,
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
        f"-DGGML_VULKAN={'ON' if accelerator is Accelerator.VULKAN or also_vulkan else 'OFF'}",
    ]
    if accelerator is Accelerator.CUDA:
        architectures = _cuda_architectures()
        if architectures:
            flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(architectures)}")
    run_logged(flags, log_path=log_path, cwd=root)


def _cuda_architectures() -> tuple[str, ...]:
    """Compile for all installed NVIDIA GPU generations without hardcoded flags."""

    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = (
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        )
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    values = set()
    for line in result.stdout.splitlines():
        parts = line.strip().split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            return ()
        values.add("".join(parts))
    return tuple(sorted(values))


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
            "llama-completion (or the legacy llama-cli) is not built; run `osai build-llama` first"
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
