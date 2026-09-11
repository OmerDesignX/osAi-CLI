#!/usr/bin/env python3
"""Install the osAi framework and build its bundled engines for this hardware."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import venv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_ROOT = PROJECT_ROOT / "requirements"
SUPPORTED_PYTHON = (3, 10), (3, 14)


class SetupError(RuntimeError):
    """A setup prerequisite or requested configuration is invalid."""


@dataclass(frozen=True, slots=True)
class SetupPlan:
    system: str
    architecture: str
    macos_major: int | None
    cuda_major: int | None
    vulkan_available: bool
    requirements: str
    mlx_accelerator: str | None
    llama_accelerator: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_plan(
    *,
    system: str,
    architecture: str,
    macos_major: int | None,
    cuda_major: int | None,
    vulkan_available: bool,
    mlx_accelerator: str = "auto",
    llama_accelerator: str = "auto",
) -> SetupPlan:
    """Choose install and build settings without importing optional runtimes."""

    machine = architecture.casefold()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    if system == "Darwin" and (macos_major is None or macos_major < 12):
        raise SetupError("osAi requires macOS 12 Monterey or newer")
    if system not in {"Darwin", "Linux", "Windows"}:
        raise SetupError(f"unsupported operating system: {system}")

    automatic_mlx: str | None
    if apple_silicon and macos_major is not None and macos_major >= 14:
        automatic_mlx = "metal"
    elif system == "Linux" and cuda_major in {12, 13}:
        automatic_mlx = "cuda"
    elif system == "Linux":
        automatic_mlx = "cpu"
    else:
        automatic_mlx = None

    selected_mlx = automatic_mlx if mlx_accelerator == "auto" else mlx_accelerator
    if selected_mlx == "off":
        selected_mlx = None
    if selected_mlx == "metal" and not (
        apple_silicon and macos_major is not None and macos_major >= 14
    ):
        raise SetupError("MLX Metal requires Apple silicon and macOS 14 or newer")
    if selected_mlx == "cuda" and not (
        system == "Linux" and cuda_major in {12, 13}
    ):
        raise SetupError("MLX CUDA requires Linux and a CUDA 12 or CUDA 13 toolkit")
    if selected_mlx == "cpu" and system != "Linux":
        raise SetupError("the bundled MLX CPU build is supported on Linux")
    if selected_mlx not in {None, "metal", "cuda", "cpu"}:
        raise SetupError("MLX accelerator must be auto, off, metal, cuda, or cpu")

    available_llama = {
        "metal": system == "Darwin" and macos_major is not None and macos_major >= 12,
        "cuda": cuda_major is not None,
        "vulkan": vulkan_available,
        "cpu": True,
    }
    if llama_accelerator == "auto":
        if available_llama["metal"]:
            selected_llama = "metal"
        elif available_llama["cuda"]:
            selected_llama = "cuda"
        elif available_llama["vulkan"]:
            selected_llama = "vulkan"
        else:
            selected_llama = "cpu"
    else:
        selected_llama = llama_accelerator
        if selected_llama not in available_llama:
            raise SetupError("llama.cpp accelerator must be auto, metal, cuda, vulkan, or cpu")
        if not available_llama[selected_llama]:
            raise SetupError(f"requested llama.cpp accelerator is unavailable: {selected_llama}")

    if selected_mlx is None:
        requirements = "requirements-llama.txt"
    elif selected_mlx == "cuda":
        assert cuda_major is not None
        requirements = f"requirements-linux-cuda{cuda_major}.txt"
    else:
        requirements = "requirements.txt"
    return SetupPlan(
        system=system,
        architecture=architecture,
        macos_major=macos_major,
        cuda_major=cuda_major,
        vulkan_available=vulkan_available,
        requirements=requirements,
        mlx_accelerator=selected_mlx,
        llama_accelerator=selected_llama,
    )


def detect_plan(
    *, mlx_accelerator: str = "auto", llama_accelerator: str = "auto"
) -> SetupPlan:
    system = platform.system()
    if system == "Windows":
        windows_version = getattr(sys, "getwindowsversion", None)
        if windows_version is not None and windows_version().major < 10:
            raise SetupError("osAi requires Windows 10 or Windows 11")
    return build_plan(
        system=system,
        architecture=platform.machine(),
        macos_major=_macos_major() if system == "Darwin" else None,
        cuda_major=_cuda_major(),
        vulkan_available=_vulkan_available(),
        mlx_accelerator=mlx_accelerator,
        llama_accelerator=llama_accelerator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Python environment, install osAi, build vendored MLX when "
            "supported, build llama.cpp, and run diagnostics."
        )
    )
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--venv",
        type=Path,
        default=PROJECT_ROOT / ".venv",
        help="virtual environment to create or reuse (default: .venv)",
    )
    environment.add_argument(
        "--current-environment",
        action="store_true",
        help="install into the Python environment running this script",
    )
    parser.add_argument("--wheel", type=Path, help="wheel to install; auto-detected in dist/")
    parser.add_argument(
        "--mlx-accelerator",
        choices=["auto", "off", "metal", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--llama-accelerator",
        choices=["auto", "metal", "cuda", "vulkan", "cpu"],
        default="auto",
    )
    parser.add_argument("--jobs", type=int, help="parallel llama.cpp build jobs")
    parser.add_argument("--dev", action="store_true", help="install test and lint tools")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="disable package-index access; requires a complete local wheelhouse",
    )
    parser.add_argument("--wheelhouse", type=Path, help="local dependency wheel directory")
    parser.add_argument(
        "--skip-mlx-build",
        action="store_true",
        help="use the selected pip MLX package instead of building the bundled source",
    )
    parser.add_argument(
        "--skip-llama-build",
        action="store_true",
        help="install only; do not compile bundled llama.cpp",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print detected settings and commands without changing the system",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_supported_python()
        if args.jobs is not None and args.jobs < 1:
            raise SetupError("--jobs must be at least 1")
        wheelhouse = _wheelhouse(args.offline, args.wheelhouse)
        plan = detect_plan(
            mlx_accelerator=args.mlx_accelerator,
            llama_accelerator=args.llama_accelerator,
        )
        requirements = REQUIREMENTS_ROOT / plan.requirements
        if not requirements.is_file():
            raise SetupError(f"missing requirements file: {requirements}")
        _require_vendored_sources(plan, args.skip_mlx_build, args.skip_llama_build)
        install_target = _install_target(args.wheel)
        target_python = _target_python(args, dry_run=args.dry_run)
        commands = _setup_commands(
            target_python=target_python,
            plan=plan,
            requirements=requirements,
            install_target=install_target,
            wheelhouse=wheelhouse,
            dev=args.dev,
            skip_mlx_build=args.skip_mlx_build,
            skip_llama_build=args.skip_llama_build,
            jobs=args.jobs,
        )
        _print_plan(plan, target_python, install_target, args)
        if args.dry_run:
            for command, _ in commands:
                print(f"+ {_display_command(command)}")
            return 0
        if not args.skip_llama_build and shutil.which("cmake") is None:
            raise SetupError("CMake is required to build bundled llama.cpp")
        environment = _child_environment()
        for command, extra_environment in commands:
            merged_environment = environment | extra_environment
            _run(command, environment=merged_environment)
        _print_next_step(args, target_python)
        return 0
    except (OSError, SetupError, subprocess.CalledProcessError) as exc:
        print(f"osai setup: {exc}", file=sys.stderr)
        return 2


def _setup_commands(
    *,
    target_python: Path,
    plan: SetupPlan,
    requirements: Path,
    install_target: Path,
    wheelhouse: Path | None,
    dev: bool,
    skip_mlx_build: bool,
    skip_llama_build: bool,
    jobs: int | None,
) -> list[tuple[list[str], dict[str, str]]]:
    commands: list[tuple[list[str], dict[str, str]]] = []
    index_arguments = _index_arguments(wheelhouse)

    def pip_install(*arguments: str) -> list[str]:
        return [
            str(target_python),
            "-m",
            "pip",
            "install",
            *index_arguments,
            *arguments,
        ]

    commands.append((pip_install("--upgrade", "pip", "setuptools", "wheel"), {}))
    commands.append((pip_install("-r", str(requirements)), {}))
    if dev:
        commands.append(
            (
                pip_install("-r", str(REQUIREMENTS_ROOT / "requirements-dev.txt")),
                {},
            )
        )
    if plan.mlx_accelerator is not None and not skip_mlx_build:
        mlx_environment = {"PYPI_RELEASE": "1"}
        if plan.system == "Linux":
            use_cuda = "ON" if plan.mlx_accelerator == "cuda" else "OFF"
            mlx_environment["CMAKE_ARGS"] = (
                f"-DMLX_BUILD_CUDA={use_cuda} -DMLX_BUILD_CPU=ON"
            )
        commands.append(
            (
                pip_install(
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(PROJECT_ROOT / "vendor" / "mlx"),
                ),
                mlx_environment,
            )
        )
        commands.append(
            (
                pip_install(
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(PROJECT_ROOT / "vendor" / "mlx-lm"),
                ),
                {},
            )
        )
        commands.append(
            (
                pip_install(
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    str(PROJECT_ROOT / "vendor" / "mlx-vlm"),
                ),
                {},
            )
        )
    install_arguments = ["--no-deps", "--force-reinstall"]
    if install_target == PROJECT_ROOT:
        install_arguments.append("--no-build-isolation")
    install_arguments.append(str(install_target))
    commands.append((pip_install(*install_arguments), {}))
    if not skip_llama_build:
        build_command = [
            str(target_python),
            "-m",
            "osai",
            "build-llama",
            "--accelerator",
            plan.llama_accelerator,
        ]
        if jobs is not None:
            build_command.extend(["--jobs", str(jobs)])
        commands.append((build_command, {}))
    commands.append(([str(target_python), "-m", "osai", "doctor"], {}))
    return commands


def _target_python(args: argparse.Namespace, *, dry_run: bool) -> Path:
    if args.current_environment:
        # Preserve a virtual-environment symlink instead of resolving it to the
        # system interpreter and accidentally installing outside the active venv.
        return Path(sys.executable).absolute()
    root = args.venv.expanduser().resolve()
    executable = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not root.exists():
        if dry_run:
            return executable
        print(f"Creating virtual environment: {root}")
        venv.EnvBuilder(with_pip=True).create(root)
    if not executable.is_file():
        raise SetupError(f"virtual environment has no Python executable: {executable}")
    _require_target_python(executable)
    return executable


def _install_target(explicit: Path | None) -> Path:
    if explicit is not None:
        selected = explicit.expanduser().resolve()
        if not selected.is_file() or selected.suffix != ".whl":
            raise SetupError(f"wheel does not exist: {selected}")
        return selected
    wheels = sorted((PROJECT_ROOT / "dist").glob("osai-0.1.0-*.whl"))
    if len(wheels) > 1:
        raise SetupError("multiple 0.1.0 wheels found; select one with --wheel")
    return wheels[0].resolve() if wheels else PROJECT_ROOT


def _wheelhouse(offline: bool, value: Path | None) -> Path | None:
    if value is not None:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise SetupError(f"wheelhouse directory does not exist: {resolved}")
        return resolved
    if offline:
        raise SetupError("--offline requires --wheelhouse with every Python dependency")
    return None


def _index_arguments(wheelhouse: Path | None) -> list[str]:
    if wheelhouse is None:
        return []
    return ["--no-index", "--find-links", str(wheelhouse)]


def _require_vendored_sources(
    plan: SetupPlan, skip_mlx_build: bool, skip_llama_build: bool
) -> None:
    if plan.mlx_accelerator is not None and not skip_mlx_build:
        for name in ("mlx", "mlx-lm", "mlx-vlm"):
            source = PROJECT_ROOT / "vendor" / name
            if not source.is_dir():
                raise SetupError(f"missing bundled source directory: {source}")
    llama = PROJECT_ROOT / "vendor" / "llama.cpp"
    if not skip_llama_build and not llama.is_dir():
        raise SetupError(f"missing bundled source directory: {llama}")


def _require_supported_python() -> None:
    version = sys.version_info[:2]
    if not SUPPORTED_PYTHON[0] <= version < SUPPORTED_PYTHON[1]:
        raise SetupError("osAi requires Python 3.10, 3.11, 3.12, or 3.13")


def _require_target_python(executable: Path) -> None:
    check = (
        "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] "
        "< (3, 14) else 1)"
    )
    completed = subprocess.run([str(executable), "-c", check], check=False)
    if completed.returncode != 0:
        raise SetupError(f"target environment uses an unsupported Python: {executable}")


def _macos_major() -> int | None:
    version = platform.mac_ver()[0]
    try:
        return int(version.split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _cuda_major() -> int | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    try:
        completed = subprocess.run(
            [nvcc, "--version"], text=True, capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"release\s+(\d+)", completed.stdout + completed.stderr)
    return int(match.group(1)) if match else None


def _vulkan_available() -> bool:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk and Path(sdk).is_dir():
        return True
    return any(
        shutil.which(command)
        for command in ("glslc", "glslangValidator", "vulkaninfo")
    )


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    return environment


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print(f"+ {_display_command(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _print_plan(
    plan: SetupPlan,
    target_python: Path,
    install_target: Path,
    args: argparse.Namespace,
) -> None:
    payload = plan.as_dict()
    payload.update(
        {
            "python": str(target_python),
            "install_target": str(install_target),
            "offline": bool(args.offline),
            "build_bundled_mlx": bool(
                plan.mlx_accelerator is not None and not args.skip_mlx_build
            ),
            "build_bundled_llama_cpp": not args.skip_llama_build,
        }
    )
    print("Detected setup:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_next_step(args: argparse.Namespace, target_python: Path) -> None:
    print("\nosAi setup completed successfully.")
    if args.current_environment:
        print(f"Run: {target_python} -m osai doctor")
        return
    root = args.venv.expanduser().resolve()
    if os.name == "nt":
        print(f"Activate: {root / 'Scripts' / 'Activate.ps1'}")
    else:
        print(f"Activate: source {shlex.quote(str(root / 'bin' / 'activate'))}")
    print("Then run: osai models")


if __name__ == "__main__":
    raise SystemExit(main())
