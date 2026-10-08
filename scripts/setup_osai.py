#!/usr/bin/env python3
"""Install the osAi framework and build its bundled engines for this hardware."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import urllib.request
import venv
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_ROOT = PROJECT_ROOT / "requirements"
SUPPORTED_PYTHON = (3, 10), (3, 14)
WINDOWS_LLVM_URL = (
    "https://github.com/mstorsjo/llvm-mingw/releases/download/20260616/"
    "llvm-mingw-20260616-ucrt-x86_64.zip"
)
WINDOWS_LLVM_SHA256 = "b9b68a4d276e16fa25802aaba458e4638f64b3884c290aaccdc2d87083b6ca35"
WINDOWS_VULKAN_VERSION = "1.4.357.0"
WINDOWS_VULKAN_URL = (
    "https://sdk.lunarg.com/sdk/download/1.4.357.0/windows/vulkansdk-windows-X64-1.4.357.0.exe"
)
WINDOWS_VULKAN_SHA256 = "81f474711e9042f4cd22b31b2f7a8870db2e428b21586fb43dd80150be97310d"
WINDOWS_VC_REDIST_URL = (
    "https://download.visualstudio.microsoft.com/download/pr/"
    "bd1c8d9d-ba95-4eee-bc6e-df1fcc876373/"
    "CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B/VC_redist.x64.exe"
)
WINDOWS_VC_REDIST_SHA256 = "cc0ff0eb1dc3f5188ae6300faef32bf5beeba4bdd6e8e445a9184072096b713b"


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
    if selected_mlx == "cuda" and not (system == "Linux" and cuda_major in {12, 13}):
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


def detect_plan(*, mlx_accelerator: str = "auto", llama_accelerator: str = "auto") -> SetupPlan:
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
        _discover_local_sdks(
            install_missing=not args.dry_run,
            prefer_vulkan=args.llama_accelerator == "vulkan",
        )
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
        environment = _child_environment(target_python)
        if not args.skip_llama_build and plan.system == "Windows":
            environment.update(_windows_compiler_environment(plan, target_python))
            if plan.llama_accelerator == "vulkan" and _vulkan_available():
                _ensure_windows_vulkan_runtime(environment)
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

    commands.append((pip_install("--upgrade", "pip", "setuptools", "wheel", "cmake", "ninja"), {}))
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
            mlx_environment["CMAKE_ARGS"] = f"-DMLX_BUILD_CUDA={use_cuda} -DMLX_BUILD_CPU=ON"
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
        if plan.llama_accelerator != "cpu":
            # An available GPU must not silently become a CPU-only install.
            build_command.append("--no-cpu-fallback")
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
    version = (PROJECT_ROOT / "VERSION.txt").read_text(encoding="utf-8").strip()
    wheels = sorted((PROJECT_ROOT / "dist").glob(f"osai-{version}-*.whl"))
    if len(wheels) > 1:
        raise SetupError(f"multiple {version} wheels found; select one with --wheel")
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
    check = "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)"
    completed = subprocess.run([str(executable), "-c", check], check=False)
    if completed.returncode != 0:
        raise SetupError(f"target environment uses an unsupported Python: {executable}")


def _macos_major() -> int | None:
    version = platform.mac_ver()[0]
    try:
        return int(version.split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _discover_local_sdks(*, install_missing: bool = False, prefer_vulkan: bool = False) -> None:
    """Find conventional SDK installs without asking for environment configuration."""

    if platform.system() == "Linux" and not os.environ.get("CUDA_PATH"):
        for candidate in (Path("/usr/local/cuda"), Path("/opt/cuda")):
            if (candidate / "bin" / "nvcc").is_file():
                os.environ["CUDA_PATH"] = str(candidate)
                break
    if platform.system() != "Windows":
        return
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    if not os.environ.get("CUDA_PATH"):
        cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
        candidates = sorted(cuda_root.glob("v*"), reverse=True)
        for candidate in candidates:
            if (candidate / "bin" / "nvcc.exe").is_file():
                os.environ["CUDA_PATH"] = str(candidate)
                break
    if not os.environ.get("VULKAN_SDK"):
        for root in (Path(r"C:\VulkanSDK"), program_files / "VulkanSDK"):
            for candidate in sorted(root.glob("*"), reverse=True):
                if (candidate / "Bin" / "glslc.exe").is_file():
                    os.environ["VULKAN_SDK"] = str(candidate)
                    break
            if os.environ.get("VULKAN_SDK"):
                break
    if (
        not os.environ.get("VULKAN_SDK")
        and install_missing
        and (prefer_vulkan or _cuda_major() is None)
        and (
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "vulkan-1.dll"
        ).is_file()
    ):
        try:
            os.environ["VULKAN_SDK"] = str(_install_windows_vulkan_sdk())
        except (
            OSError,
            SetupError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            print(f"osai setup: Vulkan SDK download unavailable ({exc}); continuing", flush=True)


def _cuda_major() -> int | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        sdk = os.environ.get("CUDA_PATH")
        if sdk:
            candidate = Path(sdk) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            if candidate.is_file():
                nvcc = str(candidate)
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
    if sdk and (Path(sdk) / ("Bin/glslc.exe" if os.name == "nt" else "bin/glslc")).is_file():
        return True
    return shutil.which("glslc") is not None


def _child_environment(target_python: Path) -> dict[str, str]:
    environment = os.environ.copy()
    current_path = next(
        (value for key, value in environment.items() if key.casefold() == "path"), ""
    )
    for key in list(environment):
        if key.casefold() == "path":
            del environment[key]
    environment["PATH"] = str(target_python.parent) + os.pathsep + current_path
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


def _windows_compiler_environment(plan: SetupPlan, target_python: Path) -> dict[str, str]:
    """Select MSVC automatically, or install a verified portable C++ compiler."""

    if plan.system != "Windows":
        return {}
    developer_command = os.environ.get("OSAI_VSDEVCMD")
    if not developer_command:
        program_files = os.environ.get("PROGRAMFILES(X86)")
        if program_files:
            vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            if vswhere.is_file():
                result = subprocess.run(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                installation = result.stdout.strip()
                if result.returncode == 0 and installation:
                    developer_command = str(
                        Path(installation) / "Common7" / "Tools" / "VsDevCmd.bat"
                    )
    if developer_command and Path(developer_command).is_file():
        if any(character in developer_command for character in ('"', "\r", "\n")):
            raise SetupError("the Microsoft compiler setup path is invalid")
        command = f'call "{developer_command}" -arch=amd64 >nul && set'
        result = subprocess.run(
            command,
            shell=True,
            executable=os.environ.get("COMSPEC", "cmd.exe"),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            compiler_environment = {}
            for line in result.stdout.splitlines():
                key, separator, value = line.partition("=")
                if separator and key and not key.startswith("="):
                    compiler_environment[key] = value
            compiler_path = next(
                (value for key, value in compiler_environment.items() if key.casefold() == "path"),
                "",
            )
            for key in list(compiler_environment):
                if key.casefold() == "path":
                    del compiler_environment[key]
            compiler_environment["CC"] = "cl"
            compiler_environment["CXX"] = "cl"
            compiler_environment["CMAKE_GENERATOR"] = "Ninja"
            compiler_environment["PATH"] = str(target_python.parent) + os.pathsep + compiler_path
            print("Using detected Microsoft C++ Build Tools", flush=True)
            return compiler_environment
    print("Downloading the verified portable C++ toolchain", flush=True)
    tools = PROJECT_ROOT / "build" / "osai-tools"
    compiler = tools / "llvm-mingw-20260616-ucrt-x86_64" / "bin" / "clang++.exe"
    if not compiler.is_file():
        archive = tools / "llvm-mingw.zip"
        tools.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or _file_sha256(archive) != WINDOWS_LLVM_SHA256:
            archive.unlink(missing_ok=True)
            _download_windows_toolchain(archive)
        with zipfile.ZipFile(archive) as bundle:
            total_size = 0
            for member in bundle.infolist():
                parts = PurePosixPath(member.filename).parts
                total_size += member.file_size
                if (
                    member.filename.startswith("/")
                    or "\\" in member.filename
                    or ".." in parts
                    or any(":" in part for part in parts)
                    or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK
                    or total_size > 4_000_000_000
                ):
                    raise SetupError("the portable C++ toolchain archive has an unsafe entry")
            bundle.extractall(tools)
        if not compiler.is_file():
            raise SetupError("the portable C++ toolchain archive is incomplete")
    binary_dir = compiler.parent
    return {
        "CC": str(binary_dir / "clang.exe"),
        "CXX": str(compiler),
        "CMAKE_GENERATOR": "Ninja",
        # llvm-mingw defaults to Windows 7, which hides CreateFile2 used by
        # llama.cpp. Match osAi's Windows 10 minimum without changing MSVC.
        "CFLAGS": (os.environ.get("CFLAGS", "") + " -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00").strip(),
        "CXXFLAGS": (
            os.environ.get("CXXFLAGS", "") + " -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"
        ).strip(),
        "PATH": str(target_python.parent)
        + os.pathsep
        + str(binary_dir)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }


def _download_windows_toolchain(archive: Path) -> None:
    request = urllib.request.Request(WINDOWS_LLVM_URL, headers={"User-Agent": "osAi-CLI"})
    for attempt in range(2):
        temporary = archive.with_suffix(".part")
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                if response.url.split("/", 3)[2] not in {
                    "github.com",
                    "objects.githubusercontent.com",
                    "release-assets.githubusercontent.com",
                }:
                    raise SetupError("the C++ toolchain download redirected to an untrusted host")
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 250_000_000:
                        raise SetupError("the C++ toolchain download exceeded its size limit")
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() == WINDOWS_LLVM_SHA256:
                temporary.replace(archive)
                return
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Toolchain checksum mismatch; retrying download ({attempt + 1}/2)", flush=True)
    raise SetupError("the portable C++ toolchain failed SHA-256 verification")


def _probe_windows_glslc(
    compiler: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess:
    # Missing DLLs otherwise open a Windows error dialog and stall unattended setup.
    kernel32 = ctypes.windll.kernel32
    previous_mode = kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    try:
        return subprocess.run(
            [str(compiler), "--version"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        kernel32.SetErrorMode(previous_mode)


def _ensure_windows_vulkan_runtime(environment: dict[str, str]) -> None:
    """Preflight the shader compiler and supply its missing Microsoft C++ runtime."""

    sdk = environment.get("VULKAN_SDK")
    compiler = Path(sdk) / "Bin" / "glslc.exe" if sdk else None
    if compiler is None or not compiler.is_file():
        found = shutil.which("glslc", path=environment.get("PATH", ""))
        if not found:
            raise SetupError("the selected Vulkan shader compiler could not be found")
        compiler = Path(found)
    try:
        result = _probe_windows_glslc(compiler, environment)
        if result.returncode == 0:
            return
        if result.returncode & 0xFFFFFFFF not in {0xC0000135, 0xC0000139, 0xC0000005}:
            details = (result.stdout + result.stderr).strip()
            raise SetupError(f"Vulkan shader compiler failed ({result.returncode}): {details}")
        # copy_only SDK extraction intentionally skips system prerequisites.
        # Its bundled redistributable is older than the shader compiler needs.
        installer = _windows_vc_redist_installer()
        if not installer.is_file() or _file_sha256(installer) != WINDOWS_VC_REDIST_SHA256:
            raise SetupError("the Microsoft C++ runtime failed SHA-256 verification")
        print(
            "Installing the verified Microsoft C++ runtime required by Vulkan; "
            "Windows may request administrator approval",
            flush=True,
        )
        installed = subprocess.run(
            [str(installer), "/install", "/quiet", "/norestart"],
            env=environment,
            check=False,
            timeout=10 * 60,
        )
        if installed.returncode not in {0, 1638, 3010}:
            raise SetupError(
                "Microsoft C++ runtime installation failed "
                f"(exit code {installed.returncode}); approve the Windows administrator "
                "prompt or install the Microsoft Visual C++ x64 Redistributable and retry"
            )
        if _probe_windows_glslc(compiler, environment).returncode != 0:
            raise SetupError(
                "Vulkan shader compiler still cannot start after installing the Microsoft "
                "C++ runtime; restart Windows if requested, then retry setup"
            )
    except subprocess.TimeoutExpired as exc:
        raise SetupError("timed out preparing the Vulkan shader compiler runtime") from exc


def _windows_vc_redist_installer() -> Path:
    """Download the pinned Microsoft Visual C++ 14.44 x64 Redistributable."""

    installer = PROJECT_ROOT / "build" / "osai-tools" / "VC_redist.x64.exe"
    if installer.is_file() and _file_sha256(installer) == WINDOWS_VC_REDIST_SHA256:
        return installer
    installer.parent.mkdir(parents=True, exist_ok=True)
    temporary = installer.with_suffix(".part")
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(WINDOWS_VC_REDIST_URL, headers={"User-Agent": "osAi-CLI"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            if response.url.split("/", 3)[2] != "download.visualstudio.microsoft.com":
                raise SetupError("the Microsoft C++ runtime redirected to an untrusted host")
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 40_000_000:
                    raise SetupError("the Microsoft C++ runtime exceeded its download size limit")
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != WINDOWS_VC_REDIST_SHA256:
            raise SetupError("the Microsoft C++ runtime failed SHA-256 verification")
        temporary.replace(installer)
    finally:
        temporary.unlink(missing_ok=True)
    return installer


def _install_windows_vulkan_sdk() -> Path:
    """Copy a verified Vulkan SDK into the private source build directory."""

    tools = PROJECT_ROOT / "build" / "osai-tools"
    sdk = tools / "vulkan-sdk" / WINDOWS_VULKAN_VERSION
    if (sdk / "Bin" / "glslc.exe").is_file() and (
        sdk / "Include" / "vulkan" / "vulkan.h"
    ).is_file():
        return sdk
    if shutil.disk_usage(PROJECT_ROOT).free < 3 * 1024**3:
        raise SetupError("at least 3 GiB free disk is needed for the Vulkan SDK")
    tools.mkdir(parents=True, exist_ok=True)
    installer = tools / f"vulkansdk-{WINDOWS_VULKAN_VERSION}.exe"
    if not installer.is_file() or _file_sha256(installer) != WINDOWS_VULKAN_SHA256:
        installer.unlink(missing_ok=True)
        temporary = installer.with_suffix(".part")
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(WINDOWS_VULKAN_URL, headers={"User-Agent": "osAi-CLI"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                if response.url.split("/", 3)[2] not in {"sdk.lunarg.com", "vulkan.lunarg.com"}:
                    raise SetupError("the Vulkan SDK download redirected to an untrusted host")
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 350_000_000:
                        raise SetupError("the Vulkan SDK download exceeded its size limit")
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != WINDOWS_VULKAN_SHA256:
                raise SetupError("the Vulkan SDK failed SHA-256 verification")
            temporary.replace(installer)
        finally:
            temporary.unlink(missing_ok=True)
    sdk.parent.mkdir(parents=True, exist_ok=True)
    print("Installing the verified Vulkan SDK into the local build cache", flush=True)
    subprocess.run(
        [
            str(installer),
            "--root",
            str(sdk),
            "--accept-licenses",
            "--default-answer",
            "--confirm-command",
            "install",
            "copy_only=1",
        ],
        check=True,
        timeout=30 * 60,
    )
    if not (sdk / "Bin" / "glslc.exe").is_file():
        raise SetupError("the downloaded Vulkan SDK did not contain glslc.exe")
    return sdk


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "build_bundled_mlx": bool(plan.mlx_accelerator is not None and not args.skip_mlx_build),
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
