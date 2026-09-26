"""Project and vendored dependency discovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("OSAI_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").exists():
        return source_root

    # The desktop app installs a wheel in .venv and keeps the downloaded
    # repository beside it. Console entry points need to find that source too.
    installation = Path(sys.executable).absolute()
    if len(installation.parents) >= 3:
        extracted = installation.parents[2] / "source"
        if extracted.is_dir():
            candidates = (extracted, *sorted(path for path in extracted.iterdir() if path.is_dir()))
            for candidate in candidates:
                if (candidate / "pyproject.toml").is_file() and (
                    candidate / "vendor" / "llama.cpp" / "CMakeLists.txt"
                ).is_file():
                    return candidate.resolve()

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "vendor" / "llama.cpp").is_dir():
            return candidate
    return source_root


def llama_cpp_root() -> Path:
    return project_root() / "vendor" / "llama.cpp"


def llama_runtime_build() -> Path:
    """Writable native build cache, separate from the installed application."""

    override = os.environ.get("OSAI_LLAMA_BUILD_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "osAi" / "llama-build"


def mlx_lm_root() -> Path:
    return project_root() / "vendor" / "mlx-lm"


def mlx_vlm_root() -> Path:
    return project_root() / "vendor" / "mlx-vlm"


def llama_binary(name: str) -> Path | None:
    root = llama_cpp_root()
    suffix = ".exe" if os.name == "nt" else ""
    # llama.cpp renamed its primary text-generation executable from
    # llama-cli to llama-completion.  Accept both so osai remains
    # compatible with vendored snapshots on either side of that change.
    aliases = (name, "llama-completion") if name == "llama-cli" else (name,)
    from_cache = llama_runtime_build()
    build_roots = (
        (from_cache, root / "build")
        if (from_cache / "OSAI_BUILD.json").is_file()
        else (root / "build",)
    )
    candidates = tuple(
        candidate
        for build in build_roots
        for alias in aliases
        for candidate in (
            build / "bin" / f"{alias}{suffix}",
            build / "bin" / "Release" / f"{alias}{suffix}",
            build / "Release" / f"{alias}{suffix}",
        )
    )
    return next((path for path in candidates if path.is_file()), None)
