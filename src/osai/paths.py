"""Project and vendored dependency discovery."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("OSAI_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").exists():
        return source_root

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "vendor" / "llama.cpp").is_dir():
            return candidate
    return source_root


def llama_cpp_root() -> Path:
    return project_root() / "vendor" / "llama.cpp"


def mlx_lm_root() -> Path:
    return project_root() / "vendor" / "mlx-lm"


def llama_binary(name: str) -> Path | None:
    root = llama_cpp_root()
    suffix = ".exe" if os.name == "nt" else ""
    # llama.cpp renamed its primary text-generation executable from
    # llama-cli to llama-completion.  Accept both so osai remains
    # compatible with vendored snapshots on either side of that change.
    aliases = (name, "llama-completion") if name == "llama-cli" else (name,)
    candidates = tuple(
        candidate
        for alias in aliases
        for candidate in (
            root / "build" / "bin" / f"{alias}{suffix}",
            root / "build" / "bin" / "Release" / f"{alias}{suffix}",
            root / "build" / "Release" / f"{alias}{suffix}",
        )
    )
    return next((path for path in candidates if path.is_file()), None)
