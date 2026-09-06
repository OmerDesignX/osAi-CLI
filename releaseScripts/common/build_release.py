#!/usr/bin/env python3
"""Build and verify the osAi wheel locally without GitHub Actions."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
EGG_INFO = ROOT / "src" / "osai.egg-info"
RELEASE_ASSETS = ROOT / "release-assets"
GENERATED_PATHS = (DIST, BUILD, EGG_INFO)
SUPPORTED_PYTHON = {(3, minor) for minor in range(10, 14)}
VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")


def run(*arguments: str | Path) -> None:
    command = [str(argument) for argument in arguments]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def remove_generated(path: Path) -> None:
    allowed = {candidate.resolve(strict=False) for candidate in (*GENERATED_PATHS, RELEASE_ASSETS)}
    resolved = path.resolve(strict=False)
    if resolved not in allowed:
        raise RuntimeError(f"refusing to remove unexpected path: {resolved}")
    if path.is_symlink():
        raise RuntimeError(f"refusing to remove linked build output: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def project_version() -> str:
    version = (ROOT / "VERSION.txt").read_text(encoding="utf-8").strip()
    if VERSION_PATTERN.fullmatch(version) is None:
        raise RuntimeError(f"invalid VERSION.txt value: {version!r}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_section = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", pyproject)
    configured = (
        re.search(r'^version\s*=\s*"([^"]+)"\s*$', project_section.group(1), re.MULTILINE)
        if project_section
        else None
    )
    package = (ROOT / "src" / "osai" / "__init__.py").read_text(encoding="utf-8")
    exported = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', package, re.MULTILINE)
    if configured is None or exported is None:
        raise RuntimeError("could not read every package version")
    if {version, configured.group(1), exported.group(1)} != {version}:
        raise RuntimeError("VERSION.txt, pyproject.toml, and osai.__version__ do not match")
    return version


def environment_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def verify_wheel(wheel: Path, version: str) -> None:
    if not wheel.is_file() or wheel.stat().st_size < 50_000:
        raise RuntimeError("the release wheel is missing or unexpectedly small")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise RuntimeError("the wheel has invalid package metadata")
        if "osai/cli.py" not in names or "osai/__init__.py" not in names:
            raise RuntimeError("the wheel is missing the osAi package")
        forbidden = (".so", ".dylib", ".dll", ".exe")
        if any(name.lower().endswith(forbidden) for name in names):
            raise RuntimeError("the universal wheel unexpectedly contains a native binary")

        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        wheel_metadata = archive.read(wheel_names[0]).decode("utf-8")
        if metadata.get("Name") != "osai" or metadata.get("Version") != version:
            raise RuntimeError("the wheel name or version does not match VERSION.txt")
        if "Tag: py3-none-any" not in wheel_metadata:
            raise RuntimeError("the wheel is not marked as platform-independent")


def main() -> int:
    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        raise RuntimeError("release builds require Python 3.10, 3.11, 3.12, or 3.13")
    version = project_version()

    for path in (*GENERATED_PATHS, RELEASE_ASSETS):
        remove_generated(path)

    try:
        with tempfile.TemporaryDirectory(prefix="osai-release-") as temporary:
            environment = Path(temporary) / "venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment_python(environment)
            run(
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                "pip",
                "setuptools>=69",
                "wheel",
            )
            run(
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-e",
                ".[dev]",
            )
            run(python, "-m", "ruff", "check", "src", "tests", "scripts", "releaseScripts")
            run(python, "-m", "pytest")
            run(python, "-m", "build", "--wheel", "--no-isolation", "--outdir", DIST)

            wheels = sorted(DIST.glob("osai-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("expected exactly one osAi wheel")
            wheel = wheels[0]
            verify_wheel(wheel, version)

            RELEASE_ASSETS.mkdir(parents=True)
            destination = RELEASE_ASSETS / wheel.name
            shutil.copy2(wheel, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            checksum = RELEASE_ASSETS / f"{wheel.name}.sha256"
            checksum.write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
            print(f"\nVerified release assets:\n  {destination}\n  {checksum}")
    finally:
        for path in GENERATED_PATHS:
            remove_generated(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
