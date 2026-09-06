"""Offline integrity verification for downloaded official models."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import VerificationError
from .model_download import MODEL_MANIFEST, verify_download_manifest

_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True, slots=True)
class BundleVerification:
    root: Path
    variants: int
    files: int
    bytes: int
    lfs_pointers: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        return result


def verify_model_bundle(root: str | Path) -> BundleVerification:
    location = Path(root).expanduser().resolve()
    if not location.is_dir():
        raise VerificationError(f"model directory does not exist: {location}")
    checksum_file = location / "SHA256SUMS"
    if checksum_file.is_file():
        return _verify_legacy_snapshot(location, checksum_file)
    manifests = sorted(
        path
        for path in location.rglob(MODEL_MANIFEST)
        if ".downloads" not in path.parts
    )
    checked = 0
    total_bytes = 0
    for manifest in manifests:
        result = verify_download_manifest(manifest)
        checked += result.files
        total_bytes += result.bytes
    return BundleVerification(location, len(manifests), checked, total_bytes, 0)


def _verify_legacy_snapshot(location: Path, checksum_file: Path) -> BundleVerification:
    try:
        lines = checksum_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerificationError(f"cannot read model checksum manifest: {exc}") from exc
    checked = 0
    total_bytes = 0
    pointers = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise VerificationError(f"invalid SHA256SUMS line {line_number}") from exc
        valid_digest = len(expected) == 64 and all(
            character in "0123456789abcdef" for character in expected
        )
        if not valid_digest:
            raise VerificationError(f"invalid SHA-256 on line {line_number}")
        relative = relative.removeprefix("*").removeprefix("./")
        path = (location / relative).resolve()
        try:
            path.relative_to(location)
        except ValueError as exc:
            raise VerificationError(f"checksum path escapes model root: {relative}") from exc
        if not path.is_file():
            raise VerificationError(f"bundled model file is missing: {relative}")
        with path.open("rb") as handle:
            prefix = handle.read(len(_LFS_PREFIX))
            if prefix == _LFS_PREFIX:
                raise VerificationError(f"bundled file is still a Git LFS pointer: {relative}")
            handle.seek(0)
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise VerificationError(f"SHA-256 mismatch for bundled model file: {relative}")
        checked += 1
        total_bytes += path.stat().st_size
    return BundleVerification(location, 1, checked, total_bytes, pointers)
