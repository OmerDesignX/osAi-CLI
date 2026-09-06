"""Crash-safe local I/O, locking, and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import TrainingError


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def atomic_text(path: str | Path, value: str) -> None:
    """Replace a UTF-8 text file only after its contents reach stable storage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    size: int
    modified_ns: int
    sha256: str | None = None


def fingerprint(path: str | Path, *, full_hash: bool = False) -> FileFingerprint:
    file_path = Path(path)
    stat = file_path.stat()
    digest = sha256_file(file_path) if full_hash else None
    return FileFingerprint(
        path=str(file_path.resolve()),
        size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=digest,
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OutputLock:
    """Simple cross-platform single-writer lock based on O_EXCL."""

    def __init__(self, output: str | Path):
        self.output = Path(output)
        self.path = self.output / ".osai.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> OutputLock:
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._descriptor, f"pid={os.getpid()}\n".encode())
        except FileExistsError as exc:
            raise TrainingError(
                f"output is locked by another run (or a stale lock): {self.path}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        with suppress(FileNotFoundError):
            self.path.unlink()
