"""Read the atomic native GGUF resume snapshot without loading it into memory."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from .errors import VerificationError

MAGIC = b"OSAIRSM1"
MAX_ADAPTER_BYTES = 1 << 34
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def extract_adapter(snapshot: Path, destination: Path) -> None:
    """Extract and verify the adapter embedded in a resumable checkpoint."""

    temporary = destination.with_name(destination.name + ".pending")
    try:
        with snapshot.open("rb") as source:
            if source.read(8) != MAGIC:
                raise VerificationError("this checkpoint has no exact resume state")
            raw_size = source.read(8)
            if len(raw_size) != 8:
                raise VerificationError("the resume checkpoint is incomplete")
            size = struct.unpack("<Q", raw_size)[0]
            if not 0 < size <= MAX_ADAPTER_BYTES:
                raise VerificationError("the resume checkpoint has an invalid adapter size")
            hash_value = FNV_OFFSET
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as target:
                remaining = size
                while remaining:
                    chunk = source.read(min(1 << 20, remaining))
                    if not chunk:
                        raise VerificationError("the resume checkpoint adapter is truncated")
                    target.write(chunk)
                    for byte in chunk:
                        hash_value = ((hash_value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
                    remaining -= len(chunk)
                target.flush()
                os.fsync(target.fileno())
            saved_hash = source.read(8)
            if len(saved_hash) != 8 or struct.unpack("<Q", saved_hash)[0] != hash_value:
                raise VerificationError("the resume checkpoint adapter failed integrity checking")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
