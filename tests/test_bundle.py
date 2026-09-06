import hashlib
from pathlib import Path

import pytest

from osai.bundle import verify_model_bundle
from osai.errors import VerificationError


def test_verifies_local_checksum_manifest(tmp_path: Path):
    payload = tmp_path / "model.gguf"
    payload.write_bytes(b"GGUF-local")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  ./model.gguf\n")

    result = verify_model_bundle(tmp_path)

    assert result.files == 1
    assert result.variants == 1
    assert result.lfs_pointers == 0


def test_rejects_lfs_pointer_even_when_hash_matches(tmp_path: Path):
    payload = tmp_path / "model.gguf"
    payload.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  ./model.gguf\n")

    with pytest.raises(VerificationError, match="LFS pointer"):
        verify_model_bundle(tmp_path)


def test_empty_download_directory_is_valid(tmp_path: Path):
    result = verify_model_bundle(tmp_path)

    assert result.variants == 0
    assert result.files == 0
    assert result.bytes == 0
