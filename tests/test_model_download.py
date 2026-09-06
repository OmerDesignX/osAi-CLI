import hashlib
import io
import json
from pathlib import Path

import pytest

import osai.model_download as downloads
from osai.errors import ModelDownloadError, VerificationError


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url


def test_variant_file_lists_match_published_shard_counts():
    gguf = downloads.variant_for("llama.cpp", "small")
    mlx = downloads.variant_for("mlx", "small")

    assert len(downloads.files_for_variant(gguf)) == 2
    assert downloads.files_for_variant(gguf)[-1].endswith("00002-of-00002.gguf")
    assert len(downloads.files_for_variant(mlx)) == 27
    assert downloads.files_for_variant(mlx)[-1].endswith("00021-of-00021.safetensors")


def test_downloads_verifies_and_atomically_activates_selected_variant(
    monkeypatch, tmp_path: Path
):
    variant = downloads.variant_for("llama.cpp", "small")
    remote_files = downloads.files_for_variant(variant)
    payloads = {
        path: f"GGUF fixture {index}".encode()
        for index, path in enumerate(remote_files, 1)
    }
    release = {
        "release": "1.0",
        "variants": [
            {
                "runtime": "llama.cpp",
                "tier": "small",
                "path": variant.repository_path,
                "bytes": variant.bytes,
                "files": list(remote_files),
            }
        ],
    }
    sums = "\n".join(
        f"{hashlib.sha256(payload).hexdigest()}  ./{path}"
        for path, payload in payloads.items()
    )

    def open_response(url: str):
        if url.endswith("/release.json"):
            return _Response(json.dumps(release).encode(), url)
        if url.endswith("/SHA256SUMS"):
            return _Response(sums.encode(), url)
        for path, payload in payloads.items():
            if url.endswith(path):
                return _Response(payload, url)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(downloads, "_open_response", open_response)
    monkeypatch.setattr(downloads, "_check_disk_budget", lambda *_: None)
    progress = []

    result = downloads.download_model_variant(
        tmp_path, variant, progress=progress.append
    )

    assert result.downloaded is True
    assert result.model == tmp_path / "GGUF" / "small" / Path(variant.repository_path).name
    assert result.model.read_bytes() == payloads[remote_files[0]]
    assert progress[-1].percent == 100
    verified = downloads.verify_download_manifest(result.manifest)
    assert verified.files == 2
    assert not any((tmp_path / ".downloads").glob("llama.cpp-small-*"))

    monkeypatch.setattr(
        downloads,
        "_open_response",
        lambda _url: (_ for _ in ()).throw(AssertionError("network used")),
    )
    existing = downloads.ensure_official_model(
        tmp_path, runtime="llama.cpp", tier="small"
    )
    assert existing.downloaded is False


def test_checksum_failure_does_not_activate_partial_model(monkeypatch, tmp_path: Path):
    variant = downloads.variant_for("llama.cpp", "small")
    remote_files = downloads.files_for_variant(variant)
    release = {
        "release": "1.0",
        "variants": [
            {
                "runtime": "llama.cpp",
                "tier": "small",
                "path": variant.repository_path,
                "bytes": variant.bytes,
            }
        ],
    }

    def open_response(url: str):
        if url.endswith("/release.json"):
            return _Response(json.dumps(release).encode(), url)
        if url.endswith("/SHA256SUMS"):
            sums = "\n".join(f"{'0' * 64}  ./{path}" for path in remote_files)
            return _Response(sums.encode(), url)
        return _Response(b"not the expected model", url)

    monkeypatch.setattr(downloads, "_open_response", open_response)
    monkeypatch.setattr(downloads, "_check_disk_budget", lambda *_: None)

    with pytest.raises(VerificationError, match="SHA-256"):
        downloads.download_model_variant(tmp_path, variant)

    assert not variant.destination(tmp_path).exists()
    assert not any((tmp_path / ".downloads").glob("llama.cpp-small-*"))


def test_published_file_list_cannot_escape_model_folder():
    variant = downloads.variant_for("mlx", "small")
    with pytest.raises(VerificationError, match="leaves"):
        downloads._published_files(
            {"files": [f"{variant.repository_path}/../secret"]}, variant
        )


def test_download_redirect_must_remain_on_github():
    response = _Response(b"data", "https://example.com/model.gguf")
    with pytest.raises(VerificationError, match="untrusted host"):
        downloads._validate_response_url(response)


def test_console_progress_renders_a_bar():
    output = io.StringIO()
    progress = downloads.ConsoleProgress(output)
    progress(downloads.DownloadProgress(50, "model.gguf", 50, 100))
    progress(downloads.DownloadProgress(100, "Ready", 100, 100))

    rendered = output.getvalue()
    assert "[############------------]" in rendered
    assert "100%" in rendered


def test_offline_environment_prevents_missing_model_download(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OSAI_OFFLINE", "1")
    monkeypatch.setattr(
        downloads,
        "_open_response",
        lambda _url: (_ for _ in ()).throw(AssertionError("network used")),
    )

    with pytest.raises(ModelDownloadError, match="OSAI_OFFLINE"):
        downloads.ensure_official_model(
            tmp_path, runtime="llama.cpp", tier="small"
        )
