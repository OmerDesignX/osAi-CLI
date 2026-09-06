import os
from pathlib import Path

import pytest

from osai.catalog import bundled_entry, custom_entry
from osai.errors import ConfigurationError


def test_custom_models_live_in_named_format_folders(tmp_path: Path):
    folder = tmp_path / "my-model"
    (folder / "mlx").mkdir(parents=True)
    (folder / "gguf").mkdir()
    (folder / "mlx" / "config.json").write_text("{}")
    (folder / "gguf" / "model-Q4_K_M.gguf").write_bytes(b"GGUF")

    entry = custom_entry("my-model", tmp_path)

    assert entry.mlx == folder / "mlx"
    assert entry.gguf == folder / "gguf" / "model-Q4_K_M.gguf"


def test_official_gguf_models_use_atomic_tier_directories(tmp_path: Path):
    entry = bundled_entry("small", tmp_path)

    assert entry.source == "official"
    assert entry.gguf == (
        tmp_path
        / "GGUF"
        / "small"
        / "osCode-GGUF-Small-Q4_K_M-00001-of-00002.gguf"
    )


def test_custom_model_name_cannot_escape_root(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="custom model name"):
        custom_entry("../escape", tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks may require Windows privileges")
def test_custom_format_symlink_cannot_escape_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-model"
    outside.mkdir()
    folder = tmp_path / "my-model"
    folder.mkdir()
    (folder / "mlx").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="must remain inside"):
        custom_entry("my-model", tmp_path)
