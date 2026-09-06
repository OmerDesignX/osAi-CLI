import json
from pathlib import Path

import pytest

from osai.errors import ConfigurationError, VerificationError
from osai.fusion import (
    FUSION_MANIFEST,
    GGUF_FUSION_KIND,
    MLX_FUSION_KIND,
    create_gguf_fusion_bundle,
    create_mlx_fusion_bundle,
    resolve_gguf_fusion_bundle,
    resolve_mlx_fusion_adapter,
)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"quantization":{"bits":5}}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"quantized-weights")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"exact-residual")
    return base, adapter


def test_mlx_fusion_keeps_base_bytes_and_embeds_adapter(tmp_path: Path):
    base, adapter = _sources(tmp_path)
    result = create_mlx_fusion_bundle(base, adapter, tmp_path / "merged")

    assert (result.path / "model.safetensors").read_bytes() == b"quantized-weights"
    assert (result.adapter / "adapters.safetensors").read_bytes() == b"exact-residual"
    assert resolve_mlx_fusion_adapter(result.path) == result.adapter
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["kind"] == MLX_FUSION_KIND
    assert manifest["base_weights_unchanged"] is True
    assert manifest["requires_full_precision_intermediate"] is False


def test_mlx_fusion_refuses_an_already_fused_base(tmp_path: Path):
    base, adapter = _sources(tmp_path)
    (base / FUSION_MANIFEST).write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="already fused"):
        create_mlx_fusion_bundle(base, adapter, tmp_path / "merged")


def test_mlx_fusion_rejects_adapter_path_traversal(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / FUSION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": MLX_FUSION_KIND,
                "adapter_path": "../adapter",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="unsafe"):
        resolve_mlx_fusion_adapter(model)


def test_gguf_fusion_keeps_split_base_and_adapter_bytes(tmp_path: Path):
    first = tmp_path / "base-00001-of-00002.gguf"
    second = tmp_path / "base-00002-of-00002.gguf"
    adapter = tmp_path / "adapter.gguf"
    first.write_bytes(b"GGUF-first")
    second.write_bytes(b"GGUF-second")
    adapter.write_bytes(b"GGUF-residual")

    created = create_gguf_fusion_bundle(
        first, (first, second), adapter, tmp_path / "merged"
    )
    resolved = resolve_gguf_fusion_bundle(created.path)

    assert resolved.model.read_bytes() == first.read_bytes()
    assert [path.read_bytes() for path in resolved.shards] == [
        first.read_bytes(),
        second.read_bytes(),
    ]
    assert resolved.adapter.read_bytes() == adapter.read_bytes()
    manifest = json.loads(resolved.manifest.read_text(encoding="utf-8"))
    assert manifest["kind"] == GGUF_FUSION_KIND
    assert manifest["requires_full_precision_intermediate"] is False


def test_gguf_fusion_rejects_path_traversal(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / FUSION_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": GGUF_FUSION_KIND,
                "model_path": "../model.gguf",
                "adapter_path": "adapter.gguf",
                "shards": ["../model.gguf"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="unsafe"):
        resolve_gguf_fusion_bundle(model)
