from pathlib import Path

import pytest

from osai.config import ModelFormat
from osai.errors import VerificationError
from osai.formats import ModelInspection, QuantizationSpec
from osai.merge import _verify_gguf_tensor_types, _verify_merged_metadata


def inspection(*, scheme="Q4_K_M", context=262144):
    return ModelInspection(
        format=ModelFormat.GGUF,
        path=Path("model-Q4_K_M.gguf"),
        architecture="qwen35",
        quantization=QuantizationSpec(scheme),
        size_bytes=100,
        shards=(Path("model-Q4_K_M.gguf"),),
        block_count=32,
        embedding_length=2560,
        context_length=context,
    )


def test_gguf_merge_preserves_every_tensor_type():
    _verify_gguf_tensor_types({"a": 12, "b": 14}, {"a": 12, "b": 14})


def test_gguf_merge_rejects_changed_tensor_type():
    with pytest.raises(VerificationError, match="tensor types changed"):
        _verify_gguf_tensor_types({"a": 12, "b": 14}, {"a": 12, "b": 1})


def test_gguf_merge_rejects_changed_tensor_set():
    with pytest.raises(VerificationError, match="tensor set changed"):
        _verify_gguf_tensor_types({"a": 12}, {"b": 12})


def test_merged_metadata_must_keep_context():
    with pytest.raises(VerificationError, match="context_length"):
        _verify_merged_metadata(inspection(), inspection(context=4096))
