from pathlib import Path
from types import SimpleNamespace

import pytest

from osai.backends.llama_utils import (
    _parse_loss,
    _select_lora_tensor_shapes,
    _write_corpus,
)
from osai.errors import VerificationError


def test_llama_corpus_is_local_and_large_enough(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"hello local model"}\n')
    destination = _write_corpus(source, tmp_path / "corpus.txt", 32)
    text = destination.read_text()
    assert "hello local model" in text
    assert len(text) >= 32 * 16


def test_structured_llama_corpus_preserves_record_boundaries_without_repeating(
    tmp_path: Path,
):
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    destination = _write_corpus(
        source,
        tmp_path / "corpus.txt",
        64,
        repeat_to_minimum=False,
        record_separator="\n<|osai_record_end|>\n",
    )
    assert destination.read_text() == "user: q\nassistant: a\n"


def test_llama_loss_parser_uses_precise_output_row(tmp_path: Path):
    output = "       0  1.1485  0.138449  0.066124\nFinal estimate: PPL = 1.1485"
    assert _parse_loss(output, tmp_path / "train.log") == 0.138449


def test_hybrid_gguf_targets_follow_available_projection_blocks():
    shapes = {
        "blk.24.attn_q.weight": (8, 8),
        "blk.24.attn_v.weight": (8, 8),
        "blk.27.attn_q.weight": (8, 8),
        "blk.27.attn_v.weight": (8, 8),
        "blk.28.ffn_down.weight": (8, 16),
        "blk.29.ffn_down.weight": (8, 16),
        "blk.30.ffn_down.weight": (8, 16),
    }
    settings = SimpleNamespace(
        num_layers=2,
        target_modules=("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"),
    )

    selected = _select_lora_tensor_shapes(shapes, 31, settings)

    assert set(selected) == {
        "blk.24.attn_q.weight",
        "blk.24.attn_v.weight",
        "blk.27.attn_q.weight",
        "blk.27.attn_v.weight",
        "blk.29.ffn_down.weight",
        "blk.30.ffn_down.weight",
    }


def test_hybrid_gguf_skips_an_unavailable_projection(capsys):
    settings = SimpleNamespace(
        num_layers=1,
        target_modules=("self_attn.q_proj", "mlp.down_proj"),
    )
    selected = _select_lora_tensor_shapes({"blk.3.ffn_down.weight": (8, 16)}, 4, settings)
    assert set(selected) == {"blk.3.ffn_down.weight"}
    assert "omits requested projection" in capsys.readouterr().err


def test_hybrid_gguf_requires_at_least_one_compatible_projection():
    settings = SimpleNamespace(
        num_layers=1,
        target_modules=("self_attn.q_proj",),
    )
    with pytest.raises(VerificationError, match="available projections"):
        _select_lora_tensor_shapes({"output.weight": (8, 8)}, 4, settings)
