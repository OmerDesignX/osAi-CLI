from pathlib import Path

from osai.backends.llama_utils import _parse_loss, _write_corpus


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
        '{"messages":[{"role":"user","content":"q"},'
        '{"role":"assistant","content":"a"}]}\n'
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
