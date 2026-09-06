import pytest

from osai.errors import ConfigurationError
from osai.learning_proof import _normalize, prove_learning


def test_learning_proof_normalizes_case_and_whitespace():
    assert _normalize("  The CODE\n is   Blue ") == "the code is blue"


def test_learning_proof_rejects_auto_engine_before_execution(tmp_path):
    with pytest.raises(ConfigurationError, match="requires"):
        prove_learning(
            engine="auto",
            model=tmp_path / "model",
            adapter=tmp_path / "adapter",
            prompt="question",
            expected="answer",
            output=tmp_path / "proof.json",
        )
