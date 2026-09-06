import json
from pathlib import Path

import pytest

from osai.dataset import validate_dataset
from osai.errors import ConfigurationError


def _line(path: Path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_chat_dataset(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
    )
    summary = validate_dataset(tmp_path)
    assert summary.schema == "chat"
    assert summary.train_examples == 1


def test_bad_role_is_rejected(tmp_path: Path):
    _line(tmp_path / "train.jsonl", {"messages": [{"role": "intruder", "content": "x"}]})
    with pytest.raises(ConfigurationError, match="invalid message role"):
        validate_dataset(tmp_path)


def test_mixed_schema_is_rejected(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "one"}) + "\n" + json.dumps({"prompt": "p", "completion": "c"}) + "\n"
    )
    with pytest.raises(ConfigurationError, match="mixed dataset schemas"):
        validate_dataset(tmp_path)


def test_blank_line_is_rejected(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text('{"text":"one"}\n\n')
    with pytest.raises(ConfigurationError, match="blank lines"):
        validate_dataset(tmp_path)
