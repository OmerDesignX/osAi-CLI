import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

from osai.dataset import validate_dataset
from osai.dataset_source import (
    dataset_files,
    largest_training_context,
    prepare_dataset_source,
)


def test_folder_uses_every_file_and_converts_parquet_locally(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "train.jsonl").write_text(
        json.dumps({"prompt": "first", "completion": "answer"}) + "\n",
        encoding="utf-8",
    )
    nested = source / "partitions"
    nested.mkdir()
    (nested / "more.ndjson").write_text(
        json.dumps({"prompt": "second", "completion": "answer"}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"prompt": ["third", "fourth"], "completion": ["a", "b"]}),
        source / "shard.parquet",
    )
    (source / "validation.jsonl").write_text(
        json.dumps({"prompt": "check", "completion": "answer"}) + "\n",
        encoding="utf-8",
    )

    prepared = prepare_dataset_source(source, tmp_path / "session" / "dataset")
    summary = validate_dataset(prepared)
    assert summary.train_examples == 4
    assert summary.valid_examples == 1
    assert len(dataset_files(source)) == 4
    assert not (source / "shard.jsonl").exists()
    assert len((prepared / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_single_parquet_file_becomes_train_jsonl(tmp_path: Path):
    source = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"text": ["one", "two"]}), source)
    prepared = prepare_dataset_source(source, tmp_path / "prepared")
    assert validate_dataset(prepared).train_examples == 2


def test_json_array_is_streamed_into_training_split(tmp_path: Path):
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            [
                {"prompt": "one", "completion": "first"},
                {"prompt": "two", "completion": "second"},
            ]
        ),
        encoding="utf-8",
    )
    prepared = prepare_dataset_source(source, tmp_path / "prepared")
    assert validate_dataset(prepared).train_examples == 2


def test_largest_context_uses_model_tokenizer(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": value}) + "\n" for value in ("short", "a longer training record")
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "short": 1, "a": 2, "longer": 3, "training": 4, "record": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    model = tmp_path / "model"
    model.mkdir()
    tokenizer.save(str(model / "tokenizer.json"))

    result = largest_training_context(source, model)
    assert result["records"] == 2
    assert result["largest_tokens"] == 4
    assert result["context"] == 68
    assert result["exact"] is True
