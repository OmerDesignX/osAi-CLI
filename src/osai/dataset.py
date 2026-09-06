"""Local JSONL validation shared by the training backends."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    path: Path
    schema: str
    train_examples: int
    valid_examples: int
    test_examples: int


def validate_dataset(path: str | Path, *, require_test: bool = False) -> DatasetSummary:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"dataset must be a directory: {root}")

    counts: dict[str, int] = {}
    observed_schema: str | None = None
    for split in ("train", "valid", "test"):
        split_path = root / f"{split}.jsonl"
        if not split_path.exists():
            counts[split] = 0
            continue
        count = 0
        for line_number, item in _read_jsonl(split_path):
            schema = _validate_example(item, split_path, line_number)
            if observed_schema is None:
                observed_schema = schema
            elif schema != observed_schema:
                raise ConfigurationError(
                    f"mixed dataset schemas are not supported: expected {observed_schema}, "
                    f"found {schema} at {split_path}:{line_number}"
                )
            count += 1
        counts[split] = count

    if counts["train"] == 0:
        raise ConfigurationError(f"training dataset is missing or empty: {root / 'train.jsonl'}")
    if require_test and counts["test"] == 0:
        raise ConfigurationError(f"test dataset is missing or empty: {root / 'test.jsonl'}")
    return DatasetSummary(
        path=root,
        schema=observed_schema or "unknown",
        train_examples=counts["train"],
        valid_examples=counts["valid"],
        test_examples=counts["test"],
    )


def text_corpus(path: str | Path, split: str = "test") -> Iterator[str]:
    split_path = Path(path) / f"{split}.jsonl"
    if not split_path.exists():
        raise ConfigurationError(f"dataset split does not exist: {split_path}")
    for line_number, item in _read_jsonl(split_path):
        schema = _validate_example(item, split_path, line_number)
        if schema == "text":
            yield item["text"]
        elif schema == "completion":
            yield f"{item['prompt']}\n{item['completion']}"
        else:
            yield "\n".join(
                str(message.get("content", ""))
                for message in item["messages"]
                if message.get("content")
            )


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ConfigurationError(
                        f"blank lines are not accepted by the MLX loader: {path}:{line_number}"
                    )
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ConfigurationError(f"expected a JSON object at {path}:{line_number}")
                yield line_number, item
    except OSError as exc:
        raise ConfigurationError(f"cannot read dataset file {path}: {exc}") from exc


def _validate_example(item: dict[str, Any], path: Path, line: int) -> str:
    if isinstance(item.get("text"), str) and item["text"].strip():
        return "text"
    if all(
        isinstance(item.get(key), str) and item[key].strip() for key in ("prompt", "completion")
    ):
        return "completion"
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ConfigurationError(f"message {index} is not an object at {path}:{line}")
            if message.get("role") not in _ROLES:
                raise ConfigurationError(
                    f"invalid message role {message.get('role')!r} at {path}:{line}"
                )
            if not isinstance(message.get("content"), str) and "tool_calls" not in message:
                raise ConfigurationError(
                    f"message {index} needs string content or tool_calls at {path}:{line}"
                )
        return "chat"
    raise ConfigurationError(
        f"unsupported example at {path}:{line}; expected text, prompt/completion, or messages"
    )
