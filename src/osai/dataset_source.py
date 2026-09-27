"""Stream selected dataset files into the canonical local split layout."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .dataset import normalize_sft_example, normalized_record_text, read_jsonl
from .errors import ConfigurationError
from .paths import llama_binary

_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".parquet"}
_METADATA_FILES = {
    "dataset.json",
    "dataset_info.json",
    "manifest.json",
    "metadata.json",
    "schema.json",
    "config.json",
}
_MEDIA_KEYS = {
    "image",
    "images",
    "image_url",
    "audio",
    "audios",
    "audio_url",
    "video",
    "videos",
    "video_url",
}
_REMOTE = {"http", "https", "ftp", "ftps", "s3", "gs", "data"}


def dataset_files(source: str | Path) -> list[tuple[str, Path]]:
    """Discover every supported file, in stable order, without following links."""
    root = Path(source).expanduser().resolve()
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = [item for item in root.rglob("*") if item.is_file() and not item.is_symlink()]
    else:
        raise ConfigurationError(f"dataset source does not exist: {root}")
    files: list[tuple[str, Path]] = []
    for item in sorted(candidates, key=lambda value: str(value).casefold()):
        if root.is_dir() and any(
            part.startswith(".") or part in {"__pycache__", "node_modules"}
            for part in item.relative_to(root).parts[:-1]
        ):
            continue
        if item.suffix.casefold() not in _EXTENSIONS or (
            root.is_dir() and item.name.casefold() in _METADATA_FILES
        ):
            continue
        stem = item.stem.casefold()
        if re.match(r"^(valid|validation|dev)(?:$|[-_.0-9])", stem):
            split = "valid"
        elif re.match(r"^test(?:$|[-_.0-9])", stem):
            split = "test"
        else:
            split = "train"
        files.append((split, item))
    if not files:
        raise ConfigurationError(f"no JSON, JSONL, NDJSON, or Parquet files found in {root}")
    if not any(split == "train" for split, _ in files):
        raise ConfigurationError(f"no training files found in {root}")
    return files


def prepare_dataset_source(source: str | Path, destination: Path) -> Path:
    """Merge all selected files; Parquet rows are converted locally in batches."""
    root = Path(source).expanduser().resolve()
    files = dataset_files(root)
    if root.is_dir() and all(
        item.parent == root and item.name in {"train.jsonl", "valid.jsonl", "test.jsonl"}
        for _, item in files
    ):
        return root
    if destination.exists():
        raise ConfigurationError(f"prepared dataset path already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    counts = {"train": 0, "valid": 0, "test": 0}
    try:
        for split, item in files:
            print(f"osai: preparing dataset split={split} file={item}", file=sys.stderr, flush=True)
            with (destination / f"{split}.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as output:
                for line_number, row in enumerate(_source_rows(item), 1):
                    if not isinstance(row, dict):
                        raise ConfigurationError(f"expected a JSON object at {item}:{line_number}")
                    row = _absolute_media(row, item.parent)
                    output.write(json.dumps(row, ensure_ascii=False, default=_json_value) + "\n")
                    counts[split] += 1
        if not counts["train"]:
            raise ConfigurationError(f"training dataset is empty: {root}")
    except Exception:
        # Keep the staged files for diagnostics; never change the user's source.
        raise
    print(
        "osai: prepared dataset "
        f"files={len(files)} train={counts['train']} valid={counts['valid']} test={counts['test']}",
        file=sys.stderr,
        flush=True,
    )
    return destination


def _source_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        for _, row in read_jsonl(path):
            yield row
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ConfigurationError(
                "Parquet conversion needs pyarrow; reinstall osAi CLI"
            ) from exc
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256):
                yield from batch.to_pylist()
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot read Parquet dataset {path}: {exc}") from exc
        return
    try:
        import ijson

        with path.open("rb") as handle:
            header = handle.read(4096)
            offset = 3 if header.startswith(b"\xef\xbb\xbf") else 0
            first = header[offset:].lstrip(b" \t\r\n")
            handle.seek(offset)
            if first.startswith(b"["):
                yield from ijson.items(handle, "item")
                return
            if not first.startswith(b"{"):
                raise ConfigurationError(f"JSON dataset must contain objects: {path}")
            container = next(
                (
                    value
                    for prefix, event, value in ijson.parse(handle)
                    if prefix == ""
                    and event == "map_key"
                    and value in {"train", "data", "records", "examples", "items"}
                ),
                None,
            )
            handle.seek(offset)
            if container is not None:
                yield from ijson.items(handle, f"{container}.item")
            else:
                # A single record must be materialized for normalization.
                yield json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError, ijson.JSONError) as exc:
        raise ConfigurationError(f"cannot read JSON dataset {path}: {exc}") from exc


def _json_value(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        import base64

        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"unsupported dataset value: {type(value).__name__}")


def _absolute_media(value: Any, parent: Path, *, media: bool = False) -> Any:
    if isinstance(value, list):
        return [_absolute_media(item, parent, media=media) for item in value]
    if isinstance(value, str) and media:
        parsed = urlparse(value)
        if parsed.scheme.casefold() in _REMOTE or value.startswith("//"):
            return value
        candidate = Path(value).expanduser()
        return str((parent / candidate).resolve()) if not candidate.is_absolute() else value
    if not isinstance(value, dict):
        return value
    result = {}
    part_type = str(value.get("type", "")).casefold().removesuffix("_url")
    part_media = part_type in {"image", "audio", "video"}
    for key, child in value.items():
        result[key] = _absolute_media(
            child,
            parent,
            media=key in _MEDIA_KEYS or ((part_media or media) and key in {"url", "path"}),
        )
    return result


def largest_training_context(
    source: str | Path,
    tokenizer_root: Path | None = None,
    gguf_model: Path | None = None,
) -> dict[str, Any]:
    """Count the largest normalized record without loading the dataset into memory."""
    files = [(split, item) for split, item in dataset_files(source) if split == "train"]
    if gguf_model is not None:
        binary = llama_binary("llama-tokenize")
        if binary is None:
            raise ConfigurationError(
                "the model tokenizer is missing; run osai build-llama and retry"
            )
        process = subprocess.Popen(
            [str(binary), "--model", str(gguf_model), "--osai-record-counts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        count = 0
        try:
            assert process.stdin is not None
            for _, item in files:
                for line_number, row in enumerate(_source_rows(item), 1):
                    normalized = normalize_sft_example(row, item, line_number)
                    content = normalized_record_text(normalized.record).encode("utf-8")
                    process.stdin.write(len(content).to_bytes(8, "little"))
                    process.stdin.write(content)
                    count += 1
                    if count % 500 == 0:
                        print(
                            f"osai: full content scan records={count}",
                            file=sys.stderr,
                            flush=True,
                        )
            process.stdin.close()
            assert process.stdout is not None
            output = process.stdout.read().decode("utf-8", errors="replace")
            status = process.wait()
        except Exception as exc:
            status = process.poll()
            if status is None:
                process.kill()
            process.wait()
            detail = (
                process.stderr.read().decode("utf-8", errors="replace")[-1000:]
                if process.stderr is not None
                else ""
            )
            if isinstance(exc, OSError):
                raise ConfigurationError(
                    f"llama.cpp tokenizer stopped after {count} records "
                    f"(exit={status}): {exc}; {detail}"
                ) from exc
            raise
        result = re.search(r"osai_max_tokens=(\d+) records=(\d+)", output)
        if status != 0 or result is None or int(result[2]) != count:
            detail = (
                process.stderr.read().decode("utf-8", errors="replace")[-1000:]
                if process.stderr is not None
                else ""
            )
            raise ConfigurationError(
                "llama.cpp could not count every training record with this model tokenizer: "
                + detail
            )
        largest = int(result[1])
        return {
            "records": count,
            "files": len(files),
            "largest_tokens": largest,
            "context": max(32, largest + 64),
            "exact": True,
            "largest_file": str(files[0][1]),
        }
    tokenizer = None
    if tokenizer_root is not None:
        candidates = [tokenizer_root / "tokenizer.json", tokenizer_root.parent / "tokenizer.json"]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    from tokenizers import Tokenizer

                    tokenizer = Tokenizer.from_file(str(candidate))
                except (ImportError, OSError, ValueError) as exc:
                    raise ConfigurationError(
                        f"cannot load model tokenizer {candidate}: {exc}"
                    ) from exc
                break
    largest = 0
    largest_file = ""
    count = 0
    for _, item in files:
        for line_number, row in enumerate(_source_rows(item), 1):
            normalized = normalize_sft_example(row, item, line_number)
            content = normalized_record_text(normalized.record)
            # With no tokenizer file, one UTF-8 byte per token is a safe upper bound.
            tokens = (
                len(tokenizer.encode(content, add_special_tokens=False).ids)
                if tokenizer is not None
                else len(content.encode("utf-8"))
            )
            if tokens > largest:
                largest = tokens
                largest_file = str(item)
            count += 1
            if count % 500 == 0:
                print(
                    f"osai: full content scan records={count}",
                    file=sys.stderr,
                    flush=True,
                )
    if count == 0:
        raise ConfigurationError(f"training dataset is empty: {source}")
    return {
        "records": count,
        "files": len(files),
        "largest_tokens": largest,
        "context": max(32, largest + 64),
        "exact": tokenizer is not None,
        "largest_file": largest_file,
    }
