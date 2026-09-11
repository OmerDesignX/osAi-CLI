"""Local dataset validation and canonicalization shared by every backend."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import ConfigurationError

_ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
    "human": "user",
    "instruction": "user",
    "question": "user",
    "user": "user",
}
_MEDIA_KEYS = {
    "image": "image",
    "images": "image",
    "audio": "audio",
    "audios": "audio",
    "video": "video",
    "videos": "video",
}
_MEDIA_TYPES = {"image", "image_url", "audio", "audio_url", "video", "video_url"}
_REMOTE_SCHEMES = {"http", "https", "ftp", "ftps", "s3", "gs"}
_MEDIA_MARKER_LINE = re.compile(r"(?m)^\s*<(?:image|video|audio)>\s*$\n?")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    path: Path
    schema: str
    train_examples: int
    valid_examples: int
    test_examples: int
    formats: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("text",)


@dataclass(frozen=True, slots=True)
class NormalizedExample:
    record: dict[str, Any]
    format: str
    family: str
    modalities: frozenset[str]


def validate_dataset(path: str | Path, *, require_test: bool = False) -> DatasetSummary:
    """Validate all local splits and describe the schemas and modalities present.

    Equivalent supervised layouts may be mixed because they are canonicalized to
    the same ``messages`` representation before a backend reads them. Raw language
    modelling rows cannot be mixed with supervised rows because their loss masks
    have different semantics.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"dataset must be a directory: {root}")

    counts: dict[str, int] = {}
    observed_family: str | None = None
    formats: set[str] = set()
    modalities = {"text"}
    for split in ("train", "valid", "test"):
        split_path = root / f"{split}.jsonl"
        if not split_path.exists():
            counts[split] = 0
            continue
        count = 0
        for line_number, item in read_jsonl(split_path):
            example = normalize_sft_example(item, split_path, line_number)
            if observed_family is None:
                observed_family = example.family
            elif example.family != observed_family:
                raise ConfigurationError(
                    "raw text and supervised examples cannot be mixed because their "
                    f"loss masks differ: {split_path}:{line_number}"
                )
            formats.add(example.format)
            modalities.update(example.modalities)
            count += 1
        counts[split] = count

    if counts["train"] == 0:
        raise ConfigurationError(f"training dataset is missing or empty: {root / 'train.jsonl'}")
    if require_test and counts["test"] == 0:
        raise ConfigurationError(f"test dataset is missing or empty: {root / 'test.jsonl'}")
    ordered_formats = tuple(sorted(formats))
    schema = ordered_formats[0] if len(ordered_formats) == 1 else "mixed-supervised"
    return DatasetSummary(
        path=root,
        schema=schema,
        train_examples=counts["train"],
        valid_examples=counts["valid"],
        test_examples=counts["test"],
        formats=ordered_formats,
        modalities=("text", *sorted(modalities - {"text"})),
    )


def require_text_training(dataset: DatasetSummary) -> None:
    """Reject media tensors instead of silently discarding them in text trainers."""

    media = sorted(set(dataset.modalities) - {"text"})
    if media:
        raise ConfigurationError(
            "the dataset was parsed as multimodal ("
            + ", ".join(media)
            + "), but osAi's bundled MLX LM and llama.cpp trainers update language "
            "models only; media would be ignored, so this run was stopped"
        )


def iter_normalized_examples(path: str | Path, split: str) -> Iterator[dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    split_path = root / f"{split}.jsonl"
    if not split_path.exists():
        raise ConfigurationError(f"dataset split does not exist: {split_path}")
    for line_number, item in read_jsonl(split_path):
        yield normalize_sft_example(item, split_path, line_number).record


def text_corpus(path: str | Path, split: str = "test") -> Iterator[str]:
    for record in iter_normalized_examples(path, split):
        yield normalized_record_text(record)


def normalized_record_text(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str):
        return record["text"].strip()
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ConfigurationError("canonical supervised record is missing messages")
    return "\n".join(
        f"{message['role']}: {message.get('content', '')}".rstrip()
        for message in messages
        if message.get("content") or message.get("tool_calls")
    )


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
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


def normalize_sft_example(
    item: dict[str, Any], path: Path, line: int
) -> NormalizedExample:
    """Map common LLM/VLM JSON schemas to MLX-compatible canonical records."""

    if not isinstance(item, dict):
        raise ConfigurationError(f"expected a JSON object at {path}:{line}")
    top_modalities = _top_level_modalities(item, path, line)

    if isinstance(item.get("messages"), list):
        messages, message_modalities = normalize_conversation(
            item["messages"], path, line, "messages"
        )
        _require_supervised_messages(messages, path, line)
        return _supervised_record(
            messages,
            "chat",
            top_modalities | message_modalities,
            tools=item.get("tools"),
        )

    for key in ("conversations", "conversation", "dialog", "dialogue"):
        if isinstance(item.get(key), list):
            messages, message_modalities = normalize_conversation(
                item[key], path, line, key
            )
            _require_supervised_messages(messages, path, line)
            return _supervised_record(
                messages,
                "sharegpt" if key in {"conversations", "conversation"} else "dialogue",
                top_modalities | message_modalities,
                tools=item.get("tools"),
            )

    if "prompt" in item and "completion" in item:
        messages, modalities = _prompt_completion_messages(
            item["prompt"], item["completion"], path, line
        )
        return _supervised_record(
            messages, "completion", top_modalities | modalities, tools=item.get("tools")
        )

    instruction = _nonempty_string(item.get("instruction"))
    output = _first_text(item, ("output", "response", "answer"))
    if instruction and output:
        context = item.get("input", item.get("context", ""))
        if context is not None and not isinstance(context, str):
            raise ConfigurationError(
                f"instruction context/input must be a string at {path}:{line}"
            )
        prompt = _instruction_prompt(instruction, context or "")
        schema = "alpaca" if "output" in item or "input" in item else "dolly"
        return _pair_record(prompt, output, schema, top_modalities)

    question = _first_text(item, ("question", "query"))
    answer = _first_text(item, ("answer", "response", "output", "target"))
    if question and answer:
        schema = "question-answer" if "question" in item else "query-response"
        return _pair_record(question, answer, schema, top_modalities)

    source = _first_text(item, ("source", "src"))
    target = _first_text(item, ("target", "tgt"))
    if source and target:
        return _pair_record(source, target, "source-target", top_modalities)

    prompt_value = _first_value(item, ("prompt", "question", "query", "instruction"))
    chosen = _first_value(item, ("chosen", "preferred", "accepted", "winner"))
    rejected = _first_value(
        item, ("rejected", "non_preferred", "dispreferred", "unpreferred", "loser")
    )
    if chosen is not None and rejected is not None:
        if prompt_value is not None:
            messages, modalities = _prompt_completion_messages(
                prompt_value, chosen, path, line
            )
            return _supervised_record(
                messages, "preference-chosen", top_modalities | modalities
            )
        if isinstance(chosen, list):
            messages, modalities = normalize_conversation(
                chosen, path, line, "chosen"
            )
            _require_supervised_messages(messages, path, line)
            return _supervised_record(
                messages, "preference-chosen", top_modalities | modalities
            )

    ranked_choice = _ranked_choice(item)
    if prompt_value is not None and ranked_choice is not None:
        messages, modalities = _prompt_completion_messages(
            prompt_value, ranked_choice, path, line
        )
        return _supervised_record(
            messages, "preference-chosen", top_modalities | modalities
        )

    prompt = _nonempty_string(item.get("prompt"))

    response = _first_text(item, ("response", "completion"))
    if prompt and response:
        return _pair_record(prompt, response, "prompt-response", top_modalities)

    if top_modalities:
        caption = _first_text(item, ("caption", "text", "transcription", "transcript"))
        if caption:
            media = next(iter(sorted(top_modalities)))
            prompt_for_media = {
                "audio": "Transcribe the audio.",
                "image": "Describe the image.",
                "video": "Describe the video.",
            }.get(media, "Describe the media.")
            return _pair_record(
                prompt_for_media, caption, f"{media}-caption", top_modalities
            )

    text = _nonempty_string(item.get("text"))
    if text:
        return NormalizedExample(
            {"text": text}, "text", "text", frozenset({"text"})
        )

    raise ConfigurationError(
        f"unsupported example at {path}:{line}; expected text, prompt/completion, "
        "messages, ShareGPT conversations, instruction/input/output, "
        "instruction/context/response, question/answer, query/response, "
        "source/target, a preference row, or a caption paired with local media"
    )


def normalize_conversation(
    value: Any, path: Path, line: int, label: str
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{label} must be a non-empty list at {path}:{line}")
    messages: list[dict[str, Any]] = []
    modalities = {"text"}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"{label} message {index} is not an object at {path}:{line}"
            )
        raw_role = raw.get("role", raw.get("from", raw.get("speaker")))
        role = _ROLE_ALIASES.get(str(raw_role).strip().casefold())
        if role is None:
            raise ConfigurationError(
                f"invalid message role {raw_role!r} at {path}:{line}"
            )
        content_value = raw.get("content", raw.get("value", raw.get("text")))
        content, content_modalities = _normalize_content(
            content_value, path, line, f"{label} message {index}"
        )
        modalities.update(content_modalities)
        tool_calls = raw.get("tool_calls", raw.get("function_call"))
        if not content and tool_calls is None:
            raise ConfigurationError(
                f"{label} message {index} needs content or tool_calls at {path}:{line}"
            )
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls is not None:
            if not isinstance(tool_calls, (dict, list)):
                raise ConfigurationError(
                    f"{label} message {index} tool_calls must be an object or list "
                    f"at {path}:{line}"
                )
            message["tool_calls"] = tool_calls
        if isinstance(raw.get("name"), str) and raw["name"].strip():
            message["name"] = raw["name"].strip()
        messages.append(message)
    return messages, frozenset(modalities)


def conversation_text(messages: Sequence[dict[str, Any]], *, response: bool = False) -> str:
    if response and len(messages) == 1 and messages[0].get("role") == "assistant":
        return str(messages[0].get("content", "")).strip()
    return "\n".join(
        f"{message['role']}: {message.get('content', '')}".rstrip()
        for message in messages
        if message.get("content") or message.get("tool_calls")
    ).strip()


def value_as_text(
    value: Any,
    path: Path,
    line: int,
    label: str,
    *,
    response: bool = False,
) -> str:
    text = _nonempty_string(value)
    if text:
        return text
    if isinstance(value, list):
        messages, modalities = normalize_conversation(value, path, line, label)
        media = sorted(set(modalities) - {"text"})
        if media:
            raise ConfigurationError(
                f"alignment {label} contains {', '.join(media)} media at {path}:{line}; "
                "the current alignment optimizers require language-model text"
            )
        rendered = conversation_text(messages, response=response)
        if rendered:
            return rendered
    raise ConfigurationError(f"{label} must contain non-empty text at {path}:{line}")


def prepare_mlx_dataset(path: str | Path, destination: str | Path) -> Path:
    """Materialize one homogeneous MLX-LM dataset from every accepted input schema."""

    dataset = validate_dataset(path)
    require_text_training(dataset)
    target = Path(destination).expanduser().resolve()
    try:
        target.mkdir(parents=True, exist_ok=False)
        for split in ("train", "valid", "test"):
            source = dataset.path / f"{split}.jsonl"
            if not source.is_file():
                continue
            output = target / f"{split}.jsonl"
            with output.open("x", encoding="utf-8") as handle:
                for record in iter_normalized_examples(dataset.path, split):
                    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                    handle.write("\n")
    except OSError as exc:
        raise ConfigurationError(f"cannot prepare canonical dataset at {target}: {exc}") from exc
    return target


def prepare_mlx_vlm_dataset(
    path: str | Path,
    destination: str | Path,
    *,
    video_fps: float = 2.0,
    video_max_frames: int = 32,
) -> Path:
    """Materialize local, canonical VLM rows with resolved media references.

    Media paths are made absolute so selecting or staging one JSON/JSONL file in
    the desktop app cannot change what a relative reference points to. Embedded
    base64/data-URI media is decoded into the private session working directory.
    """

    dataset = validate_dataset(path)
    media = set(dataset.modalities) - {"text"}
    if not media:
        raise ConfigurationError("VLM dataset preparation requires image, video, or audio rows")
    if video_fps <= 0:
        raise ConfigurationError("video_fps must be positive")
    if video_max_frames < 2:
        raise ConfigurationError("video_max_frames must be at least 2")
    target = Path(destination).expanduser().resolve()
    media_root = target / "media"
    try:
        target.mkdir(parents=True, exist_ok=False)
        media_root.mkdir()
        for split in ("train", "valid", "test"):
            source = dataset.path / f"{split}.jsonl"
            if not source.is_file():
                continue
            output = target / f"{split}.jsonl"
            with output.open("x", encoding="utf-8") as handle:
                for line_number, item in read_jsonl(source):
                    normalized = normalize_sft_example(item, source, line_number)
                    record = _without_media_markers(normalized.record)
                    references = _collect_media_references(
                        item, source, line_number, media_root
                    )
                    expected = set(normalized.modalities) - {"text"}
                    missing = sorted(name for name in expected if not references[name])
                    if missing:
                        raise ConfigurationError(
                            f"could not preserve {', '.join(missing)} references at "
                            f"{source}:{line_number}"
                        )
                    if references["image"]:
                        record["images"] = references["image"]
                    if references["video"]:
                        record["videos"] = references["video"]
                        record["video_fps"] = video_fps
                        record["video_max_frames"] = video_max_frames
                    if references["audio"]:
                        record["audio"] = references["audio"]
                    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                    handle.write("\n")
    except OSError as exc:
        raise ConfigurationError(f"cannot prepare multimodal dataset at {target}: {exc}") from exc
    return target


def _without_media_markers(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    messages = result.get("messages")
    if isinstance(messages, list):
        cleaned = []
        for message in messages:
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, str):
                copied["content"] = _MEDIA_MARKER_LINE.sub("", content).strip()
            cleaned.append(copied)
        result["messages"] = cleaned
    return result


def _collect_media_references(
    item: dict[str, Any],
    source: Path,
    line: int,
    media_root: Path,
) -> dict[str, list[str]]:
    result = {"image": [], "video": [], "audio": []}

    def add(modality: str, raw: Any, label: str) -> None:
        values = raw if isinstance(raw, list) else [raw]
        for index, value in enumerate(values):
            resolved = _materialize_media_reference(
                value, modality, source, line, f"{label}[{index}]", media_root
            )
            if resolved not in result[modality]:
                result[modality].append(resolved)

    for key, modality in _MEDIA_KEYS.items():
        if item.get(key) is not None:
            add(modality, item[key], key)

    def visit(value: Any, label: str) -> None:
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")
            return
        if not isinstance(value, dict):
            return
        part_type = str(value.get("type", "")).casefold()
        if part_type in _MEDIA_TYPES:
            modality = part_type.removesuffix("_url")
            reference = value.get(
                modality,
                value.get(f"{modality}_url", value.get("url", value.get("path"))),
            )
            add(modality, reference, label)
            return
        for key in (
            "messages",
            "conversations",
            "conversation",
            "dialog",
            "dialogue",
            "prompt",
            "completion",
            "chosen",
            "rejected",
            "content",
        ):
            if key in value:
                visit(value[key], f"{label}.{key}")

    visit(item, "row")
    return result


def _materialize_media_reference(
    reference: Any,
    modality: str,
    source: Path,
    line: int,
    label: str,
    media_root: Path,
) -> str:
    mime_type: str | None = None
    embedded: Any = None
    if isinstance(reference, dict):
        mime_type = _nonempty_string(
            reference.get("mime_type", reference.get("mime", reference.get("content_type")))
        ) or None
        embedded = reference.get("bytes")
        if embedded is None:
            reference = reference.get("url") or reference.get("path")
    if embedded is not None:
        if isinstance(embedded, list) and all(isinstance(value, int) for value in embedded):
            payload = bytes(embedded)
        elif isinstance(embedded, str):
            try:
                payload = base64.b64decode(embedded, validate=True)
            except ValueError as exc:
                raise ConfigurationError(
                    f"invalid base64 media for {label} at {source}:{line}"
                ) from exc
        else:
            raise ConfigurationError(f"invalid embedded media for {label} at {source}:{line}")
        return str(_write_embedded_media(payload, modality, mime_type, media_root))
    if not isinstance(reference, str):
        raise ConfigurationError(f"{label} needs a media path at {source}:{line}")
    value = reference.strip()
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
            mime_type = header[5:].split(";", 1)[0] or None
            payload = base64.b64decode(encoded, validate=";base64" in header.casefold())
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"invalid data URI media for {label} at {source}:{line}"
            ) from exc
        return str(_write_embedded_media(payload, modality, mime_type, media_root))
    parsed = urlparse(value)
    candidate = Path(parsed.path if parsed.scheme.casefold() == "file" else value).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return str(candidate.resolve())


def _write_embedded_media(
    payload: bytes,
    modality: str,
    mime_type: str | None,
    media_root: Path,
) -> Path:
    if not payload:
        raise ConfigurationError("embedded media cannot be empty")
    extension = mimetypes.guess_extension(mime_type or "") or {
        "image": ".png",
        "audio": ".wav",
        "video": ".mp4",
    }[modality]
    destination = media_root / f"{hashlib.sha256(payload).hexdigest()}{extension}"
    if not destination.exists():
        destination.write_bytes(payload)
    return destination.resolve()


def _supervised_record(
    messages: list[dict[str, Any]],
    schema: str,
    modalities: frozenset[str],
    *,
    tools: Any = None,
) -> NormalizedExample:
    record: dict[str, Any] = {"messages": messages}
    if tools is not None:
        if not isinstance(tools, (dict, list)):
            raise ConfigurationError("tools must be an object or list")
        record["tools"] = tools
    return NormalizedExample(record, schema, "supervised", modalities | {"text"})


def _pair_record(
    prompt: str, completion: str, schema: str, modalities: frozenset[str]
) -> NormalizedExample:
    return _supervised_record(
        [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": completion.strip()},
        ],
        schema,
        modalities | {"text"},
    )


def _prompt_completion_messages(
    prompt: Any, completion: Any, path: Path, line: int
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    modalities = {"text"}
    if isinstance(prompt, list):
        prompt_messages, prompt_modalities = normalize_conversation(
            prompt, path, line, "prompt"
        )
        modalities.update(prompt_modalities)
    else:
        prompt_messages = [
            {"role": "user", "content": value_as_text(prompt, path, line, "prompt")}
        ]
    if isinstance(completion, list):
        completion_messages, completion_modalities = normalize_conversation(
            completion, path, line, "completion"
        )
        modalities.update(completion_modalities)
    else:
        completion_messages = [
            {
                "role": "assistant",
                "content": value_as_text(completion, path, line, "completion"),
            }
        ]
    messages = [*prompt_messages, *completion_messages]
    _require_supervised_messages(messages, path, line)
    return messages, frozenset(modalities)


def _require_supervised_messages(
    messages: Sequence[dict[str, Any]], path: Path, line: int
) -> None:
    if not any(message.get("role") == "assistant" for message in messages):
        raise ConfigurationError(
            f"supervised conversation needs at least one assistant response at {path}:{line}"
        )


def _normalize_content(
    value: Any, path: Path, line: int, label: str
) -> tuple[str, frozenset[str]]:
    if isinstance(value, str):
        return value.strip(), frozenset({"text"})
    if value is None:
        return "", frozenset({"text"})
    if not isinstance(value, list):
        raise ConfigurationError(
            f"{label} content must be text or a list of content parts at {path}:{line}"
        )
    fragments: list[str] = []
    modalities = {"text"}
    for part_index, part in enumerate(value):
        if isinstance(part, str):
            if part.strip():
                fragments.append(part.strip())
            continue
        if not isinstance(part, dict):
            raise ConfigurationError(
                f"{label} content part {part_index} is invalid at {path}:{line}"
            )
        part_type = str(part.get("type", "text")).casefold()
        if part_type in {"text", "input_text", "output_text"}:
            text = part.get("text", part.get("content"))
            if not isinstance(text, str):
                raise ConfigurationError(
                    f"{label} text part {part_index} needs text at {path}:{line}"
                )
            if text.strip():
                fragments.append(text.strip())
            continue
        if part_type in _MEDIA_TYPES:
            modality = part_type.removesuffix("_url")
            reference = part.get(
                modality,
                part.get(
                    f"{modality}_url",
                    part.get("url", part.get("path")),
                ),
            )
            _validate_media_reference(reference, path, line, f"{label} {modality}")
            modalities.add(modality)
            fragments.append(f"<{modality}>")
            continue
        if part_type in {"tool_call", "tool_result", "function"}:
            text = part.get("text", part.get("content", part.get("output")))
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
                continue
        raise ConfigurationError(
            f"unsupported {label} content part type {part_type!r} at {path}:{line}"
        )
    return "\n".join(fragments), frozenset(modalities)


def _top_level_modalities(item: dict[str, Any], path: Path, line: int) -> frozenset[str]:
    modalities: set[str] = set()
    for key, modality in _MEDIA_KEYS.items():
        if key not in item or item[key] is None:
            continue
        raw_values = item[key] if isinstance(item[key], list) else [item[key]]
        if not raw_values:
            raise ConfigurationError(f"{key} cannot be empty at {path}:{line}")
        for index, reference in enumerate(raw_values):
            _validate_media_reference(reference, path, line, f"{key}[{index}]")
        modalities.add(modality)
    return frozenset(modalities)


def _validate_media_reference(reference: Any, path: Path, line: int, label: str) -> None:
    if isinstance(reference, dict):
        embedded = reference.get("bytes")
        if embedded is not None:
            if isinstance(embedded, (str, list)) and embedded:
                return
            raise ConfigurationError(f"{label} embedded bytes are empty at {path}:{line}")
        reference = reference.get("url") or reference.get("path")
    if not isinstance(reference, str) or not reference.strip():
        raise ConfigurationError(f"{label} needs a local path or embedded data at {path}:{line}")
    value = reference.strip()
    parsed = urlparse(value)
    if parsed.scheme.casefold() in _REMOTE_SCHEMES or value.startswith("//"):
        raise ConfigurationError(
            f"remote media is disabled; {label} must stay local at {path}:{line}"
        )
    if parsed.scheme.casefold() == "data":
        return
    candidate = Path(parsed.path if parsed.scheme.casefold() == "file" else value).expanduser()
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    if not candidate.exists() or not candidate.is_file():
        raise ConfigurationError(f"local media file does not exist for {label}: {candidate}")


def _first_text(item: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _nonempty_string(item.get(key))
        if value:
            return value
    return ""


def _first_value(item: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _nonempty_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _instruction_prompt(instruction: str, context: str) -> str:
    context = context.strip()
    return f"{instruction.strip()}\n\nContext:\n{context}" if context else instruction.strip()


def _ranked_choice(item: dict[str, Any]) -> Any:
    if "response_j" not in item or "response_k" not in item:
        return None
    label = item.get("label")
    if isinstance(label, (str, int, bool)) and label in {
        1,
        "1",
        "j",
        "J",
        "response_j",
    }:
        return item["response_j"]
    if isinstance(label, (str, int, bool)) and label in {
        0,
        "0",
        "k",
        "K",
        "response_k",
    }:
        return item["response_k"]
    return None
