import json
from pathlib import Path

import pytest

from osai.dataset import (
    iter_normalized_examples,
    prepare_mlx_dataset,
    prepare_mlx_vlm_dataset,
    require_text_training,
    text_corpus,
    validate_dataset,
)
from osai.errors import ConfigurationError


def _line(path: Path, value, *, encoding: str = "utf-8"):
    path.write_text(json.dumps(value) + "\n", encoding=encoding)


@pytest.mark.parametrize(
    ("row", "schema", "prompt", "answer"),
    [
        ({"prompt": "p", "completion": "c"}, "completion", "p", "c"),
        (
            {"instruction": "p", "input": "ctx", "output": "c"},
            "alpaca",
            "p\n\nContext:\nctx",
            "c",
        ),
        (
            {"instruction": "p", "context": "ctx", "response": "c"},
            "dolly",
            "p\n\nContext:\nctx",
            "c",
        ),
        ({"question": "p", "answer": "c"}, "question-answer", "p", "c"),
        ({"query": "p", "response": "c"}, "query-response", "p", "c"),
        ({"src": "p", "tgt": "c"}, "source-target", "p", "c"),
        (
            {"prompt": "p", "chosen": "c", "rejected": "r"},
            "preference-chosen",
            "p",
            "c",
        ),
    ],
)
def test_common_supervised_pairs_are_normalized(
    tmp_path: Path, row: dict, schema: str, prompt: str, answer: str
):
    _line(tmp_path / "train.jsonl", row)
    summary = validate_dataset(tmp_path)
    record = next(iter_normalized_examples(tmp_path, "train"))
    assert summary.schema == schema
    assert summary.modalities == ("text",)
    assert record["messages"] == [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


@pytest.mark.parametrize(
    ("field", "messages", "schema"),
    [
        (
            "messages",
            [
                {"role": "developer", "content": "rules"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "chat",
        ),
        (
            "conversations",
            [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ],
            "sharegpt",
        ),
        (
            "dialogue",
            [
                {"speaker": "human", "text": "hi"},
                {"speaker": "bot", "text": "hello"},
            ],
            "dialogue",
        ),
    ],
)
def test_openai_sharegpt_and_dialogue_formats(
    tmp_path: Path, field: str, messages: list[dict], schema: str
):
    _line(tmp_path / "train.jsonl", {field: messages})
    summary = validate_dataset(tmp_path)
    record = next(iter_normalized_examples(tmp_path, "train"))
    assert summary.schema == schema
    assert record["messages"][-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_conversational_prompt_completion_and_text_parts(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {
            "prompt": [
                {"role": "system", "content": "rules"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                },
            ],
            "completion": [
                {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
            ],
        },
    )
    record = next(iter_normalized_examples(tmp_path, "train"))
    assert record["messages"][1]["content"] == "first\nsecond"
    assert record["messages"][-1]["content"] == "done"


def test_conversational_preference_uses_chosen_response_for_sft(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {
            "prompt": [{"role": "user", "content": "Pick one"}],
            "chosen": [{"role": "assistant", "content": "Good"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
        },
    )
    record = next(iter_normalized_examples(tmp_path, "train"))
    assert record["messages"] == [
        {"role": "user", "content": "Pick one"},
        {"role": "assistant", "content": "Good"},
    ]


def test_tool_call_chat_is_preserved(tmp_path: Path):
    tools = [{"type": "function", "function": {"name": "weather"}}]
    _line(
        tmp_path / "train.jsonl",
        {
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "weather"}}],
                },
            ],
            "tools": tools,
        },
    )
    record = next(iter_normalized_examples(tmp_path, "train"))
    assert record["tools"] == tools
    assert record["messages"][-1]["tool_calls"][0]["function"]["name"] == "weather"


def test_compatible_supervised_schemas_can_be_mixed_and_are_homogeneous_for_mlx(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"prompt": "one", "completion": "1"},
                {"instruction": "two", "output": "2"},
                {
                    "conversations": [
                        {"from": "human", "value": "three"},
                        {"from": "gpt", "value": "3"},
                    ]
                },
            )
        ),
        encoding="utf-8",
    )
    summary = validate_dataset(source)
    assert summary.schema == "mixed-supervised"
    assert summary.formats == ("alpaca", "completion", "sharegpt")

    prepared = prepare_mlx_dataset(source, tmp_path / "prepared")
    rows = [json.loads(line) for line in (prepared / "train.jsonl").read_text().splitlines()]
    assert [row["messages"][0]["content"] for row in rows] == ["one", "two", "three"]
    assert all(set(row) == {"messages"} for row in rows)


def test_raw_text_is_supported_but_cannot_mix_with_supervised_loss(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "one"})
        + "\n"
        + json.dumps({"prompt": "p", "completion": "c"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="cannot be mixed"):
        validate_dataset(tmp_path)


def test_bom_jsonl_and_text_corpus(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {"instruction": "Question", "response": "Answer"},
        encoding="utf-8-sig",
    )
    assert list(text_corpus(tmp_path, "train")) == [
        "user: Question\nassistant: Answer"
    ]


def test_bad_role_and_prompt_only_chat_are_rejected(tmp_path: Path):
    _line(tmp_path / "train.jsonl", {"messages": [{"role": "intruder", "content": "x"}]})
    with pytest.raises(ConfigurationError, match="invalid message role"):
        validate_dataset(tmp_path)
    _line(tmp_path / "train.jsonl", {"messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(ConfigurationError, match="assistant response"):
        validate_dataset(tmp_path)


def test_blank_line_is_rejected(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text('{"text":"one"}\n\n')
    with pytest.raises(ConfigurationError, match="blank lines"):
        validate_dataset(tmp_path)


def test_instruction_context_must_be_text(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {"instruction": "Question", "context": ["bad"], "response": "Answer"},
    )
    with pytest.raises(ConfigurationError, match="context/input must be a string"):
        validate_dataset(tmp_path)


@pytest.mark.parametrize(
    ("media_key", "answer_key", "answer", "expected_schema"),
    [
        ("image", "caption", "a blue square", "image-caption"),
        ("audio", "transcription", "hello", "audio-caption"),
        ("video", "caption", "a short clip", "video-caption"),
    ],
)
def test_local_multimodal_caption_formats_are_parsed_but_not_silently_dropped(
    tmp_path: Path,
    media_key: str,
    answer_key: str,
    answer: str,
    expected_schema: str,
):
    media = tmp_path / f"sample.{media_key}"
    media.write_bytes(b"local")
    _line(tmp_path / "train.jsonl", {media_key: media.name, answer_key: answer})
    summary = validate_dataset(tmp_path)
    assert summary.schema == expected_schema
    assert summary.modalities == ("text", media_key)
    with pytest.raises(ConfigurationError, match="would be ignored"):
        require_text_training(summary)


def test_llava_and_multimodal_message_layouts_are_recognized(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    _line(
        tmp_path / "train.jsonl",
        {
            "image": image.name,
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "A diagram."},
            ],
        },
    )
    summary = validate_dataset(tmp_path)
    assert summary.schema == "sharegpt"
    assert summary.modalities == ("text", "image")


def test_remote_or_missing_media_is_rejected_locally(tmp_path: Path):
    _line(
        tmp_path / "train.jsonl",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/x"}}
                    ],
                },
                {"role": "assistant", "content": "x"},
            ]
        },
    )
    with pytest.raises(ConfigurationError, match="remote media is disabled"):
        validate_dataset(tmp_path)
    _line(tmp_path / "train.jsonl", {"image": "missing.png", "caption": "x"})
    with pytest.raises(ConfigurationError, match="does not exist"):
        validate_dataset(tmp_path)


def test_vlm_preparation_preserves_local_image_video_and_audio(tmp_path: Path):
    source = tmp_path / "dataset"
    source.mkdir()
    for name in ("picture.png", "clip.mp4", "voice.wav"):
        (source / name).write_bytes(name.encode())
    _line(
        source / "train.jsonl",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": "picture.png"},
                        {"type": "video", "video": "clip.mp4"},
                        {"type": "audio", "audio": "voice.wav"},
                        {"type": "text", "text": "Explain these."},
                    ],
                },
                {"role": "assistant", "content": "Done."},
            ]
        },
    )
    prepared = prepare_mlx_vlm_dataset(
        source, tmp_path / "prepared", video_fps=1.5, video_max_frames=12
    )
    row = json.loads((prepared / "train.jsonl").read_text())
    assert row["messages"][0]["content"] == "Explain these."
    assert row["images"] == [str((source / "picture.png").resolve())]
    assert row["videos"] == [str((source / "clip.mp4").resolve())]
    assert row["audio"] == [str((source / "voice.wav").resolve())]
    assert row["video_fps"] == 1.5
    assert row["video_max_frames"] == 12


def test_vlm_preparation_materializes_embedded_image(tmp_path: Path):
    payload = b"not-a-real-png-but-locally-materialized"
    _line(
        tmp_path / "train.jsonl",
        {
            "image": {"bytes": list(payload), "mime_type": "image/png"},
            "caption": "local bytes",
        },
    )
    prepared = prepare_mlx_vlm_dataset(tmp_path, tmp_path / "prepared")
    row = json.loads((prepared / "train.jsonl").read_text())
    materialized = Path(row["images"][0])
    assert materialized.parent == prepared / "media"
    assert materialized.read_bytes() == payload
