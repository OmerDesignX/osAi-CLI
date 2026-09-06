import json
from pathlib import Path

import pytest

from osai.alignment import AlignmentType, load_alignment_dataset
from osai.errors import ConfigurationError, TrainingError
from osai.rollouts import RolloutSettings, collect_live_rollouts


def _write_rows(root: Path, rows: list[dict]) -> None:
    root.mkdir()
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter.gguf"
    path.write_bytes(b"adapter tensors")
    return path


def test_pairwise_rollout_uses_fine_tuned_policy_answer(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "Capital?", "chosen": "Paris", "rejected": "London"}],
    )
    source = load_alignment_dataset(source_root, "dpo")
    seen = []

    def generate(requests):
        seen.extend(requests)
        return ["Paris is the capital." for _ in requests]

    result = collect_live_rollouts(
        source,
        tmp_path / "rollouts",
        engine="llama.cpp",
        adapter=_adapter(tmp_path),
        settings=RolloutSettings(samples_per_prompt=1),
        generate=generate,
    )
    assert len(seen) == 1
    assert result.dataset.examples[0].chosen == "Paris is the capital."
    row = json.loads(result.data_file.read_text(encoding="utf-8"))
    assert row["rollout"]["generated"] is True
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["local_only"] is True
    assert manifest["network_used"] is False
    assert manifest["generated_examples"] == 1


def test_ppo_rollout_scores_generated_answers_from_local_references(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "Capital?", "chosen": "Paris", "rejected": "London"}],
    )
    source = load_alignment_dataset(source_root, "ppo")
    result = collect_live_rollouts(
        source,
        tmp_path / "rollouts",
        engine="mlx",
        adapter=_adapter(tmp_path),
        settings=RolloutSettings(samples_per_prompt=2),
        generate=lambda _: ["Paris", "London"],
    )
    assert result.dataset.schema == "reward"
    assert [item.response for item in result.dataset.examples] == ["Paris", "London"]
    rewards = [item.reward for item in result.dataset.examples]
    assert rewards[0] > 0
    assert rewards[1] < 0


@pytest.mark.parametrize("method", ["reinforce", "rloo", "grpo"])
def test_live_reward_methods_accept_preference_sources(tmp_path: Path, method: str):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "Capital?", "chosen": "Paris", "rejected": "London"}],
    )
    source = load_alignment_dataset(source_root, method, live_rollouts=True)
    result = collect_live_rollouts(
        source,
        tmp_path / "rollouts",
        engine="mlx",
        adapter=_adapter(tmp_path),
        settings=RolloutSettings(samples_per_prompt=2),
        generate=lambda _: ["Paris", "London"],
    )
    assert result.dataset.alignment_type is AlignmentType(method)
    assert len(result.dataset.examples) == 2


def test_group_live_rollouts_require_multiple_answers(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "p", "chosen": "yes", "rejected": "no"}],
    )
    source = load_alignment_dataset(source_root, "grpo", live_rollouts=True)
    with pytest.raises(ConfigurationError, match="at least two"):
        collect_live_rollouts(
            source,
            tmp_path / "rollouts",
            engine="mlx",
            adapter=_adapter(tmp_path),
            settings=RolloutSettings(samples_per_prompt=1),
            generate=lambda _: ["yes"],
        )


@pytest.mark.parametrize("method", ["rloo", "grpo"])
def test_group_live_rollouts_expand_one_reward_reference(tmp_path: Path, method: str):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "p", "response": "good answer", "reward": 1.0}],
    )
    source = load_alignment_dataset(source_root, method, live_rollouts=True)
    result = collect_live_rollouts(
        source,
        tmp_path / "rollouts",
        engine="mlx",
        adapter=_adapter(tmp_path),
        settings=RolloutSettings(samples_per_prompt=2),
        generate=lambda _: ["good answer", "unrelated"],
    )
    assert len(result.dataset.examples) == 2


def test_rollout_generator_must_return_every_answer(tmp_path: Path):
    source_root = tmp_path / "source"
    _write_rows(
        source_root,
        [{"prompt": "p", "chosen": "yes", "rejected": "no"}],
    )
    source = load_alignment_dataset(source_root, "dpo")
    with pytest.raises(TrainingError, match="different number"):
        collect_live_rollouts(
            source,
            tmp_path / "rollouts",
            engine="mlx",
            adapter=_adapter(tmp_path),
            settings=RolloutSettings(samples_per_prompt=2),
            generate=lambda _: ["yes"],
        )
