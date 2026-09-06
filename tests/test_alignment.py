import json
import math
from pathlib import Path

import pytest

from osai.alignment import (
    AlignmentType,
    load_alignment_dataset,
    preference_gradients,
    preference_loss,
    reward_advantages,
)
from osai.backends.alignment import AlignmentOptions
from osai.errors import ConfigurationError


def write_rows(root: Path, rows: list[dict]):
    root.mkdir()
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_auto_alignment_chooses_dpo_for_preference_pairs(tmp_path: Path):
    write_rows(tmp_path / "pairs", [{"prompt": "p", "chosen": "c", "rejected": "r"}])
    dataset = load_alignment_dataset(tmp_path / "pairs")
    assert dataset.alignment_type is AlignmentType.DPO
    assert dataset.schema == "preference"


def test_auto_alignment_chooses_ppo_for_reward_rows(tmp_path: Path):
    write_rows(tmp_path / "rewards", [{"prompt": "p", "response": "r", "reward": 0.75}])
    dataset = load_alignment_dataset(tmp_path / "rewards")
    assert dataset.alignment_type is AlignmentType.PPO


@pytest.mark.parametrize(
    "method",
    [
        "dpo",
        "ipo",
        "simpo",
        "orpo",
        "cpo",
        "kto",
        "ppo",
        "reinforce",
        "rloo",
        "grpo",
    ],
)
def test_alignment_losses_are_finite(method: str):
    loss = preference_loss(method, -0.2, -0.8, -0.4, -0.5, reward=1.0)
    assert math.isfinite(loss)


@pytest.mark.parametrize(
    "method",
    [
        "dpo",
        "ipo",
        "simpo",
        "orpo",
        "cpo",
        "kto",
        "ppo",
        "reinforce",
        "rloo",
        "grpo",
    ],
)
def test_preference_gradients_match_finite_differences(method: str):
    policy_chosen, policy_rejected = -1.0, -2.0
    reference_chosen, reference_rejected = -1.2, -2.1
    kwargs = {"beta": 0.2, "gamma": 0.3, "reward": 0.7, "clip": 0.2}
    chosen, rejected = preference_gradients(
        method,
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        **kwargs,
    )
    epsilon = 1e-6

    def loss(pc: float, pr: float) -> float:
        return preference_loss(
            method, pc, pr, reference_chosen, reference_rejected, **kwargs
        )

    chosen_fd = (
        loss(policy_chosen + epsilon, policy_rejected)
        - loss(policy_chosen - epsilon, policy_rejected)
    ) / (2 * epsilon)
    rejected_fd = (
        loss(policy_chosen, policy_rejected + epsilon)
        - loss(policy_chosen, policy_rejected - epsilon)
    ) / (2 * epsilon)
    assert chosen == pytest.approx(chosen_fd, rel=1e-5, abs=1e-6)
    assert rejected == pytest.approx(rejected_fd, rel=1e-5, abs=1e-6)


def test_non_ppo_method_rejects_reward_rows(tmp_path: Path):
    write_rows(tmp_path / "data", [{"prompt": "p", "response": "r", "reward": 1}])
    with pytest.raises(ConfigurationError, match="requires prompt/chosen/rejected"):
        load_alignment_dataset(tmp_path / "data", "dpo")


def test_alignment_options_reject_unknown_optimizer():
    with pytest.raises(ConfigurationError, match="optimizer"):
        AlignmentOptions(optimizer="invalid").validate()


def test_reinforce_accepts_reward_rows(tmp_path: Path):
    write_rows(
        tmp_path / "rewards",
        [{"prompt": "p", "response": "answer", "reward": 1.0}],
    )
    dataset = load_alignment_dataset(tmp_path / "rewards", "reinforce")
    assert dataset.alignment_type is AlignmentType.REINFORCE


@pytest.mark.parametrize("reward", [1.0, -1.0])
def test_kto_accepts_binary_feedback(tmp_path: Path, reward: float):
    root = tmp_path / str(reward)
    write_rows(
        root,
        [{"prompt": "p", "response": "answer", "reward": reward}],
    )
    dataset = load_alignment_dataset(root, "kto")
    assert dataset.alignment_type is AlignmentType.KTO
    assert dataset.schema == "reward"


def test_kto_rejects_neutral_feedback(tmp_path: Path):
    write_rows(
        tmp_path / "rewards",
        [{"prompt": "p", "response": "answer", "reward": 0.0}],
    )
    with pytest.raises(ConfigurationError, match="positive or negative"):
        load_alignment_dataset(tmp_path / "rewards", "kto")


def test_kto_binary_feedback_moves_desirable_and_undesirable_in_opposite_directions():
    desirable, _ = preference_gradients("kto", -1.0, 0.0, -1.0, 0.0, reward=1.0)
    undesirable, _ = preference_gradients("kto", -1.0, 0.0, -1.0, 0.0, reward=-1.0)
    assert desirable < 0.0
    assert undesirable > 0.0


def test_rloo_uses_other_group_rewards_as_baseline(tmp_path: Path):
    rows = [
        {"prompt": "p", "response": "best", "reward": 1.0},
        {"prompt": "p", "response": "middle", "reward": 0.0},
        {"prompt": "p", "response": "worst", "reward": -1.0},
    ]
    write_rows(tmp_path / "rewards", rows)
    dataset = load_alignment_dataset(tmp_path / "rewards", "rloo")
    assert dataset.alignment_type is AlignmentType.RLOO
    assert reward_advantages(
        "rloo", [row["reward"] for row in rows], [row["prompt"] for row in rows]
    ) == pytest.approx((1.5, 0.0, -1.5))


def test_grpo_standardizes_each_prompt_group(tmp_path: Path):
    rows = [
        {"prompt": "p", "response": "best", "reward": 1.0},
        {"prompt": "p", "response": "worst", "reward": -1.0},
    ]
    write_rows(tmp_path / "rewards", rows)
    dataset = load_alignment_dataset(tmp_path / "rewards", "grpo")
    assert dataset.alignment_type is AlignmentType.GRPO
    assert reward_advantages(
        "grpo", [row["reward"] for row in rows], [row["prompt"] for row in rows]
    ) == pytest.approx((1.0, -1.0))


def test_gpro_misspelling_is_rejected(tmp_path: Path):
    write_rows(
        tmp_path / "rewards",
        [
            {"prompt": "p", "response": "best", "reward": 1.0},
            {"prompt": "p", "response": "worst", "reward": -1.0},
        ],
    )
    with pytest.raises(ConfigurationError, match="alignment type must be"):
        load_alignment_dataset(tmp_path / "rewards", "gpro")


def test_orpo_remains_finite_for_saturated_sequence_probabilities():
    loss = preference_loss("orpo", -1e-12, -1000.0, -0.1, -1.0)
    gradients = preference_gradients("orpo", -1e-12, -1000.0, -0.1, -1.0)
    assert math.isfinite(loss)
    assert all(math.isfinite(value) for value in gradients)


@pytest.mark.parametrize("method", ["rloo", "grpo"])
def test_group_methods_require_two_rows_per_prompt(tmp_path: Path, method: str):
    write_rows(
        tmp_path / method,
        [{"prompt": "p", "response": "answer", "reward": 1.0}],
    )
    with pytest.raises(ConfigurationError, match="at least two"):
        load_alignment_dataset(tmp_path / method, method)


def test_grpo_rejects_zero_variance_reward_group(tmp_path: Path):
    write_rows(
        tmp_path / "grpo",
        [
            {"prompt": "p", "response": "one", "reward": 1.0},
            {"prompt": "p", "response": "two", "reward": 1.0},
        ],
    )
    with pytest.raises(ConfigurationError, match="non-zero variance"):
        load_alignment_dataset(tmp_path / "grpo", "grpo")


@pytest.mark.parametrize("method", ["reinforce", "rloo", "grpo"])
def test_reward_methods_reject_preference_pairs(tmp_path: Path, method: str):
    write_rows(
        tmp_path / method,
        [{"prompt": "p", "chosen": "yes", "rejected": "no"}],
    )
    with pytest.raises(ConfigurationError, match="prompt/response/reward"):
        load_alignment_dataset(tmp_path / method, method)
