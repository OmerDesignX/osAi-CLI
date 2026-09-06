"""Local alignment dataset validation and objective definitions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

_PROBABILITY_EPSILON = 1e-4


class AlignmentType(str, Enum):
    AUTO = "auto"
    DPO = "dpo"
    IPO = "ipo"
    SIMPO = "simpo"
    ORPO = "orpo"
    CPO = "cpo"
    KTO = "kto"
    PPO = "ppo"
    REINFORCE = "reinforce"
    RLOO = "rloo"
    GRPO = "grpo"


_REWARD_METHODS = {
    AlignmentType.KTO,
    AlignmentType.PPO,
    AlignmentType.REINFORCE,
    AlignmentType.RLOO,
    AlignmentType.GRPO,
}


@dataclass(frozen=True, slots=True)
class AlignmentExample:
    prompt: str
    chosen: str | None = None
    rejected: str | None = None
    response: str | None = None
    reward: float | None = None
    old_logprob: float | None = None


@dataclass(frozen=True, slots=True)
class AlignmentDataset:
    path: Path
    schema: str
    alignment_type: AlignmentType
    examples: tuple[AlignmentExample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "schema": self.schema,
            "alignment_type": self.alignment_type.value,
            "examples": len(self.examples),
        }


def load_alignment_dataset(
    path: str | Path,
    requested: str | AlignmentType = AlignmentType.AUTO,
    *,
    live_rollouts: bool = False,
) -> AlignmentDataset:
    root = Path(path).expanduser().resolve()
    source = root / "train.jsonl"
    if not root.is_dir() or not source.is_file():
        raise ConfigurationError(f"alignment dataset must contain train.jsonl: {root}")
    try:
        selected = _alignment_type(requested)
    except ValueError as exc:
        raise ConfigurationError(
            "alignment type must be auto, dpo, ipo, simpo, orpo, cpo, kto, "
            "ppo, reinforce, rloo, or grpo"
        ) from exc

    examples: list[AlignmentExample] = []
    schema: str | None = None
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ConfigurationError(
                        f"blank alignment dataset line: {source}:{line_number}"
                    )
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"invalid alignment JSON at {source}:{line_number}: {exc.msg}"
                    ) from exc
                example, item_schema = _parse_example(item, source, line_number)
                if schema is None:
                    schema = item_schema
                elif schema != item_schema:
                    raise ConfigurationError(
                        f"mixed alignment schemas at {source}:{line_number}"
                    )
                examples.append(example)
    except OSError as exc:
        raise ConfigurationError(f"cannot read alignment dataset {source}: {exc}") from exc
    if not examples:
        raise ConfigurationError(f"alignment dataset is empty: {source}")

    assert schema is not None
    if selected is AlignmentType.AUTO:
        selected = AlignmentType.DPO if schema == "preference" else AlignmentType.PPO
    if schema == "reward" and selected not in _REWARD_METHODS:
        raise ConfigurationError(f"{selected.value} requires prompt/chosen/rejected records")
    if schema == "preference" and selected in {
        AlignmentType.REINFORCE,
        AlignmentType.RLOO,
        AlignmentType.GRPO,
    } and not live_rollouts:
        raise ConfigurationError(
            f"{selected.value} requires prompt/response/reward records"
        )
    if schema == "reward" and selected is AlignmentType.KTO and any(
        item.reward == 0.0 for item in examples
    ):
        raise ConfigurationError("kto reward labels must be positive or negative, not zero")
    dataset = AlignmentDataset(root, schema, selected, tuple(examples))
    if (
        selected in {AlignmentType.RLOO, AlignmentType.GRPO}
        and schema == "reward"
        and not live_rollouts
    ):
        reward_advantages(
            selected,
            [float(item.reward) for item in examples if item.reward is not None],
            [item.prompt for item in examples],
        )
    return dataset


def reward_advantages(
    alignment_type: str | AlignmentType,
    rewards: list[float] | tuple[float, ...],
    prompts: list[str] | tuple[str, ...],
) -> tuple[float, ...]:
    """Build fixed sequence-level advantages for local reward-scored rollouts."""

    method = _alignment_type(alignment_type)
    if method not in _REWARD_METHODS:
        raise ConfigurationError(f"{method.value} does not use scalar reward records")
    if not rewards or len(rewards) != len(prompts):
        raise ConfigurationError("reward advantages require equally sized non-empty inputs")
    values = tuple(float(value) for value in rewards)
    if not all(math.isfinite(value) for value in values):
        raise ConfigurationError("reward advantages must be finite")
    if method in {AlignmentType.KTO, AlignmentType.PPO, AlignmentType.REINFORCE}:
        return values

    groups: dict[str, list[int]] = {}
    for index, prompt in enumerate(prompts):
        groups.setdefault(prompt, []).append(index)
    advantages = [0.0] * len(values)
    for prompt, indices in groups.items():
        if len(indices) < 2:
            raise ConfigurationError(
                f"{method.value} needs at least two reward rows for prompt {prompt!r}"
            )
        group_rewards = [values[index] for index in indices]
        if method is AlignmentType.RLOO:
            total = math.fsum(group_rewards)
            for index in indices:
                baseline = (total - values[index]) / (len(indices) - 1)
                advantages[index] = values[index] - baseline
            continue
        mean = math.fsum(group_rewards) / len(group_rewards)
        variance = math.fsum((value - mean) ** 2 for value in group_rewards) / len(
            group_rewards
        )
        standard_deviation = math.sqrt(variance)
        if standard_deviation <= 1e-12:
            raise ConfigurationError(
                f"grpo rewards need non-zero variance for prompt {prompt!r}"
            )
        for index in indices:
            advantages[index] = (values[index] - mean) / standard_deviation
    return tuple(advantages)


def preference_loss(
    alignment_type: str | AlignmentType,
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    *,
    beta: float = 0.1,
    gamma: float = 0.5,
    reward: float | None = None,
    clip: float = 0.2,
) -> float:
    """Return one sequence-level preference objective (lower is better)."""

    method = _alignment_type(alignment_type)
    if beta <= 0 or clip <= 0:
        raise ConfigurationError("alignment beta and PPO clip must be positive")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    relative_margin = policy_margin - reference_margin

    if method is AlignmentType.DPO:
        return -_log_sigmoid(beta * relative_margin)
    if method is AlignmentType.IPO:
        return (relative_margin - 1.0 / (2.0 * beta)) ** 2
    if method is AlignmentType.SIMPO:
        return -_log_sigmoid(beta * policy_margin - gamma)
    if method is AlignmentType.CPO:
        return -_log_sigmoid(beta * policy_margin) - beta * policy_chosen
    if method is AlignmentType.ORPO:
        odds_margin = _log_odds(policy_chosen) - _log_odds(policy_rejected)
        return -policy_chosen - beta * _log_sigmoid(odds_margin)
    if method is AlignmentType.KTO:
        if reward is not None:
            if reward == 0.0:
                raise ConfigurationError(
                    "kto reward labels must be positive or negative, not zero"
                )
            delta = policy_chosen - reference_chosen
            weight = abs(reward)
            signed_delta = delta if reward > 0.0 else -delta
            return weight * (1.0 - _sigmoid(beta * signed_delta))
        kl = max(0.0, 0.5 * (
            (policy_chosen - reference_chosen)
            + (policy_rejected - reference_rejected)
        ))
        desirable = 1.0 - _sigmoid(beta * (policy_chosen - reference_chosen - kl))
        undesirable = 1.0 - _sigmoid(beta * (kl - policy_rejected + reference_rejected))
        return 0.5 * (desirable + undesirable)
    if method is AlignmentType.PPO:
        advantage = 1.0 if reward is None else reward
        log_ratio = max(-20.0, min(20.0, policy_chosen - reference_chosen))
        ratio = math.exp(log_ratio)
        clipped = max(1.0 - clip, min(1.0 + clip, ratio))
        return -min(ratio * advantage, clipped * advantage)
    if method in {AlignmentType.REINFORCE, AlignmentType.RLOO}:
        advantage = 1.0 if reward is None else reward
        return -advantage * policy_chosen
    if method is AlignmentType.GRPO:
        advantage = 1.0 if reward is None else reward
        log_ratio = max(-20.0, min(20.0, policy_chosen - reference_chosen))
        ratio = math.exp(log_ratio)
        clipped = max(1.0 - clip, min(1.0 + clip, ratio))
        reference_log_ratio = max(
            -20.0, min(20.0, reference_chosen - policy_chosen)
        )
        reference_ratio = math.exp(reference_log_ratio)
        kl = reference_ratio - reference_log_ratio - 1.0
        return -min(ratio * advantage, clipped * advantage) + beta * kl
    raise ConfigurationError("auto alignment type must be resolved before optimization")


def preference_gradients(
    alignment_type: str | AlignmentType,
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    *,
    beta: float = 0.1,
    gamma: float = 0.5,
    reward: float | None = None,
    clip: float = 0.2,
) -> tuple[float, float]:
    """Differentiate the objective by chosen/rejected sequence log-probability."""

    method = _alignment_type(alignment_type)
    if beta <= 0 or clip <= 0:
        raise ConfigurationError("alignment beta and PPO clip must be positive")
    margin = policy_chosen - policy_rejected
    relative = margin - (reference_chosen - reference_rejected)

    if method is AlignmentType.DPO:
        derivative = -beta * _sigmoid(-beta * relative)
        return derivative, -derivative
    if method is AlignmentType.IPO:
        derivative = 2.0 * (relative - 1.0 / (2.0 * beta))
        return derivative, -derivative
    if method is AlignmentType.SIMPO:
        derivative = -beta * _sigmoid(-(beta * margin - gamma))
        return derivative, -derivative
    if method is AlignmentType.CPO:
        derivative = -beta * _sigmoid(-beta * margin)
        return derivative - beta, -derivative
    if method is AlignmentType.ORPO:
        chosen_probability = min(
            1.0 - _PROBABILITY_EPSILON,
            max(_PROBABILITY_EPSILON, math.exp(min(0.0, policy_chosen))),
        )
        rejected_probability = min(
            1.0 - _PROBABILITY_EPSILON,
            max(_PROBABILITY_EPSILON, math.exp(min(0.0, policy_rejected))),
        )
        odds_margin = _log_odds(policy_chosen) - _log_odds(policy_rejected)
        odds_derivative = -beta * _sigmoid(-odds_margin)
        return (
            -1.0 + odds_derivative / (1.0 - chosen_probability),
            -odds_derivative / (1.0 - rejected_probability),
        )
    if method is AlignmentType.KTO:
        if reward is not None:
            if reward == 0.0:
                raise ConfigurationError(
                    "kto reward labels must be positive or negative, not zero"
                )
            sigmoid = _sigmoid(beta * (policy_chosen - reference_chosen))
            derivative = abs(reward) * beta * sigmoid * (1.0 - sigmoid)
            return (-derivative if reward > 0.0 else derivative), 0.0
        raw_kl = 0.5 * (
            (policy_chosen - reference_chosen)
            + (policy_rejected - reference_rejected)
        )
        kl = max(0.0, raw_kl)
        dkl = 0.5 if raw_kl > 0.0 else 0.0
        desirable_arg = beta * (policy_chosen - reference_chosen - kl)
        undesirable_arg = beta * (kl - policy_rejected + reference_rejected)
        desirable_sigmoid = _sigmoid(desirable_arg)
        undesirable_sigmoid = _sigmoid(undesirable_arg)
        desirable_derivative = -desirable_sigmoid * (1.0 - desirable_sigmoid)
        undesirable_derivative = -undesirable_sigmoid * (1.0 - undesirable_sigmoid)
        chosen = 0.5 * beta * (
            desirable_derivative * (1.0 - dkl)
            + undesirable_derivative * dkl
        )
        rejected = 0.5 * beta * (
            desirable_derivative * -dkl
            + undesirable_derivative * (dkl - 1.0)
        )
        return chosen, rejected
    if method is AlignmentType.PPO:
        advantage = 1.0 if reward is None else reward
        log_ratio = policy_chosen - reference_chosen
        if log_ratio <= -20.0 or log_ratio >= 20.0:
            return 0.0, 0.0
        ratio = math.exp(log_ratio)
        clipped = max(1.0 - clip, min(1.0 + clip, ratio))
        if ratio * advantage <= clipped * advantage:
            return -advantage * ratio, 0.0
        return 0.0, 0.0
    if method in {AlignmentType.REINFORCE, AlignmentType.RLOO}:
        advantage = 1.0 if reward is None else reward
        return -advantage, 0.0
    if method is AlignmentType.GRPO:
        advantage = 1.0 if reward is None else reward
        raw_log_ratio = policy_chosen - reference_chosen
        log_ratio = max(-20.0, min(20.0, raw_log_ratio))
        ratio = math.exp(log_ratio)
        clipped = max(1.0 - clip, min(1.0 + clip, ratio))
        surrogate_derivative = (
            advantage * ratio if ratio * advantage <= clipped * advantage else 0.0
        )
        reference_log_ratio = max(
            -20.0, min(20.0, reference_chosen - policy_chosen)
        )
        reference_ratio = math.exp(reference_log_ratio)
        kl_derivative = beta * (1.0 - reference_ratio)
        if not -20.0 < raw_log_ratio < 20.0:
            surrogate_derivative = 0.0
            kl_derivative = 0.0
        return -surrogate_derivative + kl_derivative, 0.0
    raise ConfigurationError("auto alignment type must be resolved before optimization")


def _alignment_type(value: str | AlignmentType) -> AlignmentType:
    if isinstance(value, AlignmentType):
        return value
    return AlignmentType(value)


def example_as_dict(example: AlignmentExample) -> dict[str, Any]:
    return {key: value for key, value in asdict(example).items() if value is not None}


def _parse_example(item: Any, path: Path, line: int) -> tuple[AlignmentExample, str]:
    if not isinstance(item, dict):
        raise ConfigurationError(f"expected an alignment JSON object at {path}:{line}")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError(f"alignment record needs a non-empty prompt at {path}:{line}")
    chosen = item.get("chosen")
    rejected = item.get("rejected")
    if all(isinstance(value, str) and value.strip() for value in (chosen, rejected)):
        return AlignmentExample(prompt, chosen=chosen, rejected=rejected), "preference"
    response = item.get("response")
    reward = item.get("reward")
    old_logprob = item.get("old_logprob")
    if (
        isinstance(response, str)
        and response.strip()
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
        and (
            old_logprob is None
            or (
                isinstance(old_logprob, (int, float))
                and not isinstance(old_logprob, bool)
                and math.isfinite(float(old_logprob))
            )
        )
    ):
        return (
            AlignmentExample(
                prompt,
                response=response,
                reward=float(reward),
                old_logprob=float(old_logprob) if old_logprob is not None else None,
            ),
            "reward",
        )
    raise ConfigurationError(
        f"alignment record must contain prompt/chosen/rejected or "
        f"prompt/response/reward at {path}:{line}"
    )


def _log_sigmoid(value: float) -> float:
    return -math.log1p(math.exp(-abs(value))) + min(value, 0.0)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _log_odds(log_probability: float) -> float:
    # log(-expm1(log_p)) evaluates log(1 - p) without rounding p to one.
    bounded = min(math.log1p(-_PROBABILITY_EPSILON), log_probability)
    return bounded - math.log(-math.expm1(bounded))
