"""Fresh local-policy rollout collection for alignment."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .alignment import (
    AlignmentDataset,
    AlignmentExample,
    AlignmentType,
    load_alignment_dataset,
)
from .errors import ConfigurationError, TrainingError
from .io import atomic_json, atomic_text, sha256_file

_PAIRWISE_METHODS = {
    AlignmentType.DPO,
    AlignmentType.IPO,
    AlignmentType.SIMPO,
    AlignmentType.ORPO,
    AlignmentType.CPO,
}
_GROUP_METHODS = {AlignmentType.RLOO, AlignmentType.GRPO}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    source_index: int
    sample_index: int
    prompt: str
    seed: int


@dataclass(frozen=True, slots=True)
class RolloutSettings:
    enabled: bool = True
    samples_per_prompt: int = 2
    max_tokens: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 0

    def validate(self, method: AlignmentType) -> None:
        if self.samples_per_prompt < 1 or self.max_tokens < 1:
            raise ConfigurationError("rollout sample and token counts must be at least 1")
        if self.seed < 0:
            raise ConfigurationError("rollout seed must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ConfigurationError("rollout temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ConfigurationError("rollout top-p must be greater than 0 and at most 1")
        if self.enabled and method in _GROUP_METHODS and self.samples_per_prompt < 2:
            raise ConfigurationError(
                f"{method.value} live rollouts need at least two samples per prompt"
            )


@dataclass(frozen=True, slots=True)
class RolloutResult:
    dataset: AlignmentDataset
    data_file: Path
    manifest: Path
    generated_examples: int
    scorer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "live-local",
            "data_file": str(self.data_file),
            "manifest": str(self.manifest),
            "generated_examples": self.generated_examples,
            "scorer": self.scorer,
        }


RolloutGenerator = Callable[[tuple[RolloutRequest, ...]], Sequence[str]]


def collect_live_rollouts(
    source: AlignmentDataset,
    destination: str | Path,
    *,
    engine: str,
    adapter: str | Path,
    settings: RolloutSettings,
    generate: RolloutGenerator,
) -> RolloutResult:
    """Generate current-policy answers, score them locally, and persist provenance."""

    method = source.alignment_type
    settings.validate(method)
    if not settings.enabled:
        raise ConfigurationError("live rollout collection was called while disabled")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(adapter).expanduser().resolve()
    requests = tuple(
        RolloutRequest(
            source_index=source_index,
            sample_index=sample_index,
            prompt=example.prompt,
            seed=settings.seed + source_index * settings.samples_per_prompt + sample_index,
        )
        for source_index, example in enumerate(source.examples)
        for sample_index in range(settings.samples_per_prompt)
    )
    generated = tuple(str(value).strip() for value in generate(requests))
    if len(generated) != len(requests):
        raise TrainingError(
            "local rollout generator returned a different number of answers than requested"
        )
    empty = [index for index, value in enumerate(generated) if not value]
    if empty:
        raise TrainingError(
            "local rollout generator returned an empty answer for request "
            + ", ".join(str(index) for index in empty[:5])
        )

    rows: list[dict[str, Any]] = []
    for request, response in zip(requests, generated, strict=True):
        source_example = source.examples[request.source_index]
        row = _rollout_row(method, source_example, response)
        row["rollout"] = {
            "generated": True,
            "source_index": request.source_index,
            "sample_index": request.sample_index,
            "seed": request.seed,
            "engine": engine,
        }
        rows.append(row)

    data_file = root / "train.jsonl"
    atomic_text(
        data_file,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    # Reload through the public validator so generated data cannot bypass any
    # objective-specific constraints such as GRPO reward variance.
    dataset = load_alignment_dataset(root, method)
    scorer = (
        "pairwise-reference-selection"
        if method in _PAIRWISE_METHODS
        else "local-reference-similarity"
    )
    manifest_path = root / "manifest.json"
    source_file = source.path / "train.jsonl"
    adapter_digest = (
        sha256_file(adapter_path / "adapters.safetensors")
        if adapter_path.is_dir()
        else sha256_file(adapter_path)
    )
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "live-local",
            "local_only": True,
            "network_used": False,
            "engine": engine,
            "alignment_type": method.value,
            "source_dataset": str(source.path),
            "source_sha256": sha256_file(source_file),
            "fine_tuned_adapter": str(adapter_path),
            "fine_tuned_adapter_sha256": adapter_digest,
            "settings": asdict(settings),
            "scorer": scorer,
            "generated_examples": len(rows),
            "rollout_sha256": sha256_file(data_file),
        },
    )
    return RolloutResult(dataset, data_file, manifest_path, len(rows), scorer)


def _rollout_row(
    method: AlignmentType,
    source: AlignmentExample,
    response: str,
) -> dict[str, Any]:
    if method in _PAIRWISE_METHODS:
        if source.chosen is None or source.rejected is None:
            raise ConfigurationError(
                f"{method.value} live rollouts require prompt/chosen/rejected source rows"
            )
        chosen_score = _similarity(response, source.chosen)
        rejected_score = _similarity(response, source.rejected)
        if chosen_score >= rejected_score:
            chosen, rejected = response, source.rejected
        else:
            chosen, rejected = source.chosen, response
        return {"prompt": source.prompt, "chosen": chosen, "rejected": rejected}

    reward = _local_reward(source, response)
    if method is AlignmentType.KTO and reward == 0.0:
        reward = -1e-6
    return {"prompt": source.prompt, "response": response, "reward": reward}


def _local_reward(source: AlignmentExample, response: str) -> float:
    if source.chosen is not None and source.rejected is not None:
        return _similarity(response, source.chosen) - _similarity(
            response, source.rejected
        )
    if source.response is not None and source.reward is not None:
        similarity = _similarity(response, source.response)
        return float(source.reward) * max(similarity, 1e-6)
    raise ConfigurationError("live rollout scoring requires preference or reward references")


def _similarity(candidate: str, reference: str) -> float:
    candidate_words = _TOKEN_RE.findall(candidate.casefold())
    reference_words = _TOKEN_RE.findall(reference.casefold())
    if not candidate_words or not reference_words:
        return 0.0
    candidate_counts = Counter(candidate_words)
    reference_counts = Counter(reference_words)
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_words)
    recall = overlap / len(reference_words)
    token_f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    sequence = SequenceMatcher(None, " ".join(candidate_words), " ".join(reference_words)).ratio()
    return 0.7 * token_f1 + 0.3 * sequence
