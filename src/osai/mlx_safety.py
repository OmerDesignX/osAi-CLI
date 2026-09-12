"""Memory-safe MLX batching helpers shared with the vendored trainer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

Token = TypeVar("Token")


def truncate_completion_aware(
    sequence: Sequence[Token], prompt_offset: int, max_seq_length: int
) -> tuple[Sequence[Token], int]:
    """Fit a sample while retaining prompt context and supervised answer tokens."""

    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    if len(sequence) <= max_seq_length:
        return sequence, prompt_offset
    if prompt_offset <= 0 or prompt_offset >= len(sequence):
        return sequence[:max_seq_length], min(prompt_offset, max_seq_length)

    completion_length = len(sequence) - prompt_offset
    completion_budget = min(completion_length, max(1, max_seq_length // 2))
    prompt_budget = max_seq_length - completion_budget
    start = max(0, prompt_offset - prompt_budget)
    fitted = sequence[start : start + max_seq_length]
    return fitted, max(0, prompt_offset - start)
