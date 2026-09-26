"""Memory-safe MLX batching helpers shared with the vendored trainer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

Token = TypeVar("Token")


class WindowedDataset:
    """Expose every supervised token through bounded, one-token-overlap windows.

    The overlap is input context only: the shifted language-model loss starts at
    the next token, so no target is trained twice. Prompt-only windows are
    omitted, but the window at the answer boundary keeps preceding context.
    """

    def __init__(self, dataset, max_seq_length: int):
        if max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        self.dataset = dataset
        self.windows: list[tuple[int, int, int, int]] = []
        for index in range(len(dataset)):
            sequence, offset = dataset[index]
            length = len(sequence)
            target = max(1, offset)
            first = True
            while target < length:
                start = max(0, target - max_seq_length // 2) if first else target - 1
                end = min(length, start + max_seq_length)
                self.windows.append((index, start, end, max(0, offset - start)))
                target = end
                first = False

    def __len__(self):
        return len(self.windows)

    def itemlen(self, index: int) -> int:
        _, start, end, _ = self.windows[index]
        return end - start

    def __getitem__(self, index: int):
        row, start, end, offset = self.windows[index]
        return self.dataset[row][0][start:end], offset


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
