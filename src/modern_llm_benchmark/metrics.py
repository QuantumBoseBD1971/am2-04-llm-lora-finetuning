"""Simple generation evaluation metrics."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalise(text: str) -> str:
    """Lowercase, trim and collapse whitespace."""
    return " ".join(text.strip().lower().split())


def exact_match(expected: str, predicted: str) -> float:
    """Return 1.0 for normalised exact match, else 0.0."""
    return float(normalise(expected) == normalise(predicted))


def token_f1(expected: str, predicted: str) -> float:
    """Compute bag-of-token F1."""
    expected_tokens = TOKEN_PATTERN.findall(normalise(expected))
    predicted_tokens = TOKEN_PATTERN.findall(normalise(predicted))

    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    expected_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}

    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1

    overlap = sum(
        min(count, predicted_counts.get(token, 0))
        for token, count in expected_counts.items()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return float(2 * precision * recall / (precision + recall))
