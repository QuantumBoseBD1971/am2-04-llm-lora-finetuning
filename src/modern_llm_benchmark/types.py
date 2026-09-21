"""Core benchmark data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    example_id: str
    instruction: str
    expected: str


@dataclass(frozen=True)
class GenerationResult:
    example_id: str
    prompt_type: str
    output: str
