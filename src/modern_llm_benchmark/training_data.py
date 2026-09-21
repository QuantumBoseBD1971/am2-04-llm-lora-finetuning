"""Formatting helpers for supervised instruction fine-tuning."""

from __future__ import annotations

from modern_llm_benchmark.types import Example


def format_training_pair(example: Example) -> dict[str, str]:
    """Convert an Example to source/target text."""
    return {
        "source": (
            "Follow the instruction exactly. Return only the answer and no explanation.\n\n"
            f"Instruction: {example.instruction}\n"
            "Answer:"
        ),
        "target": example.expected,
    }


def build_training_records(examples: list[Example]) -> list[dict[str, str]]:
    """Convert examples to supervised fine-tuning records."""
    return [format_training_pair(example) for example in examples]
