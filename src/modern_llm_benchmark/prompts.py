"""Prompt construction for zero-shot and few-shot evaluation."""

from __future__ import annotations

from modern_llm_benchmark.types import Example

SYSTEM_TEXT = (
    "Follow the instruction exactly. Return only the answer and no explanation."
)


def zero_shot_prompt(example: Example) -> str:
    """Construct a minimal zero-shot prompt."""
    return (
        f"{SYSTEM_TEXT}\n\n"
        f"Instruction: {example.instruction}\n"
        "Answer:"
    )


def few_shot_prompt(
    example: Example,
    demonstrations: list[Example],
) -> str:
    """Construct a fixed few-shot prompt."""
    blocks = [SYSTEM_TEXT]

    for demo in demonstrations:
        blocks.append(
            f"Instruction: {demo.instruction}\n"
            f"Answer: {demo.expected}"
        )

    blocks.append(
        f"Instruction: {example.instruction}\n"
        "Answer:"
    )
    return "\n\n".join(blocks)
