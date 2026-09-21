"""Deterministic benchmark examples for CI."""

from modern_llm_benchmark.types import Example


def benchmark_examples() -> list[Example]:
    return [
        Example(
            example_id="e1",
            instruction="Return the opposite of 'increase'.",
            expected="decrease",
        ),
        Example(
            example_id="e2",
            instruction="Convert 'HELLO' to lowercase.",
            expected="hello",
        ),
        Example(
            example_id="e3",
            instruction="Return only the number: two plus three.",
            expected="5",
        ),
        Example(
            example_id="e4",
            instruction="Return the plural of 'battery'.",
            expected="batteries",
        ),
    ]


def demonstration_examples() -> list[Example]:
    return [
        Example(
            example_id="d1",
            instruction="Return the opposite of 'hot'.",
            expected="cold",
        ),
        Example(
            example_id="d2",
            instruction="Convert 'WORLD' to lowercase.",
            expected="world",
        ),
        Example(
            example_id="d3",
            instruction="Return only the number: one plus one.",
            expected="2",
        ),
    ]
