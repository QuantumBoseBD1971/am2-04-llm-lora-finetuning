"""Shared prompting benchmark runner."""

from __future__ import annotations

import pandas as pd

from modern_llm_benchmark.generation import Generator
from modern_llm_benchmark.metrics import exact_match, token_f1
from modern_llm_benchmark.prompts import few_shot_prompt, zero_shot_prompt
from modern_llm_benchmark.types import Example


def run_prompt_benchmark(
    examples: list[Example],
    generator: Generator,
    demonstrations: list[Example],
) -> pd.DataFrame:
    """Evaluate zero-shot and few-shot prompting on the same examples."""
    rows = []

    for prompt_type in ["zero_shot", "few_shot"]:
        for example in examples:
            if prompt_type == "zero_shot":
                prompt = zero_shot_prompt(example)
            else:
                prompt = few_shot_prompt(example, demonstrations)

            output = generator.generate(prompt)
            rows.append(
                {
                    "example_id": example.example_id,
                    "prompt_type": prompt_type,
                    "expected": example.expected,
                    "output": output,
                    "exact_match": exact_match(example.expected, output),
                    "token_f1": token_f1(example.expected, output),
                }
            )

    return pd.DataFrame(rows)
