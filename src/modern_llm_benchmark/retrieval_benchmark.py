"""Compare zero-shot, fixed few-shot and retrieved few-shot prompting."""

from __future__ import annotations

import pandas as pd

from modern_llm_benchmark.generation import Generator
from modern_llm_benchmark.metrics import exact_match, token_f1
from modern_llm_benchmark.prompts import few_shot_prompt, zero_shot_prompt
from modern_llm_benchmark.retrieval import ExampleRetriever
from modern_llm_benchmark.retrieved_prompting import retrieval_assisted_prompt
from modern_llm_benchmark.types import Example


def run_retrieval_prompt_benchmark(
    examples: list[Example],
    generator: Generator,
    demonstrations: list[Example],
    retriever: ExampleRetriever,
    retrieved_k: int = 2,
) -> pd.DataFrame:
    """Evaluate all prompting strategies under one harness."""
    rows = []

    for example in examples:
        strategies = {
            "zero_shot": zero_shot_prompt(example),
            "fixed_few_shot": few_shot_prompt(example, demonstrations),
        }

        retrieved_prompt, retrieved_examples = retrieval_assisted_prompt(
            example,
            retriever,
            top_k=retrieved_k,
        )
        strategies["retrieved_few_shot"] = retrieved_prompt

        for prompt_type, prompt in strategies.items():
            output = generator.generate(prompt)
            rows.append(
                {
                    "example_id": example.example_id,
                    "prompt_type": prompt_type,
                    "expected": example.expected,
                    "output": output,
                    "exact_match": exact_match(example.expected, output),
                    "token_f1": token_f1(example.expected, output),
                    "retrieved_example_ids": (
                        ",".join(item.example_id for item in retrieved_examples)
                        if prompt_type == "retrieved_few_shot"
                        else ""
                    ),
                }
            )

    return pd.DataFrame(rows)
