"""Run real Hugging Face prompting with semantic demonstration retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_llm_benchmark.fixtures import benchmark_examples, demonstration_examples
from modern_llm_benchmark.generation import TransformersGenerator
from modern_llm_benchmark.metrics import exact_match, token_f1
from modern_llm_benchmark.prompts import few_shot_prompt, zero_shot_prompt
from modern_llm_benchmark.retrieval import ExampleRetriever, SentenceTransformerEncoder
from modern_llm_benchmark.retrieved_prompting import retrieval_assisted_prompt

RESULTS_PATH = Path("results/tables/phase2_real_prompting.csv")


def main() -> None:
    examples = benchmark_examples()
    demonstrations = demonstration_examples()

    generator = TransformersGenerator()
    retriever = ExampleRetriever(
        demonstrations,
        SentenceTransformerEncoder(),
    )

    rows = []
    for example in examples:
        strategies = {
            "zero_shot": zero_shot_prompt(example),
            "fixed_few_shot": few_shot_prompt(example, demonstrations),
        }

        retrieved_prompt, retrieved_examples = retrieval_assisted_prompt(
            example,
            retriever,
            top_k=2,
        )
        strategies["retrieved_few_shot"] = retrieved_prompt

        for strategy, prompt in strategies.items():
            output = generator.generate(prompt)
            rows.append(
                {
                    "example_id": example.example_id,
                    "strategy": strategy,
                    "expected": example.expected,
                    "output": output,
                    "exact_match": exact_match(example.expected, output),
                    "token_f1": token_f1(example.expected, output),
                    "retrieved_example_ids": (
                        ",".join(item.example_id for item in retrieved_examples)
                        if strategy == "retrieved_few_shot"
                        else ""
                    ),
                }
            )

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    print(
        table.groupby("strategy")[["exact_match", "token_f1"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
