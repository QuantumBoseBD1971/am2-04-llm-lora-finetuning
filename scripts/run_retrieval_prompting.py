"""Run the Phase 2 retrieval-assisted prompting benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modern_llm_benchmark.fixtures import benchmark_examples, demonstration_examples
from modern_llm_benchmark.retrieval import ExampleRetriever
from modern_llm_benchmark.retrieval_benchmark import run_retrieval_prompt_benchmark

RESULTS_PATH = Path("results/tables/phase2_retrieval_prompting.csv")


class KeywordEncoder:
    """Deterministic semantic stand-in used for the CI-safe demo."""

    vocabulary = [
        "opposite",
        "lowercase",
        "number",
        "plus",
        "plural",
    ]

    def encode(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    float(term in lower)
                    for term in self.vocabulary
                ]
            )
        return np.asarray(rows, dtype=float)


class DeterministicGenerator:
    """Deterministic benchmark generator."""

    def generate(self, prompt: str) -> str:
        rules = {
            "opposite of 'increase'": "decrease",
            "convert 'hello' to lowercase": "hello",
            "two plus three": "5",
            "plural of 'battery'": "batteries",
        }
        lower = prompt.lower()
        for marker, answer in rules.items():
            if marker in lower:
                return answer
        return "unknown"


def main() -> None:
    examples = benchmark_examples()
    demonstrations = demonstration_examples()
    retriever = ExampleRetriever(demonstrations, KeywordEncoder())

    table = run_retrieval_prompt_benchmark(
        examples,
        DeterministicGenerator(),
        demonstrations,
        retriever,
        retrieved_k=2,
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    summary = (
        table.groupby("prompt_type")[["exact_match", "token_f1"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
