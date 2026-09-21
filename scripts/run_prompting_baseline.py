"""Run the Phase 1 prompting benchmark with a CI-safe deterministic generator."""

from __future__ import annotations

from pathlib import Path

from modern_llm_benchmark.benchmark import run_prompt_benchmark
from modern_llm_benchmark.fixtures import benchmark_examples, demonstration_examples

RESULTS_PATH = Path("results/tables/phase1_prompting_baseline.csv")


class DeterministicGenerator:
    """Small local test double that mimics an instruction-following model."""

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
    table = run_prompt_benchmark(
        benchmark_examples(),
        DeterministicGenerator(),
        demonstration_examples(),
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
