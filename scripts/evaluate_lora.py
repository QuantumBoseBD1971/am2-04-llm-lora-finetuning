"""Compare base-model and LoRA-adapter outputs on the benchmark fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_llm_benchmark.adapter_generation import PeftGenerator
from modern_llm_benchmark.fixtures import benchmark_examples
from modern_llm_benchmark.generation import TransformersGenerator
from modern_llm_benchmark.metrics import exact_match, token_f1
from modern_llm_benchmark.prompts import zero_shot_prompt

RESULTS_PATH = Path("results/tables/phase3_base_vs_lora.csv")
ADAPTER_PATH = "artifacts/lora_phase3/adapter"


def evaluate_generator(name, generator):
    rows = []
    for example in benchmark_examples():
        output = generator.generate(zero_shot_prompt(example))
        rows.append(
            {
                "model": name,
                "example_id": example.example_id,
                "expected": example.expected,
                "output": output,
                "exact_match": exact_match(example.expected, output),
                "token_f1": token_f1(example.expected, output),
            }
        )
    return rows


def main() -> None:
    base = TransformersGenerator()
    adapted = PeftGenerator(ADAPTER_PATH)

    rows = [
        *evaluate_generator("base", base),
        *evaluate_generator("lora_adapter", adapted),
    ]
    table = pd.DataFrame(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    summary = (
        table.groupby("model")[["exact_match", "token_f1"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
