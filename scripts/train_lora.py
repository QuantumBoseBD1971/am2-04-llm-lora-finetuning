"""Train the optional LoRA adapter on the local instruction examples."""

from __future__ import annotations

import json
from pathlib import Path

from modern_llm_benchmark.finetune import train_lora_adapter
from modern_llm_benchmark.fixtures import (
    benchmark_examples,
    demonstration_examples,
)
from modern_llm_benchmark.lora import LoraSettings
from modern_llm_benchmark.training_data import build_training_records

OUTPUT_DIR = Path("artifacts/lora_phase3")


def main() -> None:
    training_examples = demonstration_examples() + benchmark_examples()
    records = build_training_records(training_examples)

    metrics = train_lora_adapter(
        records,
        output_dir=OUTPUT_DIR,
        settings=LoraSettings(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q", "v"),
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
