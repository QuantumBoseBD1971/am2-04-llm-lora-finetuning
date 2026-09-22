"""Build a compact assessor-facing evidence pack from executed LLM adaptation results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
ARTIFACTS_DIR = Path("artifacts/lora_phase3")
EVIDENCE_DIR = Path("evidence")
EVIDENCE_TABLES = EVIDENCE_DIR / "tables"


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def available_table(name: str) -> Path | None:
    current = TABLES_DIR / name
    if current.exists():
        return current
    historical = EVIDENCE_TABLES / name
    if historical.exists():
        return historical
    return None


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    EVIDENCE_TABLES.mkdir(parents=True, exist_ok=True)

    selected = [
        "phase1_prompting_baseline.csv",
        "phase2_retrieval_prompting.csv",
        "phase2_real_prompting.csv",
        "phase3_base_vs_lora.csv",
        "phase4_error_summary.csv",
    ]
    for name in selected:
        copy_if_exists(TABLES_DIR / name, EVIDENCE_TABLES / name)

    training_metrics = ARTIFACTS_DIR / "training_metrics.json"
    if training_metrics.exists():
        copy_if_exists(training_metrics, EVIDENCE_DIR / "lora_training_metrics.json")

    lines = [
        "# Real Experiment Results",
        "",
        "This evidence pack consolidates outputs from completed prompting, retrieval and LoRA workflow runs.",
        "",
    ]

    real_prompting = available_table("phase2_real_prompting.csv")
    if real_prompting is not None:
        frame = pd.read_csv(real_prompting)
        lines.extend(["## Real Hugging Face prompting", ""])
        for strategy, group in frame.groupby("strategy"):
            lines.append(
                f"- {strategy}: exact match={fmt(group['exact_match'].mean())}, "
                f"token F1={fmt(group['token_f1'].mean())}"
            )
        lines.extend(["", "### Generated outputs", ""])
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.example_id} / {row.strategy}: expected={row.expected!r}, output={row.output!r}"
            )
        lines.append("")

    lora = available_table("phase3_base_vs_lora.csv")
    if lora is not None:
        frame = pd.read_csv(lora)
        lines.extend(["## Base model vs LoRA adapter", ""])
        for model, group in frame.groupby("model"):
            lines.append(
                f"- {model}: exact match={fmt(group['exact_match'].mean())}, "
                f"token F1={fmt(group['token_f1'].mean())}"
            )
        lines.append("")

    training_path = EVIDENCE_DIR / "lora_training_metrics.json"
    if training_path.exists():
        metrics = json.loads(training_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "## LoRA training efficiency",
                "",
                f"- Base model: **{metrics.get('base_model', 'n/a')}**",
                f"- Trainable parameters: **{metrics.get('trainable_parameters', 'n/a')}**",
                f"- Total parameters: **{metrics.get('total_parameters', 'n/a')}**",
                f"- Trainable percentage: **{fmt(metrics.get('trainable_percentage'))}%**",
                f"- Training loss: **{fmt(metrics.get('train_loss'))}**",
                "",
            ]
        )

    summary = {
        "project": "am2-04-llm-lora-finetuning",
        "deterministic_prompting_executed": available_table("phase1_prompting_baseline.csv") is not None,
        "retrieval_prompting_executed": available_table("phase2_retrieval_prompting.csv") is not None,
        "real_hf_prompting_executed": real_prompting is not None,
        "lora_training_executed": training_path.exists(),
        "lora_evaluation_executed": lora is not None,
    }
    (EVIDENCE_DIR / "final_project_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines.extend(
        [
            "## Evidence files",
            "",
            "- tables/phase1_prompting_baseline.csv",
            "- tables/phase2_retrieval_prompting.csv",
            "- tables/phase2_real_prompting.csv",
            "- tables/phase3_base_vs_lora.csv",
            "- tables/phase4_error_summary.csv",
            "- lora_training_metrics.json",
            "- final_project_summary.json",
            "",
            "Large adapter/checkpoint files remain excluded from Git.",
        ]
    )

    (EVIDENCE_DIR / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Evidence pack written to {EVIDENCE_DIR.resolve()}")


if __name__ == "__main__":
    main()
