"""Synthesize generated benchmark artefacts into a final project summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modern_llm_benchmark.robustness import summarise_errors

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_PATH = RESULTS_DIR / "final_project_summary.json"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    summary: dict[str, object] = {
        "project": "am2-04-llm-lora-finetuning",
        "generated_from": "local reproducible result artefacts",
    }

    prompting = _read_csv("phase1_prompting_baseline.csv")
    if prompting is not None and not prompting.empty:
        summary["phase1"] = (
            prompting.groupby("prompt_type")[["exact_match", "token_f1"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

    retrieval = _read_csv("phase2_retrieval_prompting.csv")
    if retrieval is not None and not retrieval.empty:
        summary["phase2"] = (
            retrieval.groupby("prompt_type")[["exact_match", "token_f1"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

    lora = _read_csv("phase3_base_vs_lora.csv")
    if lora is not None and not lora.empty:
        summary["phase3"] = (
            lora.groupby("model")[["exact_match", "token_f1"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        summarise_errors(lora, "model").to_csv(
            TABLES_DIR / "phase4_error_summary.csv",
            index=False,
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
