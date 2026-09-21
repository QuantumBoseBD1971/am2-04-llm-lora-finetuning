from pathlib import Path

import pandas as pd

from modern_llm_benchmark.robustness import add_error_flags, summarise_errors
from modern_llm_benchmark.tracking import append_run, create_run, read_runs


def test_error_flags_and_summary() -> None:
    frame = pd.DataFrame(
        {
            "model": ["base", "base"],
            "expected": ["a", "b"],
            "output": ["a", "x"],
            "exact_match": [1.0, 0.0],
            "token_f1": [1.0, 0.0],
        }
    )

    enriched = add_error_flags(frame)
    assert enriched["is_error"].tolist() == [False, True]

    summary = summarise_errors(frame, "model")
    assert int(summary.iloc[0]["error_count"]) == 1


def test_experiment_tracking_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    run = create_run(
        experiment="unit-test",
        strategy="lora",
        metrics={"exact_match": 0.75},
        params={"rank": 8},
        artefacts={"adapter": "adapter-v1"},
    )

    append_run(run, path)
    loaded = read_runs(path)

    assert len(loaded) == 1
    assert loaded[0].run_id == run.run_id
    assert loaded[0].artefacts["adapter"] == "adapter-v1"
