"""Robustness and error-analysis helpers for adaptation benchmarks."""

from __future__ import annotations

import pandas as pd


def add_error_flags(results: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic error flags to a benchmark result table."""
    required = {"expected", "output", "exact_match", "token_f1"}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"Missing required result columns: {sorted(missing)}")

    out = results.copy()
    out["is_error"] = out["exact_match"] < 1.0
    out["partial_match"] = (out["exact_match"] < 1.0) & (out["token_f1"] > 0.0)
    out["empty_output"] = out["output"].fillna("").astype(str).str.strip().eq("")
    return out


def summarise_errors(results: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarise exact-match and token-F1 performance by strategy/model."""
    if group_column not in results.columns:
        raise KeyError(f"Grouping column '{group_column}' is missing.")

    enriched = add_error_flags(results)
    return (
        enriched.groupby(group_column, as_index=False)
        .agg(
            examples=("is_error", "size"),
            error_count=("is_error", "sum"),
            mean_exact_match=("exact_match", "mean"),
            mean_token_f1=("token_f1", "mean"),
            partial_match_count=("partial_match", "sum"),
            empty_output_count=("empty_output", "sum"),
        )
    )
