"""Small summaries for generated regime labels."""

from __future__ import annotations

import pandas as pd


def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "regime_label", "regime_name"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    summary = (
        labels.groupby(["method", "regime_label", "regime_name"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["method", "regime_label"])
    )
    summary["share"] = summary["n_rows"] / summary.groupby("method")["n_rows"].transform("sum")
    return summary


def count_label_switches(labels: pd.DataFrame) -> pd.DataFrame:
    if "method" not in labels.columns or "regime_label" not in labels.columns:
        raise ValueError("labels must contain method and regime_label")
    rows = []
    for method, group in labels.groupby("method", sort=True):
        group = group.sort_values("date") if "date" in group.columns else group
        switches = int(group["regime_label"].ne(group["regime_label"].shift()).sum() - 1)
        rows.append({"method": method, "label_switches": max(switches, 0)})
    return pd.DataFrame(rows)
