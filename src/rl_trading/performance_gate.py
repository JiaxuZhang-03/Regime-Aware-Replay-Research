from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceGateConfig:
    min_final_value: float = 0.90
    max_drawdown: float = 0.35
    max_turnover: float = 1.25
    score_drawdown_penalty: float = 1.0
    score_turnover_penalty: float = 0.05


def apply_performance_gate(
    summary: pd.DataFrame,
    cfg: PerformanceGateConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or PerformanceGateConfig()
    out = summary.copy()
    final_value = _numeric(out, "final_portfolio_value")
    drawdown = _numeric(out, "max_drawdown")
    turnover = _numeric(out, "mean_turnover")

    out["robust_score"] = (
        final_value
        - cfg.score_drawdown_penalty * drawdown
        - cfg.score_turnover_penalty * turnover
    )

    reasons: list[str] = []
    passes: list[bool] = []
    for i in range(len(out)):
        row_reasons = []
        if np.isfinite(final_value[i]) and final_value[i] < cfg.min_final_value:
            row_reasons.append("low_final_value")
        if np.isfinite(drawdown[i]) and drawdown[i] > cfg.max_drawdown:
            row_reasons.append("high_drawdown")
        if np.isfinite(turnover[i]) and turnover[i] > cfg.max_turnover:
            row_reasons.append("high_turnover")
        passes.append(len(row_reasons) == 0)
        reasons.append(";".join(row_reasons) if row_reasons else "pass")

    out["passes_performance_gate"] = passes
    out["gate_failure_reasons"] = reasons
    out["gate_min_final_value"] = cfg.min_final_value
    out["gate_max_drawdown"] = cfg.max_drawdown
    out["gate_max_turnover"] = cfg.max_turnover
    return out


def select_best_policies(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    working = summary.copy()
    if "passes_performance_gate" not in working.columns:
        working = apply_performance_gate(working)

    group_cols = [c for c in ["label_method", "seed"] if c in working.columns]
    if not group_cols:
        group_cols = ["_all"]
        working["_all"] = "all"

    rows = []
    for _, group in working.groupby(group_cols, dropna=False):
        passing = group[group["passes_performance_gate"].astype(bool)]
        candidates = passing if not passing.empty else group
        best = candidates.sort_values("robust_score", ascending=False).iloc[0].copy()
        best["selection_pool_size"] = int(len(group))
        best["selection_used_fallback"] = bool(passing.empty)
        rows.append(best)

    selected = pd.DataFrame(rows)
    if "_all" in selected.columns:
        selected = selected.drop(columns=["_all"])
    return selected.reset_index(drop=True)


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
