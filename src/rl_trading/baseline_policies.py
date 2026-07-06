from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

from src.regime_labeling.features import MarketFeatureConfig, build_market_features
from src.rl_trading.dqn_replay import build_asset_returns, find_col
from src.rl_trading.policy_safety import (
    apply_policy_safety,
    normalize_long_only,
    regime_anchor_weights,
    safety_config_from_object,
)


BASELINE_POLICIES = ("cash", "equal_weight", "regime_anchor", "vol_target")


@dataclass
class BaselineConfig:
    market_csv: str = "data/market_indices_20080601_20260531/market_regime_features_wide.csv"
    labels_csv: str = "outputs/regime_labels/all_regime_labels.csv"
    output_root: str = "outputs/baseline_policies"
    label_method: str = "rule_based"
    policy: str = "regime_anchor"
    seed: int = 0
    tradable_symbols: tuple[str, ...] = ("DIA", "SPY", "QQQ")
    primary_symbol: str = "SPY"
    transaction_cost_bps: float = 10.0
    max_steps: int = 0

    safety_enabled: bool = True
    safety_min_cash_weight: float = 0.03
    safety_max_asset_weight: float = 0.85
    safety_max_turnover: float = 0.75
    safety_regime_blend: float = 0.0
    safety_risk_on_cash: float = 0.05
    safety_sideways_cash: float = 0.25
    safety_high_vol_cash: float = 0.55
    safety_risk_off_cash: float = 0.70

    vol_target_ann_vol: float = 0.12
    vol_target_window: int = 20


def run_many(
    base_cfg: BaselineConfig,
    policies: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    summaries = []
    for policy in policies:
        if policy not in BASELINE_POLICIES:
            raise ValueError(f"unknown baseline policy: {policy}")
        for seed in seeds:
            cfg = BaselineConfig(**base_cfg.__dict__)
            cfg.policy = policy
            cfg.seed = seed
            out_dir = run_single_policy(cfg)
            summaries.append(pd.read_csv(out_dir / "summary.csv"))

    summary = pd.concat(summaries, ignore_index=True)
    analysis_dir = Path(base_cfg.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / f"{base_cfg.label_method}_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[summary] wrote {summary_path}")
    return summary


def run_single_policy(cfg: BaselineConfig) -> Path:
    if cfg.policy not in BASELINE_POLICIES:
        raise ValueError(f"policy must be one of: {BASELINE_POLICIES}")

    df, symbols = _load_market_df(cfg)
    rng = np.random.default_rng(cfg.seed)
    _ = rng  # reserved for future stochastic baselines

    output_dir = Path(cfg.output_root) / f"{cfg.label_method}_{cfg.policy}_seed{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_names = ["cash", *symbols]
    safety_cfg = safety_config_from_object(cfg)
    cost_rate = cfg.transaction_cost_bps / 10000.0

    prev_weights = np.zeros(len(weight_names), dtype=np.float32)
    prev_weights[0] = 1.0
    portfolio_value = 1.0
    peak_value = 1.0
    max_steps = len(df) - 2
    if cfg.max_steps > 0:
        max_steps = min(max_steps, int(cfg.max_steps))

    rows: list[dict[str, Any]] = []
    for step in range(max_steps):
        current = df.iloc[step]
        next_row = df.iloc[step + 1]
        proposed = _policy_weights(cfg, df, step, symbols)
        weights, safety_info = apply_policy_safety(
            proposed_weights=proposed,
            previous_weights=prev_weights,
            regime_label=int(current["regime_label"]),
            regime_name=str(current["regime_name"]),
            cfg=safety_cfg,
        )

        asset_returns = np.array([float(next_row[f"ret_{s}"]) for s in symbols], dtype=np.float32)
        gross_return = float(np.dot(weights[1:], asset_returns))
        turnover = float(np.abs(weights - prev_weights).sum())
        cost = cost_rate * turnover
        reward = math.log(max(1e-12, 1.0 + gross_return)) - cost
        portfolio_value *= math.exp(reward)
        peak_value = max(peak_value, portfolio_value)
        drawdown = 1.0 - portfolio_value / max(peak_value, 1e-12)

        row = {
            "step": step,
            "date": str(current["date"].date()),
            "policy": cfg.policy,
            "regime_label": int(current["regime_label"]),
            "regime_name": str(current["regime_name"]),
            "reward": float(reward),
            "portfolio_value": float(portfolio_value),
            "drawdown": float(drawdown),
            "turnover": float(turnover),
            "gross_return": float(gross_return),
            "cost": float(cost),
            **safety_info,
        }
        for name, value in zip(weight_names, weights):
            row[f"weight_{name}"] = float(value)
        for name, value in zip(weight_names, proposed):
            row[f"proposed_weight_{name}"] = float(value)
        rows.append(row)
        prev_weights = weights.copy()

    trade_df = pd.DataFrame(rows)
    trade_df.to_csv(output_dir / "trading_log.csv", index=False)

    summary = {
        "label_method": cfg.label_method,
        "replay": "policy",
        "seed": cfg.seed,
        "policy": cfg.policy,
        "final_portfolio_value": float(trade_df["portfolio_value"].iloc[-1]),
        "max_drawdown": float(trade_df["drawdown"].max()),
        "mean_turnover": float(trade_df["turnover"].mean()),
        "mean_reward": float(trade_df["reward"].mean()),
        "mean_cash_weight": float(trade_df["weight_cash"].mean()),
        "mean_proposed_cash_weight": float(trade_df["proposed_weight_cash"].mean()),
        "mean_safety_turnover_cap_delta": float(
            (trade_df["safety_turnover_before_cap"] - trade_df["safety_turnover_after_cap"]).mean()
        ),
        "safety_enabled": bool(cfg.safety_enabled),
        "safety_regime_blend": cfg.safety_regime_blend,
        "vol_target_ann_vol": cfg.vol_target_ann_vol,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    metadata = {
        "config": {**cfg.__dict__, "tradable_symbols": list(cfg.tradable_symbols)},
        "symbols": symbols,
        "weight_names": weight_names,
        "n_rows": int(len(df)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"[done] baseline | {cfg.label_method} | {cfg.policy} | seed={cfg.seed} | "
        f"final_value={summary['final_portfolio_value']:.4f} | "
        f"max_dd={summary['max_drawdown']:.4f}"
    )
    return output_dir


def _policy_weights(
    cfg: BaselineConfig,
    df: pd.DataFrame,
    step: int,
    symbols: list[str],
) -> np.ndarray:
    n_assets = len(symbols)
    if cfg.policy == "cash":
        out = np.zeros(n_assets + 1, dtype=np.float32)
        out[0] = 1.0
        return out

    if cfg.policy == "equal_weight":
        out = np.zeros(n_assets + 1, dtype=np.float32)
        out[1:] = 1.0 / max(n_assets, 1)
        return out

    current = df.iloc[step]
    if cfg.policy == "regime_anchor":
        return regime_anchor_weights(
            int(current["regime_label"]),
            str(current["regime_name"]),
            n_assets,
            safety_config_from_object(cfg),
        )

    if cfg.policy == "vol_target":
        exposure = _vol_target_exposure(cfg, df, step, symbols)
        out = np.zeros(n_assets + 1, dtype=np.float32)
        out[0] = 1.0 - exposure
        out[1:] = exposure / max(n_assets, 1)
        return normalize_long_only(out)

    raise ValueError(f"unknown policy: {cfg.policy}")


def _vol_target_exposure(
    cfg: BaselineConfig,
    df: pd.DataFrame,
    step: int,
    symbols: list[str],
) -> float:
    if step <= 1:
        return 0.5
    start = max(0, step - int(cfg.vol_target_window))
    ret_cols = [f"ret_{s}" for s in symbols]
    equal_returns = df.iloc[start:step][ret_cols].mean(axis=1)
    realized = float(equal_returns.std() * np.sqrt(252.0))
    if not np.isfinite(realized) or realized <= 1e-8:
        return 1.0
    return float(np.clip(cfg.vol_target_ann_vol / realized, 0.0, 1.0))


def _load_market_df(cfg: BaselineConfig) -> tuple[pd.DataFrame, list[str]]:
    market_path = Path(cfg.market_csv)
    label_path = Path(cfg.labels_csv)
    if not market_path.exists():
        raise FileNotFoundError(f"market_csv not found: {market_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"labels_csv not found: {label_path}")

    raw = pd.read_csv(market_path)
    if "date" not in raw.columns:
        raise ValueError("market_csv must contain a date column.")
    raw["date"] = pd.to_datetime(raw["date"])

    features = build_market_features(
        raw,
        MarketFeatureConfig(date_col="date", primary_symbol=cfg.primary_symbol),
    )
    features["date"] = pd.to_datetime(features["date"])

    labels = pd.read_csv(label_path)
    if "date" not in labels.columns:
        raise ValueError("labels_csv must contain a date column.")
    labels["date"] = pd.to_datetime(labels["date"])
    if "method" in labels.columns:
        labels = labels[labels["method"].astype(str) == cfg.label_method].copy()
    if "regime_label" not in labels.columns:
        possible = find_col(labels, ["label", "regime", "state"])
        if possible is None:
            raise ValueError("labels file must contain regime_label, label, regime, or state.")
        labels["regime_label"] = labels[possible]

    num_label = pd.to_numeric(labels["regime_label"], errors="coerce")
    if num_label.isna().any():
        labels["regime_label"] = pd.Categorical(labels["regime_label"].astype(str)).codes
    else:
        labels["regime_label"] = num_label.astype(int)
    if "regime_name" not in labels.columns:
        labels["regime_name"] = labels["regime_label"].astype(str)

    labels = labels[["date", "regime_label", "regime_name"]].drop_duplicates("date")
    returns = build_asset_returns(raw, cfg.tradable_symbols)
    returns["date"] = pd.to_datetime(returns["date"])

    df = features.merge(labels, on="date", how="inner").merge(returns, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    symbols = []
    for s in cfg.tradable_symbols:
        su = s.upper()
        if f"ret_{su}" in df.columns:
            symbols.append(su)
    if not symbols:
        raise ValueError("No tradable returns found.")
    return df, symbols
