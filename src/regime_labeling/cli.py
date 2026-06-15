"""Command line entry point for generating regime labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .diagnostics import count_label_switches, summarize_labels
from .features import MarketFeatureConfig
from .hmm import HMMConfig, label_hmm
from .recap_ard import RecapCusumConfig, label_recap_cusum
from .rule_based import RuleBasedConfig, label_rule_based


METHODS = ("rule_based", "hmm", "recap_cusum")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    feature_config = MarketFeatureConfig(
        date_col=args.date_col,
        primary_symbol=args.primary_symbol,
    )

    selected_methods = METHODS if args.method == "all" else (args.method,)
    outputs = []
    for method in selected_methods:
        labels = _run_method(raw, method, feature_config, args)
        output_path = output_dir / f"{method}_labels.csv"
        labels.to_csv(output_path, index=False)
        _write_metadata(output_dir / f"{method}_metadata.json", input_path, labels, method, args)
        outputs.append(labels)
        print(f"wrote {output_path} ({len(labels)} rows)")

    combined = pd.concat(outputs, ignore_index=True)
    if args.method == "all":
        all_path = output_dir / "all_regime_labels.csv"
        combined.to_csv(all_path, index=False)
        print(f"wrote {all_path} ({len(combined)} rows)")

    summary = summarize_labels(combined)
    summary_path = output_dir / "label_summary.csv"
    summary.to_csv(summary_path, index=False)
    switches = count_label_switches(combined)
    switches_path = output_dir / "label_switches.csv"
    switches.to_csv(switches_path, index=False)
    print(f"wrote {summary_path}")
    print(f"wrote {switches_path}")


def _run_method(
    raw: pd.DataFrame,
    method: str,
    feature_config: MarketFeatureConfig,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if method == "rule_based":
        return label_rule_based(
            raw,
            feature_config,
            RuleBasedConfig(
                lookback=args.rule_lookback,
                min_periods=args.rule_min_periods,
                vol_quantile=args.rule_vol_quantile,
                vix_quantile=args.rule_vix_quantile,
            ),
        )
    if method == "hmm":
        return label_hmm(
            raw,
            feature_config,
            HMMConfig(
                n_states=args.hmm_states,
                n_iter=args.hmm_iter,
                random_seed=args.random_seed,
            ),
        )
    if method == "recap_cusum":
        return label_recap_cusum(
            raw,
            feature_config,
            RecapCusumConfig(
                reference_window=args.cusum_reference_window,
                min_segment=args.cusum_min_segment,
                drift=args.cusum_drift,
                threshold=args.cusum_threshold,
            ),
        )
    raise ValueError(f"unknown method: {method}")


def _write_metadata(
    path: Path,
    input_path: Path,
    labels: pd.DataFrame,
    method: str,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "method": method,
        "input": str(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(labels)),
        "date_min": str(labels["date"].min()) if "date" in labels.columns else None,
        "date_max": str(labels["date"].max()) if "date" in labels.columns else None,
        "args": vars(args),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate market-regime labels.")
    parser.add_argument(
        "--input",
        default="data/market_indices_20080601_20260531/market_regime_features_wide.csv",
        help="CSV with market features.",
    )
    parser.add_argument("--output-dir", default="outputs/regime_labels")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--primary-symbol", default="SPY")
    parser.add_argument("--random-seed", type=int, default=7)

    parser.add_argument("--rule-lookback", type=int, default=252)
    parser.add_argument("--rule-min-periods", type=int, default=60)
    parser.add_argument("--rule-vol-quantile", type=float, default=0.75)
    parser.add_argument("--rule-vix-quantile", type=float, default=0.75)

    parser.add_argument("--hmm-states", type=int, default=3)
    parser.add_argument("--hmm-iter", type=int, default=80)

    parser.add_argument("--cusum-reference-window", type=int, default=60)
    parser.add_argument("--cusum-min-segment", type=int, default=20)
    parser.add_argument("--cusum-drift", type=float, default=0.25)
    parser.add_argument("--cusum-threshold", type=float, default=8.0)
    return parser.parse_args(argv)
