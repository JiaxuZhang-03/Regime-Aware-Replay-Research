from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_trading.dqn_replay import ExperimentConfig, run_many


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-data DQN baselines: online, uniform replay, PER, regime-aware replay."
    )

    parser.add_argument(
        "--market-csv",
        default="data/market_indices_20080601_20260531/market_regime_features_wide.csv",
    )
    parser.add_argument(
        "--labels-csv",
        default="outputs/regime_labels/all_regime_labels.csv",
    )
    parser.add_argument("--output-root", default="outputs/dqn_replay")

    parser.add_argument("--label-method", default="rule_based", choices=["rule_based", "hmm", "recap_cusum"])

    # Default now runs all four methods:
    # online  = DQN-only / no replay
    # uniform = DQN + Uniform Replay
    # per     = DQN + PER
    # regime  = DQN + Regime-aware Replay
    parser.add_argument("--replays", default="online,uniform,per,regime")

    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tradable-symbols", default="DIA,SPY,QQQ")
    parser.add_argument("--primary-symbol", default="SPY")

    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)

    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=256)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-update-freq", type=int, default=500)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=8000)

    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-eps", type=float, default=1e-6)

    parser.add_argument("--regime-same-ratio", type=float, default=0.50)
    parser.add_argument("--regime-high-td-ratio", type=float, default=0.25)
    parser.add_argument("--regime-recent-ratio", type=float, default=0.15)
    parser.add_argument("--regime-random-ratio", type=float, default=0.10)
    parser.add_argument("--regime-recent-window", type=int, default=252)

    args = parser.parse_args()

    replays = parse_csv_list(args.replays)
    valid_replays = {"online", "uniform", "per", "regime"}
    invalid = [r for r in replays if r not in valid_replays]
    if invalid:
        raise ValueError(f"Invalid replay methods: {invalid}. Valid methods: {sorted(valid_replays)}")

    cfg = ExperimentConfig(
        market_csv=args.market_csv,
        labels_csv=args.labels_csv,
        output_root=args.output_root,
        label_method=args.label_method,
        tradable_symbols=tuple(parse_csv_list(args.tradable_symbols)),
        primary_symbol=args.primary_symbol,
        transaction_cost_bps=args.transaction_cost_bps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eps=args.per_eps,
        regime_same_ratio=args.regime_same_ratio,
        regime_high_td_ratio=args.regime_high_td_ratio,
        regime_recent_ratio=args.regime_recent_ratio,
        regime_random_ratio=args.regime_random_ratio,
        regime_recent_window=args.regime_recent_window,
    )

    summary = run_many(
        base_cfg=cfg,
        replays=replays,
        seeds=parse_int_list(args.seeds),
    )

    print("\n=== DQN Replay Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()