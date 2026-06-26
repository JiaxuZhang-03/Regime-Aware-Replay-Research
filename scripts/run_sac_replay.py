from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_trading.sac_replay import SACExperimentConfig, expand_tuning_grid, run_many


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-data SAC baselines with continuous portfolio weights: "
            "uniform replay, PER, regime-aware replay, and DEER."
        )
    )

    parser.add_argument(
        "--market-csv",
        default="data/market_indices_20080601_20260531/market_regime_features_wide.csv",
    )
    parser.add_argument(
        "--labels-csv",
        default="outputs/regime_labels/all_regime_labels.csv",
    )
    parser.add_argument("--output-root", default="outputs/sac_replay")
    parser.add_argument("--run-name", default="")

    parser.add_argument(
        "--label-method",
        default="rule_based,hmm,recap_cusum",
        help="Comma-separated label methods to run: rule_based,hmm,recap_cusum.",
    )
    parser.add_argument("--replays", default="uniform,per,regime,deer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tradable-symbols", default="DIA,SPY,QQQ")
    parser.add_argument("--primary-symbol", default="SPY")

    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-temperature", type=float, default=1.0)

    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=512)
    parser.add_argument("--start-steps", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--updates-per-step", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--target-update-interval", type=int, default=1)

    parser.add_argument("--init-alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument(
        "--disable-auto-entropy-tuning",
        action="store_true",
        help="Use fixed init_alpha instead of learned temperature.",
    )

    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-eps", type=float, default=1e-6)

    parser.add_argument("--regime-same-ratio", type=float, default=0.50)
    parser.add_argument("--regime-high-td-ratio", type=float, default=0.25)
    parser.add_argument("--regime-recent-ratio", type=float, default=0.15)
    parser.add_argument("--regime-random-ratio", type=float, default=0.10)
    parser.add_argument("--regime-recent-window", type=int, default=252)

    parser.add_argument("--deer-s0", type=float, default=0.8)
    parser.add_argument("--deer-half-life", type=int, default=5)
    parser.add_argument("--deer-s-floor", type=float, default=0.05)
    parser.add_argument("--deer-lambda", type=float, default=1.0)
    parser.add_argument("--deer-zmax", type=float, default=5.0)
    parser.add_argument("--deer-probe-tau", type=float, default=0.01)
    parser.add_argument("--deer-scale-refresh-freq", type=int, default=5)
    parser.add_argument("--deer-probe-size", type=int, default=2048)
    parser.add_argument("--deer-scale-rho", type=float, default=0.9)
    parser.add_argument("--deer-scale-floor", type=float, default=1e-8)
    parser.add_argument("--deer-min-post-samples", type=int, default=4)

    parser.add_argument(
        "--tuning-grid",
        default="",
        help=(
            "Optional JSON grid. Use configs/sac_tuning_grid.json as a template. "
            "Values override SACExperimentConfig fields."
        ),
    )

    args = parser.parse_args()

    replays = parse_csv_list(args.replays)
    valid_replays = {"uniform", "per", "regime", "deer"}
    invalid = [r for r in replays if r not in valid_replays]
    if invalid:
        raise ValueError(f"Invalid replay methods: {invalid}. Valid methods: {sorted(valid_replays)}")

    label_methods = parse_csv_list(args.label_method)
    valid_label_methods = {"rule_based", "hmm", "recap_cusum"}
    invalid_label_methods = [m for m in label_methods if m not in valid_label_methods]
    if invalid_label_methods:
        raise ValueError(
            f"Invalid label methods: {invalid_label_methods}. "
            f"Valid methods: {sorted(valid_label_methods)}"
        )

    valid_cfg_fields = {f.name for f in fields(SACExperimentConfig)}
    summaries = []
    grid = expand_tuning_grid(args.tuning_grid or None)

    for grid_idx, overrides in enumerate(grid):
        unknown = sorted(set(overrides).difference(valid_cfg_fields))
        if unknown:
            raise ValueError(f"Unknown tuning-grid fields: {unknown}")

        for label_method in label_methods:
            cfg = SACExperimentConfig(
                market_csv=args.market_csv,
                labels_csv=args.labels_csv,
                output_root=args.output_root,
                run_name=args.run_name,
                label_method=label_method,
                tradable_symbols=tuple(parse_csv_list(args.tradable_symbols)),
                primary_symbol=args.primary_symbol,
                transaction_cost_bps=args.transaction_cost_bps,
                action_temperature=args.action_temperature,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                start_steps=args.start_steps,
                max_steps=args.max_steps,
                updates_per_step=args.updates_per_step,
                gamma=args.gamma,
                tau=args.tau,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                alpha_lr=args.alpha_lr,
                hidden_dim=args.hidden_dim,
                grad_clip_norm=args.grad_clip_norm,
                target_update_interval=args.target_update_interval,
                auto_entropy_tuning=not args.disable_auto_entropy_tuning,
                init_alpha=args.init_alpha,
                target_entropy_scale=args.target_entropy_scale,
                per_alpha=args.per_alpha,
                per_beta_start=args.per_beta_start,
                per_beta_end=args.per_beta_end,
                per_eps=args.per_eps,
                regime_same_ratio=args.regime_same_ratio,
                regime_high_td_ratio=args.regime_high_td_ratio,
                regime_recent_ratio=args.regime_recent_ratio,
                regime_random_ratio=args.regime_random_ratio,
                regime_recent_window=args.regime_recent_window,
                deer_s0=args.deer_s0,
                deer_half_life=args.deer_half_life,
                deer_s_floor=args.deer_s_floor,
                deer_lambda=args.deer_lambda,
                deer_zmax=args.deer_zmax,
                deer_probe_tau=args.deer_probe_tau,
                deer_scale_refresh_freq=args.deer_scale_refresh_freq,
                deer_probe_size=args.deer_probe_size,
                deer_scale_rho=args.deer_scale_rho,
                deer_scale_floor=args.deer_scale_floor,
                deer_min_post_samples=args.deer_min_post_samples,
            )

            for key, value in overrides.items():
                setattr(cfg, key, value)

            if overrides and not cfg.run_name:
                cfg.run_name = f"grid{grid_idx:03d}"

            summaries.append(
                run_many(
                    base_cfg=cfg,
                    replays=replays,
                    seeds=parse_int_list(args.seeds),
                )
            )

    summary = summaries[0] if len(summaries) == 1 else pd.concat(summaries, ignore_index=True)
    print("\n=== SAC Replay Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
