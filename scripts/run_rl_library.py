from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_trading.baseline_policies import BaselineConfig
from src.rl_trading.baseline_policies import run_many as run_baseline_many
from src.rl_trading.dqn_replay import ExperimentConfig as DQNConfig
from src.rl_trading.dqn_replay import run_many as run_dqn_many
from src.rl_trading.model_registry import MODEL_REGISTRY, available_models, compatible_replays, validate_models
from src.rl_trading.performance_gate import PerformanceGateConfig, apply_performance_gate, select_best_policies
from src.rl_trading.sac_replay import SACExperimentConfig
from src.rl_trading.sac_replay import run_many as run_sac_many


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local RL model library for regime-aware replay generalization experiments."
    )
    parser.add_argument(
        "--models",
        default="dqn,sac,regime_anchor,vol_target,cash,equal_weight",
        help=f"Comma-separated models: {','.join(available_models())}",
    )
    parser.add_argument("--market-csv", default="data/market_indices_20080601_20260531/market_regime_features_wide.csv")
    parser.add_argument("--labels-csv", default="outputs/regime_labels/all_regime_labels.csv")
    parser.add_argument("--output-root", default="outputs/rl_library")
    parser.add_argument("--label-method", default="rule_based")
    parser.add_argument("--replays", default="uniform,per,regime,deer")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--tradable-symbols", default="DIA,SPY,QQQ")
    parser.add_argument("--primary-symbol", default="SPY")

    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--disable-policy-safety", action="store_true")
    parser.add_argument("--safety-min-cash-weight", type=float, default=0.03)
    parser.add_argument("--safety-max-asset-weight", type=float, default=0.85)
    parser.add_argument("--safety-max-turnover", type=float, default=0.75)
    parser.add_argument("--safety-regime-blend", type=float, default=0.20)
    parser.add_argument("--safety-risk-on-cash", type=float, default=0.05)
    parser.add_argument("--safety-sideways-cash", type=float, default=0.25)
    parser.add_argument("--safety-high-vol-cash", type=float, default=0.55)
    parser.add_argument("--safety-risk-off-cash", type=float, default=0.70)

    parser.add_argument("--dqn-lr", type=float, default=1e-3)
    parser.add_argument("--dqn-target-update-freq", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=8000)

    parser.add_argument("--sac-actor-lr", type=float, default=3e-4)
    parser.add_argument("--sac-critic-lr", type=float, default=3e-4)
    parser.add_argument("--sac-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--sac-start-steps", type=int, default=128)
    parser.add_argument("--sac-updates-per-step", type=int, default=1)
    parser.add_argument("--sac-action-temperature", type=float, default=1.0)
    parser.add_argument("--vol-target-ann-vol", type=float, default=0.12)
    parser.add_argument("--vol-target-window", type=int, default=20)

    parser.add_argument("--gate-min-final-value", type=float, default=0.90)
    parser.add_argument("--gate-max-drawdown", type=float, default=0.35)
    parser.add_argument("--gate-max-turnover", type=float, default=1.25)

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

    args = parser.parse_args()

    models = validate_models(parse_csv_list(args.models))
    replays = parse_csv_list(args.replays)
    seeds = parse_int_list(args.seeds)
    label_methods = parse_csv_list(args.label_method)
    symbols = tuple(parse_csv_list(args.tradable_symbols))

    summaries = []
    for model in models:
        spec = MODEL_REGISTRY[model]
        model_replays = compatible_replays(model, replays)
        if not model_replays:
            print(f"[skip] {model}: no compatible replay method in {replays}")
            continue
        for label_method in label_methods:
            if spec.family == "baseline":
                cfg = BaselineConfig(
                    market_csv=args.market_csv,
                    labels_csv=args.labels_csv,
                    output_root=str(Path(args.output_root) / "baselines"),
                    label_method=label_method,
                    policy=model,
                    tradable_symbols=symbols,
                    primary_symbol=args.primary_symbol,
                    transaction_cost_bps=args.transaction_cost_bps,
                    max_steps=args.max_steps,
                    safety_enabled=not args.disable_policy_safety,
                    safety_min_cash_weight=args.safety_min_cash_weight,
                    safety_max_asset_weight=args.safety_max_asset_weight,
                    safety_max_turnover=args.safety_max_turnover,
                    safety_regime_blend=0.0,
                    safety_risk_on_cash=args.safety_risk_on_cash,
                    safety_sideways_cash=args.safety_sideways_cash,
                    safety_high_vol_cash=args.safety_high_vol_cash,
                    safety_risk_off_cash=args.safety_risk_off_cash,
                    vol_target_ann_vol=args.vol_target_ann_vol,
                    vol_target_window=args.vol_target_window,
                )
                summary = run_baseline_many(cfg, [model], seeds)
            elif model == "dqn":
                cfg = DQNConfig(
                    market_csv=args.market_csv,
                    labels_csv=args.labels_csv,
                    output_root=str(Path(args.output_root) / "dqn"),
                    label_method=label_method,
                    tradable_symbols=symbols,
                    primary_symbol=args.primary_symbol,
                    transaction_cost_bps=args.transaction_cost_bps,
                    safety_enabled=not args.disable_policy_safety,
                    safety_min_cash_weight=args.safety_min_cash_weight,
                    safety_max_asset_weight=args.safety_max_asset_weight,
                    safety_max_turnover=args.safety_max_turnover,
                    safety_regime_blend=args.safety_regime_blend,
                    safety_risk_on_cash=args.safety_risk_on_cash,
                    safety_sideways_cash=args.safety_sideways_cash,
                    safety_high_vol_cash=args.safety_high_vol_cash,
                    safety_risk_off_cash=args.safety_risk_off_cash,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    max_steps=args.max_steps,
                    gamma=args.gamma,
                    lr=args.dqn_lr,
                    hidden_dim=args.hidden_dim,
                    target_update_freq=args.dqn_target_update_freq,
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
                summary = run_dqn_many(cfg, model_replays, seeds)
            elif model == "sac":
                cfg = SACExperimentConfig(
                    market_csv=args.market_csv,
                    labels_csv=args.labels_csv,
                    output_root=str(Path(args.output_root) / "sac"),
                    run_name="library",
                    label_method=label_method,
                    tradable_symbols=symbols,
                    primary_symbol=args.primary_symbol,
                    transaction_cost_bps=args.transaction_cost_bps,
                    action_temperature=args.sac_action_temperature,
                    safety_enabled=not args.disable_policy_safety,
                    safety_min_cash_weight=args.safety_min_cash_weight,
                    safety_max_asset_weight=args.safety_max_asset_weight,
                    safety_max_turnover=args.safety_max_turnover,
                    safety_regime_blend=args.safety_regime_blend,
                    safety_risk_on_cash=args.safety_risk_on_cash,
                    safety_sideways_cash=args.safety_sideways_cash,
                    safety_high_vol_cash=args.safety_high_vol_cash,
                    safety_risk_off_cash=args.safety_risk_off_cash,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    start_steps=args.sac_start_steps,
                    max_steps=args.max_steps,
                    updates_per_step=args.sac_updates_per_step,
                    gamma=args.gamma,
                    actor_lr=args.sac_actor_lr,
                    critic_lr=args.sac_critic_lr,
                    alpha_lr=args.sac_alpha_lr,
                    hidden_dim=args.hidden_dim,
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
                summary = run_sac_many(cfg, model_replays, seeds)
            else:
                raise AssertionError(f"unhandled model: {model}")

            summary = summary.copy()
            summary.insert(0, "model", model)
            summary.insert(1, "model_family", spec.family)
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No model runs were executed.")

    combined = pd.concat(summaries, ignore_index=True)
    combined = apply_performance_gate(
        combined,
        PerformanceGateConfig(
            min_final_value=args.gate_min_final_value,
            max_drawdown=args.gate_max_drawdown,
            max_turnover=args.gate_max_turnover,
        ),
    )
    selected = select_best_policies(combined)
    analysis_dir = Path(args.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "rl_library_summary.csv"
    selected_path = analysis_dir / "selected_policies.csv"
    combined.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    print(f"\n[rl-library] wrote {summary_path}")
    print(f"[rl-library] wrote {selected_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
