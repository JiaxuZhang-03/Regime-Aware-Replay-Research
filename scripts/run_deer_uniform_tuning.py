from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sac_oos import (
    _mean,
    _summary_metrics,
    buy_hold_metrics,
    clone_cfg,
    parse_csv_list,
    parse_int_list,
    run_test_phase,
    run_train_phase,
)
from src.rl_trading.sac_replay import MarketSACEnv, SACAgent, SACExperimentConfig, make_buffer, set_seed


def parse_float_grid(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def parse_int_grid(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def fmt_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def config_id_from_params(params: dict[str, Any]) -> str:
    return (
        f"s0{fmt_float(params['deer_s0'])}"
        f"_hl{int(params['deer_half_life'])}"
        f"_lam{fmt_float(params['deer_lambda'])}"
        f"_z{fmt_float(params['deer_zmax'])}"
        f"_post{int(params['deer_min_post_samples'])}"
    )


def grid_or_single_float(text: str, default: float) -> list[float]:
    return parse_float_grid(text) if text.strip() else [float(default)]


def grid_or_single_int(text: str, default: int) -> list[int]:
    return parse_int_grid(text) if text.strip() else [int(default)]


def model_config_id_from_params(params: dict[str, Any]) -> str:
    return (
        f"sac_lr{fmt_float(params['actor_lr'])}"
        f"_ent{fmt_float(params['target_entropy_scale'])}"
        f"_a{fmt_float(params['init_alpha'])}"
        f"_temp{fmt_float(params['action_temperature'])}"
        f"_h{int(params['hidden_dim'])}"
        f"_upd{int(params['updates_per_step'])}"
    )


def robust_score(final_value: float, drawdown: float, turnover: float) -> float:
    if pd.isna(final_value) or pd.isna(drawdown) or pd.isna(turnover):
        return np.nan
    return float(final_value) - float(drawdown) - 0.05 * float(turnover)


def iter_deer_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    configs: list[tuple[str, dict[str, Any]]] = []
    for s0, half_life, lamb, zmax, min_post in itertools.product(
        parse_float_grid(args.deer_s0_grid),
        parse_int_grid(args.deer_half_life_grid),
        parse_float_grid(args.deer_lambda_grid),
        parse_float_grid(args.deer_zmax_grid),
        parse_int_grid(args.deer_min_post_samples_grid),
    ):
        params: dict[str, Any] = {
            "deer_s0": s0,
            "deer_half_life": half_life,
            "deer_s_floor": args.deer_s_floor,
            "deer_lambda": lamb,
            "deer_zmax": zmax,
            "deer_probe_tau": args.deer_probe_tau,
            "deer_scale_refresh_freq": args.deer_scale_refresh_freq,
            "deer_probe_size": args.deer_probe_size,
            "deer_scale_rho": args.deer_scale_rho,
            "deer_scale_floor": args.deer_scale_floor,
            "deer_min_post_samples": min_post,
        }
        configs.append((config_id_from_params(params), params))
    if args.max_configs > 0:
        configs = configs[: int(args.max_configs)]
    return configs


def iter_model_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    configs: list[tuple[str, dict[str, Any]]] = []
    for actor_lr, target_entropy, init_alpha, action_temp, hidden_dim, updates_per_step in itertools.product(
        grid_or_single_float(args.model_actor_lr_grid, args.actor_lr),
        grid_or_single_float(args.model_target_entropy_scale_grid, args.target_entropy_scale),
        grid_or_single_float(args.model_init_alpha_grid, args.init_alpha),
        grid_or_single_float(args.model_action_temperature_grid, args.action_temperature),
        grid_or_single_int(args.model_hidden_dim_grid, args.hidden_dim),
        grid_or_single_int(args.model_updates_per_step_grid, args.updates_per_step),
    ):
        params: dict[str, Any] = {
            "actor_lr": actor_lr,
            "critic_lr": actor_lr,
            "alpha_lr": actor_lr,
            "target_entropy_scale": target_entropy,
            "init_alpha": init_alpha,
            "action_temperature": action_temp,
            "hidden_dim": hidden_dim,
            "updates_per_step": updates_per_step,
        }
        configs.append((model_config_id_from_params(params), params))
    if args.max_model_configs > 0:
        configs = configs[: int(args.max_model_configs)]
    return configs


def build_cfg(
    args: argparse.Namespace,
    replay: str,
    seed: int,
    run_name: str,
    deer_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
) -> SACExperimentConfig:
    params: dict[str, Any] = {}
    if deer_params is not None:
        params.update(deer_params)
    model = model_params or {}
    return SACExperimentConfig(
        market_csv=args.market_csv,
        labels_csv=args.labels_csv,
        output_root=args.output_root,
        run_name=run_name,
        label_method=args.label_method,
        tradable_symbols=tuple(parse_csv_list(args.tradable_symbols)),
        primary_symbol=args.primary_symbol,
        transaction_cost_bps=args.transaction_cost_bps,
        action_temperature=float(model.get("action_temperature", args.action_temperature)),
        safety_enabled=not args.disable_policy_safety,
        safety_min_cash_weight=args.safety_min_cash_weight,
        safety_max_asset_weight=args.safety_max_asset_weight,
        safety_max_turnover=args.safety_max_turnover,
        safety_regime_blend=args.safety_regime_blend,
        safety_risk_on_cash=args.safety_risk_on_cash,
        safety_sideways_cash=args.safety_sideways_cash,
        safety_high_vol_cash=args.safety_high_vol_cash,
        safety_risk_off_cash=args.safety_risk_off_cash,
        replay=replay,
        seed=seed,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        start_steps=args.start_steps,
        updates_per_step=int(model.get("updates_per_step", args.updates_per_step)),
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=float(model.get("actor_lr", args.actor_lr)),
        critic_lr=float(model.get("critic_lr", args.critic_lr)),
        alpha_lr=float(model.get("alpha_lr", args.alpha_lr)),
        hidden_dim=int(model.get("hidden_dim", args.hidden_dim)),
        auto_entropy_tuning=not args.disable_auto_entropy_tuning,
        init_alpha=float(model.get("init_alpha", args.init_alpha)),
        target_entropy_scale=float(model.get("target_entropy_scale", args.target_entropy_scale)),
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_eps=args.per_eps,
        regime_same_ratio=args.regime_same_ratio,
        regime_high_td_ratio=args.regime_high_td_ratio,
        regime_recent_ratio=args.regime_recent_ratio,
        regime_random_ratio=args.regime_random_ratio,
        regime_recent_window=args.regime_recent_window,
        **params,
    )


def make_split_envs(
    cfg: SACExperimentConfig,
    args: argparse.Namespace,
    include_test: bool,
) -> tuple[MarketSACEnv, MarketSACEnv, MarketSACEnv | None]:
    train_env = MarketSACEnv(clone_cfg(cfg, start_date=args.train_start, end_date=args.train_end))
    val_env = MarketSACEnv(clone_cfg(cfg, start_date=args.val_start, end_date=args.val_end))
    val_env.feature_mean = train_env.feature_mean.copy()
    val_env.feature_std = train_env.feature_std.copy()

    test_env = None
    if include_test:
        test_env = MarketSACEnv(clone_cfg(cfg, start_date=args.test_start, end_date=args.test_end))
        test_env.feature_mean = train_env.feature_mean.copy()
        test_env.feature_std = train_env.feature_std.copy()
    return train_env, val_env, test_env


def run_candidate(
    args: argparse.Namespace,
    replay: str,
    seed: int,
    config_id: str,
    deer_params: dict[str, Any] | None,
    phase: str,
    include_test: bool,
    model_config_id: str = "fixed",
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_name = f"{phase}_{replay}_{config_id}_seed{seed}"
    cfg = build_cfg(
        args,
        replay=replay,
        seed=seed,
        run_name=run_name,
        deer_params=deer_params,
        model_params=model_params,
    )
    set_seed(seed)
    train_env, val_env, test_env = make_split_envs(cfg, args, include_test=include_test)

    agent = SACAgent(train_env.state_dim, train_env.action_dim, cfg)
    buffer = make_buffer(cfg)

    train_log, replay_log = run_train_phase(train_env, agent, buffer, cfg, args.train_max_steps)
    val_log = run_test_phase(val_env, agent, args.val_max_steps)
    test_log = pd.DataFrame()
    if include_test:
        assert test_env is not None
        test_log = run_test_phase(test_env, agent, args.test_max_steps)

    out_dir = Path(args.output_root) / phase / f"{replay}_{config_id}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log.to_csv(out_dir / "train_trading_log.csv", index=False)
    replay_log.to_csv(out_dir / "train_replay_diagnostics.csv", index=False)
    val_log.to_csv(out_dir / "val_trading_log.csv", index=False)
    if include_test:
        test_log.to_csv(out_dir / "test_trading_log.csv", index=False)

    metadata = {
        "target_model": "SAC",
        "comparison_axis": "tuned_deer_line_vs_fixed_uniform",
        "selection_uses_test": False,
        "phase": phase,
        "include_test": include_test,
        "config": {**cfg.__dict__, "tradable_symbols": list(cfg.tradable_symbols)},
        "deer_params": deer_params,
        "model_config_id": model_config_id,
        "model_params": model_params,
        "train_rows": int(len(train_env.df)),
        "val_rows": int(len(val_env.df)),
        "test_rows": 0 if test_env is None else int(len(test_env.df)),
        "symbols": train_env.symbols,
        "feature_cols": train_env.feature_cols,
        "val_test_use_train_feature_scaler": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = summarize_candidate(
        phase=phase,
        replay=replay,
        config_id=config_id,
        model_config_id=model_config_id,
        label_method=args.label_method,
        seed=seed,
        train_log=train_log,
        replay_log=replay_log,
        val_log=val_log,
        test_log=test_log,
        val_env=val_env,
        test_env=test_env,
        deer_params=deer_params,
        model_params=model_params,
    )
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(
        f"[{phase}] {replay} | cfg={config_id} | seed={seed} | "
        f"val={summary['val_final_portfolio_value']:.4f}"
        + (
            f" | test={summary['test_final_portfolio_value']:.4f}"
            if include_test and not pd.isna(summary.get("test_final_portfolio_value", np.nan))
            else ""
        )
    )
    return summary


def summarize_candidate(
    phase: str,
    replay: str,
    config_id: str,
    model_config_id: str,
    label_method: str,
    seed: int,
    train_log: pd.DataFrame,
    replay_log: pd.DataFrame,
    val_log: pd.DataFrame,
    test_log: pd.DataFrame,
    val_env: MarketSACEnv,
    test_env: MarketSACEnv | None,
    deer_params: dict[str, Any] | None,
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "phase": phase,
        "target_model": "sac",
        "label_method": label_method,
        "replay": replay,
        "config_id": config_id,
        "model_config_id": model_config_id,
        "seed": seed,
        "mean_train_mismatch": _mean(replay_log, "mismatch_rate"),
        "mean_train_post_boundary_sample_rate": _mean(replay_log, "post_boundary_sample_rate"),
        "mean_train_doe": _mean(replay_log, "mean_doe"),
        "mean_train_z_doe": _mean(replay_log, "mean_z_doe"),
        "mean_train_z_td": _mean(replay_log, "mean_z_td"),
        "mean_train_priority": _mean(replay_log, "mean_priority"),
        "mean_train_alpha": _mean(replay_log, "alpha"),
        "mean_train_entropy": _mean(replay_log, "entropy"),
        **_summary_metrics("train", train_log),
        **_summary_metrics("val", val_log),
        **_summary_metrics("test", test_log),
    }
    for key, value in buy_hold_metrics(val_env).items():
        summary[f"val_{key}"] = value
    if test_env is not None:
        for key, value in buy_hold_metrics(test_env).items():
            summary[f"test_{key}"] = value
    summary["val_robust_score"] = robust_score(
        summary["val_final_portfolio_value"],
        summary["val_max_drawdown"],
        summary["val_mean_turnover"],
    )
    summary["test_robust_score"] = robust_score(
        summary["test_final_portfolio_value"],
        summary["test_max_drawdown"],
        summary["test_mean_turnover"],
    )
    if deer_params:
        for key, value in deer_params.items():
            summary[key] = value
    if model_params:
        for key, value in model_params.items():
            summary[f"model_{key}"] = value
    return summary


def pair_with_uniform(deer_df: pd.DataFrame, uniform_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    uniform_cols = [
        "seed",
        f"{prefix}_final_portfolio_value",
        f"{prefix}_max_drawdown",
        f"{prefix}_mean_turnover",
        f"{prefix}_robust_score",
    ]
    paired = deer_df.merge(
        uniform_df[uniform_cols],
        on="seed",
        suffixes=("_deer", "_uniform"),
        how="left",
    )
    paired[f"{prefix}_delta_final"] = (
        paired[f"{prefix}_final_portfolio_value_deer"] - paired[f"{prefix}_final_portfolio_value_uniform"]
    )
    paired[f"{prefix}_delta_robust"] = paired[f"{prefix}_robust_score_deer"] - paired[f"{prefix}_robust_score_uniform"]
    paired[f"{prefix}_delta_drawdown"] = paired[f"{prefix}_max_drawdown_deer"] - paired[f"{prefix}_max_drawdown_uniform"]
    paired[f"{prefix}_delta_turnover"] = paired[f"{prefix}_mean_turnover_deer"] - paired[f"{prefix}_mean_turnover_uniform"]
    paired[f"{prefix}_wins_final"] = paired[f"{prefix}_delta_final"] > 0
    paired[f"{prefix}_wins_robust"] = paired[f"{prefix}_delta_robust"] > 0
    return paired


def win_rate(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def aggregate_val_configs(args: argparse.Namespace, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_spec: dict[str, Any] = {
        "n": ("seed", "size"),
        "model_config_id": ("model_config_id", "first"),
        "mean_val_delta_final": ("val_delta_final", "mean"),
        "min_val_delta_final": ("val_delta_final", "min"),
        "mean_val_delta_robust": ("val_delta_robust", "mean"),
        "min_val_delta_robust": ("val_delta_robust", "min"),
        "mean_val_delta_drawdown": ("val_delta_drawdown", "mean"),
        "mean_val_delta_turnover": ("val_delta_turnover", "mean"),
        "val_win_rate_final": ("val_wins_final", win_rate),
        "val_win_rate_robust": ("val_wins_robust", win_rate),
        "mean_deer_val_final": ("val_final_portfolio_value_deer", "mean"),
        "mean_uniform_val_final": ("val_final_portfolio_value_uniform", "mean"),
        "mean_deer_val_dd": ("val_max_drawdown_deer", "mean"),
        "mean_uniform_val_dd": ("val_max_drawdown_uniform", "mean"),
        "mean_deer_val_turnover": ("val_mean_turnover_deer", "mean"),
        "mean_uniform_val_turnover": ("val_mean_turnover_uniform", "mean"),
        "mean_train_mismatch": ("mean_train_mismatch", "mean"),
        "mean_train_post_boundary_sample_rate": ("mean_train_post_boundary_sample_rate", "mean"),
        "deer_s0": ("deer_s0", "first"),
        "deer_half_life": ("deer_half_life", "first"),
        "deer_lambda": ("deer_lambda", "first"),
        "deer_zmax": ("deer_zmax", "first"),
        "deer_min_post_samples": ("deer_min_post_samples", "first"),
    }
    for col in [
        "model_actor_lr",
        "model_critic_lr",
        "model_alpha_lr",
        "model_target_entropy_scale",
        "model_init_alpha",
        "model_action_temperature",
        "model_hidden_dim",
        "model_updates_per_step",
    ]:
        if col in paired.columns:
            agg_spec[col] = (col, "first")

    grouped = paired.groupby("config_id").agg(**agg_spec).reset_index()
    if args.selection_objective in {"final", "stable_final", "tail_final"}:
        grouped["passes_val_selection_gate"] = (
            (grouped["n"] >= len(parse_int_list(args.seeds)))
            & (grouped["val_win_rate_final"] >= args.min_val_win_rate)
            & (grouped["mean_val_delta_final"] >= args.min_val_delta_final)
            & (grouped["min_val_delta_final"] >= args.min_val_min_delta_final)
            & (grouped["mean_deer_val_dd"] <= grouped["mean_uniform_val_dd"] + args.max_val_dd_regret)
        )
        if args.selection_objective == "tail_final":
            sort_cols = ["val_win_rate_final", "min_val_delta_final", "mean_val_delta_final", "mean_val_delta_robust"]
        elif args.selection_objective == "stable_final":
            sort_cols = ["val_win_rate_final", "mean_val_delta_final", "mean_val_delta_robust"]
        else:
            sort_cols = ["mean_val_delta_final", "val_win_rate_final", "mean_val_delta_robust"]
    else:
        grouped["passes_val_selection_gate"] = (
            (grouped["n"] >= len(parse_int_list(args.seeds)))
            & (grouped["val_win_rate_robust"] >= args.min_val_win_rate)
            & (grouped["mean_val_delta_robust"] >= args.min_val_delta_robust)
            & (grouped["mean_val_delta_final"] >= args.min_val_delta_final)
            & (grouped["mean_deer_val_dd"] <= grouped["mean_uniform_val_dd"] + args.max_val_dd_regret)
        )
        sort_cols = ["mean_val_delta_robust", "val_win_rate_robust", "mean_val_delta_final"]
    passing = grouped[grouped["passes_val_selection_gate"]].copy()
    pool = passing if not passing.empty else grouped
    selected = pool.sort_values(
        sort_cols,
        ascending=False,
    ).head(1).copy()
    selected["selection_used_fallback"] = bool(passing.empty)
    selected["selection_objective"] = args.selection_objective
    return grouped, selected


def summarize_final_oos(paired: pd.DataFrame) -> pd.DataFrame:
    out = {
        "n": int(len(paired)),
        "mean_test_delta_final": float(paired["test_delta_final"].mean()),
        "min_test_delta_final": float(paired["test_delta_final"].min()),
        "test_win_rate_final": float(paired["test_wins_final"].mean()),
        "mean_test_delta_robust": float(paired["test_delta_robust"].mean()),
        "min_test_delta_robust": float(paired["test_delta_robust"].min()),
        "test_win_rate_robust": float(paired["test_wins_robust"].mean()),
        "mean_deer_test_final": float(paired["test_final_portfolio_value_deer"].mean()),
        "mean_uniform_test_final": float(paired["test_final_portfolio_value_uniform"].mean()),
        "mean_deer_test_dd": float(paired["test_max_drawdown_deer"].mean()),
        "mean_uniform_test_dd": float(paired["test_max_drawdown_uniform"].mean()),
        "mean_deer_test_turnover": float(paired["test_mean_turnover_deer"].mean()),
        "mean_uniform_test_turnover": float(paired["test_mean_turnover_uniform"].mean()),
    }
    return pd.DataFrame([out])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune DEER replay against uniform replay with a fixed SAC target model and strict train/val/test."
    )
    parser.add_argument("--market-csv", default="data/market_indices_20080601_20260531/market_regime_features_wide.csv")
    parser.add_argument("--labels-csv", default="outputs/regime_labels/all_regime_labels.csv")
    parser.add_argument("--output-root", default="outputs/deer_uniform_tuning")
    parser.add_argument("--label-method", default="recap_cusum")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-model-configs", type=int, default=0)
    parser.add_argument("--skip-final-test", action="store_true")

    parser.add_argument("--train-start", default="2008-06-02")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--val-start", default="2021-01-04")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-03")
    parser.add_argument("--test-end", default="2026-05-28")
    parser.add_argument("--train-max-steps", type=int, default=0)
    parser.add_argument("--val-max-steps", type=int, default=0)
    parser.add_argument("--test-max-steps", type=int, default=0)
    parser.add_argument("--tradable-symbols", default="DIA,SPY,QQQ")
    parser.add_argument("--primary-symbol", default="SPY")

    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-temperature", type=float, default=1.5)
    parser.add_argument("--disable-policy-safety", action="store_true")
    parser.add_argument("--safety-min-cash-weight", type=float, default=0.03)
    parser.add_argument("--safety-max-asset-weight", type=float, default=0.85)
    parser.add_argument("--safety-max-turnover", type=float, default=0.75)
    parser.add_argument("--safety-regime-blend", type=float, default=0.20)
    parser.add_argument("--safety-risk-on-cash", type=float, default=0.05)
    parser.add_argument("--safety-sideways-cash", type=float, default=0.25)
    parser.add_argument("--safety-high-vol-cash", type=float, default=0.55)
    parser.add_argument("--safety-risk-off-cash", type=float, default=0.70)

    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--init-alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--disable-auto-entropy-tuning", action="store_true")

    parser.add_argument("--model-actor-lr-grid", default="")
    parser.add_argument("--model-target-entropy-scale-grid", default="")
    parser.add_argument("--model-init-alpha-grid", default="")
    parser.add_argument("--model-action-temperature-grid", default="")
    parser.add_argument("--model-hidden-dim-grid", default="")
    parser.add_argument("--model-updates-per-step-grid", default="")

    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-eps", type=float, default=1e-6)
    parser.add_argument("--regime-same-ratio", type=float, default=0.75)
    parser.add_argument("--regime-high-td-ratio", type=float, default=0.10)
    parser.add_argument("--regime-recent-ratio", type=float, default=0.10)
    parser.add_argument("--regime-random-ratio", type=float, default=0.05)
    parser.add_argument("--regime-recent-window", type=int, default=252)

    parser.add_argument("--deer-s0-grid", default="0.8,1.2")
    parser.add_argument("--deer-half-life-grid", default="3,8")
    parser.add_argument("--deer-lambda-grid", default="1.0,2.0")
    parser.add_argument("--deer-zmax-grid", default="5.0")
    parser.add_argument("--deer-min-post-samples-grid", default="4,8")
    parser.add_argument("--deer-s-floor", type=float, default=0.05)
    parser.add_argument("--deer-probe-tau", type=float, default=0.01)
    parser.add_argument("--deer-scale-refresh-freq", type=int, default=5)
    parser.add_argument("--deer-probe-size", type=int, default=2048)
    parser.add_argument("--deer-scale-rho", type=float, default=0.9)
    parser.add_argument("--deer-scale-floor", type=float, default=1e-8)

    parser.add_argument("--min-val-win-rate", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-val-delta-robust", type=float, default=0.0)
    parser.add_argument("--min-val-delta-final", type=float, default=0.0)
    parser.add_argument("--min-val-min-delta-final", type=float, default=-1e9)
    parser.add_argument("--max-val-dd-regret", type=float, default=0.03)
    parser.add_argument(
        "--selection-objective",
        choices=["robust", "final", "stable_final", "tail_final"],
        default="robust",
    )

    args = parser.parse_args()
    seeds = parse_int_list(args.seeds)
    deer_replay_configs = iter_deer_configs(args)
    deer_model_configs = iter_model_configs(args)
    tuned_deer_configs: list[tuple[str, str, dict[str, Any], str, dict[str, Any]]] = []
    for deer_id, deer_params in deer_replay_configs:
        for model_id, model_params in deer_model_configs:
            tuned_deer_configs.append((f"{deer_id}__{model_id}", deer_id, deer_params, model_id, model_params))
    if not tuned_deer_configs:
        raise ValueError("No DEER configs generated.")

    analysis_dir = Path(args.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "protocol.json").write_text(
        json.dumps(
            {
                "target_model": "SAC",
                "comparison": "fixed uniform line vs tuned DEER line",
                "uniform_line": "fixed baseline args; no DEER/model grid tuning",
                "deer_line": "DEER replay params crossed with DEER-only model params",
                "selection_uses_test": False,
                "train_window": [args.train_start, args.train_end],
                "val_window": [args.val_start, args.val_end],
                "test_window": [args.test_start, args.test_end],
                "deer_replay_config_count": len(deer_replay_configs),
                "deer_model_config_count": len(deer_model_configs),
                "tuned_deer_config_count": len(tuned_deer_configs),
                "seeds": seeds,
                "selection_objective": args.selection_objective,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[protocol] fixed target=SAC | train={args.train_start}..{args.train_end} | "
        f"val={args.val_start}..{args.val_end} | test={args.test_start}..{args.test_end}"
    )
    print(
        f"[protocol] fixed uniform line; tuning {len(tuned_deer_configs)} DEER-line configs "
        f"against uniform over seeds={seeds}"
    )

    uniform_tune = [
        run_candidate(
            args,
            "uniform",
            seed,
            "uniform",
            None,
            phase="tune",
            include_test=False,
            model_config_id="fixed_uniform",
            model_params=None,
        )
        for seed in seeds
    ]
    deer_tune: list[dict[str, Any]] = []
    for config_id, _deer_id, deer_params, model_id, model_params in tuned_deer_configs:
        for seed in seeds:
            deer_tune.append(
                run_candidate(
                    args,
                    "deer",
                    seed,
                    config_id,
                    deer_params,
                    phase="tune",
                    include_test=False,
                    model_config_id=model_id,
                    model_params=model_params,
                )
            )

    uniform_df = pd.DataFrame(uniform_tune)
    deer_df = pd.DataFrame(deer_tune)
    uniform_df.to_csv(analysis_dir / "uniform_tune_summary.csv", index=False)
    deer_df.to_csv(analysis_dir / "deer_tune_summary.csv", index=False)

    paired_val = pair_with_uniform(deer_df, uniform_df, prefix="val")
    paired_val.to_csv(analysis_dir / "deer_vs_uniform_val_paired.csv", index=False)
    by_config, selected = aggregate_val_configs(args, paired_val)
    by_config.to_csv(analysis_dir / "deer_tune_by_config.csv", index=False)
    selected.to_csv(analysis_dir / "selected_deer_config.csv", index=False)

    print("[selection] selected DEER config by validation paired robust delta")
    print(selected.to_string(index=False))

    if args.skip_final_test:
        return

    selected_config_id = str(selected["config_id"].iloc[0])
    selected_tuple = next(item for item in tuned_deer_configs if item[0] == selected_config_id)
    _, _selected_deer_id, selected_deer_params, selected_model_id, selected_model_params = selected_tuple

    final_uniform = [
        run_candidate(
            args,
            "uniform",
            seed,
            "uniform",
            None,
            phase="final",
            include_test=True,
            model_config_id="fixed_uniform",
            model_params=None,
        )
        for seed in seeds
    ]
    final_deer = [
        run_candidate(
            args,
            "deer",
            seed,
            selected_config_id,
            selected_deer_params,
            phase="final",
            include_test=True,
            model_config_id=selected_model_id,
            model_params=selected_model_params,
        )
        for seed in seeds
    ]
    final_uniform_df = pd.DataFrame(final_uniform)
    final_deer_df = pd.DataFrame(final_deer)
    final_uniform_df.to_csv(analysis_dir / "uniform_final_oos_summary.csv", index=False)
    final_deer_df.to_csv(analysis_dir / "deer_final_oos_summary.csv", index=False)

    final_paired = pair_with_uniform(final_deer_df, final_uniform_df, prefix="test")
    final_paired.to_csv(analysis_dir / "deer_vs_uniform_final_oos_paired.csv", index=False)
    final_summary = summarize_final_oos(final_paired)
    final_summary.insert(0, "selected_config_id", selected_config_id)
    final_summary.insert(1, "selected_model_config_id", selected_model_id)
    final_summary.to_csv(analysis_dir / "final_oos_comparison_summary.csv", index=False)

    print("[final-oos] paired test comparison")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
