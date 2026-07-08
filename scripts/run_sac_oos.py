from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_trading.performance_gate import PerformanceGateConfig, apply_performance_gate
from src.rl_trading.sac_replay import (
    ContinuousDEERReplayBuffer,
    ContinuousPERBuffer,
    MarketSACEnv,
    SACAgent,
    SACExperimentConfig,
    beta_at,
    deer_score_from_new_count,
    make_buffer,
    set_seed,
    summarize_replay_batch,
)


def parse_csv_list(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def clone_cfg(cfg: SACExperimentConfig, **overrides: Any) -> SACExperimentConfig:
    data = cfg.__dict__.copy()
    data.update(overrides)
    return SACExperimentConfig(**data)


def run_train_phase(
    env: MarketSACEnv,
    agent: SACAgent,
    buffer: Any,
    cfg: SACExperimentConfig,
    max_steps_override: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = env.reset()
    max_steps = len(env.df) - 1
    if max_steps_override > 0:
        max_steps = min(max_steps, int(max_steps_override))

    trade_logs: list[dict[str, Any]] = []
    replay_logs: list[dict[str, Any]] = []

    current_regime_for_boundary = env.current_regime()
    current_boundary = 0
    deer_n_new = 0
    deer_s_score = 0.0

    for step in range(max_steps):
        step_regime = env.current_regime()
        boundary_changed = step > 0 and step_regime != current_regime_for_boundary
        if boundary_changed:
            current_boundary += 1
            current_regime_for_boundary = step_regime
            deer_n_new = 0
            deer_s_score = cfg.deer_s0
            if cfg.replay == "deer":
                agent.freeze_reference()

        if step < cfg.start_steps:
            action = agent.random_action()
            policy_mode = "random_start"
        else:
            action = agent.act(state, deterministic=False)
            policy_mode = "sac_train"

        next_state, reward, done, info = env.step(action)
        transition = {
            "state": state,
            "action": action.astype(np.float32),
            "action_weights": info["action_weights"].astype(np.float32),
            "proposed_action_weights": info["proposed_action_weights"].astype(np.float32),
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "date": info["date"],
            "regime_label": info["regime_label"],
            "regime_name": info["regime_name"],
            "time_index": info["time_index"],
            "boundary_id": current_boundary,
        }

        trade_logs.append(_trade_row(step, info, reward, current_boundary, boundary_changed, deer_s_score, policy_mode, env))
        buffer.add(transition)

        if cfg.replay == "deer" and current_boundary > 0:
            deer_n_new += 1
            deer_s_score = deer_score_from_new_count(deer_n_new, cfg)

        if len(buffer) >= max(cfg.warmup_steps, cfg.batch_size):
            for update_idx in range(max(1, int(cfg.updates_per_step))):
                beta = beta_at(step, max_steps, cfg) if cfg.replay in {"per", "regime", "deer"} else 0.0
                batch = buffer.sample(
                    cfg.batch_size,
                    beta=beta,
                    current_regime=env.current_regime(),
                    current_step=step,
                    current_boundary=current_boundary,
                )
                update = agent.update(batch)

                if cfg.replay == "deer":
                    assert isinstance(buffer, ContinuousDEERReplayBuffer)
                    if current_boundary > 0:
                        need_scale_refresh = (
                            buffer.scale_td is None
                            or buffer.scale_doe is None
                            or (agent.update_count % max(1, cfg.deer_scale_refresh_freq) == 0)
                        )
                        if need_scale_refresh and len(buffer) > 0:
                            probe = buffer.uniform_probe_batch(cfg.deer_probe_size)
                            buffer.refresh_scales(
                                td_errors=agent.compute_td_errors(probe),
                                doe_values=agent.compute_q_discrepancy(probe),
                                allow_doe=True,
                            )
                        doe_values = agent.compute_q_discrepancy(batch)
                    else:
                        doe_values = np.zeros_like(update["td_errors"], dtype=np.float32)
                    deer_info = buffer.update_deer_priorities(
                        indices=batch["indices"],
                        td_errors=update["td_errors"],
                        doe_values=doe_values,
                        current_boundary=current_boundary,
                        s_score=deer_s_score,
                    )
                    batch["doe_values"] = doe_values.astype(np.float32)
                    batch["z_td_values"] = np.asarray(deer_info["z_td"], dtype=np.float32)
                    batch["z_doe_values"] = np.asarray(deer_info["z_doe"], dtype=np.float32)
                    batch["deer_is_post_change"] = np.asarray(deer_info["is_post_change"], dtype=bool)
                    batch["deer_priority_mode"] = np.asarray(deer_info["source_mode"], dtype=object)
                    batch["priorities"] = np.asarray(deer_info["priority"], dtype=np.float32)
                elif cfg.replay in {"per", "regime"}:
                    assert isinstance(buffer, ContinuousPERBuffer)
                    buffer.update_priorities(batch["indices"], update["td_errors"])

                replay_diag = summarize_replay_batch(
                    batch=batch,
                    current_regime=env.current_regime(),
                    current_step=step,
                    td_errors=update["td_errors"],
                )
                replay_diag.update(
                    {
                        "step": step,
                        "date": env.current_date(),
                        "current_regime": env.current_regime(),
                        "current_boundary": current_boundary,
                        "update_idx": update_idx,
                        "critic_loss": update["critic_loss"],
                        "actor_loss": update["actor_loss"],
                        "alpha": update["alpha"],
                        "entropy": update["entropy"],
                        "mean_q": update["mean_q"],
                        "beta": beta,
                        "deer_s_score": deer_s_score if cfg.replay == "deer" else np.nan,
                        "deer_n_new": deer_n_new if cfg.replay == "deer" else np.nan,
                        "scale_td": getattr(buffer, "scale_td", np.nan),
                        "scale_doe": getattr(buffer, "scale_doe", np.nan),
                    }
                )
                replay_logs.append(replay_diag)

        state = next_state
        if done:
            break

    return pd.DataFrame(trade_logs), pd.DataFrame(replay_logs)


def run_test_phase(
    env: MarketSACEnv,
    agent: SACAgent,
    max_steps_override: int = 0,
) -> pd.DataFrame:
    state = env.reset()
    max_steps = len(env.df) - 1
    if max_steps_override > 0:
        max_steps = min(max_steps, int(max_steps_override))

    trade_logs: list[dict[str, Any]] = []
    current_regime_for_boundary = env.current_regime()
    current_boundary = 0

    for step in range(max_steps):
        step_regime = env.current_regime()
        boundary_changed = step > 0 and step_regime != current_regime_for_boundary
        if boundary_changed:
            current_boundary += 1
            current_regime_for_boundary = step_regime

        action = agent.act(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        trade_logs.append(
            _trade_row(
                step=step,
                info=info,
                reward=reward,
                boundary_id=current_boundary,
                boundary_changed=boundary_changed,
                deer_s_score=np.nan,
                policy_mode="sac_oos_deterministic",
                env=env,
            )
        )
        state = next_state
        if done:
            break

    return pd.DataFrame(trade_logs)


def _trade_row(
    step: int,
    info: dict[str, Any],
    reward: float,
    boundary_id: int,
    boundary_changed: bool,
    deer_s_score: float,
    policy_mode: str,
    env: MarketSACEnv,
) -> dict[str, Any]:
    row = {
        "step": step,
        "date": info["date"],
        "regime_label": info["regime_label"],
        "regime_name": info["regime_name"],
        "boundary_id": boundary_id,
        "boundary_changed": bool(boundary_changed),
        "deer_s_score": deer_s_score,
        "reward": reward,
        "portfolio_value": info["portfolio_value"],
        "drawdown": info["drawdown"],
        "turnover": info["turnover"],
        "gross_return": info["gross_return"],
        "cost": info["cost"],
        "policy_mode": policy_mode,
        "safety_blend": info["safety_blend"],
        "safety_turnover_before_cap": info["safety_turnover_before_cap"],
        "safety_turnover_after_cap": info["safety_turnover_after_cap"],
        "safety_anchor_cash": info["safety_anchor_cash"],
    }
    for name in env.weight_names:
        row[f"weight_{name}"] = info[f"weight_{name}"]
        row[f"proposed_weight_{name}"] = info[f"proposed_weight_{name}"]
        row[f"raw_action_{name}"] = info[f"raw_action_{name}"]
    return row


def _mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean(skipna=True))


def _summary_metrics(prefix: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        f"{prefix}_start_date": str(df["date"].iloc[0]) if not df.empty else "",
        f"{prefix}_end_date": str(df["date"].iloc[-1]) if not df.empty else "",
        f"{prefix}_steps": int(len(df)),
        f"{prefix}_final_portfolio_value": float(df["portfolio_value"].iloc[-1]) if not df.empty else np.nan,
        f"{prefix}_max_drawdown": float(df["drawdown"].max()) if not df.empty else np.nan,
        f"{prefix}_mean_turnover": _mean(df, "turnover"),
        f"{prefix}_mean_reward": _mean(df, "reward"),
        f"{prefix}_mean_cash_weight": _mean(df, "weight_cash"),
    }


def buy_hold_metrics(env: MarketSACEnv) -> dict[str, float]:
    out: dict[str, float] = {}
    df = env.df.copy()
    for sym in env.symbols:
        close_cols = [f"adjclose_{sym}", f"close_{sym}", sym]
        col = next((c for c in close_cols if c in df.columns), None)
        if col is None:
            ret_col = f"ret_{sym}"
            if ret_col not in df.columns:
                continue
            returns = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0).copy()
            if not returns.empty:
                returns.iloc[0] = 0.0
            ratio = float((1.0 + returns).cumprod().iloc[-1])
        else:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            ratio = float(vals.iloc[-1] / vals.iloc[0])
        out[f"buyhold_{sym}_final_value"] = ratio
        out[f"buyhold_{sym}_return_pct"] = 100.0 * (ratio - 1.0)

    ret_cols = [f"ret_{sym}" for sym in env.symbols if f"ret_{sym}" in df.columns]
    if ret_cols:
        returns = df[ret_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if not returns.empty:
            returns.iloc[0, :] = 0.0
        equal_weight = (1.0 + returns.mean(axis=1)).cumprod().iloc[-1]
        out["buyhold_equal_weight_final_value"] = float(equal_weight)
        out["buyhold_equal_weight_return_pct"] = 100.0 * (float(equal_weight) - 1.0)
    return out


def run_single_oos(
    base_cfg: SACExperimentConfig,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    train_max_steps: int,
    test_max_steps: int,
) -> Path:
    set_seed(base_cfg.seed)
    train_cfg = clone_cfg(base_cfg, start_date=train_start, end_date=train_end)
    test_cfg = clone_cfg(base_cfg, start_date=test_start, end_date=test_end)

    train_env = MarketSACEnv(train_cfg)
    test_env = MarketSACEnv(test_cfg)
    test_env.feature_mean = train_env.feature_mean.copy()
    test_env.feature_std = train_env.feature_std.copy()

    if train_env.state_dim != test_env.state_dim or train_env.action_dim != test_env.action_dim:
        raise ValueError("Train/test environment dimensions differ.")

    agent = SACAgent(train_env.state_dim, train_env.action_dim, train_cfg)
    buffer = make_buffer(train_cfg)

    out_dir = Path(base_cfg.output_root) / (
        f"{base_cfg.label_method}_{base_cfg.replay}_seed{base_cfg.seed}_{base_cfg.run_name or 'oos'}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log, replay_log = run_train_phase(train_env, agent, buffer, train_cfg, train_max_steps)
    test_log = run_test_phase(test_env, agent, test_max_steps)

    train_log.to_csv(out_dir / "train_trading_log.csv", index=False)
    replay_log.to_csv(out_dir / "train_replay_diagnostics.csv", index=False)
    test_log.to_csv(out_dir / "test_trading_log.csv", index=False)

    summary = {
        "label_method": base_cfg.label_method,
        "replay": base_cfg.replay,
        "seed": base_cfg.seed,
        "run_name": base_cfg.run_name or "oos",
        "train_window_start": train_start,
        "train_window_end": train_end,
        "test_window_start": test_start,
        "test_window_end": test_end,
        "final_portfolio_value": float(test_log["portfolio_value"].iloc[-1]),
        "max_drawdown": float(test_log["drawdown"].max()),
        "mean_turnover": _mean(test_log, "turnover"),
        "safety_enabled": bool(base_cfg.safety_enabled),
        "safety_regime_blend": base_cfg.safety_regime_blend,
        "mean_mismatch_rate": _mean(replay_log, "mismatch_rate"),
        "mean_td_error": _mean(replay_log, "mean_td_error"),
        **_summary_metrics("train", train_log),
        **_summary_metrics("test", test_log),
        **buy_hold_metrics(test_env),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    metadata = {
        "config": {**base_cfg.__dict__, "tradable_symbols": list(base_cfg.tradable_symbols)},
        "train_rows": int(len(train_env.df)),
        "test_rows": int(len(test_env.df)),
        "symbols": train_env.symbols,
        "weight_names": train_env.weight_names,
        "feature_cols": train_env.feature_cols,
        "test_uses_train_feature_scaler": True,
        "test_learning_frozen": True,
        "test_policy_mode": "deterministic_actor",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"[oos] {base_cfg.label_method} | SAC-{base_cfg.replay} | seed={base_cfg.seed} | "
        f"test_final={summary['test_final_portfolio_value']:.4f} | "
        f"test_dd={summary['test_max_drawdown']:.4f}"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on a past window and evaluate frozen policy OOS.")
    parser.add_argument("--market-csv", default="data/market_indices_20080601_20260531/market_regime_features_wide.csv")
    parser.add_argument("--labels-csv", default="outputs/regime_labels/all_regime_labels.csv")
    parser.add_argument("--output-root", default="outputs/sac_oos")
    parser.add_argument("--run-name", default="oos")
    parser.add_argument("--label-method", default="rule_based,hmm,recap_cusum")
    parser.add_argument("--replays", default="uniform,per,regime,deer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-start", default="2008-06-02")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--test-start", default="2022-01-03")
    parser.add_argument("--test-end", default="2026-05-28")
    parser.add_argument("--train-max-steps", type=int, default=0)
    parser.add_argument("--test-max-steps", type=int, default=0)
    parser.add_argument("--tradable-symbols", default="DIA,SPY,QQQ")
    parser.add_argument("--primary-symbol", default="SPY")

    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--action-temperature", type=float, default=1.5)
    parser.add_argument("--disable-policy-safety", action="store_true")
    parser.add_argument("--safety-min-cash-weight", type=float, default=0.25)
    parser.add_argument("--safety-max-asset-weight", type=float, default=0.45)
    parser.add_argument("--safety-max-turnover", type=float, default=0.35)
    parser.add_argument("--safety-regime-blend", type=float, default=0.60)
    parser.add_argument("--safety-risk-on-cash", type=float, default=0.30)
    parser.add_argument("--safety-sideways-cash", type=float, default=0.60)
    parser.add_argument("--safety-high-vol-cash", type=float, default=0.85)
    parser.add_argument("--safety-risk-off-cash", type=float, default=0.95)

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

    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-eps", type=float, default=1e-6)
    parser.add_argument("--regime-same-ratio", type=float, default=0.75)
    parser.add_argument("--regime-high-td-ratio", type=float, default=0.10)
    parser.add_argument("--regime-recent-ratio", type=float, default=0.10)
    parser.add_argument("--regime-random-ratio", type=float, default=0.05)
    parser.add_argument("--regime-recent-window", type=int, default=252)

    parser.add_argument("--gate-min-final-value", type=float, default=0.90)
    parser.add_argument("--gate-max-drawdown", type=float, default=0.35)
    parser.add_argument("--gate-max-turnover", type=float, default=1.25)

    args = parser.parse_args()

    summaries = []
    for label_method in parse_csv_list(args.label_method):
        for replay in parse_csv_list(args.replays):
            for seed in parse_int_list(args.seeds):
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
                    updates_per_step=args.updates_per_step,
                    gamma=args.gamma,
                    tau=args.tau,
                    actor_lr=args.actor_lr,
                    critic_lr=args.critic_lr,
                    alpha_lr=args.alpha_lr,
                    hidden_dim=args.hidden_dim,
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
                )
                out_dir = run_single_oos(
                    cfg,
                    train_start=args.train_start,
                    train_end=args.train_end,
                    test_start=args.test_start,
                    test_end=args.test_end,
                    train_max_steps=args.train_max_steps,
                    test_max_steps=args.test_max_steps,
                )
                summaries.append(pd.read_csv(out_dir / "summary.csv"))

    summary = pd.concat(summaries, ignore_index=True)
    summary = apply_performance_gate(
        summary,
        PerformanceGateConfig(
            min_final_value=args.gate_min_final_value,
            max_drawdown=args.gate_max_drawdown,
            max_turnover=args.gate_max_turnover,
        ),
    )
    analysis_dir = Path(args.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "sac_oos_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_replay = summary.groupby(["label_method", "replay"]).agg(
        n=("replay", "size"),
        pass_rate=("passes_performance_gate", "mean"),
        mean_test_final=("test_final_portfolio_value", "mean"),
        min_test_final=("test_final_portfolio_value", "min"),
        mean_test_dd=("test_max_drawdown", "mean"),
        max_test_dd=("test_max_drawdown", "max"),
        mean_turnover=("test_mean_turnover", "mean"),
        mean_train_mismatch=("mean_mismatch_rate", "mean"),
    ).reset_index()
    by_replay.to_csv(analysis_dir / "sac_oos_by_label_replay.csv", index=False)

    overall = summary.groupby("replay").agg(
        n=("replay", "size"),
        pass_rate=("passes_performance_gate", "mean"),
        mean_test_final=("test_final_portfolio_value", "mean"),
        min_test_final=("test_final_portfolio_value", "min"),
        mean_test_dd=("test_max_drawdown", "mean"),
        max_test_dd=("test_max_drawdown", "max"),
        mean_turnover=("test_mean_turnover", "mean"),
        mean_train_mismatch=("mean_mismatch_rate", "mean"),
    ).reset_index()
    overall.to_csv(analysis_dir / "sac_oos_by_replay.csv", index=False)

    print(f"[summary] wrote {summary_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
