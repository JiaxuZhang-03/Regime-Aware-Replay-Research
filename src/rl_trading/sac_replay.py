from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import itertools
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.regime_labeling.features import MarketFeatureConfig, build_market_features
from src.rl_trading.dqn_replay import (
    build_asset_returns,
    find_col,
    robust_positive_scale,
    sigmoid_np,
)
from src.rl_trading.policy_safety import apply_policy_safety, safety_config_from_object


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class SACExperimentConfig:
    market_csv: str = "data/market_indices_20080601_20260531/market_regime_features_wide.csv"
    labels_csv: str = "outputs/regime_labels/all_regime_labels.csv"
    output_root: str = "outputs/sac_replay"
    run_name: str = ""

    label_method: str = "rule_based"
    tradable_symbols: tuple[str, ...] = ("DIA", "SPY", "QQQ")
    primary_symbol: str = "SPY"

    transaction_cost_bps: float = 10.0
    action_temperature: float = 1.0
    safety_enabled: bool = True
    safety_min_cash_weight: float = 0.03
    safety_max_asset_weight: float = 0.85
    safety_max_turnover: float = 0.75
    safety_regime_blend: float = 0.20
    safety_risk_on_cash: float = 0.05
    safety_sideways_cash: float = 0.25
    safety_high_vol_cash: float = 0.55
    safety_risk_off_cash: float = 0.70

    # Methods:
    # uniform = SAC + Uniform Replay
    # per     = SAC + Prioritized Experience Replay
    # regime  = SAC + Regime-aware Replay mixture
    # deer    = SAC + DEER-style TD-error / Q-discrepancy priority
    replay: str = "uniform"
    seed: int = 0

    buffer_size: int = 50000
    batch_size: int = 128
    warmup_steps: int = 512
    start_steps: int = 512
    max_steps: int = 0
    updates_per_step: int = 1

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    grad_clip_norm: float = 10.0
    target_update_interval: int = 1

    auto_entropy_tuning: bool = True
    init_alpha: float = 0.2
    target_entropy_scale: float = 1.0

    # PER parameters
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6

    # Regime-aware replay mixture
    regime_same_ratio: float = 0.50
    regime_high_td_ratio: float = 0.25
    regime_recent_ratio: float = 0.15
    regime_random_ratio: float = 0.10
    regime_recent_window: int = 252

    # DEER-style replay parameters. External regime labels define change points.
    deer_s0: float = 0.8
    deer_half_life: int = 5
    deer_s_floor: float = 0.05
    deer_lambda: float = 1.0
    deer_zmax: float = 5.0
    deer_probe_tau: float = 0.01
    deer_scale_refresh_freq: int = 5
    deer_probe_size: int = 2048
    deer_scale_rho: float = 0.9
    deer_scale_floor: float = 1e-8
    deer_min_post_samples: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def deer_score_from_new_count(n_new: int, cfg: SACExperimentConfig) -> float:
    if n_new <= 0:
        return 0.0
    age_new = max(int(n_new) - 1, 0)
    score = float(cfg.deer_s0) * (2.0 ** (-age_new / max(1, int(cfg.deer_half_life))))
    if score < float(cfg.deer_s_floor):
        return 0.0
    return score


class MarketSACEnv:
    """Continuous-action long-only portfolio environment.

    The SAC actor outputs bounded continuous logits. The environment converts
    those logits to portfolio weights over cash plus tradable assets with a
    softmax transform. Cash has zero return; asset weights earn next-day returns.
    """

    def __init__(self, cfg: SACExperimentConfig):
        self.cfg = cfg

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

        self.symbols = []
        for s in cfg.tradable_symbols:
            su = s.upper()
            col = f"ret_{su}"
            if col in df.columns:
                self.symbols.append(su)

        if not self.symbols:
            raise ValueError(
                "No tradable returns found. Need columns like ret_SPY / SPY_ret / close_SPY / SPY_close."
            )

        self.df = df
        self.feature_cols = [
            c for c in ["ret_short", "ret_long", "vol", "trend", "vix", "turbulence"] if c in df.columns
        ]
        if not self.feature_cols:
            raise ValueError("No market feature columns found after build_market_features().")

        feature_df = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        self.feature_mean = feature_df.mean()
        self.feature_std = feature_df.std().replace(0.0, 1.0).fillna(1.0)

        self.weight_names = ["cash", *self.symbols]
        self.action_dim = len(self.weight_names)
        self.state_dim = len(self.feature_cols) + self.action_dim
        self.cost_rate = cfg.transaction_cost_bps / 10000.0
        self.action_temperature = max(float(cfg.action_temperature), 1e-6)
        self.safety_config = safety_config_from_object(cfg)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.prev_weights = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_weights[0] = 1.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._state()

    def _state(self) -> np.ndarray:
        row = self.df.iloc[self.t]
        raw_feats = pd.to_numeric(row[self.feature_cols], errors="coerce")
        feats = ((raw_feats - self.feature_mean) / self.feature_std)
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        return np.concatenate([feats, self.prev_weights]).astype(np.float32)

    def action_to_weights(self, action: np.ndarray) -> np.ndarray:
        logits = np.asarray(action, dtype=np.float32).reshape(-1)
        if logits.shape[0] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got shape {logits.shape}.")
        logits = np.clip(logits / self.action_temperature, -20.0, 20.0)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / max(float(exp_logits.sum()), 1e-12)
        return weights.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        current = self.df.iloc[self.t]
        next_t = self.t + 1
        done = next_t >= len(self.df) - 1

        next_row = self.df.iloc[next_t]
        proposed_weights = self.action_to_weights(raw_action)
        new_weights, safety_info = apply_policy_safety(
            proposed_weights=proposed_weights,
            previous_weights=self.prev_weights,
            regime_label=int(current["regime_label"]),
            regime_name=str(current["regime_name"]),
            cfg=self.safety_config,
        )
        asset_weights = new_weights[1:]
        turnover = float(np.abs(new_weights - self.prev_weights).sum())

        asset_returns = np.array([float(next_row[f"ret_{s}"]) for s in self.symbols], dtype=np.float32)
        gross_return = float(np.dot(asset_weights, asset_returns))
        cost = self.cost_rate * turnover
        reward = math.log(max(1e-12, 1.0 + gross_return)) - cost

        self.portfolio_value *= math.exp(reward)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 1.0 - self.portfolio_value / max(self.peak_value, 1e-12)

        info: dict[str, Any] = {
            "date": str(current["date"].date()),
            "time_index": int(self.t),
            "regime_label": int(current["regime_label"]),
            "regime_name": str(current["regime_name"]),
            "portfolio_value": float(self.portfolio_value),
            "drawdown": float(drawdown),
            "turnover": float(turnover),
            "gross_return": float(gross_return),
            "cost": float(cost),
            "action": raw_action.copy(),
            "proposed_action_weights": proposed_weights.copy(),
            "action_weights": new_weights.copy(),
            **safety_info,
        }
        for name, weight in zip(self.weight_names, new_weights):
            info[f"weight_{name}"] = float(weight)
        for name, weight in zip(self.weight_names, proposed_weights):
            info[f"proposed_weight_{name}"] = float(weight)
        for name, value in zip(self.weight_names, raw_action):
            info[f"raw_action_{name}"] = float(value)

        self.t = next_t
        self.prev_weights = new_weights.copy()
        return self._state(), float(reward), done, info

    def current_regime(self) -> int:
        return int(self.df.iloc[self.t]["regime_label"])

    def current_date(self) -> str:
        return str(self.df.iloc[self.t]["date"].date())


class ContinuousReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.storage: list[dict[str, Any]] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, item: dict[str, Any]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.0,
        current_regime: int | None = None,
        current_step: int | None = None,
        current_boundary: int | None = None,
    ) -> dict[str, Any]:
        n = len(self.storage)
        idx = self.rng.choice(n, size=batch_size, replace=n < batch_size)
        weights = np.ones(len(idx), dtype=np.float32)
        batch = self._pack(idx, weights)
        batch["sample_sources"] = np.array(["uniform"] * len(idx), dtype=object)
        return batch

    def _pack(self, idx: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
        batch = [self.storage[int(i)] for i in idx]
        return {
            "states": np.stack([b["state"] for b in batch]).astype(np.float32),
            "actions": np.stack([b["action"] for b in batch]).astype(np.float32),
            "action_weights": np.stack([b["action_weights"] for b in batch]).astype(np.float32),
            "rewards": np.array([b["reward"] for b in batch], dtype=np.float32),
            "next_states": np.stack([b["next_state"] for b in batch]).astype(np.float32),
            "dones": np.array([b["done"] for b in batch], dtype=np.float32),
            "regime_labels": np.array([b["regime_label"] for b in batch], dtype=np.int64),
            "boundary_ids": np.array([int(b.get("boundary_id", 0)) for b in batch], dtype=np.int64),
            "time_indices": np.array([b["time_index"] for b in batch], dtype=np.int64),
            "dates": [b["date"] for b in batch],
            "indices": np.array(idx, dtype=np.int64),
            "weights": weights.astype(np.float32),
            "stored_td_error": np.array([float(b.get("td_error", np.nan)) for b in batch], dtype=np.float32),
            "stored_doe_raw": np.array([float(b.get("doe_raw", np.nan)) for b in batch], dtype=np.float32),
            "stored_doe_normalized": np.array([float(b.get("doe_normalized", np.nan)) for b in batch], dtype=np.float32),
        }


class ContinuousPERBuffer(ContinuousReplayBuffer):
    def __init__(self, capacity: int, seed: int, alpha: float = 0.6, eps: float = 1e-6):
        super().__init__(capacity, seed)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, item: dict[str, Any]) -> None:
        insert_pos = self.pos
        super().add(item)
        self.priorities[insert_pos] = self.max_priority

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        current_regime: int | None = None,
        current_step: int | None = None,
        current_boundary: int | None = None,
    ) -> dict[str, Any]:
        n = len(self.storage)
        p = np.maximum(self.priorities[:n], self.eps)
        probs = p ** self.alpha
        probs = probs / probs.sum()

        idx = self.rng.choice(n, size=batch_size, replace=n < batch_size, p=probs)
        weights = (n * probs[idx]) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        batch = self._pack(idx, weights.astype(np.float32))
        batch["priorities"] = p[idx].astype(np.float32)
        batch["sample_probs"] = probs[idx].astype(np.float32)
        batch["sample_sources"] = np.array(["per"] * len(idx), dtype=object)
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        new_p = np.abs(td_errors).astype(np.float32) + self.eps
        self.priorities[indices] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))


class ContinuousRegimeAwareReplayBuffer(ContinuousPERBuffer):
    def __init__(
        self,
        capacity: int,
        seed: int,
        alpha: float,
        eps: float,
        same_ratio: float,
        high_td_ratio: float,
        recent_ratio: float,
        random_ratio: float,
        recent_window: int,
    ):
        super().__init__(capacity=capacity, seed=seed, alpha=alpha, eps=eps)
        total = same_ratio + high_td_ratio + recent_ratio + random_ratio
        if total <= 0:
            raise ValueError("Regime-aware ratios must sum to a positive value.")
        self.same_ratio = same_ratio / total
        self.high_td_ratio = high_td_ratio / total
        self.recent_ratio = recent_ratio / total
        self.random_ratio = random_ratio / total
        self.recent_window = int(recent_window)

    def _choice(
        self,
        candidates: np.ndarray,
        k: int,
        probs: np.ndarray | None = None,
    ) -> np.ndarray:
        if k <= 0:
            return np.array([], dtype=np.int64)

        if len(candidates) == 0:
            candidates = np.arange(len(self.storage), dtype=np.int64)
            probs = None

        replace = len(candidates) < k
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float64)
            probs = probs / probs.sum()
        return self.rng.choice(candidates, size=k, replace=replace, p=probs).astype(np.int64)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        current_regime: int | None = None,
        current_step: int | None = None,
        current_boundary: int | None = None,
    ) -> dict[str, Any]:
        n = len(self.storage)
        all_idx = np.arange(n, dtype=np.int64)

        if current_regime is None:
            current_regime = int(self.storage[-1]["regime_label"])
        if current_step is None:
            current_step = int(self.storage[-1]["time_index"])

        regime_labels = np.array([b["regime_label"] for b in self.storage], dtype=np.int64)
        time_indices = np.array([b["time_index"] for b in self.storage], dtype=np.int64)

        same_candidates = all_idx[regime_labels == current_regime]
        recent_candidates = all_idx[time_indices >= current_step - self.recent_window]

        priorities = np.maximum(self.priorities[:n], self.eps)
        priority_probs = priorities ** self.alpha
        priority_probs = priority_probs / priority_probs.sum()

        n_same = int(round(batch_size * self.same_ratio))
        n_high = int(round(batch_size * self.high_td_ratio))
        n_recent = int(round(batch_size * self.recent_ratio))
        n_random = batch_size - n_same - n_high - n_recent

        idx_same = self._choice(same_candidates, n_same)
        idx_high = self._choice(all_idx, n_high, probs=priority_probs)
        idx_recent = self._choice(recent_candidates, n_recent)
        idx_random = self._choice(all_idx, n_random)

        idx = np.concatenate([idx_same, idx_high, idx_recent, idx_random]).astype(np.int64)
        sources = np.array(
            ["same_regime"] * len(idx_same)
            + ["high_td"] * len(idx_high)
            + ["recent"] * len(idx_recent)
            + ["random"] * len(idx_random),
            dtype=object,
        )

        if len(idx) < batch_size:
            extra = self._choice(all_idx, batch_size - len(idx))
            idx = np.concatenate([idx, extra])
            sources = np.concatenate([sources, np.array(["fill_random"] * len(extra), dtype=object)])

        if len(idx) > batch_size:
            idx = idx[:batch_size]
            sources = sources[:batch_size]

        weights = np.ones(len(idx), dtype=np.float32)
        batch = self._pack(idx, weights)
        batch["priorities"] = priorities[idx].astype(np.float32)
        batch["sample_probs"] = priority_probs[idx].astype(np.float32)
        batch["sample_sources"] = sources
        return batch


class ContinuousDEERReplayBuffer(ContinuousPERBuffer):
    def __init__(self, capacity: int, seed: int, cfg: SACExperimentConfig):
        super().__init__(capacity=capacity, seed=seed, alpha=cfg.per_alpha, eps=cfg.per_eps)
        self.cfg = cfg
        self.scale_td: float | None = None
        self.scale_doe: float | None = None

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        current_regime: int | None = None,
        current_step: int | None = None,
        current_boundary: int | None = None,
    ) -> dict[str, Any]:
        n = len(self.storage)
        min_post = max(0, int(self.cfg.deer_min_post_samples))

        if current_boundary is None or int(current_boundary) <= 0 or min_post <= 0:
            batch = super().sample(
                batch_size=batch_size,
                beta=beta,
                current_regime=current_regime,
                current_step=current_step,
                current_boundary=current_boundary,
            )
            batch["sample_sources"] = np.array(["deer_per"] * len(batch["indices"]), dtype=object)
            return batch

        all_idx = np.arange(n, dtype=np.int64)
        priorities = np.maximum(self.priorities[:n], self.eps)
        global_probs = priorities ** self.alpha
        global_probs = global_probs / global_probs.sum()

        boundary_ids = np.array([int(b.get("boundary_id", 0)) for b in self.storage], dtype=np.int64)
        post_candidates = all_idx[boundary_ids == int(current_boundary)]
        n_post = min(min_post, len(post_candidates), int(batch_size))

        idx_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []

        if n_post > 0:
            post_probs = global_probs[post_candidates].astype(np.float64)
            post_probs = post_probs / post_probs.sum()
            idx_post = self.rng.choice(
                post_candidates,
                size=n_post,
                replace=len(post_candidates) < n_post,
                p=post_probs,
            ).astype(np.int64)
            idx_parts.append(idx_post)
            source_parts.append(np.array(["deer_forced_post"] * len(idx_post), dtype=object))
        else:
            idx_post = np.array([], dtype=np.int64)

        n_rest = int(batch_size) - int(sum(len(x) for x in idx_parts))
        if n_rest > 0:
            remaining = np.setdiff1d(all_idx, idx_post, assume_unique=False)
            if len(remaining) == 0:
                remaining = all_idx
            rest_probs = global_probs[remaining].astype(np.float64)
            rest_probs = rest_probs / rest_probs.sum()
            idx_rest = self.rng.choice(
                remaining,
                size=n_rest,
                replace=len(remaining) < n_rest,
                p=rest_probs,
            ).astype(np.int64)
            idx_parts.append(idx_rest)
            source_parts.append(np.array(["deer_per_remainder"] * len(idx_rest), dtype=object))

        idx = np.concatenate(idx_parts).astype(np.int64)
        sources = np.concatenate(source_parts).astype(object)

        weights = (n * global_probs[idx]) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        batch = self._pack(idx, weights.astype(np.float32))
        batch["priorities"] = priorities[idx].astype(np.float32)
        batch["sample_probs"] = global_probs[idx].astype(np.float32)
        batch["sample_sources"] = sources
        return batch

    def uniform_probe_batch(self, probe_size: int) -> dict[str, Any]:
        n = len(self.storage)
        k = min(int(probe_size), n)
        if k <= 0:
            raise ValueError("Cannot create a probe batch from an empty replay buffer.")
        idx = self.rng.choice(n, size=k, replace=False).astype(np.int64)
        return self._pack(idx, np.ones(k, dtype=np.float32))

    def refresh_scales(self, td_errors: np.ndarray, doe_values: np.ndarray, allow_doe: bool) -> None:
        cfg = self.cfg
        self.scale_td = robust_positive_scale(
            np.abs(td_errors),
            self.scale_td,
            rho=cfg.deer_scale_rho,
            eps=cfg.per_eps,
            floor=cfg.deer_scale_floor,
        )
        if allow_doe:
            self.scale_doe = robust_positive_scale(
                np.abs(doe_values),
                self.scale_doe,
                rho=cfg.deer_scale_rho,
                eps=cfg.per_eps,
                floor=cfg.deer_scale_floor,
            )

    def update_deer_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        doe_values: np.ndarray,
        current_boundary: int,
        s_score: float,
    ) -> dict[str, np.ndarray | float]:
        cfg = self.cfg
        idx = np.asarray(indices, dtype=np.int64)
        td_abs = np.abs(np.asarray(td_errors, dtype=np.float32))
        doe_abs = np.abs(np.asarray(doe_values, dtype=np.float32))

        if self.scale_td is None or self.scale_td <= cfg.deer_scale_floor:
            z_td = td_abs.copy()
        else:
            z_td = np.clip(td_abs / max(self.scale_td, cfg.deer_scale_floor), 0.0, cfg.deer_zmax)

        if self.scale_doe is None or self.scale_doe <= cfg.deer_scale_floor:
            z_doe = np.zeros_like(doe_abs, dtype=np.float32)
        else:
            z_doe = np.clip(doe_abs / max(self.scale_doe, cfg.deer_scale_floor), 0.0, cfg.deer_zmax)

        post = np.array(
            [int(self.storage[int(i)].get("boundary_id", 0)) == int(current_boundary) for i in idx],
            dtype=bool,
        )

        if int(current_boundary) <= 0 or self.scale_doe is None:
            new_p = td_abs + cfg.per_eps
            source_mode = np.array(["deer_per_fallback"] * len(idx), dtype=object)
        else:
            s = float(np.clip(s_score, 0.0, 1.0))
            p_old = 2.0 * sigmoid_np(-float(cfg.deer_lambda) * z_doe) + cfg.per_eps
            p_new = (
                (1.0 - s) * (2.0 * sigmoid_np(z_td) - 1.0)
                + s * (2.0 * sigmoid_np(float(cfg.deer_lambda) * z_doe) - 1.0)
                + cfg.per_eps
            )
            new_p = np.where(post, p_new, p_old).astype(np.float32)
            source_mode = np.where(post, "deer_post_change", "deer_pre_change").astype(object)

        new_p = np.maximum(new_p.astype(np.float32), cfg.per_eps)
        self.priorities[idx] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))

        for j, i in enumerate(idx):
            rec = self.storage[int(i)]
            rec["td_error"] = float(td_abs[j])
            rec["doe_raw"] = float(doe_abs[j])
            rec["td_normalized"] = float(z_td[j])
            rec["doe_normalized"] = float(z_doe[j])
            rec["priority"] = float(new_p[j])
            rec["deer_is_post_change"] = bool(post[j])
            rec["deer_s_score"] = float(s_score)

        return {
            "priority": new_p,
            "z_td": z_td.astype(np.float32),
            "z_doe": z_doe.astype(np.float32),
            "is_post_change": post,
            "source_mode": source_mode,
            "scale_td": np.array([np.nan if self.scale_td is None else self.scale_td], dtype=np.float32),
            "scale_doe": np.array([np.nan if self.scale_doe is None else self.scale_doe], dtype=np.float32),
        }


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: SACExperimentConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)

        self.actor = ActorNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1 = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2 = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_q1 = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_q2 = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        hard_update(self.target_q1, self.q1)
        hard_update(self.target_q2, self.q2)

        self.q1_probe = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2_probe = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        hard_update(self.q1_probe, self.q1)
        hard_update(self.q2_probe, self.q2)
        self.q1_reference = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2_reference = CriticNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        hard_update(self.q1_reference, self.q1_probe)
        hard_update(self.q2_reference, self.q2_probe)
        for module in (self.q1_probe, self.q2_probe, self.q1_reference, self.q2_reference):
            for p in module.parameters():
                p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.target_entropy = -float(action_dim) * float(cfg.target_entropy_scale)
        init_alpha = max(float(cfg.init_alpha), 1e-8)
        self.log_alpha = torch.tensor(math.log(init_alpha), dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad_(cfg.auto_entropy_tuning)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def freeze_reference(self) -> None:
        hard_update(self.q1_reference, self.q1_probe)
        hard_update(self.q2_reference, self.q2_probe)

    def _ema_update_probe(self) -> None:
        tau = float(self.cfg.deer_probe_tau)
        soft_update(self.q1_probe, self.q1, tau)
        soft_update(self.q2_probe, self.q2, tau)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                action = self.actor.deterministic(x)
            else:
                action, _ = self.actor.sample(x)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def random_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def compute_td_errors(self, batch: dict[str, Any]) -> np.ndarray:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q
            td1 = (target - self.q1(states, actions)).abs()
            td2 = (target - self.q2(states, actions)).abs()
            td = 0.5 * (td1 + td2)
        return td.squeeze(1).cpu().numpy()

    def compute_q_discrepancy(self, batch: dict[str, Any]) -> np.ndarray:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            old_q = torch.min(
                self.q1_reference(states, actions),
                self.q2_reference(states, actions),
            )
            new_q = torch.min(
                self.q1_probe(states, actions),
                self.q2_probe(states, actions),
            )
            doe = (new_q - old_q).abs()
        return doe.squeeze(1).cpu().numpy()

    def update(self, batch: dict[str, Any]) -> dict[str, Any]:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = (weights * F.mse_loss(current_q1, target, reduction="none")).mean()
        q2_loss = (weights * F.mse_loss(current_q2, target, reduction="none")).mean()

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=self.cfg.grad_clip_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=self.cfg.grad_clip_norm)
        self.q2_opt.step()

        new_actions, log_prob = self.actor.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.cfg.grad_clip_norm)
        self.actor_opt.step()

        if self.cfg.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        self.update_count += 1
        if self.update_count % max(1, int(self.cfg.target_update_interval)) == 0:
            soft_update(self.target_q1, self.q1, self.cfg.tau)
            soft_update(self.target_q2, self.q2, self.cfg.tau)
        self._ema_update_probe()

        td_errors = 0.5 * ((target - current_q1).abs() + (target - current_q2).abs())

        return {
            "critic_loss": float((q1_loss + q2_loss).detach().cpu().item()),
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "entropy": float((-log_prob).detach().mean().cpu().item()),
            "td_errors": td_errors.detach().squeeze(1).cpu().numpy(),
            "mean_q": float(torch.min(current_q1, current_q2).detach().mean().cpu().item()),
        }


def beta_at(step: int, max_steps: int, cfg: SACExperimentConfig) -> float:
    frac = min(1.0, step / max(1, max_steps))
    return cfg.per_beta_start + frac * (cfg.per_beta_end - cfg.per_beta_start)


def summarize_replay_batch(
    batch: dict[str, Any],
    current_regime: int,
    current_step: int,
    td_errors: np.ndarray,
) -> dict[str, Any]:
    sampled_regime = batch["regime_labels"]
    sample_age = current_step - batch["time_indices"]

    out: dict[str, Any] = {
        "mismatch_rate": float(np.mean(sampled_regime != current_regime)),
        "mean_sample_age": float(np.mean(sample_age)),
        "median_sample_age": float(np.median(sample_age)),
        "mean_td_error": float(np.mean(td_errors)),
        "mean_cash_weight": float(np.mean(batch["action_weights"][:, 0])),
        "mean_max_asset_weight": float(np.mean(np.max(batch["action_weights"][:, 1:], axis=1))),
    }

    for r in sorted(np.unique(sampled_regime)):
        mask = sampled_regime == r
        out[f"sampled_regime_{int(r)}_count"] = int(mask.sum())
        out[f"td_error_regime_{int(r)}"] = float(np.mean(td_errors[mask]))

        if "priorities" in batch:
            out[f"priority_regime_{int(r)}"] = float(np.mean(batch["priorities"][mask]))

    if "priorities" in batch:
        out["mean_priority"] = float(np.mean(batch["priorities"]))
        out["mean_sample_prob"] = float(np.mean(batch["sample_probs"]))

    if "doe_values" in batch:
        out["mean_doe"] = float(np.mean(batch["doe_values"]))
    if "z_doe_values" in batch:
        out["mean_z_doe"] = float(np.mean(batch["z_doe_values"]))
    if "z_td_values" in batch:
        out["mean_z_td"] = float(np.mean(batch["z_td_values"]))
    if "deer_is_post_change" in batch:
        out["post_boundary_sample_rate"] = float(np.mean(batch["deer_is_post_change"]))
    if "deer_priority_mode" in batch:
        modes = np.asarray(batch["deer_priority_mode"], dtype=object)
        for mode in sorted(set(modes.tolist())):
            out[f"mode_{mode}_count"] = int(np.sum(modes == mode))
    if "sample_sources" in batch:
        sources = np.asarray(batch["sample_sources"], dtype=object)
        for src in sorted(set(sources.tolist())):
            out[f"source_{src}_count"] = int(np.sum(sources == src))
    return out


def make_buffer(cfg: SACExperimentConfig) -> ContinuousReplayBuffer:
    if cfg.replay == "uniform":
        return ContinuousReplayBuffer(cfg.buffer_size, cfg.seed)
    if cfg.replay == "per":
        return ContinuousPERBuffer(
            capacity=cfg.buffer_size,
            seed=cfg.seed,
            alpha=cfg.per_alpha,
            eps=cfg.per_eps,
        )
    if cfg.replay == "regime":
        return ContinuousRegimeAwareReplayBuffer(
            capacity=cfg.buffer_size,
            seed=cfg.seed,
            alpha=cfg.per_alpha,
            eps=cfg.per_eps,
            same_ratio=cfg.regime_same_ratio,
            high_td_ratio=cfg.regime_high_td_ratio,
            recent_ratio=cfg.regime_recent_ratio,
            random_ratio=cfg.regime_random_ratio,
            recent_window=cfg.regime_recent_window,
        )
    if cfg.replay == "deer":
        return ContinuousDEERReplayBuffer(
            capacity=cfg.buffer_size,
            seed=cfg.seed,
            cfg=cfg,
        )
    raise ValueError("replay must be one of: uniform, per, regime, deer")


def _output_dir_for(cfg: SACExperimentConfig) -> Path:
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    return Path(cfg.output_root) / f"{cfg.label_method}_{cfg.replay}_seed{cfg.seed}{suffix}"


def run_single_experiment(cfg: SACExperimentConfig) -> Path:
    set_seed(cfg.seed)

    env = MarketSACEnv(cfg)
    buffer = make_buffer(cfg)
    agent = SACAgent(env.state_dim, env.action_dim, cfg)

    output_dir = _output_dir_for(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = cfg.__dict__.copy()
    config_dict["tradable_symbols"] = list(cfg.tradable_symbols)

    metadata = {
        "config": config_dict,
        "symbols": env.symbols,
        "weight_names": env.weight_names,
        "feature_cols": env.feature_cols,
        "n_rows": int(len(env.df)),
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "action_transform": "softmax(raw_action / action_temperature)",
        "policy_safety": {
            "enabled": bool(cfg.safety_enabled),
            "min_cash_weight": float(cfg.safety_min_cash_weight),
            "max_asset_weight": float(cfg.safety_max_asset_weight),
            "max_turnover": float(cfg.safety_max_turnover),
            "regime_blend": float(cfg.safety_regime_blend),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    state = env.reset()
    max_steps = len(env.df) - 2
    if cfg.max_steps > 0:
        max_steps = min(max_steps, int(cfg.max_steps))

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
            policy_mode = "sac_policy"

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

        trade_row = {
            "step": step,
            "date": info["date"],
            "regime_label": info["regime_label"],
            "regime_name": info["regime_name"],
            "boundary_id": current_boundary,
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
            trade_row[f"weight_{name}"] = info[f"weight_{name}"]
            trade_row[f"proposed_weight_{name}"] = info[f"proposed_weight_{name}"]
            trade_row[f"raw_action_{name}"] = info[f"raw_action_{name}"]
        trade_logs.append(trade_row)

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
                    need_scale_refresh = (
                        buffer.scale_td is None
                        or (current_boundary > 0 and buffer.scale_doe is None)
                        or (agent.update_count % max(1, cfg.deer_scale_refresh_freq) == 0)
                    )
                    if need_scale_refresh and len(buffer) > 0:
                        probe = buffer.uniform_probe_batch(cfg.deer_probe_size)
                        td_probe = agent.compute_td_errors(probe)
                        doe_probe = agent.compute_q_discrepancy(probe)
                        buffer.refresh_scales(
                            td_errors=td_probe,
                            doe_values=doe_probe,
                            allow_doe=current_boundary > 0,
                        )

                    doe_values = agent.compute_q_discrepancy(batch)
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
                        "q1_loss": update["q1_loss"],
                        "q2_loss": update["q2_loss"],
                        "actor_loss": update["actor_loss"],
                        "alpha_loss": update["alpha_loss"],
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

    trade_df = pd.DataFrame(trade_logs)
    replay_df = pd.DataFrame(replay_logs)
    trade_df.to_csv(output_dir / "trading_log.csv", index=False)
    replay_df.to_csv(output_dir / "replay_diagnostics.csv", index=False)

    def replay_mean(column: str) -> float:
        if replay_df.empty or column not in replay_df.columns:
            return np.nan
        return float(replay_df[column].mean(skipna=True))

    summary = {
        "label_method": cfg.label_method,
        "replay": cfg.replay,
        "seed": cfg.seed,
        "run_name": cfg.run_name,
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
        "mean_mismatch_rate": replay_mean("mismatch_rate"),
        "mean_td_error": replay_mean("mean_td_error"),
        "mean_priority": replay_mean("mean_priority"),
        "mean_doe": replay_mean("mean_doe"),
        "mean_z_doe": replay_mean("mean_z_doe"),
        "mean_post_boundary_sample_rate": replay_mean("post_boundary_sample_rate"),
        "mean_alpha": replay_mean("alpha"),
        "mean_entropy": replay_mean("entropy"),
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "alpha_lr": cfg.alpha_lr,
        "init_alpha": cfg.init_alpha,
        "hidden_dim": cfg.hidden_dim,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "tau": cfg.tau,
        "action_temperature": cfg.action_temperature,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    mismatch = summary["mean_mismatch_rate"]
    mismatch_display = "NA" if np.isnan(mismatch) else f"{mismatch:.4f}"
    print(
        f"[done] {cfg.label_method} | SAC-{cfg.replay} | seed={cfg.seed} | "
        f"run={cfg.run_name or 'base'} | final_value={summary['final_portfolio_value']:.4f} | "
        f"max_dd={summary['max_drawdown']:.4f} | mismatch={mismatch_display}"
    )
    return output_dir


def run_many(
    base_cfg: SACExperimentConfig,
    replays: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    all_summaries = []
    for replay in replays:
        for seed in seeds:
            cfg = SACExperimentConfig(**base_cfg.__dict__)
            cfg.replay = replay
            cfg.seed = seed
            out_dir = run_single_experiment(cfg)
            all_summaries.append(pd.read_csv(out_dir / "summary.csv"))

    summary = pd.concat(all_summaries, ignore_index=True)
    analysis_dir = Path(base_cfg.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{base_cfg.run_name}" if base_cfg.run_name else ""
    summary_path = analysis_dir / f"{base_cfg.label_method}_sac_replay_summary{suffix}.csv"
    summary.to_csv(summary_path, index=False)
    make_plots(base_cfg.output_root, base_cfg.label_method, replays, seeds, base_cfg.run_name)
    print(f"[summary] wrote {summary_path}")
    return summary


def expand_tuning_grid(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return [{}]

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]

    if not isinstance(data, dict):
        raise ValueError("tuning grid must be a dict of parameter lists or a list of override dicts.")

    parameters = data.get("parameters", data)
    if not isinstance(parameters, dict):
        raise ValueError("tuning grid 'parameters' must be a dict.")

    keys = list(parameters.keys())
    values = []
    for key in keys:
        value = parameters[key]
        if isinstance(value, list):
            values.append(value)
        else:
            values.append([value])

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def make_plots(
    output_root: str,
    label_method: str,
    replays: list[str],
    seeds: list[int],
    run_name: str = "",
) -> None:
    analysis_dir = Path(output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = analysis_dir / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}")
        return

    suffix = f"_{run_name}" if run_name else ""

    def run_dir(replay: str, seed: int) -> Path:
        return Path(output_root) / f"{label_method}_{replay}_seed{seed}{suffix}"

    plt.figure(figsize=(12, 5))
    for replay in replays:
        for seed in seeds:
            path = run_dir(replay, seed) / "trading_log.csv"
            if path.exists():
                df = pd.read_csv(path)
                plt.plot(df["step"], df["portfolio_value"], label=f"{replay}-seed{seed}", alpha=0.85)
    plt.title("SAC Trading: Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_sac_portfolio_value{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for replay in replays:
        for seed in seeds:
            path = run_dir(replay, seed) / "replay_diagnostics.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "mismatch_rate" in df.columns and not df["mismatch_rate"].dropna().empty:
                    plt.plot(
                        df["step"],
                        df["mismatch_rate"].rolling(50, min_periods=1).mean(),
                        label=f"{replay}-seed{seed}",
                        alpha=0.85,
                    )
    plt.title("SAC Replay Diagnostics: Mismatch Rate")
    plt.xlabel("Step")
    plt.ylabel("Rolling mismatch rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_sac_mismatch_rate{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for replay in replays:
        for seed in seeds:
            path = run_dir(replay, seed) / "replay_diagnostics.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "mean_td_error" in df.columns:
                    plt.plot(
                        df["step"],
                        df["mean_td_error"].rolling(50, min_periods=1).mean(),
                        label=f"{replay}-seed{seed}",
                        alpha=0.85,
                    )
    plt.title("SAC Diagnostics: TD-error")
    plt.xlabel("Step")
    plt.ylabel("Rolling mean TD-error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_sac_td_error{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plotted = False
    for replay in ["per", "regime", "deer"]:
        if replay not in replays:
            continue
        for seed in seeds:
            path = run_dir(replay, seed) / "replay_diagnostics.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "mean_priority" in df.columns:
                    plt.plot(
                        df["step"],
                        df["mean_priority"].rolling(50, min_periods=1).mean(),
                        label=f"{replay}-seed{seed}",
                        alpha=0.85,
                    )
                    plotted = True
    if plotted:
        plt.title("SAC Priority Diagnostics: PER vs Regime-aware vs DEER")
        plt.xlabel("Step")
        plt.ylabel("Rolling mean priority")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / f"{label_method}_sac_priority{suffix}.png", dpi=160)
    plt.close()

    if "deer" in replays:
        plt.figure(figsize=(12, 5))
        plotted = False
        for seed in seeds:
            path = run_dir("deer", seed) / "replay_diagnostics.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "mean_doe" in df.columns:
                    plt.plot(
                        df["step"],
                        df["mean_doe"].rolling(50, min_periods=1).mean(),
                        label=f"deer-doe-seed{seed}",
                        alpha=0.85,
                    )
                    plotted = True
        if plotted:
            plt.title("SAC DEER Diagnostics: Empirical Q-discrepancy")
            plt.xlabel("Step")
            plt.ylabel("Rolling mean DoE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{label_method}_sac_deer_doe{suffix}.png", dpi=160)
        plt.close()

    if "regime" in replays:
        for seed in seeds:
            path = run_dir("regime", seed) / "replay_diagnostics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            source_cols = [c for c in df.columns if c.startswith("source_") and c.endswith("_count")]
            if not source_cols:
                continue
            plt.figure(figsize=(12, 5))
            for c in source_cols:
                plt.plot(
                    df["step"],
                    df[c].rolling(50, min_periods=1).mean(),
                    label=c.replace("source_", "").replace("_count", ""),
                    alpha=0.85,
                )
            plt.title(f"SAC Regime-aware Replay Source Composition, seed={seed}")
            plt.xlabel("Step")
            plt.ylabel("Rolling sample count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{label_method}_sac_regime_sources_seed{seed}{suffix}.png", dpi=160)
            plt.close()
