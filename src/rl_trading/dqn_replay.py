from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from src.regime_labeling.features import MarketFeatureConfig, build_market_features
from src.rl_trading.policy_safety import apply_policy_safety, normalize_long_only, safety_config_from_object


@dataclass
class ExperimentConfig:
    market_csv: str = "data/market_indices_20080601_20260531/market_regime_features_wide.csv"
    labels_csv: str = "outputs/regime_labels/all_regime_labels.csv"
    output_root: str = "outputs/dqn_replay"

    label_method: str = "rule_based"
    tradable_symbols: tuple[str, ...] = ("DIA", "SPY", "QQQ")
    primary_symbol: str = "SPY"

    transaction_cost_bps: float = 10.0
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
    # online  = DQN-only / no replay
    # uniform = DQN + Uniform Replay
    # per     = DQN + Prioritized Experience Replay
    # regime  = DQN + Regime-aware Replay mixture
    # deer    = DQN + DEER-style TD-error / Q-discrepancy priority
    replay: str = "uniform"
    seed: int = 0

    buffer_size: int = 50000
    batch_size: int = 64
    warmup_steps: int = 256
    max_steps: int = 0

    gamma: float = 0.99
    lr: float = 1e-3
    hidden_dim: int = 128
    target_update_freq: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000

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

    # DEER-style replay parameters. The MVP uses external regime labels as change points.
    # At a boundary, the agent freezes a Q snapshot. Priority then combines TD-error
    # and empirical Q-discrepancy: |Q_probe(s,a) - Q_reference(s,a)|.
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
    deer_initial_priority: str = "max"  # max, median, or doe

    # Mechanism experiment diagnostics.
    mechanism_probe_size: int = 128
    mechanism_event_horizon: int = 60
    mechanism_recovery_fraction: float = 1.25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def to_numeric_clean(s: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )


def build_asset_returns(raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])

    id_col = find_col(raw, ["symbol", "tic", "ticker"])

    # Long panel format: date, tic/symbol, close/adjclose
    if id_col is not None and raw["date"].duplicated().any():
        price_col = find_col(raw, ["adjclose", "adjcp", "close"])
        if price_col is None:
            raise ValueError("Long panel data requires one price column among adjclose, adjcp, close.")

        panel = raw[["date", id_col, price_col]].copy()
        panel[id_col] = panel[id_col].astype(str).str.upper()
        px = panel.pivot_table(index="date", columns=id_col, values=price_col, aggfunc="last").sort_index()
        rets = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out = pd.DataFrame({"date": rets.index})
        for s in symbols:
            su = s.upper()
            if su in rets.columns:
                out[f"ret_{su}"] = rets[su].to_numpy()
        return out.reset_index(drop=True)

    # Wide format
    out = pd.DataFrame({"date": raw["date"]})

    for s in symbols:
        su = s.upper()

        ret_col = find_col(
            raw,
            [
                f"ret_{su}",
                f"{su}_ret",
                f"ret_1d_{su}",
                f"{su}_ret_1d",
                f"return_1_{su}",
                f"{su}_return_1",
                f"return_1d_{su}",
                f"{su}_return_1d",
                f"close_return_{su}",
                f"adjcp_return_{su}",
                f"pct_change_{su}",
                f"{su}_pct_change",
            ],
        )

        if ret_col is not None:
            out[f"ret_{su}"] = to_numeric_clean(raw[ret_col]).to_numpy()
            continue

        price_col = find_col(
            raw,
            [
                f"adjclose_{su}",
                f"{su}_adjclose",
                f"adjcp_{su}",
                f"{su}_adjcp",
                f"close_{su}",
                f"{su}_close",
                f"price_{su}",
                f"{su}_price",
            ],
        )

        if price_col is not None:
            px = to_numeric_clean(raw[price_col])
            out[f"ret_{su}"] = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    return out.fillna(0.0)


class MarketDQNEnv:
    def __init__(self, cfg: ExperimentConfig):
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

        # Ensure regime_label is integer-coded
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

        self.actions, self.action_names = self._make_actions()
        self.n_actions = len(self.actions)
        self.state_dim = len(self.feature_cols) + self.n_actions

        self.cost_rate = cfg.transaction_cost_bps / 10000.0
        self.safety_config = safety_config_from_object(cfg)
        self.reset()

    def _make_actions(self) -> tuple[list[np.ndarray], list[str]]:
        n = len(self.symbols)
        actions: list[np.ndarray] = []
        names: list[str] = []

        actions.append(np.zeros(n, dtype=np.float32))
        names.append("cash")

        for i, s in enumerate(self.symbols):
            w = np.zeros(n, dtype=np.float32)
            w[i] = 1.0
            actions.append(w)
            names.append(s)

        if n >= 2:
            actions.append(np.ones(n, dtype=np.float32) / n)
            names.append("equal_weight")

        return actions, names

    def reset(self) -> np.ndarray:
        self.t = 0
        self.prev_action = 0
        self.prev_weights = self.actions[0].copy()
        self.prev_full_weights = self._full_weights_from_asset_weights(self.prev_weights)
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._state()

    def _full_weights_from_asset_weights(self, asset_weights: np.ndarray) -> np.ndarray:
        assets = np.asarray(asset_weights, dtype=np.float32).reshape(-1)
        cash = max(0.0, 1.0 - float(np.sum(np.maximum(assets, 0.0))))
        return normalize_long_only(np.concatenate([[cash], assets]).astype(np.float32))

    def _state(self) -> np.ndarray:
        row = self.df.iloc[self.t]
        raw_feats = pd.to_numeric(row[self.feature_cols], errors="coerce")
        feats = ((raw_feats - self.feature_mean) / self.feature_std)
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

        action_onehot = np.zeros(self.n_actions, dtype=np.float32)
        action_onehot[self.prev_action] = 1.0

        return np.concatenate([feats, action_onehot]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = int(action)
        current = self.df.iloc[self.t]
        next_t = self.t + 1
        done = next_t >= len(self.df) - 1

        next_row = self.df.iloc[next_t]
        proposed_full_weights = self._full_weights_from_asset_weights(self.actions[action])
        guarded_full_weights, safety_info = apply_policy_safety(
            proposed_weights=proposed_full_weights,
            previous_weights=self.prev_full_weights,
            regime_label=int(current["regime_label"]),
            regime_name=str(current["regime_name"]),
            cfg=self.safety_config,
        )
        new_weights = guarded_full_weights[1:]
        turnover = float(np.abs(guarded_full_weights - self.prev_full_weights).sum())

        asset_returns = np.array([float(next_row[f"ret_{s}"]) for s in self.symbols], dtype=np.float32)
        gross_return = float(np.dot(new_weights, asset_returns))
        cost = self.cost_rate * turnover
        reward = math.log(max(1e-12, 1.0 + gross_return)) - cost

        self.portfolio_value *= math.exp(reward)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 1.0 - self.portfolio_value / max(self.peak_value, 1e-12)

        info = {
            "date": str(current["date"].date()),
            "time_index": int(self.t),
            "regime_label": int(current["regime_label"]),
            "regime_name": str(current["regime_name"]),
            "action": action,
            "action_name": self.action_names[action],
            "portfolio_value": float(self.portfolio_value),
            "drawdown": float(drawdown),
            "turnover": float(turnover),
            "gross_return": float(gross_return),
            "cost": float(cost),
            "action_weights": guarded_full_weights.copy(),
            "proposed_action_weights": proposed_full_weights.copy(),
            **safety_info,
        }
        for name, weight in zip(["cash", *self.symbols], guarded_full_weights):
            info[f"weight_{name}"] = float(weight)
        for name, weight in zip(["cash", *self.symbols], proposed_full_weights):
            info[f"proposed_weight_{name}"] = float(weight)

        self.t = next_t
        self.prev_action = action
        self.prev_weights = new_weights.copy()
        self.prev_full_weights = guarded_full_weights.copy()

        return self._state(), float(reward), done, info

    def current_regime(self) -> int:
        return int(self.df.iloc[self.t]["regime_label"])

    def current_date(self) -> str:
        return str(self.df.iloc[self.t]["date"].date())


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.storage: list[dict[str, Any]] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, item: dict[str, Any]) -> None:
        item.setdefault("sample_count", 0)
        item.setdefault("priority_update_count", 0)
        item.setdefault("last_sample_step", -1)
        item.setdefault("last_priority_update_step", -1)
        item.setdefault("priority_sum", 0.0)
        item.setdefault("priority_observations", 0)
        item.setdefault("max_priority", float("nan"))
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
        self._record_samples(idx, current_step)
        batch["sample_sources"] = np.array(["uniform"] * len(idx), dtype=object)
        return batch

    def _record_samples(self, indices: np.ndarray, current_step: int | None) -> None:
        for i in np.asarray(indices, dtype=np.int64):
            rec = self.storage[int(i)]
            rec["sample_count"] += 1
            rec["last_sample_step"] = -1 if current_step is None else int(current_step)

    def transition_diagnostics(self, final_step: int) -> pd.DataFrame:
        rows = []
        for rec in self.storage:
            n_priority = int(rec.get("priority_observations", 0))
            rows.append({
                "transition_id": int(rec.get("transition_id", rec["time_index"])),
                "date": rec["date"], "time_index": int(rec["time_index"]),
                "sample_age": int(final_step - rec["time_index"]),
                "regime_label": int(rec["regime_label"]),
                "regime_name": rec.get("regime_name", ""),
                "boundary_id": int(rec.get("boundary_id", 0)),
                "distance_to_boundary": int(rec.get("distance_to_boundary", 0)),
                "action": int(rec["action"]), "reward": float(rec["reward"]),
                "reward_magnitude": abs(float(rec["reward"])),
                "return_sign": int(np.sign(rec["reward"])),
                "position_changed": bool(rec.get("position_changed", False)),
                "next_day_return": float(rec.get("next_day_return", np.nan)),
                "volatility": float(rec.get("volatility", np.nan)),
                "sample_count": int(rec.get("sample_count", 0)),
                "priority_update_count": int(rec.get("priority_update_count", 0)),
                "last_sample_step": int(rec.get("last_sample_step", -1)),
                "last_priority_update_step": int(rec.get("last_priority_update_step", -1)),
                "mean_priority": float(rec.get("priority_sum", 0.0)) / max(n_priority, 1),
                "max_priority": float(rec.get("max_priority", np.nan)),
                "mean_doe": float(rec.get("doe_sum", 0.0)) / max(int(rec.get("doe_observations", 0)), 1),
                "mean_td_error": float(rec.get("td_sum", 0.0)) / max(int(rec.get("td_observations", 0)), 1),
                "final_priority": float(rec.get("priority", np.nan)),
                "initial_priority": float(rec.get("initial_priority", np.nan)),
            })
        return pd.DataFrame(rows)

    def _pack(self, idx: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
        batch = [self.storage[int(i)] for i in idx]
        return {
            "states": np.stack([b["state"] for b in batch]).astype(np.float32),
            "actions": np.array([b["action"] for b in batch], dtype=np.int64),
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


class PERBuffer(ReplayBuffer):
    def __init__(self, capacity: int, seed: int, alpha: float = 0.6, eps: float = 1e-6):
        super().__init__(capacity, seed)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, item: dict[str, Any]) -> None:
        insert_pos = self.pos
        super().add(item)
        initial = float(item.get("initial_priority", self.max_priority))
        self.priorities[insert_pos] = max(initial, self.eps)
        item["initial_priority"] = float(self.priorities[insert_pos])
        item["priority"] = float(self.priorities[insert_pos])

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
        self._record_samples(idx, current_step)
        batch["priorities"] = p[idx].astype(np.float32)
        batch["sample_probs"] = probs[idx].astype(np.float32)
        batch["sample_sources"] = np.array(["per"] * len(idx), dtype=object)
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, current_step: int | None = None) -> None:
        new_p = np.abs(td_errors).astype(np.float32) + self.eps
        self.priorities[indices] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))
        for j, i in enumerate(np.asarray(indices, dtype=np.int64)):
            rec = self.storage[int(i)]
            p = float(new_p[j])
            rec["priority"] = p
            rec["priority_update_count"] += 1
            rec["last_priority_update_step"] = -1 if current_step is None else int(current_step)
            rec["priority_sum"] += p
            rec["priority_observations"] += 1
            rec["max_priority"] = p if np.isnan(rec["max_priority"]) else max(rec["max_priority"], p)


class RegimeAwareReplayBuffer(PERBuffer):
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
        src_same = ["same_regime"] * len(idx_same)

        idx_high = self._choice(all_idx, n_high, probs=priority_probs)
        src_high = ["high_td"] * len(idx_high)

        idx_recent = self._choice(recent_candidates, n_recent)
        src_recent = ["recent"] * len(idx_recent)

        idx_random = self._choice(all_idx, n_random)
        src_random = ["random"] * len(idx_random)

        idx = np.concatenate([idx_same, idx_high, idx_recent, idx_random]).astype(np.int64)
        sources = np.array(src_same + src_high + src_recent + src_random, dtype=object)

        if len(idx) < batch_size:
            extra = self._choice(all_idx, batch_size - len(idx))
            idx = np.concatenate([idx, extra])
            sources = np.concatenate([sources, np.array(["fill_random"] * len(extra), dtype=object)])

        if len(idx) > batch_size:
            idx = idx[:batch_size]
            sources = sources[:batch_size]

        # For regime-aware replay, use simple weights = 1.
        # The purpose is diagnostics and regime-controlled sampling, not exact IS correction.
        weights = np.ones(len(idx), dtype=np.float32)

        batch = self._pack(idx, weights)
        self._record_samples(idx, current_step)
        batch["priorities"] = priorities[idx].astype(np.float32)
        batch["sample_probs"] = priority_probs[idx].astype(np.float32)

        batch["sample_sources"] = sources
        return batch


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x_arr))


def robust_positive_scale(
    values: np.ndarray,
    previous: float | None,
    rho: float,
    eps: float,
    floor: float,
) -> float | None:
    """Scale-only robust normalization for non-negative diagnostics."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return previous

    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    candidate = med + 1.4826 * mad + eps
    if not np.isfinite(candidate) or candidate <= floor:
        return previous

    if previous is None:
        return float(candidate)
    return float(rho * previous + (1.0 - rho) * candidate)


def deer_score_from_new_count(n_new: int, cfg: ExperimentConfig) -> float:
    """External-label MVP schedule. First new transition after a boundary uses full S0."""
    if n_new <= 0:
        return 0.0
    age_new = max(int(n_new) - 1, 0)
    score = float(cfg.deer_s0) * (2.0 ** (-age_new / max(1, int(cfg.deer_half_life))))
    if score < float(cfg.deer_s_floor):
        return 0.0
    return score


class DEERReplayBuffer(PERBuffer):
    """DEER-style replay for the finance DQN MVP.

    This class keeps the PER sampling interface but changes priority updates.
    External regime labels define boundaries. After a boundary, transitions from the
    current boundary are treated as post-change; older transitions are treated as
    pre-change. Priority uses robust-normalized TD-error and empirical Q discrepancy.
    """

    def __init__(self, capacity: int, seed: int, cfg: ExperimentConfig):
        super().__init__(capacity=capacity, seed=seed, alpha=cfg.per_alpha, eps=cfg.per_eps)
        self.cfg = cfg
        self.scale_td: float | None = None
        self.scale_doe: float | None = None
        self.last_scale_refresh_update = -1

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
            replace_post = len(post_candidates) < n_post
            idx_post = self.rng.choice(
                post_candidates,
                size=n_post,
                replace=replace_post,
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
            replace_rest = len(remaining) < n_rest
            idx_rest = self.rng.choice(
                remaining,
                size=n_rest,
                replace=replace_rest,
                p=rest_probs,
            ).astype(np.int64)
            idx_parts.append(idx_rest)
            source_parts.append(np.array(["deer_per_remainder"] * len(idx_rest), dtype=object))

        idx = np.concatenate(idx_parts).astype(np.int64)
        sources = np.concatenate(source_parts).astype(object)

        weights = (n * global_probs[idx]) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        batch = self._pack(idx, weights.astype(np.float32))
        self._record_samples(idx, current_step)
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
        current_step: int | None = None,
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

        # Before the first detected boundary, DEER should behave like PER.
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
            rec["priority_update_count"] += 1
            rec["last_priority_update_step"] = -1 if current_step is None else int(current_step)
            rec["priority_sum"] += float(new_p[j])
            rec["priority_observations"] += 1
            rec["max_priority"] = (float(new_p[j]) if np.isnan(rec["max_priority"])
                                   else max(rec["max_priority"], float(new_p[j])))
            rec["doe_sum"] = float(rec.get("doe_sum", 0.0)) + float(doe_abs[j])
            rec["doe_observations"] = int(rec.get("doe_observations", 0)) + 1
            rec["td_sum"] = float(rec.get("td_sum", 0.0)) + float(td_abs[j])
            rec["td_observations"] = int(rec.get("td_observations", 0)) + 1

        return {
            "priority": new_p,
            "z_td": z_td.astype(np.float32),
            "z_doe": z_doe.astype(np.float32),
            "is_post_change": post,
            "source_mode": source_mode,
            "scale_td": np.array([np.nan if self.scale_td is None else self.scale_td], dtype=np.float32),
            "scale_doe": np.array([np.nan if self.scale_doe is None else self.scale_doe], dtype=np.float32),
        }


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        # q_probe is an EMA critic used for a less noisy empirical Q-discrepancy.
        # q_reference is frozen at each externally detected regime boundary.
        self.q_probe = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.q_probe.load_state_dict(self.q.state_dict())
        self.q_reference = QNetwork(state_dim, n_actions, cfg.hidden_dim).to(self.device)
        self.q_reference.load_state_dict(self.q_probe.state_dict())
        for p in self.q_probe.parameters():
            p.requires_grad_(False)
        for p in self.q_reference.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.update_count = 0
        self.n_actions = n_actions

    def freeze_reference(self) -> None:
        self.q_reference.load_state_dict(self.q_probe.state_dict())

    def _ema_update_probe(self) -> None:
        tau = float(self.cfg.deer_probe_tau)
        with torch.no_grad():
            for probe_param, q_param in zip(self.q_probe.parameters(), self.q.parameters()):
                probe_param.data.mul_(1.0 - tau).add_(q_param.data, alpha=tau)

    def compute_td_errors(self, batch: dict[str, Any]) -> np.ndarray:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_actions = self.q(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q
            td = target - q_sa
        return td.abs().cpu().numpy()

    def compute_q_discrepancy(self, batch: dict[str, Any]) -> np.ndarray:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q_old = self.q_reference(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q_new = self.q_probe(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            doe = (q_new - q_old).abs()
        return doe.cpu().numpy()

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q(x), dim=1).item())

    def q_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(states, dtype=torch.float32, device=self.device)
            return self.q(x).cpu().numpy()

    def update(self, batch: dict[str, Any]) -> dict[str, Any]:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights = torch.tensor(batch["weights"], dtype=torch.float32, device=self.device)

        q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        td = target - q_sa
        loss = (weights * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()
        self._ema_update_probe()

        self.update_count += 1
        if self.update_count % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "td_errors": td.detach().abs().cpu().numpy(),
            "mean_q": float(q_sa.detach().mean().cpu().item()),
        }


def probe_metrics(q_before: np.ndarray, q_after: np.ndarray) -> dict[str, Any]:
    """Value/policy stability metrics on the exact same fixed probe states."""
    before_actions, after_actions = q_before.argmax(axis=1), q_after.argmax(axis=1)
    before_sorted, after_sorted = np.sort(q_before, axis=1), np.sort(q_after, axis=1)
    margin_before = before_sorted[:, -1] - before_sorted[:, -2]
    margin_after = after_sorted[:, -1] - after_sorted[:, -2]
    out = {
        "q_drift": float(np.mean(np.abs(q_after - q_before))),
        "action_flip_rate": float(np.mean(before_actions != after_actions)),
        "q_margin_before": float(np.mean(margin_before)),
        "q_margin_after": float(np.mean(margin_after)),
        "q_margin_change": float(np.mean(margin_after - margin_before)),
    }
    for a in range(q_before.shape[1]):
        out[f"action_share_before_{a}"] = float(np.mean(before_actions == a))
        out[f"action_share_after_{a}"] = float(np.mean(after_actions == a))
    return out


def make_online_batch(transition: dict[str, Any]) -> dict[str, Any]:
    return {
        "states": np.expand_dims(transition["state"], axis=0).astype(np.float32),
        "actions": np.array([transition["action"]], dtype=np.int64),
        "rewards": np.array([transition["reward"]], dtype=np.float32),
        "next_states": np.expand_dims(transition["next_state"], axis=0).astype(np.float32),
        "dones": np.array([transition["done"]], dtype=np.float32),
        "regime_labels": np.array([transition["regime_label"]], dtype=np.int64),
        "boundary_ids": np.array([int(transition.get("boundary_id", 0))], dtype=np.int64),
        "time_indices": np.array([transition["time_index"]], dtype=np.int64),
        "dates": [transition["date"]],
        "indices": np.array([0], dtype=np.int64),
        "weights": np.ones(1, dtype=np.float32),
        "sample_sources": np.array(["online_latest"], dtype=object),
    }


def epsilon_at(step: int, cfg: ExperimentConfig) -> float:
    frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def beta_at(step: int, max_steps: int, cfg: ExperimentConfig) -> float:
    frac = min(1.0, step / max(1, max_steps))
    return cfg.per_beta_start + frac * (cfg.per_beta_end - cfg.per_beta_start)


def summarize_replay_batch(
    batch: dict[str, Any],
    current_regime: int,
    current_step: int,
    td_errors: np.ndarray,
    replay: str,
) -> dict[str, Any]:
    sampled_regime = batch["regime_labels"]
    sample_age = current_step - batch["time_indices"]

    if replay == "online":
        mismatch_rate = np.nan
    else:
        mismatch_rate = float(np.mean(sampled_regime != current_regime))

    out: dict[str, Any] = {
        "mismatch_rate": mismatch_rate,
        "mean_sample_age": float(np.mean(sample_age)),
        "median_sample_age": float(np.median(sample_age)),
        "mean_td_error": float(np.mean(td_errors)),
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


def make_buffer(cfg: ExperimentConfig) -> ReplayBuffer | None:
    if cfg.replay == "online":
        return None

    if cfg.replay == "uniform":
        return ReplayBuffer(cfg.buffer_size, cfg.seed)

    if cfg.replay == "per":
        return PERBuffer(
            capacity=cfg.buffer_size,
            seed=cfg.seed,
            alpha=cfg.per_alpha,
            eps=cfg.per_eps,
        )

    if cfg.replay == "regime":
        return RegimeAwareReplayBuffer(
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
        return DEERReplayBuffer(
            capacity=cfg.buffer_size,
            seed=cfg.seed,
            cfg=cfg,
        )

    raise ValueError("replay must be one of: online, uniform, per, regime, deer")


def run_single_experiment(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.seed)

    env = MarketDQNEnv(cfg)
    buffer = make_buffer(cfg)
    agent = DQNAgent(env.state_dim, env.n_actions, cfg)

    init_suffix = f"_init-{cfg.deer_initial_priority}" if cfg.replay == "deer" else ""
    output_dir = Path(cfg.output_root) / f"{cfg.label_method}_{cfg.replay}{init_suffix}_seed{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = cfg.__dict__.copy()
    config_dict["tradable_symbols"] = list(cfg.tradable_symbols)

    metadata = {
        "config": config_dict,
        "symbols": env.symbols,
        "action_names": env.action_names,
        "feature_cols": env.feature_cols,
        "n_rows": int(len(env.df)),
        "state_dim": int(env.state_dim),
        "n_actions": int(env.n_actions),
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
    mechanism_logs: list[dict[str, Any]] = []
    state_history: list[np.ndarray] = []
    active_events: list[dict[str, Any]] = []

    # Boundary state for DEER. Boundaries are detected from external regime labels
    # before taking the action at the current state, so the first transition in a
    # new regime is correctly tagged with the new boundary_id.
    current_regime_for_boundary = env.current_regime()
    current_boundary = 0
    deer_n_new = 0
    deer_s_score = 0.0
    boundary_start_step = 0

    for step in range(max_steps):
        step_regime = env.current_regime()
        boundary_changed = step > 0 and step_regime != current_regime_for_boundary
        if boundary_changed:
            current_boundary += 1
            current_regime_for_boundary = step_regime
            deer_n_new = 0
            deer_s_score = cfg.deer_s0
            boundary_start_step = step
            probe_states = np.stack((state_history or [state])[-max(1, cfg.mechanism_probe_size):])
            active_events.append({
                "boundary_id": current_boundary,
                "boundary_step": step,
                "old_regime": int(trade_logs[-1]["regime_label"]) if trade_logs else step_regime,
                "new_regime": step_regime,
                "states": probe_states,
                "q_before": agent.q_values(probe_states),
                "td_baseline": float(np.mean([x["mean_td_error"] for x in replay_logs[-20:]])) if replay_logs else np.nan,
            })
            if cfg.replay == "deer":
                agent.freeze_reference()

        eps = epsilon_at(step, cfg)
        action = agent.act(state, eps)

        next_state, reward, done, info = env.step(action)

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "date": info["date"],
            "regime_label": info["regime_label"],
            "regime_name": info["regime_name"],
            "time_index": info["time_index"],
            "boundary_id": current_boundary,
            "transition_id": step,
            "distance_to_boundary": step - boundary_start_step,
            "position_changed": bool(info["turnover"] > 0),
            "next_day_return": info["gross_return"],
            "volatility": float(env.df.iloc[info["time_index"]].get("vol", np.nan)),
        }

        trade_logs.append(
            {
                "step": step,
                "date": info["date"],
                "regime_label": info["regime_label"],
                "regime_name": info["regime_name"],
                "boundary_id": current_boundary,
                "boundary_changed": bool(boundary_changed),
                "deer_s_score": deer_s_score,
                "action": info["action"],
                "action_name": info["action_name"],
                "reward": reward,
                "portfolio_value": info["portfolio_value"],
                "drawdown": info["drawdown"],
                "turnover": info["turnover"],
                "gross_return": info["gross_return"],
                "cost": info["cost"],
                "epsilon": eps,
                "safety_blend": info["safety_blend"],
                "safety_turnover_before_cap": info["safety_turnover_before_cap"],
                "safety_turnover_after_cap": info["safety_turnover_after_cap"],
                "safety_anchor_cash": info["safety_anchor_cash"],
                **{f"weight_{name}": info[f"weight_{name}"] for name in ["cash", *env.symbols]},
                **{
                    f"proposed_weight_{name}": info[f"proposed_weight_{name}"]
                    for name in ["cash", *env.symbols]
                },
            }
        )

        if cfg.replay == "online":
            batch = make_online_batch(transition)
            update = agent.update(batch)

            replay_diag = summarize_replay_batch(
                batch=batch,
                current_regime=env.current_regime(),
                current_step=step,
                td_errors=update["td_errors"],
                replay=cfg.replay,
            )
            replay_diag.update(
                {
                    "step": step,
                    "date": env.current_date(),
                    "current_regime": env.current_regime(),
                    "loss": update["loss"],
                    "mean_q": update["mean_q"],
                    "epsilon": eps,
                    "beta": 0.0,
                }
            )
            replay_logs.append(replay_diag)

        else:
            assert buffer is not None
            if isinstance(buffer, PERBuffer):
                if cfg.deer_initial_priority == "median" and len(buffer) > 0:
                    transition["initial_priority"] = float(np.median(buffer.priorities[:len(buffer)]))
                elif cfg.deer_initial_priority == "doe" and cfg.replay == "deer" and current_boundary > 0:
                    transition["initial_priority"] = float(agent.compute_q_discrepancy(make_online_batch(transition))[0] + cfg.per_eps)
            buffer.add(transition)

            if cfg.replay == "deer" and current_boundary > 0:
                deer_n_new += 1
                deer_s_score = deer_score_from_new_count(deer_n_new, cfg)

            if len(buffer) >= max(cfg.warmup_steps, cfg.batch_size):
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
                    assert isinstance(buffer, DEERReplayBuffer)

                    # Refresh robust scales from an independent uniform probe set.
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
                        current_step=step,
                    )
                    batch["doe_values"] = doe_values.astype(np.float32)
                    batch["z_td_values"] = np.asarray(deer_info["z_td"], dtype=np.float32)
                    batch["z_doe_values"] = np.asarray(deer_info["z_doe"], dtype=np.float32)
                    batch["deer_is_post_change"] = np.asarray(deer_info["is_post_change"], dtype=bool)
                    batch["deer_priority_mode"] = np.asarray(deer_info["source_mode"], dtype=object)
                    batch["priorities"] = np.asarray(deer_info["priority"], dtype=np.float32)

                elif cfg.replay in {"per", "regime"}:
                    assert isinstance(buffer, PERBuffer)
                    buffer.update_priorities(batch["indices"], update["td_errors"], current_step=step)

                replay_diag = summarize_replay_batch(
                    batch=batch,
                    current_regime=env.current_regime(),
                    current_step=step,
                    td_errors=update["td_errors"],
                    replay=cfg.replay,
                )

                replay_diag.update(
                    {
                        "step": step,
                        "date": env.current_date(),
                        "current_regime": env.current_regime(),
                        "current_boundary": current_boundary,
                        "loss": update["loss"],
                        "mean_q": update["mean_q"],
                        "epsilon": eps,
                        "beta": beta,
                        "deer_s_score": deer_s_score if cfg.replay == "deer" else np.nan,
                        "deer_n_new": deer_n_new if cfg.replay == "deer" else np.nan,
                        "scale_td": getattr(buffer, "scale_td", np.nan),
                        "scale_doe": getattr(buffer, "scale_doe", np.nan),
                    }
                )
                replay_logs.append(replay_diag)

        current_td = replay_logs[-1]["mean_td_error"] if replay_logs and replay_logs[-1].get("step") == step else np.nan
        for event in active_events:
            offset = step - event["boundary_step"]
            if 0 <= offset <= cfg.mechanism_event_horizon:
                metrics = probe_metrics(event["q_before"], agent.q_values(event["states"]))
                metrics.update({
                    "step": step, "date": info["date"], "event_offset": offset,
                    "boundary_id": event["boundary_id"], "boundary_step": event["boundary_step"],
                    "old_regime": event["old_regime"], "new_regime": event["new_regime"],
                    "probe_size": len(event["states"]), "td_error": current_td,
                    "td_baseline": event["td_baseline"],
                    "td_error_shock": current_td / max(event["td_baseline"], 1e-12) if np.isfinite(event["td_baseline"]) else np.nan,
                })
                mechanism_logs.append(metrics)
        active_events = [e for e in active_events if step - e["boundary_step"] < cfg.mechanism_event_horizon]
        state_history.append(state.copy())

        state = next_state
        if done:
            break

    trade_df = pd.DataFrame(trade_logs)
    replay_df = pd.DataFrame(replay_logs)
    mechanism_df = pd.DataFrame(mechanism_logs)
    if replay_df.empty:
        replay_df = pd.DataFrame(columns=["step", "date", "mismatch_rate", "mean_td_error"])
    if mechanism_df.empty:
        mechanism_df = pd.DataFrame(columns=["step", "date", "event_offset", "boundary_id", "q_drift", "action_flip_rate", "q_margin_change", "td_error", "td_error_shock"])

    trade_df.to_csv(output_dir / "trading_log.csv", index=False)
    replay_df.to_csv(output_dir / "replay_diagnostics.csv", index=False)
    mechanism_df.to_csv(output_dir / "mechanism_events.csv", index=False)
    if buffer is not None:
        buffer.transition_diagnostics(max_steps - 1).to_csv(output_dir / "transition_diagnostics.csv", index=False)

    if not replay_df.empty and "mismatch_rate" in replay_df.columns:
        mean_mismatch_rate = float(replay_df["mismatch_rate"].mean(skipna=True))
    else:
        mean_mismatch_rate = np.nan

    if not replay_df.empty and "mean_td_error" in replay_df.columns:
        mean_td_error = float(replay_df["mean_td_error"].mean())
    else:
        mean_td_error = np.nan

    if not replay_df.empty and "mean_priority" in replay_df.columns:
        mean_priority = float(replay_df["mean_priority"].mean())
    else:
        mean_priority = np.nan

    mean_doe = float(replay_df["mean_doe"].mean()) if not replay_df.empty and "mean_doe" in replay_df.columns else np.nan
    mean_z_doe = float(replay_df["mean_z_doe"].mean()) if not replay_df.empty and "mean_z_doe" in replay_df.columns else np.nan
    mean_post_boundary_sample_rate = (
        float(replay_df["post_boundary_sample_rate"].mean())
        if not replay_df.empty and "post_boundary_sample_rate" in replay_df.columns
        else np.nan
    )

    summary = {
        "label_method": cfg.label_method,
        "replay": cfg.replay,
        "seed": cfg.seed,
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
        "mean_mismatch_rate": mean_mismatch_rate,
        "mean_td_error": mean_td_error,
        "mean_priority": mean_priority,
        "mean_doe": mean_doe,
        "mean_z_doe": mean_z_doe,
        "mean_post_boundary_sample_rate": mean_post_boundary_sample_rate,
    }

    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    mismatch_display = "NA" if np.isnan(mean_mismatch_rate) else f"{mean_mismatch_rate:.4f}"

    print(
        f"[done] {cfg.label_method} | {cfg.replay} | seed={cfg.seed} | "
        f"final_value={summary['final_portfolio_value']:.4f} | "
        f"max_dd={summary['max_drawdown']:.4f} | "
        f"mismatch={mismatch_display}"
    )

    return output_dir


def run_many(
    base_cfg: ExperimentConfig,
    replays: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    all_summaries = []

    for replay in replays:
        for seed in seeds:
            cfg = ExperimentConfig(**base_cfg.__dict__)
            cfg.replay = replay
            cfg.seed = seed

            out_dir = run_single_experiment(cfg)
            all_summaries.append(pd.read_csv(out_dir / "summary.csv"))

    summary = pd.concat(all_summaries, ignore_index=True)

    analysis_dir = Path(base_cfg.output_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / f"{base_cfg.label_method}_dqn_replay_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_plots(base_cfg.output_root, base_cfg.label_method, replays, seeds, base_cfg.deer_initial_priority)
    analyze_mechanism_outputs(base_cfg.output_root, base_cfg.label_method)

    print(f"[summary] wrote {summary_path}")
    return summary


def analyze_mechanism_outputs(output_root: str, label_method: str) -> None:
    """Build paper-ready mechanism tables from completed runs (no plotting dependency)."""
    root, rows, transitions, summaries_all = Path(output_root), [], [], []
    for run in sorted(root.glob(f"{label_method}_*_seed*")):
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8")) if (run / "metadata.json").exists() else {"config": {}}
        cfg = meta.get("config", {})
        common = {"run": run.name, "replay": cfg.get("replay", "unknown"), "seed": cfg.get("seed", np.nan),
                  "initial_priority_setting": cfg.get("deer_initial_priority", "max")}
        event_path = run / "mechanism_events.csv"
        if event_path.exists() and event_path.stat().st_size:
            event = pd.read_csv(event_path)
            if not event.empty:
                event = event.assign(**common)
                rows.append(event)
        transition_path = run / "transition_diagnostics.csv"
        if transition_path.exists() and transition_path.stat().st_size:
            trans = pd.read_csv(transition_path).assign(**common)
            transitions.append(trans)
        summary_path = run / "summary.csv"
        if summary_path.exists():
            summaries_all.append(pd.read_csv(summary_path).assign(run=run.name,
                                                                  initial_priority_setting=common["initial_priority_setting"]))

    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if summaries_all:
        pd.concat(summaries_all, ignore_index=True).to_csv(
            out / f"{label_method}_all_runs_summary.csv", index=False
        )
    if rows:
        events = pd.concat(rows, ignore_index=True)
        metric_cols = [c for c in ["q_drift", "action_flip_rate", "q_margin_before", "q_margin_after", "q_margin_change", "td_error", "td_error_shock"] if c in events]
        metric_cols += [c for c in events if c.startswith("action_share_")]
        curves = events.groupby(["replay", "initial_priority_setting", "event_offset"])[metric_cols].agg(["mean", "std", "count"])
        curves.columns = ["_".join(c) for c in curves.columns]
        curves.reset_index().to_csv(out / f"{label_method}_boundary_aligned_curves.csv", index=False)
        summaries = []
        for keys, group in events.groupby(["run", "boundary_id"]):
            ordered = group.sort_values("event_offset")
            baseline = ordered["td_baseline"].iloc[0]
            recovered = ordered[(ordered["event_offset"] > 0) & (ordered["td_error"] <= baseline * 1.25)] if np.isfinite(baseline) else ordered.iloc[0:0]
            summaries.append({**{c: ordered[c].iloc[0] for c in ["run", "replay", "seed", "initial_priority_setting"]},
                              "boundary_id": keys[1], "mean_post_q_drift": ordered["q_drift"].mean(),
                              "mean_action_flip_rate": ordered["action_flip_rate"].mean(),
                              "max_td_error_shock": ordered["td_error_shock"].max(),
                              "td_recovery_updates": recovered["event_offset"].iloc[0] if len(recovered) else np.nan})
        pd.DataFrame(summaries).to_csv(out / f"{label_method}_boundary_summary.csv", index=False)

    if transitions:
        trans = pd.concat(transitions, ignore_index=True)
        numeric = [c for c in ["sample_age", "distance_to_boundary", "mean_td_error", "reward_magnitude", "next_day_return", "volatility", "position_changed", "sample_count", "priority_update_count", "mean_priority", "max_priority", "mean_doe"] if c in trans]
        comparisons = []
        for run, group in trans.groupby("run"):
            for score, label in [("mean_doe", "doe"), ("final_priority", "priority")]:
                valid = group[np.isfinite(group[score])]
                if len(valid) < 5:
                    continue
                lo, hi = valid[score].quantile([0.2, 0.8])
                for bucket, selected in [("low_20", valid[valid[score] <= lo]), ("high_20", valid[valid[score] >= hi])]:
                    comparisons.append({"run": run, "score": label, "group": bucket, "n": len(selected),
                                        **{f"mean_{c}": pd.to_numeric(selected[c], errors="coerce").mean() for c in numeric}})
        pd.DataFrame(comparisons).to_csv(out / f"{label_method}_high_low_comparison.csv", index=False)
        concentration = []
        for run, group in trans.groupby("run"):
            counts = np.sort(group["sample_count"].to_numpy(dtype=float))[::-1]
            total = counts.sum()
            priorities = np.maximum(pd.to_numeric(group["final_priority"], errors="coerce").fillna(0).to_numpy(), 0)
            priority_total = priorities.sum()
            concentration.append({"run": run, **{f"top_{p}pct_share": float(counts[:max(1, math.ceil(len(counts)*p/100))].sum()/total) if total else 0.0 for p in [1, 5, 10]},
                                  "sampling_entropy": float(-(lambda p: np.sum(p[p > 0] * np.log(p[p > 0])))(counts / total)) if total else np.nan,
                                  "priority_entropy": float(-(lambda p: np.sum(p[p > 0] * np.log(p[p > 0])))(priorities / priority_total)) if priority_total else np.nan})
        pd.DataFrame(concentration).to_csv(out / f"{label_method}_replay_concentration.csv", index=False)
        trans.sort_values(["run", "sample_count"], ascending=[True, False]).groupby("run").head(100).to_csv(
            out / f"{label_method}_top_replayed_samples.csv", index=False
        )

    # Paper-facing plots are generated from saved CSVs so analysis can be rerun
    # without retraining the agents.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[mechanism plots skipped] matplotlib unavailable: {exc}")
        return

    if rows:
        style = events["replay"].astype(str) + events["initial_priority_setting"].map(
            lambda x: f"-{x}" if x != "max" else ""
        )
        events = events.assign(method=style)
        for metric, ylabel in [("q_drift", "Mean absolute Q drift"),
                               ("action_flip_rate", "Action flip rate"),
                               ("td_error_shock", "TD-error / pre-boundary baseline"),
                               ("q_margin_change", "Q-margin change")]:
            plt.figure(figsize=(9, 5))
            for method, group in events.groupby("method"):
                curve = group.groupby("event_offset")[metric].agg(["mean", "sem"]).reset_index()
                plt.plot(curve["event_offset"], curve["mean"], label=method)
                plt.fill_between(curve["event_offset"], curve["mean"] - curve["sem"],
                                 curve["mean"] + curve["sem"], alpha=0.15)
            plt.axvline(0, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Updates after regime boundary")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"{label_method}_{metric}_around_boundaries.png", dpi=180)
            plt.close()

        action_cols = sorted(c for c in events if c.startswith("action_share_after_"))
        if action_cols:
            deer = events[events["replay"] == "deer"]
            plt.figure(figsize=(9, 5))
            for col in action_cols:
                curve = deer.groupby("event_offset")[col].mean()
                plt.plot(curve.index, curve.values, label=col.replace("action_share_after_", "action "))
            plt.xlabel("Updates after regime boundary")
            plt.ylabel("Greedy-action share on fixed probes")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"{label_method}_policy_distribution_after_boundaries.png", dpi=180)
            plt.close()

    if transitions:
        deer = trans[trans["replay"] == "deer"].copy()
        for x, y, name in [("mean_doe", "sample_count", "sampling_count_vs_doe"),
                           ("mean_td_error", "sample_count", "sampling_count_vs_td_error"),
                           ("mean_doe", "priority_update_count", "priority_updates_vs_doe")]:
            valid = deer[np.isfinite(deer[x]) & np.isfinite(deer[y])]
            if valid.empty:
                continue
            plt.figure(figsize=(8, 5))
            for setting, group in valid.groupby("initial_priority_setting"):
                plt.scatter(group[x], group[y], s=7, alpha=0.25, label=setting)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"{label_method}_{name}.png", dpi=180)
            plt.close()

        valid_priority = deer[np.isfinite(deer["final_priority"])]
        if not valid_priority.empty:
            plt.figure(figsize=(9, 5))
            valid_priority.boxplot(column="final_priority", by="regime_label", grid=False)
            plt.suptitle("")
            plt.title("DEER final priority by regime")
            plt.ylabel("Final priority")
            plt.tight_layout()
            plt.savefig(out / f"{label_method}_priority_by_regime.png", dpi=180)
            plt.close()

        plt.figure(figsize=(8, 5))
        grid = np.linspace(0, 100, 201)
        concentration_curves: dict[str, list[np.ndarray]] = {}
        for run, group in trans.groupby("run"):
            counts = np.sort(group["sample_count"].to_numpy(dtype=float))[::-1]
            if counts.sum() <= 0:
                continue
            replay = str(group["replay"].iloc[0])
            setting = str(group["initial_priority_setting"].iloc[0])
            method = f"DEER-{setting}" if replay == "deer" else replay.upper()
            x = np.concatenate([[0.0], np.arange(1, len(counts) + 1) / len(counts) * 100])
            y = np.concatenate([[0.0], np.cumsum(counts) / counts.sum()])
            concentration_curves.setdefault(method, []).append(np.interp(grid, x, y))
        for method, curves in concentration_curves.items():
            values = np.vstack(curves)
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
            plt.plot(grid, mean, linewidth=2, label=method)
            plt.fill_between(grid, mean - sem, mean + sem, alpha=0.12)
        plt.xlabel("Top transitions (%)")
        plt.ylabel("Cumulative share of replay")
        plt.legend(title="Replay method")
        plt.tight_layout()
        plt.savefig(out / f"{label_method}_replay_concentration_curve.png", dpi=180)
        plt.close()


def make_plots(output_root: str, label_method: str, replays: list[str], seeds: list[int], deer_initial_priority: str = "max") -> None:
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

    def run_dir(replay: str, seed: int) -> Path:
        suffix = f"_init-{deer_initial_priority}" if replay == "deer" else ""
        return Path(output_root) / f"{label_method}_{replay}{suffix}_seed{seed}"

    plt.figure(figsize=(12, 5))
    for replay in replays:
        for seed in seeds:
            path = run_dir(replay, seed) / "trading_log.csv"
            if path.exists():
                df = pd.read_csv(path)
                plt.plot(df["step"], df["portfolio_value"], label=f"{replay}-seed{seed}", alpha=0.85)
    plt.title("DQN Trading: Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_portfolio_value.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    for replay in replays:
        if replay == "online":
            continue
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
    plt.title("Replay Diagnostics: Mismatch Rate")
    plt.xlabel("Step")
    plt.ylabel("Rolling mismatch rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_mismatch_rate.png", dpi=160)
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
    plt.title("DQN Diagnostics: TD-error")
    plt.xlabel("Step")
    plt.ylabel("Rolling mean TD-error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / f"{label_method}_td_error.png", dpi=160)
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
        plt.title("Priority Diagnostics: PER vs Regime-aware vs DEER")
        plt.xlabel("Step")
        plt.ylabel("Rolling mean priority")
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_dir / f"{label_method}_priority.png", dpi=160)
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
            plt.title("DEER Diagnostics: Empirical Q-discrepancy")
            plt.xlabel("Step")
            plt.ylabel("Rolling mean DoE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{label_method}_deer_doe.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 5))
        plotted = False
        for seed in seeds:
            path = run_dir("deer", seed) / "replay_diagnostics.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "deer_s_score" in df.columns:
                    plt.plot(
                        df["step"],
                        df["deer_s_score"].rolling(10, min_periods=1).mean(),
                        label=f"deer-S-seed{seed}",
                        alpha=0.85,
                    )
                    plotted = True
        if plotted:
            plt.title("DEER Diagnostics: S score schedule")
            plt.xlabel("Step")
            plt.ylabel("S score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{label_method}_deer_s_score.png", dpi=160)
        plt.close()

    # Regime-aware sample source composition
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
            plt.title(f"Regime-aware Replay Source Composition, seed={seed}")
            plt.xlabel("Step")
            plt.ylabel("Rolling sample count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(analysis_dir / f"{label_method}_regime_sources_seed{seed}.png", dpi=160)
            plt.close()
