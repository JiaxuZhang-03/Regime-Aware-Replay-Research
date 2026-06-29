from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortfolioSafetyConfig:
    """Model-agnostic policy guard inspired by ReCAP's regime gate.

    The guard keeps a small regime-conditioned anchor policy library and blends
    each model action with the anchor for the current market regime. It then
    applies long-only cash, concentration, and turnover constraints.
    """

    enabled: bool = True
    min_cash_weight: float = 0.03
    max_asset_weight: float = 0.85
    max_turnover: float = 0.75
    regime_blend: float = 0.20
    risk_on_cash: float = 0.05
    sideways_cash: float = 0.25
    high_vol_cash: float = 0.55
    risk_off_cash: float = 0.70


def safety_config_from_object(obj: Any) -> PortfolioSafetyConfig:
    return PortfolioSafetyConfig(
        enabled=bool(getattr(obj, "safety_enabled", True)),
        min_cash_weight=float(getattr(obj, "safety_min_cash_weight", 0.03)),
        max_asset_weight=float(getattr(obj, "safety_max_asset_weight", 0.85)),
        max_turnover=float(getattr(obj, "safety_max_turnover", 0.75)),
        regime_blend=float(getattr(obj, "safety_regime_blend", 0.20)),
        risk_on_cash=float(getattr(obj, "safety_risk_on_cash", 0.05)),
        sideways_cash=float(getattr(obj, "safety_sideways_cash", 0.25)),
        high_vol_cash=float(getattr(obj, "safety_high_vol_cash", 0.55)),
        risk_off_cash=float(getattr(obj, "safety_risk_off_cash", 0.70)),
    )


def normalize_long_only(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 1e-12:
        out = np.zeros_like(w)
        out[0] = 1.0
        return out
    return (w / total).astype(np.float32)


def regime_anchor_weights(
    regime_label: int | None,
    regime_name: str | None,
    n_assets: int,
    cfg: PortfolioSafetyConfig,
) -> np.ndarray:
    cash = _cash_for_regime(regime_label, regime_name, cfg)
    anchor = np.zeros(n_assets + 1, dtype=np.float32)
    anchor[0] = cash
    if n_assets > 0:
        anchor[1:] = (1.0 - cash) / n_assets
    return normalize_long_only(anchor)


def apply_policy_safety(
    proposed_weights: np.ndarray,
    previous_weights: np.ndarray,
    regime_label: int | None,
    regime_name: str | None,
    cfg: PortfolioSafetyConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    proposed = normalize_long_only(proposed_weights)
    previous = normalize_long_only(previous_weights)
    if proposed.shape != previous.shape:
        raise ValueError(f"proposed and previous weight shapes differ: {proposed.shape} vs {previous.shape}")

    if not cfg.enabled:
        turnover = float(np.abs(proposed - previous).sum())
        return proposed, {
            "safety_blend": 0.0,
            "safety_turnover_before_cap": turnover,
            "safety_turnover_after_cap": turnover,
            "safety_anchor_cash": np.nan,
        }

    n_assets = max(0, proposed.shape[0] - 1)
    blend = float(np.clip(cfg.regime_blend, 0.0, 1.0))
    anchor = regime_anchor_weights(regime_label, regime_name, n_assets, cfg)
    guarded = normalize_long_only((1.0 - blend) * proposed + blend * anchor)
    guarded = _enforce_cash_and_concentration(guarded, cfg)

    turnover_before = float(np.abs(guarded - previous).sum())
    max_turnover = float(max(cfg.max_turnover, 0.0))
    if max_turnover > 0.0 and turnover_before > max_turnover:
        scale = max_turnover / max(turnover_before, 1e-12)
        guarded = normalize_long_only(previous + scale * (guarded - previous))
        guarded = _enforce_cash_and_concentration(guarded, cfg)

    turnover_after = float(np.abs(guarded - previous).sum())
    return guarded.astype(np.float32), {
        "safety_blend": blend,
        "safety_turnover_before_cap": turnover_before,
        "safety_turnover_after_cap": turnover_after,
        "safety_anchor_cash": float(anchor[0]),
    }


def _cash_for_regime(
    regime_label: int | None,
    regime_name: str | None,
    cfg: PortfolioSafetyConfig,
) -> float:
    name = (regime_name or "").lower()
    if name in {"risk_on", "bull"}:
        return float(cfg.risk_on_cash)
    if name in {"sideways", "neutral"}:
        return float(cfg.sideways_cash)
    if name in {"high_vol", "volatile", "stress"}:
        return float(cfg.high_vol_cash)
    if name in {"risk_off", "bear"}:
        return float(cfg.risk_off_cash)

    label = None if regime_label is None else int(regime_label)
    # Shared taxonomy fallback: 0 risk_on, 1 sideways, 2 high_vol, 3 risk_off.
    if label == 0:
        return float(cfg.risk_on_cash)
    if label == 1:
        return float(cfg.sideways_cash)
    if label == 2:
        return float(cfg.high_vol_cash)
    if label == 3:
        return float(cfg.risk_off_cash)
    return float(cfg.sideways_cash)


def _enforce_cash_and_concentration(
    weights: np.ndarray,
    cfg: PortfolioSafetyConfig,
) -> np.ndarray:
    w = normalize_long_only(weights)
    if len(w) <= 1:
        return w

    min_cash = float(np.clip(cfg.min_cash_weight, 0.0, 1.0))
    max_asset = float(np.clip(cfg.max_asset_weight, 0.0, 1.0))
    w[0] = max(w[0], min_cash)
    w[1:] = np.minimum(w[1:], max_asset)
    w = normalize_long_only(w)

    # A second pass handles normalization drift after the cash floor.
    if w[0] < min_cash:
        asset_total = float(w[1:].sum())
        w[0] = min_cash
        if asset_total > 1e-12:
            w[1:] = w[1:] / asset_total * (1.0 - min_cash)
        else:
            w[1:] = 0.0
    if np.any(w[1:] > max_asset):
        w[1:] = np.minimum(w[1:], max_asset)
        w = normalize_long_only(w)
    return w.astype(np.float32)
