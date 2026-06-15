"""Transparent trend/volatility rule-based regime labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import MarketFeatureConfig, build_market_features
from .taxonomy import label_for_name


@dataclass(frozen=True)
class RuleBasedConfig:
    lookback: int = 252
    min_periods: int = 60
    vol_quantile: float = 0.75
    vix_quantile: float = 0.75
    ret_threshold: float = 0.0
    trend_threshold: float = 0.0


def label_rule_based(
    df: pd.DataFrame,
    feature_config: MarketFeatureConfig | None = None,
    rule_config: RuleBasedConfig | None = None,
) -> pd.DataFrame:
    """Label regimes using rolling trend, return, VIX, and volatility rules."""

    rule_config = rule_config or RuleBasedConfig()
    features = build_market_features(df, feature_config)

    vol_threshold = _rolling_threshold(
        features["vol"], rule_config.lookback, rule_config.min_periods, rule_config.vol_quantile
    )
    vix_threshold = _rolling_threshold(
        features["vix"], rule_config.lookback, rule_config.min_periods, rule_config.vix_quantile
    )

    high_vol = (features["vol"] >= vol_threshold) | (features["vix"] >= vix_threshold)
    positive_trend = (
        (features["trend"] > rule_config.trend_threshold)
        & (features["ret_long"] > rule_config.ret_threshold)
    )
    negative_trend = (
        (features["trend"] < -rule_config.trend_threshold)
        | (features["ret_long"] < -rule_config.ret_threshold)
    )

    names = np.full(len(features), "sideways", dtype=object)
    names[positive_trend & ~high_vol] = "risk_on"
    names[high_vol & ~negative_trend] = "high_vol"
    names[negative_trend] = "risk_off"
    names[high_vol & (features["ret_short"] >= 0.0) & (features["ret_long"] >= 0.0)] = "high_vol"

    out = features.copy()
    out["method"] = "rule_based"
    out["regime_name"] = names
    out["regime_label"] = [label_for_name(name) for name in names]
    out["vol_threshold"] = vol_threshold
    out["vix_threshold"] = vix_threshold
    return _standard_columns(out)


def _rolling_threshold(
    series: pd.Series,
    lookback: int,
    min_periods: int,
    quantile: float,
) -> pd.Series:
    rolling = series.rolling(lookback, min_periods=min_periods).quantile(quantile).shift(1)
    expanding = series.expanding(min_periods=1).quantile(quantile).shift(1)
    threshold = rolling.fillna(expanding)
    return threshold.fillna(series.median())


def _standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading = ["date", "method", "regime_label", "regime_name"]
    return df[[*leading, *[column for column in df.columns if column not in leading]]]
