"""Market-level feature extraction for regime labelers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketFeatureConfig:
    date_col: str = "date"
    primary_symbol: str = "SPY"
    fallback_symbols: tuple[str, ...] = ("GSPC", "DIA", "QQQ", "DJI", "IXIC")


def build_market_features(
    df: pd.DataFrame,
    config: MarketFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build one market-level row per date.

    The function accepts either the existing wide index feature file or a long
    panel such as the DOW30 RECAP feature file. Returned columns are normalized
    to a small common schema used by all three labelers.
    """

    config = config or MarketFeatureConfig()
    if config.date_col not in df.columns:
        raise ValueError(f"missing date column: {config.date_col}")

    working = df.copy()
    working[config.date_col] = pd.to_datetime(working[config.date_col])
    working = working.sort_values(config.date_col)

    id_col = _find_id_column(working)
    if id_col and working[config.date_col].duplicated().any():
        working = _collapse_long_panel(working, config, id_col)

    features = pd.DataFrame({config.date_col: working[config.date_col]})
    symbols = (config.primary_symbol, *config.fallback_symbols)

    price_col = _pick_column(
        working,
        _with_symbols("adjclose", symbols)
        + _with_symbols("close", symbols)
        + ["adjcp", "close"],
    )
    price = _numeric_series(working, price_col) if price_col else None

    features["ret_short"] = _feature_or_price_return(
        working,
        _with_symbols("ret_20d", symbols) + ["return_5", "ret_5d", "return_1", "ret_1d"],
        price,
        periods=20,
    )
    features["ret_long"] = _feature_or_price_return(
        working,
        _with_symbols("ret_60d", symbols) + ["ret_60d"],
        price,
        periods=60,
    )
    features["vol"] = _feature_or_price_vol(
        working,
        _with_symbols("vol_20d", symbols) + ["vol_20d", "vol_60d"],
        price,
    )
    features["trend"] = _feature_or_price_trend(
        working,
        _with_symbols("trend_price_200", symbols)
        + _with_symbols("trend_20_60", symbols)
        + ["trend_price_200", "trend_20_60"],
        price,
    )
    features["vix"] = _optional_feature(
        working,
        ["adjclose_VIX", "vix", "VIX", "close_VIX", "adjcp_VIX"],
        default=0.0,
    )
    features["turbulence"] = _optional_feature(
        working,
        ["turbulence", "Turbulence"],
        default=0.0,
    )

    for column in features.columns:
        if column == config.date_col:
            continue
        series = pd.to_numeric(features[column], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)
        features[column] = series.ffill().bfill().fillna(0.0)

    return features.reset_index(drop=True)


def feature_matrix(
    features: pd.DataFrame,
    columns: Iterable[str],
    standardize: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return a dense matrix, selected column names, means, and scales."""

    selected = [column for column in columns if column in features.columns]
    if not selected:
        raise ValueError("no requested feature columns are present")

    matrix = features[selected].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    col_medians = np.nanmedian(matrix, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(col_medians, np.where(missing)[1])

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    if standardize:
        matrix = (matrix - means) / scales
    return matrix, selected, means, scales


def _find_id_column(df: pd.DataFrame) -> str | None:
    for column in ("symbol", "tic", "ticker"):
        if column in df.columns:
            return column
    return None


def _collapse_long_panel(
    df: pd.DataFrame,
    config: MarketFeatureConfig,
    id_col: str,
) -> pd.DataFrame:
    primary_rows = df[df[id_col].astype(str).str.upper() == config.primary_symbol.upper()]
    if not primary_rows.empty:
        return primary_rows.drop_duplicates(config.date_col).sort_values(config.date_col)

    numeric_cols = [
        column
        for column in df.columns
        if column not in {config.date_col, id_col}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    collapsed = df.groupby(config.date_col, as_index=False)[numeric_cols].mean()
    return collapsed.sort_values(config.date_col)


def _with_symbols(prefix: str, symbols: Iterable[str]) -> list[str]:
    return [f"{prefix}_{symbol}" for symbol in symbols]


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        column = lower_lookup.get(candidate.lower())
        if column is not None:
            return column
    return None


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _optional_feature(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: float,
) -> pd.Series:
    column = _pick_column(df, candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype=float)
    return _numeric_series(df, column)


def _feature_or_price_return(
    df: pd.DataFrame,
    candidates: Iterable[str],
    price: pd.Series | None,
    periods: int,
) -> pd.Series:
    column = _pick_column(df, candidates)
    if column is not None:
        return _numeric_series(df, column)
    if price is not None:
        return price.pct_change(periods=periods).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _feature_or_price_vol(
    df: pd.DataFrame,
    candidates: Iterable[str],
    price: pd.Series | None,
) -> pd.Series:
    column = _pick_column(df, candidates)
    if column is not None:
        return _numeric_series(df, column)
    if price is not None:
        return price.pct_change().rolling(20, min_periods=5).std().fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _feature_or_price_trend(
    df: pd.DataFrame,
    candidates: Iterable[str],
    price: pd.Series | None,
) -> pd.Series:
    column = _pick_column(df, candidates)
    if column is not None:
        return _numeric_series(df, column)
    if price is not None:
        ma_200 = price.rolling(200, min_periods=20).mean()
        return (price / ma_200 - 1.0).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)
