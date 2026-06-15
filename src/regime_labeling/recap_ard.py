"""ReCAP-inspired adaptive regime detection labels.

ReCAP's ARD module uses CUSUM-style change detection on market-level features.
Here we keep the same labeling spirit for replay diagnostics: detect adaptive
segments, then assign each segment to a small interpretable regime taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import MarketFeatureConfig, build_market_features, feature_matrix
from .taxonomy import label_for_name


@dataclass(frozen=True)
class RecapCusumConfig:
    feature_columns: tuple[str, ...] = ("ret_short", "ret_long", "trend", "vol", "vix", "turbulence")
    reference_window: int = 60
    min_segment: int = 20
    drift: float = 0.25
    threshold: float = 8.0


def label_recap_cusum(
    df: pd.DataFrame,
    feature_config: MarketFeatureConfig | None = None,
    cusum_config: RecapCusumConfig | None = None,
) -> pd.DataFrame:
    cusum_config = cusum_config or RecapCusumConfig()
    features = build_market_features(df, feature_config)
    matrix, selected, _, _ = feature_matrix(features, cusum_config.feature_columns, standardize=False)
    change_points = _detect_change_points(matrix, cusum_config)

    segment_id = np.zeros(len(features), dtype=int)
    current_segment = 0
    change_flags = np.zeros(len(features), dtype=bool)
    for idx in range(len(features)):
        if idx in change_points:
            current_segment += 1
            change_flags[idx] = True
        segment_id[idx] = current_segment

    names = _name_segments(features, segment_id)
    labels = np.array([label_for_name(name) for name in names], dtype=int)

    out = features.copy()
    out["method"] = "recap_cusum"
    out["regime_label"] = labels
    out["regime_name"] = names
    out["segment_id"] = segment_id
    out["change_point"] = change_flags
    out["cusum_features"] = ",".join(selected)
    leading = ["date", "method", "regime_label", "regime_name", "segment_id", "change_point"]
    return out[[*leading, *[column for column in out.columns if column not in leading]]]


def _detect_change_points(matrix: np.ndarray, config: RecapCusumConfig) -> set[int]:
    n_samples, n_features = matrix.shape
    change_points: set[int] = set()
    pos = np.zeros(n_features)
    neg = np.zeros(n_features)
    last_change = 0

    start = max(config.reference_window, 2)
    for idx in range(start, n_samples):
        if idx - last_change < config.min_segment:
            continue

        ref_start = max(last_change, idx - config.reference_window)
        reference = matrix[ref_start:idx]
        if len(reference) < max(5, min(config.reference_window, config.min_segment) // 2):
            continue
        means = reference.mean(axis=0)
        scales = reference.std(axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        z = (matrix[idx] - means) / scales

        pos = np.maximum(0.0, pos + z - config.drift)
        neg = np.maximum(0.0, neg - z - config.drift)
        if np.any(pos > config.threshold) or np.any(neg > config.threshold):
            change_points.add(idx)
            pos[:] = 0.0
            neg[:] = 0.0
            last_change = idx

    return change_points


def _name_segments(features: pd.DataFrame, segment_id: np.ndarray) -> np.ndarray:
    vol_cutoff = features["vol"].quantile(0.75)
    vix_cutoff = features["vix"].quantile(0.75)
    segment_names: dict[int, str] = {}

    for segment in np.unique(segment_id):
        rows = features[segment_id == segment]
        ret = float(rows["ret_long"].mean())
        trend = float(rows["trend"].mean())
        vol = float(rows["vol"].mean())
        vix = float(rows["vix"].mean())
        high_vol = (vol >= vol_cutoff) or (vix >= vix_cutoff)

        if high_vol and (ret < 0.0 or trend < 0.0):
            name = "risk_off"
        elif high_vol:
            name = "high_vol"
        elif ret > 0.0 and trend > 0.0:
            name = "risk_on"
        elif ret < 0.0 or trend < 0.0:
            name = "risk_off"
        else:
            name = "sideways"
        segment_names[int(segment)] = name

    return np.array([segment_names[int(segment)] for segment in segment_id], dtype=object)
