"""Shared regime names and label ids.

The ids are intentionally simple because replay-buffer diagnostics only need a
stable equality test between the current regime and sampled transition regimes.
"""

from __future__ import annotations

REGIME_NAME_TO_LABEL = {
    "risk_on": 0,
    "sideways": 1,
    "high_vol": 2,
    "risk_off": 3,
}

LABEL_TO_REGIME_NAME = {value: key for key, value in REGIME_NAME_TO_LABEL.items()}


def label_for_name(name: str) -> int:
    return REGIME_NAME_TO_LABEL.get(name, REGIME_NAME_TO_LABEL["sideways"])


def name_for_label(label: int) -> str:
    return LABEL_TO_REGIME_NAME.get(int(label), "sideways")
