from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regime_labeling.hmm import HMMConfig, label_hmm
from regime_labeling.recap_ard import RecapCusumConfig, label_recap_cusum
from regime_labeling.rule_based import RuleBasedConfig, label_rule_based


class LabelerSmokeTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=160, freq="B")
        first = np.linspace(0.01, 0.08, 80)
        second = np.linspace(-0.02, -0.12, 80)
        ret_long = np.r_[first, second]
        vol = np.r_[np.full(80, 0.01), np.full(80, 0.05)]
        vix = np.r_[np.full(80, 14.0), np.full(80, 35.0)]
        trend = np.r_[np.full(80, 0.05), np.full(80, -0.08)]
        self.df = pd.DataFrame(
            {
                "date": dates,
                "ret_20d_SPY": ret_long / 2.0,
                "ret_60d_SPY": ret_long,
                "vol_20d_SPY": vol,
                "trend_price_200_SPY": trend,
                "adjclose_VIX": vix,
            }
        )

    def test_rule_based_outputs_labels(self):
        labels = label_rule_based(
            self.df,
            rule_config=RuleBasedConfig(lookback=40, min_periods=10),
        )
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue({"date", "method", "regime_label", "regime_name"}.issubset(labels.columns))
        self.assertGreater(labels["regime_name"].nunique(), 1)

    def test_hmm_outputs_probabilities(self):
        labels = label_hmm(self.df, hmm_config=HMMConfig(n_states=3, n_iter=8))
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue({"hmm_state", "prob_label_0", "prob_label_1", "prob_label_2"}.issubset(labels.columns))
        probs = labels[["prob_label_0", "prob_label_1", "prob_label_2"]].sum(axis=1)
        self.assertTrue(np.allclose(probs, 1.0))

    def test_recap_cusum_outputs_segments(self):
        labels = label_recap_cusum(
            self.df,
            cusum_config=RecapCusumConfig(reference_window=20, min_segment=10, threshold=4.0),
        )
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue({"segment_id", "change_point"}.issubset(labels.columns))
        self.assertGreaterEqual(labels["segment_id"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
