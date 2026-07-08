import unittest

import pandas as pd

from src.rl_trading.performance_gate import (
    PerformanceGateConfig,
    apply_performance_gate,
    select_best_policies,
    select_naive_policies,
)


class PerformanceGateSelectionTests(unittest.TestCase):
    def test_naive_and_gated_selection_can_diverge(self):
        summary = pd.DataFrame(
            [
                {
                    "label_method": "rule_based",
                    "seed": 0,
                    "model": "sac",
                    "replay": "deer",
                    "final_portfolio_value": 1.40,
                    "max_drawdown": 0.50,
                    "mean_turnover": 0.20,
                },
                {
                    "label_method": "rule_based",
                    "seed": 0,
                    "model": "sac",
                    "replay": "regime",
                    "final_portfolio_value": 1.25,
                    "max_drawdown": 0.20,
                    "mean_turnover": 0.10,
                },
            ]
        )
        gated_input = apply_performance_gate(
            summary,
            PerformanceGateConfig(min_final_value=0.90, max_drawdown=0.35, max_turnover=1.25),
        )

        naive = select_naive_policies(gated_input)
        gated = select_best_policies(gated_input)

        self.assertEqual(naive.loc[0, "replay"], "deer")
        self.assertEqual(naive.loc[0, "selection_mode"], "naive")
        self.assertEqual(gated.loc[0, "replay"], "regime")
        self.assertEqual(gated.loc[0, "selection_mode"], "gated")
        self.assertFalse(bool(naive.loc[0, "passes_performance_gate"]))
        self.assertTrue(bool(gated.loc[0, "passes_performance_gate"]))


if __name__ == "__main__":
    unittest.main()
