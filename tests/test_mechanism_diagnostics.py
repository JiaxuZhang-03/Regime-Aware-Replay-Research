import unittest

import numpy as np

from src.rl_trading.dqn_replay import PERBuffer, probe_metrics


def transition(i: int) -> dict:
    return {
        "state": np.array([i, 0], dtype=np.float32),
        "action": 0,
        "reward": 0.01,
        "next_state": np.array([i + 1, 0], dtype=np.float32),
        "done": False,
        "regime_label": 0,
        "regime_name": "test",
        "boundary_id": 0,
        "time_index": i,
        "date": f"2020-01-{i + 1:02d}",
        "transition_id": i,
    }


class MechanismDiagnosticsTests(unittest.TestCase):
    def test_probe_metrics_detect_flip_and_margin(self):
        before = np.array([[2.0, 1.0], [0.0, 3.0]])
        after = np.array([[0.0, 2.0], [0.0, 4.0]])
        result = probe_metrics(before, after)
        self.assertEqual(result["action_flip_rate"], 0.5)
        self.assertGreater(result["q_drift"], 0.0)
        self.assertAlmostEqual(result["action_share_before_0"], 0.5)

    def test_transition_audit_tracks_sampling_and_priority_updates(self):
        buffer = PERBuffer(capacity=10, seed=7)
        for i in range(4):
            buffer.add(transition(i))
        batch = buffer.sample(4, current_step=9)
        buffer.update_priorities(batch["indices"], np.ones(4), current_step=9)
        audit = buffer.transition_diagnostics(final_step=9)
        self.assertEqual(int(audit["sample_count"].sum()), 4)
        self.assertEqual(int(audit["priority_update_count"].sum()), 4)
        self.assertTrue((audit.loc[audit.sample_count > 0, "last_sample_step"] == 9).all())


if __name__ == "__main__":
    unittest.main()
