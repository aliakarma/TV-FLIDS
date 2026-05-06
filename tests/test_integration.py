"""
Tests: end-to-end simulation sanity checks.
Run: python tests/test_integration.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEndToEnd(unittest.TestCase):
    def test_fedavg_produces_metrics(self):
        from experiments.run_experiment import run_experiment
        result = run_experiment(
            strategy_name="fedavg",
            attack_config_name="no_attack",
            seed=42,
            num_rounds=2,
            verbose=False,
        )
        self.assertIn(
            "final_accuracy",
            result,
            "final_accuracy missing - evaluate_fn not running",
        )
        self.assertGreater(result["final_accuracy"], 0.0)
        self.assertIn("final_f1_macro", result)

    def test_tvflids_produces_metrics(self):
        from experiments.run_experiment import run_experiment
        result = run_experiment(
            strategy_name="tvflids",
            attack_config_name="label_flip_30",
            seed=42,
            num_rounds=2,
            verbose=False,
        )
        self.assertGreater(result.get("final_accuracy", 0), 0.0)
        self.assertIn("final_attack_success_rate", result)

    def test_tvflids_outperforms_fedavg_under_attack(self):
        from experiments.run_experiment import run_experiment
        r_fedavg = run_experiment(
            "fedavg",
            "label_flip_30",
            seed=42,
            num_rounds=20,
            verbose=False,
        )
        r_tvflids = run_experiment(
            "tvflids",
            "label_flip_30",
            seed=42,
            num_rounds=20,
            verbose=False,
        )
        self.assertGreaterEqual(
            r_tvflids.get("final_accuracy", 0),
            r_fedavg.get("final_accuracy", 0),
            "TV-FLIDS worse than FedAvg - check trust/verification logic",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
