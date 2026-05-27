"""
tests/test_smoke.py
Lightweight smoke test for CI environments.
Uses synthetic data to avoid downloading NSL-KDD.
Run: python tests/test_smoke.py
"""
import os
import sys
import unittest
import numpy as np
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSyntheticSmoke(unittest.TestCase):
    """Verify pipeline runs end-to-end on synthetic data."""

    def test_model_forward_backward(self):
        """MLP forward + loss backward runs without error."""
        import torch
        import torch.nn as nn
        from models.mlp import IDSMLP

        model = IDSMLP(input_dim=41, num_classes=5)
        x = torch.randn(32, 41)
        y = torch.randint(0, 5, (32,))
        out = model(x)
        self.assertEqual(out.shape, (32, 5))
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        self.assertFalse(torch.isnan(loss))

    def test_trust_scorer_full_cycle(self):
        """TrustScorer update → aggregation weights in a single round."""
        from models.mlp import IDSMLP
        from trust.trust_scorer import TrustScorer
        import numpy as np

        model = IDSMLP(41, 5)
        gp = model.get_parameters()
        ts = TrustScorer(num_clients=5)
        updates = [[p + np.random.randn(*p.shape).astype("float32") * 0.01
                    for p in gp] for _ in range(5)]
        ref = [np.zeros_like(p) for p in gp]
        sim  = ts.compute_similarity_scores(updates, ref)
        acc  = ts.compute_accuracy_scores(1.0, [0.8] * 5)
        anom = ts.compute_anomaly_scores(updates)
        ts.update_trust([0, 1, 2, 3, 4], sim, acc, anom)
        w = ts.get_aggregation_weights([0, 1, 2, 3, 4])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_meta_gradient_updates_weights(self):
        """AdaptiveTrustScorer weights change after 3 meta-gradient steps."""
        import torch
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer

        ats = AdaptiveTrustScorer(num_clients=5, meta_lr=0.1)
        initial = ats.get_current_weights()["alpha"]

        def val_fn(alpha, beta, gamma):
            sim  = torch.tensor([0.9, 0.1, 0.5], dtype=torch.float32)
            acc  = torch.tensor([0.8, 0.2, 0.5], dtype=torch.float32)
            anom = torch.tensor([0.1, 0.8, 0.3], dtype=torch.float32)
            raw = torch.clamp(alpha * sim + beta * acc - gamma * anom, 0, 1)
            w = raw / (raw.sum() + 1e-8)
            losses = torch.tensor([0.3, 1.2, 0.7], dtype=torch.float32)
            return (w * losses).sum()

        for _ in range(3):
            ats.meta_update(val_fn)

        updated = ats.get_current_weights()["alpha"]
        self.assertNotAlmostEqual(initial, updated, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
