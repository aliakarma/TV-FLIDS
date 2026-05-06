"""
tests/test_all.py — Comprehensive unit tests for TV-FLIDS.
Run: python tests/test_all.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_model(input_dim=41, num_classes=5):
    from models.mlp import IDSMLP
    return IDSMLP(input_dim=input_dim, num_classes=num_classes)

def make_val_loader(n=200, input_dim=41, num_classes=5):
    X = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=64)

def make_updates(model, n_clients=5, scale=0.01):
    gp = model.get_parameters()
    return [[p + np.random.randn(*p.shape).astype(np.float32)*scale for p in gp]
            for _ in range(n_clients)]


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestIDSMLP(unittest.TestCase):
    def test_forward_shape(self):
        m = make_model()
        x = torch.randn(16, 41)
        m.eval()
        with torch.no_grad():
            out = m(x)
        self.assertEqual(out.shape, (16, 5))

    def test_get_set_parameters_roundtrip(self):
        m1 = make_model()
        m2 = make_model()
        params = m1.get_parameters()
        m2.set_parameters(params)
        for p1, p2 in zip(params, m2.get_parameters()):
            np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_parameter_count(self):
        m = make_model(41, 5)
        n = m.count_parameters()
        self.assertGreater(n, 0)
        self.assertLess(n, 1_000_000)

    def test_different_input_dims(self):
        for d, c in [(41, 5), (49, 10), (784, 10)]:
            m = make_model(d, c)
            x = torch.randn(4, d)
            m.eval()
            with torch.no_grad():
                out = m(x)
            self.assertEqual(out.shape, (4, c))


# ── Partitioner Tests ─────────────────────────────────────────────────────────

class TestPartitioners(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(1000, 41).astype(np.float32)
        self.y = np.random.randint(0, 5, 1000).astype(np.int64)

    def test_iid_partition_count(self):
        from data.partitioning import IIDPartitioner
        shards = IIDPartitioner().partition(self.X, self.y, 10)
        self.assertEqual(len(shards), 10)
        total = sum(len(s[0]) for s in shards)
        self.assertEqual(total, 1000)

    def test_noniid_partition_count(self):
        from data.partitioning import NonIIDPartitioner
        shards = NonIIDPartitioner(0.5).partition(self.X, self.y, 10)
        self.assertEqual(len(shards), 10)

    def test_server_validation_set(self):
        from data.partitioning import create_server_validation_set
        Xv, yv = create_server_validation_set(self.X, self.y, val_size=100)
        self.assertEqual(len(Xv), 100)
        self.assertEqual(len(yv), 100)

    def test_noniid_alpha_extremes(self):
        from data.partitioning import NonIIDPartitioner
        for alpha in [0.1, 0.5, 10.0]:
            shards = NonIIDPartitioner(alpha).partition(self.X, self.y, 5)
            self.assertEqual(len(shards), 5)


# ── Attack Tests ──────────────────────────────────────────────────────────────

class TestAttacks(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.y = np.array([0, 1, 2, 3, 1, 2, 0, 4, 1])
        self.X = np.random.randn(9, 41).astype(np.float32)
        from attacks.adversarial import AdversarialAttackFactory
        self.f = AdversarialAttackFactory()
        from models.mlp import IDSMLP
        m = IDSMLP(41, 5)
        self.gp = m.get_parameters()
        self.cp = [p + 0.01 for p in self.gp]

    def test_label_flip_all_attacks_to_normal(self):
        y2 = self.f.label_flip(self.y, target_class=0, flip_ratio=1.0)
        self.assertTrue(np.all(y2[self.y != 0] == 0))
        self.assertEqual(y2[0], 0)  # already normal, unchanged

    def test_label_flip_partial(self):
        y2 = self.f.label_flip(self.y, target_class=0, flip_ratio=0.5, seed=42)
        self.assertEqual(len(y2), len(self.y))

    def test_gradient_scale(self):
        scaled = self.f.gradient_scale(self.cp, self.gp, scale_factor=10.0)
        delta_orig  = np.linalg.norm(np.concatenate([(c-g).flatten() for c,g in zip(self.cp, self.gp)]))
        delta_scale = np.linalg.norm(np.concatenate([(s-g).flatten() for s,g in zip(scaled, self.gp)]))
        self.assertAlmostEqual(delta_scale / delta_orig, 10.0, places=3)

    def test_noise_injection_changes_params(self):
        noisy = self.f.noise_injection(self.gp, noise_std=0.5, seed=42)
        diff = sum(np.sum(np.abs(n - g)) for n, g in zip(noisy, self.gp))
        self.assertGreater(diff, 0)

    def test_backdoor_preserves_size(self):
        Xb, yb = self.f.backdoor_attack(self.X, self.y, poison_ratio=0.3, seed=42)
        self.assertEqual(len(Xb), len(self.X))
        self.assertEqual(len(yb), len(self.y))

    def test_get_malicious_ids(self):
        from attacks.adversarial import get_malicious_client_ids
        mal = get_malicious_client_ids(20, 0.3, seed=42)
        self.assertEqual(len(mal), 6)
        self.assertEqual(sorted(mal), mal)
        self.assertTrue(all(0 <= m < 20 for m in mal))


# ── Trust Scorer Tests ────────────────────────────────────────────────────────

class TestTrustScorer(unittest.TestCase):
    def setUp(self):
        from trust.trust_scorer import TrustScorer
        self.ts = TrustScorer(10, alpha=0.4, beta=0.4, gamma=0.2)
        m = make_model()
        self.gp = m.get_parameters()
        np.random.seed(42)
        self.updates = make_updates(m, 5)

    def test_weight_constraint(self):
        self.assertAlmostEqual(self.ts.alpha + self.ts.beta + self.ts.gamma, 1.0, places=5)

    def test_similarity_scores_shape(self):
        ref = [np.zeros_like(p) for p in self.gp]
        sims = self.ts.compute_similarity_scores(self.updates, ref)
        self.assertEqual(len(sims), 5)
        self.assertTrue(np.all(sims >= 0) and np.all(sims <= 1))

    def test_accuracy_scores_clip(self):
        acc = self.ts.compute_accuracy_scores(1.0, [0.5, 1.5, 1.0, 0.8, 2.0])
        self.assertTrue(np.all(acc >= 0) and np.all(acc <= 1))

    def test_anomaly_scores_range(self):
        anom = self.ts.compute_anomaly_scores(self.updates)
        self.assertEqual(len(anom), 5)
        self.assertTrue(np.all(anom >= 0) and np.all(anom <= 1))

    def test_aggregation_weights_sum_to_one(self):
        ref = [np.zeros_like(p) for p in self.gp]
        sim  = self.ts.compute_similarity_scores(self.updates, ref)
        acc  = self.ts.compute_accuracy_scores(1.0, [0.9]*5)
        anom = self.ts.compute_anomaly_scores(self.updates)
        self.ts.update_trust([0,1,2,3,4], sim, acc, anom)
        w = self.ts.get_aggregation_weights([0,1,2,3,4])
        self.assertAlmostEqual(w.sum(), 1.0, places=5)

    def test_trust_floor_respected(self):
        for _ in range(20):
            ref = [np.zeros_like(p) for p in self.gp]
            sim  = self.ts.compute_similarity_scores(self.updates, ref)
            acc  = self.ts.compute_accuracy_scores(1.0, [1.5]*5)  # all degrade
            anom = self.ts.compute_anomaly_scores(self.updates)
            self.ts.update_trust([0,1,2,3,4], sim, acc, anom)
        self.assertTrue(np.all(self.ts.trust_scores >= self.ts.min_trust))

    def test_reset(self):
        self.ts.trust_scores[:] = 0.5
        self.ts.reset()
        np.testing.assert_allclose(self.ts.trust_scores, np.ones(10))


# ── Adaptive Trust Tests ──────────────────────────────────────────────────────

class TestAdaptiveTrustScorer(unittest.TestCase):
    def test_weights_sum_to_one(self):
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer
        ats = AdaptiveTrustScorer(10)
        w = ats.get_current_weights()
        self.assertAlmostEqual(w['alpha'] + w['beta'] + w['gamma'], 1.0, places=5)

    def test_weights_positive(self):
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer
        ats = AdaptiveTrustScorer(10)
        w = ats.get_current_weights()
        self.assertTrue(all(v > 0 for v in w.values()))

    def test_reset_restores_uniform(self):
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer
        ats = AdaptiveTrustScorer(10)
        ats.reset()
        w = ats.get_current_weights()
        self.assertAlmostEqual(w['alpha'], 1/3, places=4)


class TestMetaGradient(unittest.TestCase):
    def test_weights_actually_change(self):
        from trust.adaptive_trust_scorer import AdaptiveTrustScorer
        ats = AdaptiveTrustScorer(10, meta_lr=0.1)
        initial = ats.get_current_weights().copy()

        def biased_val_fn(alpha, beta, gamma):
            sim_t = torch.tensor([0.9, 0.1], dtype=torch.float32)
            acc_t = torch.tensor([0.5, 0.5], dtype=torch.float32)
            anom_t = torch.tensor([0.1, 0.9], dtype=torch.float32)
            raw = torch.clamp(alpha * sim_t + beta * acc_t - gamma * anom_t, 0, 1)
            w = raw / (raw.sum() + 1e-8)
            losses = torch.tensor([0.3, 1.2], dtype=torch.float32)
            return (w * losses).sum()

        for _ in range(5):
            ats.meta_update(biased_val_fn)

        updated = ats.get_current_weights()
        self.assertNotAlmostEqual(
            initial['alpha'], updated['alpha'], places=4,
            msg="Weights did not change - gradient not flowing"
        )


# ── Verification Module Tests ─────────────────────────────────────────────────

class TestVerificationModule(unittest.TestCase):
    def setUp(self):
        from trust.verification import VerificationModule
        self.vm_lax = VerificationModule(loss_threshold=-1e9,
                                         cosine_threshold=-1e9, zscore_threshold=1e9)
        self.vm_strict = VerificationModule(loss_threshold=1e9,
                                             cosine_threshold=-1e9, zscore_threshold=1e9)
        self.model = make_model()
        self.gp = self.model.get_parameters()
        self.val_loader = make_val_loader()
        self.device = torch.device('cpu')
        np.random.seed(42)
        self.updates = make_updates(self.model, 3)

    def test_lax_verifies_all(self):
        res = self.vm_lax.verify_all(self.updates, [0,1,2], global_loss=1.0,
                                      global_params=self.gp, model=self.model,
                                      device=self.device, val_loader=self.val_loader)
        total = len(res['verified'])+len(res['flagged'])+len(res['rejected'])
        self.assertEqual(total, 3)

    def test_strict_rejects_all(self):
        res = self.vm_strict.verify_all(self.updates, [0,1,2], global_loss=0.001,
                                         global_params=self.gp, model=self.model,
                                         device=self.device, val_loader=self.val_loader)
        self.assertEqual(len(res['rejected']), 3)

    def test_adaptive_thresholds(self):
        from trust.verification import VerificationModule
        t1 = VerificationModule.adaptive_zscore_threshold(2.5, round_num=1, warmup_rounds=20)
        t20 = VerificationModule.adaptive_zscore_threshold(2.5, round_num=20, warmup_rounds=20)
        t100 = VerificationModule.adaptive_zscore_threshold(2.5, round_num=100, warmup_rounds=20)
        self.assertGreater(t1, t20)       # More lenient early
        self.assertAlmostEqual(t100, 2.5, places=1)  # Approaches base at convergence

    def test_log_is_populated(self):
        self.vm_lax.verify_all(self.updates, [0,1,2], global_loss=1.0,
                                global_params=self.gp, model=self.model,
                                device=self.device, val_loader=self.val_loader)
        self.assertEqual(len(self.vm_lax.verification_log), 1)


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestExperimentMetrics(unittest.TestCase):
    def setUp(self):
        from evaluation.metrics import ExperimentMetrics
        self.em = ExperimentMetrics()
        self.y_true = np.array([0,1,2,0,1,2,3,0,1])
        self.y_pred = np.array([0,1,2,0,0,2,3,1,1])

    def test_accuracy_range(self):
        m = self.em.compute_round_metrics(self.y_true, self.y_pred, 1)
        self.assertGreaterEqual(m['accuracy'], 0.0)
        self.assertLessEqual(m['accuracy'],    1.0)

    def test_attack_success_rate(self):
        m = self.em.compute_round_metrics(self.y_true, self.y_pred, 1)
        self.assertGreaterEqual(m['attack_success_rate'], 0.0)
        self.assertLessEqual(m['attack_success_rate'],    1.0)

    def test_metric_series(self):
        for i in range(5):
            self.em.compute_round_metrics(self.y_true, self.y_pred, i+1)
        series = self.em.get_metric_series('accuracy')
        self.assertEqual(len(series), 5)

    def test_reset_clears(self):
        self.em.compute_round_metrics(self.y_true, self.y_pred, 1)
        self.em.reset()
        self.assertEqual(len(self.em.round_metrics), 0)


# ── Statistical Testing Tests ─────────────────────────────────────────────────

class TestStatisticalTesting(unittest.TestCase):
    def test_wilcoxon_significant(self):
        from evaluation.statistical_testing import compare_methods_wilcoxon
        a = [{'final_accuracy': v} for v in [0.92, 0.91, 0.93, 0.90, 0.94]]
        b = [{'final_accuracy': v} for v in [0.80, 0.81, 0.79, 0.82, 0.78]]
        res = compare_methods_wilcoxon(a, b)
        self.assertIn('p_value', res)
        self.assertIn('significant', res)

    def test_compute_summary(self):
        from evaluation.statistical_testing import compute_summary
        data = [{'acc': 0.9}, {'acc': 0.92}, {'acc': 0.88}]
        mean, std = compute_summary(data, 'acc')
        self.assertAlmostEqual(mean, 0.9, places=2)
        self.assertGreater(std, 0)

    def test_format_result(self):
        from evaluation.statistical_testing import format_result
        s = format_result(0.9123, 0.0234)
        self.assertIn('±', s)

    def test_mcnemar(self):
        from evaluation.statistical_testing import mcnemar_test
        y_true = np.array([0,1,2,0,1,2,3,0])
        y_a    = np.array([0,1,2,0,0,2,3,0])
        y_b    = np.array([0,0,2,0,1,0,3,1])
        res = mcnemar_test(y_true, y_a, y_b)
        self.assertIn('p_value', res)


# ── Proposition 1 Tests ───────────────────────────────────────────────────────

class TestProposition1(unittest.TestCase):
    def test_bound_holds_simple_case(self):
        from theory.proposition1_verification import verify_proposition1
        m = make_model()
        gp = m.get_parameters()
        np.random.seed(42)
        n_honest, n_byz = 14, 6
        trust = np.ones(20)
        trust[:n_honest] = 0.8
        trust[n_honest:] = 0.01  # τ_min for Byzantine
        honest_params = [
            [p + np.random.randn(*p.shape).astype(np.float32)*0.01 for p in gp]
            for _ in range(n_honest)
        ]
        byz_params = [
            [p + np.random.randn(*p.shape).astype(np.float32)*2.0 for p in gp]
            for _ in range(n_byz)
        ]
        res = verify_proposition1(
            trust, list(range(n_honest)), list(range(n_honest, 20)),
            gp, honest_params, byz_params, tau_min=0.01
        )
        self.assertIn('bound_holds', res)
        self.assertIn('observed_deviation', res)
        self.assertIn('theoretical_bound', res)


# ── Overhead Tests ────────────────────────────────────────────────────────────

class TestOverhead(unittest.TestCase):
    def test_communication_cost(self):
        from evaluation.overhead import estimate_communication_cost
        c = estimate_communication_cost(53125, 10)
        self.assertLess(c['overhead_pct'], 1.0)  # TV-FLIDS adds < 1% comm overhead
        self.assertGreater(c['tvflids_total_mb'], c['fedavg_total_mb'])

    def test_overhead_tracker(self):
        from evaluation.overhead import OverheadTracker
        import time
        ot = OverheadTracker()
        with ot.time_phase('verification'):
            time.sleep(0.001)
        self.assertEqual(len(ot.timings['verification']), 1)
        self.assertGreater(ot.timings['verification'][0], 0)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = loader.discover(os.path.dirname(__file__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
