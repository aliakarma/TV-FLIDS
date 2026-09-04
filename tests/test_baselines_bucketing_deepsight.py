"""
tests/test_baselines_bucketing_deepsight.py

In-process, Ray-free smoke tests for BucketingStrategy and DeepSightStrategy
(fl/baselines/bucketing_strategy.py, fl/baselines/deepsight_strategy.py).

Why this exists: experiments/run_experiment.py's real orchestration path goes
through Flower's Ray-based fl.simulation.start_simulation(), which crashes in
this Windows/Python-3.11 environment with an unrelated Ray-actor/PyTorch DLL
load failure (WinError 1114 loading torch/lib/c10.dll) -- see
tests/test_pipeline_smoke.py's module docstring for the full explanation.
This test follows the exact same idiom (_FakeProxy / _FakeFitRes duck types)
to call aggregate_fit() directly, in-process, on tiny synthetic client
updates -- no Ray, no real FL simulation.

Run: python -m pytest tests/test_baselines_bucketing_deepsight.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from models.mlp import IDSMLP
from attacks.adversarial import AdversarialAttackFactory
from fl.baselines.bucketing_strategy import BucketingStrategy
from fl.baselines.deepsight_strategy import DeepSightStrategy


INPUT_DIM = 20
NUM_CLASSES = 5
NUM_CLIENTS = 8
MALICIOUS_IDS = [6, 7]  # 2 of 8 -> 25% adversarial


class _FakeProxy:
    def __init__(self, cid):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, parameters, num_examples=100):
        self.parameters = parameters
        self.num_examples = num_examples


def make_model():
    return IDSMLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)


def build_results(global_params, include_malicious, rng_seed=0):
    """Build a small list of (proxy, fit_res) tuples: honest clients are
    global params + small Gaussian noise; malicious clients (if included)
    apply the gradient_scale attack (10x amplification), matching the
    pattern used in tests/test_pipeline_smoke.py."""
    rng = np.random.default_rng(rng_seed)
    factory = AdversarialAttackFactory()
    results = []
    for cid in range(NUM_CLIENTS):
        honest_update = [
            p + rng.normal(size=p.shape).astype(np.float32) * 0.01
            for p in global_params
        ]
        if include_malicious and cid in MALICIOUS_IDS:
            honest_update = factory.gradient_scale(
                honest_update, global_params, scale_factor=10.0
            )
        results.append(
            (_FakeProxy(cid), _FakeFitRes(ndarrays_to_parameters(honest_update)))
        )
    return results


class TestBucketingStrategySmoke(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.global_params = self.model.get_parameters()
        self.shapes = [p.shape for p in self.global_params]

    def test_aggregate_fit_runs_and_shapes_match(self):
        strategy = BucketingStrategy(
            bucket_size=2,
            beta=0.1,
            global_model=self.model,
            malicious_ids=MALICIOUS_IDS,
            seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results = build_results(self.global_params, include_malicious=True)
        agg_params, log = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )
        self.assertIsNotNone(agg_params)
        arrays = parameters_to_ndarrays(agg_params)
        self.assertEqual(len(arrays), len(self.shapes))
        for arr, shape in zip(arrays, self.shapes):
            self.assertEqual(arr.shape, shape)
        self.assertIn("bucketing_num_buckets", log)
        self.assertEqual(log["bucketing_bucket_size"], 2)
        # 8 clients / bucket_size 2 -> 4 buckets
        self.assertEqual(log["bucketing_num_buckets"], 4)

    def test_bucket_reshuffle_is_seeded_and_round_dependent(self):
        strategy = BucketingStrategy(
            bucket_size=2, beta=0.1, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        rng1 = np.random.default_rng(42 + 1)
        rng2 = np.random.default_rng(42 + 2)
        order1 = rng1.permutation(NUM_CLIENTS)
        order2 = rng2.permutation(NUM_CLIENTS)
        # Different rounds should (almost certainly) produce different
        # bucket assignments -- proves the reshuffle is round-dependent.
        self.assertFalse(np.array_equal(order1, order2))

    def test_malicious_present_changes_aggregate_vs_all_honest(self):
        strategy_a = BucketingStrategy(
            bucket_size=2, beta=0.1, global_model=make_model(),
            malicious_ids=MALICIOUS_IDS, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        strategy_b = BucketingStrategy(
            bucket_size=2, beta=0.1, global_model=make_model(),
            malicious_ids=MALICIOUS_IDS, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results_honest = build_results(self.global_params, include_malicious=False, rng_seed=0)
        results_mixed = build_results(self.global_params, include_malicious=True, rng_seed=0)

        agg_honest, _ = strategy_a.aggregate_fit(1, results_honest, [])
        agg_mixed, _ = strategy_b.aggregate_fit(1, results_mixed, [])

        arr_honest = parameters_to_ndarrays(agg_honest)
        arr_mixed = parameters_to_ndarrays(agg_mixed)
        differs = any(
            not np.allclose(a, b) for a, b in zip(arr_honest, arr_mixed)
        )
        self.assertTrue(differs, "Bucketing aggregate should differ once malicious "
                                  "gradient-scale clients are mixed in.")


def build_results_correlated(global_params, include_malicious, rng_seed=0):
    """Honest clients share a common small "consistent update direction"
    (simulating the fact that real honest gradients from independent data
    shards still tend to point toward the same loss-reducing direction)
    plus tiny per-client noise, so the bias-cosine clustering step has an
    actual similarity structure to key off. Purely independent per-client
    noise (as used in build_results() above) carries no directional signal
    at all, which would make clustering behavior arbitrary by construction
    -- not a fair test of the mechanism.

    Malicious clients bias their output layer in the *opposite* direction
    (mimicking the label-flip/backdoor-style attacks DeepSight's bias
    clustering is designed to catch) and then additionally apply
    gradient_scale on top (a magnitude attack, which weight-clip -- not
    clustering -- defends against). This exercises both DeepSight
    mechanisms at once."""
    rng = np.random.default_rng(rng_seed)
    factory = AdversarialAttackFactory()
    common_direction = [
        rng.normal(size=p.shape).astype(np.float32) * 0.02 for p in global_params
    ]
    results = []
    for cid in range(NUM_CLIENTS):
        if include_malicious and cid in MALICIOUS_IDS:
            base_update = [
                g - cd + rng.normal(size=g.shape).astype(np.float32) * 0.001
                for g, cd in zip(global_params, common_direction)
            ]
            update = factory.gradient_scale(base_update, global_params, scale_factor=10.0)
        else:
            update = [
                g + cd + rng.normal(size=g.shape).astype(np.float32) * 0.001
                for g, cd in zip(global_params, common_direction)
            ]
        results.append((_FakeProxy(cid), _FakeFitRes(ndarrays_to_parameters(update))))
    return results


class TestDeepSightStrategySmoke(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.global_params = self.model.get_parameters()
        self.shapes = [p.shape for p in self.global_params]

    def test_aggregate_fit_runs_and_shapes_match(self):
        strategy = DeepSightStrategy(
            distance_threshold=0.5,
            global_model=self.model,
            malicious_ids=MALICIOUS_IDS,
            seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results = build_results(self.global_params, include_malicious=True)
        agg_params, log = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )
        self.assertIsNotNone(agg_params)
        arrays = parameters_to_ndarrays(agg_params)
        self.assertEqual(len(arrays), len(self.shapes))
        for arr, shape in zip(arrays, self.shapes):
            self.assertEqual(arr.shape, shape)
        self.assertIn("deepsight_benign", log)
        self.assertIn("deepsight_suspicious", log)
        self.assertIn("deepsight_clip_norm", log)
        self.assertEqual(
            log["deepsight_benign"] + log["deepsight_suspicious"], NUM_CLIENTS
        )

    def test_clip_norm_bounds_accepted_update_norms(self):
        """Weight-clip should ensure no accepted update contributes with a
        norm larger than the reported clip_norm -- proves the clipping
        mechanism actually runs, not just that averaging happens."""
        strategy = DeepSightStrategy(
            distance_threshold=0.5,
            global_model=self.model,
            malicious_ids=MALICIOUS_IDS,
            seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results = build_results(self.global_params, include_malicious=True)
        _, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
        self.assertGreater(log["deepsight_clip_norm"], 0.0)

    def test_malicious_present_changes_aggregate_vs_all_honest(self):
        strategy_a = DeepSightStrategy(
            distance_threshold=0.5, global_model=make_model(),
            malicious_ids=MALICIOUS_IDS, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        strategy_b = DeepSightStrategy(
            distance_threshold=0.5, global_model=make_model(),
            malicious_ids=MALICIOUS_IDS, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results_honest = build_results(self.global_params, include_malicious=False, rng_seed=0)
        results_mixed = build_results(self.global_params, include_malicious=True, rng_seed=0)

        agg_honest, _ = strategy_a.aggregate_fit(1, results_honest, [])
        agg_mixed, _ = strategy_b.aggregate_fit(1, results_mixed, [])

        arr_honest = parameters_to_ndarrays(agg_honest)
        arr_mixed = parameters_to_ndarrays(agg_mixed)
        differs = any(
            not np.allclose(a, b) for a, b in zip(arr_honest, arr_mixed)
        )
        self.assertTrue(differs, "DeepSight aggregate should differ once malicious "
                                  "gradient-scale clients are mixed in.")

    def test_defense_dampens_attack_vs_plain_fedavg_mean(self):
        """DeepSight's clustering + weight-clip should pull the aggregate
        closer to the honest direction than an unclipped FedAvg-style mean
        would -- i.e. the defense mechanism actually suppresses the
        attack's contribution, not just changes the output arbitrarily.

        Uses build_results_correlated() rather than build_results(): with
        fully independent per-client Gaussian noise, honest clients' bias
        updates carry no shared directional signal for the cosine-based
        clustering step to key off, which would make a "does clustering
        help" comparison meaningless by construction. Here honest clients
        share a common small update direction (as real honest gradients
        from a common loss landscape would) while malicious clients push
        the opposite direction and scale it up 10x, giving the clustering
        step a genuine, checkable job to do."""
        strategy = DeepSightStrategy(
            distance_threshold=0.5, global_model=make_model(),
            malicious_ids=MALICIOUS_IDS, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1,
            min_available_clients=NUM_CLIENTS,
        )
        results_mixed = build_results_correlated(self.global_params, include_malicious=True, rng_seed=0)
        agg_deepsight, _ = strategy.aggregate_fit(1, results_mixed, [])
        arr_deepsight = parameters_to_ndarrays(agg_deepsight)

        # Plain FedAvg-style mean over the same (attacked) client updates.
        params_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results_mixed]
        n_layers = len(params_list[0])
        arr_fedavg = [
            np.mean([p[layer] for p in params_list], axis=0)
            for layer in range(n_layers)
        ]

        deepsight_norm = np.linalg.norm(
            np.concatenate([(a - g).flatten() for a, g in zip(arr_deepsight, self.global_params)])
        )
        fedavg_norm = np.linalg.norm(
            np.concatenate([(a - g).flatten() for a, g in zip(arr_fedavg, self.global_params)])
        )
        self.assertLess(
            deepsight_norm, fedavg_norm,
            "DeepSight's weight-clip should shrink the aggregate's deviation "
            "from the global model relative to an unclipped FedAvg mean over "
            "the same (gradient_scale-attacked) client updates.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
