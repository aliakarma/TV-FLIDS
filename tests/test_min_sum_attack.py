"""
tests/test_min_sum_attack.py
Unit tests verifying the Min-Sum poisoning attack (Shejwalkar & Houmansadr, NDSS 2021).
Reference: IEEE TIFS Manuscript §III-C line 145, Supplementary §S4.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from attacks.adversarial import (
    AdversarialAttackFactory,
    apply_min_sum_attack_to_params,
    apply_round_attacks,
)


class TestMinSumAttack(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dim = 100
        self.global_params = [self.rng.normal(size=(self.dim,)).astype(np.float32)]

        # 8 honest updates with positive mean
        self.honest_updates = [
            [self.rng.normal(loc=0.5, scale=0.1, size=(self.dim,)).astype(np.float32)]
            for _ in range(8)
        ]
        # 3 coalition updates
        self.coalition_updates = [
            [self.rng.normal(loc=0.5, scale=0.1, size=(self.dim,)).astype(np.float32)]
            for _ in range(3)
        ]

    def test_min_sum_constraint_satisfaction(self):
        """Min-Sum attack must satisfy: sum_{u in U} ||Delta_m - u|| <= max_{v in U} sum_{u in U} ||v - u||."""
        ref_updates = self.honest_updates
        flat_refs = np.array([AdversarialAttackFactory._flatten_params(u) for u in ref_updates])

        poisoned_params = AdversarialAttackFactory.min_sum_attack(
            ref_updates, self.global_params, variant="omniscient"
        )
        mal_delta = AdversarialAttackFactory._flatten_params(
            [p - g for p, g in zip(poisoned_params, self.global_params)]
        )

        # Compute max sum of distances among reference updates
        max_ref_sum_dist = max(sum(np.linalg.norm(v - u) for u in flat_refs) for v in flat_refs)
        mal_sum_dist = sum(np.linalg.norm(mal_delta - u) for u in flat_refs)

        self.assertLessEqual(
            mal_sum_dist, max_ref_sum_dist + 1e-4,
            f"Malicious sum distance {mal_sum_dist:.4f} exceeded reference bound {max_ref_sum_dist:.4f}"
        )

    def test_min_sum_perturbation_direction(self):
        """Min-Sum must displace in the inverse standard-deviation direction (-sgn(mean))."""
        ref_updates = self.honest_updates
        flat_refs = np.array([AdversarialAttackFactory._flatten_params(u) for u in ref_updates])
        mean_ref = np.mean(flat_refs, axis=0)

        poisoned_params = AdversarialAttackFactory.min_sum_attack(
            ref_updates, self.global_params, variant="partial"
        )
        mal_delta = AdversarialAttackFactory._flatten_params(
            [p - g for p, g in zip(poisoned_params, self.global_params)]
        )

        displacement = mal_delta - mean_ref
        # Since mean_ref is positive (~0.5), displacement should be strictly negative
        self.assertLess(float(np.dot(displacement, mean_ref)), 0.0,
                        "Min-Sum displacement should oppose the honest aggregate direction.")

    def test_min_sum_partial_vs_omniscient_isolation(self):
        """Partial mode must only use coalition updates; omniscient mode uses all honest updates."""
        all_client_params = [
            [g + u for g, u in zip(self.global_params, upd)]
            for upd in (self.honest_updates + self.coalition_updates)
        ]
        client_ids = list(range(len(all_client_params)))
        malicious_ids = [8, 9, 10]

        # Partial run
        params_partial = [p.copy() if False else [x.copy() for x in p] for p in all_client_params]
        params_partial = apply_min_sum_attack_to_params(
            params_partial, self.global_params, client_ids, malicious_ids, variant="partial"
        )

        # Omniscient run
        params_omni = [p.copy() if False else [x.copy() for x in p] for p in all_client_params]
        params_omni = apply_min_sum_attack_to_params(
            params_omni, self.global_params, client_ids, malicious_ids, variant="omniscient"
        )

        # Honest clients must remain untouched in both
        for cid in range(8):
            np.testing.assert_array_equal(params_partial[cid][0], all_client_params[cid][0])
            np.testing.assert_array_equal(params_omni[cid][0], all_client_params[cid][0])

        # Malicious updates must be modified
        for cid in malicious_ids:
            self.assertFalse(np.allclose(params_partial[cid][0], all_client_params[cid][0]))
            self.assertFalse(np.allclose(params_omni[cid][0], all_client_params[cid][0]))

    def test_apply_round_attacks_dispatch_min_sum(self):
        """apply_round_attacks must dispatch min_sum attack correctly."""
        all_client_params = [
            [g + u for g, u in zip(self.global_params, upd)]
            for upd in (self.honest_updates + self.coalition_updates)
        ]
        client_ids = list(range(len(all_client_params)))
        malicious_ids = [8, 9, 10]

        attacked = apply_round_attacks(
            client_params=all_client_params,
            global_params=self.global_params,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type="min_sum",
            attack_kwargs={"variant": "partial"},
        )

        for cid in malicious_ids:
            self.assertFalse(np.allclose(attacked[cid][0], self.global_params[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
