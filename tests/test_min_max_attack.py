"""
tests/test_min_max_attack.py
Unit tests verifying the Min-Max poisoning attack (Shejwalkar & Houmansadr, NDSS 2021).
Reference: IEEE TIFS Manuscript §III-C line 145, Supplementary §S4.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from attacks.adversarial import (
    AdversarialAttackFactory,
    apply_min_max_attack_to_params,
    apply_round_attacks,
)


class TestMinMaxAttack(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dim = 100
        self.global_params = [self.rng.normal(size=(self.dim,)).astype(np.float32)]

        self.honest_updates = [
            [self.rng.normal(loc=0.5, scale=0.1, size=(self.dim,)).astype(np.float32)]
            for _ in range(8)
        ]
        self.coalition_updates = [
            [self.rng.normal(loc=0.5, scale=0.1, size=(self.dim,)).astype(np.float32)]
            for _ in range(3)
        ]

    def test_min_max_constraint_satisfaction(self):
        """Min-Max attack must satisfy: max_{u in U} ||Delta_m - u|| <= max_{u, v in U} ||u - v||."""
        ref_updates = self.honest_updates
        flat_refs = np.array([AdversarialAttackFactory._flatten_params(u) for u in ref_updates])

        poisoned_params = AdversarialAttackFactory.min_max_attack(
            ref_updates, self.global_params, variant="omniscient", gamma_init=2.0
        )
        mal_delta = AdversarialAttackFactory._flatten_params(
            [p - g for p, g in zip(poisoned_params, self.global_params)]
        )

        # Compute max pairwise distance among reference updates
        dists = []
        for i in range(len(flat_refs)):
            for j in range(i + 1, len(flat_refs)):
                dists.append(np.linalg.norm(flat_refs[i] - flat_refs[j]))
        max_pairwise_dist = max(dists)

        cand_max_dist = max(np.linalg.norm(mal_delta - u) for u in flat_refs)

        self.assertLessEqual(
            cand_max_dist, max_pairwise_dist + 1e-4,
            f"Malicious max distance {cand_max_dist:.4f} exceeded reference bound {max_pairwise_dist:.4f}"
        )

    def test_min_max_perturbation_direction(self):
        """Min-Max must displace in the inverse standard-deviation direction."""
        ref_updates = self.honest_updates
        flat_refs = np.array([AdversarialAttackFactory._flatten_params(u) for u in ref_updates])
        mean_ref = np.mean(flat_refs, axis=0)

        poisoned_params = AdversarialAttackFactory.min_max_attack(
            ref_updates, self.global_params, variant="partial"
        )
        mal_delta = AdversarialAttackFactory._flatten_params(
            [p - g for p, g in zip(poisoned_params, self.global_params)]
        )

        displacement = mal_delta - mean_ref
        self.assertLess(float(np.dot(displacement, mean_ref)), 0.0,
                        "Min-Max displacement should oppose the honest aggregate direction.")

    def test_min_max_partial_vs_omniscient_isolation(self):
        """Partial mode must only use coalition updates; omniscient mode uses all honest updates."""
        all_client_params = [
            [g + u for g, u in zip(self.global_params, upd)]
            for upd in (self.honest_updates + self.coalition_updates)
        ]
        client_ids = list(range(len(all_client_params)))
        malicious_ids = [8, 9, 10]

        params_partial = [p.copy() if False else [x.copy() for x in p] for p in all_client_params]
        params_partial = apply_min_max_attack_to_params(
            params_partial, self.global_params, client_ids, malicious_ids, variant="partial"
        )

        params_omni = [p.copy() if False else [x.copy() for x in p] for p in all_client_params]
        params_omni = apply_min_max_attack_to_params(
            params_omni, self.global_params, client_ids, malicious_ids, variant="omniscient"
        )

        for cid in range(8):
            np.testing.assert_array_equal(params_partial[cid][0], all_client_params[cid][0])
            np.testing.assert_array_equal(params_omni[cid][0], all_client_params[cid][0])

        for cid in malicious_ids:
            self.assertFalse(np.allclose(params_partial[cid][0], all_client_params[cid][0]))
            self.assertFalse(np.allclose(params_omni[cid][0], all_client_params[cid][0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
