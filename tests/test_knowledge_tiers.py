"""
tests/test_knowledge_tiers.py
Unit tests verifying the information boundaries and mathematical properties of Knowledge Tiers K0, K1, and K2.
Reference: IEEE TIFS Manuscript §III-B, Supplementary §S4.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from attacks.knowledge import KnowledgeTier, ValidationEstimateProvider, NSLKDD_VAL_QUOTAS


class TestKnowledgeTiers(unittest.TestCase):
    def setUp(self):
        self.num_features = 20
        self.val_quotas = {0: 100, 1: 70, 2: 20, 3: 8, 4: 2}  # 200 total for fast fixture
        self.total_samples = 200

        # Synthetic server val data (D_val)
        rng_val = np.random.default_rng(100)
        self.server_val_X = rng_val.normal(size=(200, self.num_features)).astype(np.float32)
        val_y = []
        for c, count in self.val_quotas.items():
            val_y.extend([c] * count)
        self.server_val_y = np.array(val_y, dtype=np.int64)

        # Synthetic background data (training pool outside D_val and D_tune)
        rng_bg = np.random.default_rng(200)
        self.bg_X = rng_bg.normal(size=(1000, self.num_features)).astype(np.float32)
        self.bg_y = rng_bg.choice(5, size=1000, p=[0.5, 0.35, 0.1, 0.04, 0.01])

        # Synthetic coalition local data (clients 0, 1, 2)
        rng_c0 = np.random.default_rng(300)
        rng_c1 = np.random.default_rng(301)
        self.coalition_data = [
            (rng_c0.normal(size=(150, self.num_features)).astype(np.float32),
             rng_c0.choice(5, size=150, p=[0.5, 0.4, 0.08, 0.015, 0.005])),
            (rng_c1.normal(size=(150, self.num_features)).astype(np.float32),
             rng_c1.choice(5, size=150, p=[0.6, 0.3, 0.08, 0.015, 0.005])),
        ]

    def test_k0_boundary_and_zero_server_val_leakage(self):
        """K0 must use only coalition data and be 100% invariant to server_val_data."""
        # Generate with server_val_data provided (should be completely ignored by K0)
        X_k0_a, y_k0_a = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K0,
            coalition_data=self.coalition_data,
            server_val_data=(self.server_val_X, self.server_val_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )

        # Completely replace/corrupt server_val_data
        corrupted_val_X = np.ones_like(self.server_val_X) * 999.0
        corrupted_val_y = np.zeros_like(self.server_val_y)
        X_k0_b, y_k0_b = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K0,
            coalition_data=self.coalition_data,
            server_val_data=(corrupted_val_X, corrupted_val_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )

        # Output must be byte-for-byte identical (proving zero leakage of D_val into K0)
        np.testing.assert_array_equal(X_k0_a, X_k0_b)
        np.testing.assert_array_equal(y_k0_a, y_k0_b)
        self.assertEqual(len(X_k0_a), self.total_samples)

    def test_k0_raises_without_coalition_data(self):
        """K0 must reject calls that do not provide coalition data."""
        with self.assertRaises(ValueError):
            ValidationEstimateProvider.get_validation_estimate(
                tier=KnowledgeTier.K0,
                coalition_data=None,
            )

    def test_k1_boundary_and_zero_server_val_leakage(self):
        """K1 must use only background training pool and be 100% invariant to server_val_data."""
        X_k1_a, y_k1_a = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K1,
            background_data=(self.bg_X, self.bg_y),
            server_val_data=(self.server_val_X, self.server_val_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )

        corrupted_val_X = np.ones_like(self.server_val_X) * -777.0
        corrupted_val_y = np.zeros_like(self.server_val_y)
        X_k1_b, y_k1_b = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K1,
            background_data=(self.bg_X, self.bg_y),
            server_val_data=(corrupted_val_X, corrupted_val_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )

        np.testing.assert_array_equal(X_k1_a, X_k1_b)
        np.testing.assert_array_equal(y_k1_a, y_k1_b)
        self.assertEqual(len(X_k1_a), self.total_samples)

    def test_k1_raises_without_background_data(self):
        """K1 must reject calls that do not provide background data."""
        with self.assertRaises(ValueError):
            ValidationEstimateProvider.get_validation_estimate(
                tier=KnowledgeTier.K1,
                background_data=None,
            )

    def test_k2_returns_exact_server_val(self):
        """K2 must return exact server validation data D_val."""
        X_k2, y_k2 = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K2,
            server_val_data=(self.server_val_X, self.server_val_y),
            seed=42,
        )
        np.testing.assert_array_equal(X_k2, self.server_val_X)
        np.testing.assert_array_equal(y_k2, self.server_val_y)

    def test_k2_raises_without_server_val(self):
        """K2 must reject calls that do not provide server validation data."""
        with self.assertRaises(ValueError):
            ValidationEstimateProvider.get_validation_estimate(
                tier=KnowledgeTier.K2,
                server_val_data=None,
            )

    def test_quota_sampling_exactness(self):
        """Resampling must satisfy requested class counts when source classes are present."""
        X_sample, y_sample = ValidationEstimateProvider.sample_with_quotas(
            self.bg_X, self.bg_y, self.val_quotas, total_samples=self.total_samples, seed=42
        )
        self.assertEqual(len(X_sample), self.total_samples)
        for c, count in self.val_quotas.items():
            self.assertEqual(np.sum(y_sample == c), count)

    def test_seed_determinism(self):
        """Same seed must produce identical estimates; different seed produces different ordering."""
        X1, y1 = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K1,
            background_data=(self.bg_X, self.bg_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )
        X2, y2 = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K1,
            background_data=(self.bg_X, self.bg_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=42,
        )
        X3, y3 = ValidationEstimateProvider.get_validation_estimate(
            tier=KnowledgeTier.K1,
            background_data=(self.bg_X, self.bg_y),
            val_quotas=self.val_quotas,
            total_samples=self.total_samples,
            seed=999,
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        self.assertFalse(np.array_equal(X1, X3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
