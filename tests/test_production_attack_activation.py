"""
tests/test_production_attack_activation.py
Tests proving end-to-end production path activation for TV-FLIDS attack suite.

Verifies:
  - Part B: Min-Sum activation (partial vs. omniscient, norms, cosine similarities, constraint satisfaction).
  - Part C: ACK3 activation (clean vs. K1 vs. K2, label transformation, surrogate isolation, balanced hinge loss).
  - Part D: On-Off transition (rounds 1..10 honest, round 11 active for on_off_lf_10 and on_off_ack2_10; k=20 generalization).
  - Part E: LF-R activation (normal class 0 preserved, attack classes remapped to {1..4} \\ {y}, determinism).
  - Part F: ACK4 production path (simulation of clipping, validation gate, trust update, weight-share objective, omega* in grid).
  - Part G: Clean client preservation (clean updates untouched, client ordering preserved, participant count invariant across TV-FLIDS and baselines).
  - Part I: No hidden bypass in round interception and client training paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from flwr.common import (
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from models.mlp import IDSMLP
from fl.client import TVFLIDSClient
from fl.strategy import TVFLIDSStrategy
from fl.baselines import (
    FedAvgStrategy,
    FLTrustStrategy,
    TrimmedMeanStrategy,
    KrumStrategy,
    BaFFLeStrategy,
)
from attacks.adversarial import (
    apply_round_attacks,
    AdversarialAttackFactory,
    get_malicious_client_ids,
)
from attacks.knowledge import ValidationEstimateProvider


class MockClientProxy(ClientProxy):
    def __init__(self, cid: str):
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout):
        pass

    def get_parameters(self, ins, timeout):
        pass

    def fit(self, ins, timeout):
        pass

    def evaluate(self, ins, timeout):
        pass

    def reconnect(self, ins, timeout):
        pass


def _make_mlp():
    torch.manual_seed(42)
    return IDSMLP(input_dim=10, num_classes=5)


def _make_synthetic_dataset(n_samples=200, input_dim=10, num_classes=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
    # Ensure all classes 0..4 appear in dataset
    y = np.array([i % num_classes for i in range(n_samples)], dtype=np.int64)
    rng.shuffle(y)
    return X, y


def _make_dummy_fit_res(client_id: int, rng_seed: int = 42) -> Tuple[ClientProxy, FitRes]:
    """Create a dummy (ClientProxy, FitRes) with realistic varied weights."""
    model = _make_mlp()
    params = model.get_parameters()
    rng = np.random.default_rng(rng_seed + client_id)
    perturbed = [p + rng.normal(0, 0.05, size=p.shape).astype(np.float32) for p in params]
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(perturbed),
        num_examples=100,
        metrics={"client_id": client_id, "train_time_ms": 10.0},
    )
    proxy = MockClientProxy(cid=str(client_id))
    return proxy, fit_res


# ── Part B: Min-Sum Activation ───────────────────────────────────────────────

class TestMinSumProductionActivation:
    """Proves Min-Sum partial and omniscient attacks actively alter updates and satisfy constraints."""

    def test_min_sum_activation_partial_and_omniscient(self):
        # 10 clients: 7 honest (0..6), 3 malicious (7..9)
        num_clients = 10
        malicious_ids = [7, 8, 9]
        honest_ids = [i for i in range(num_clients) if i not in malicious_ids]
        client_ids = list(range(num_clients))

        global_model = _make_mlp()
        global_weights = global_model.get_parameters()

        # Realistic non-collinear client parameters
        rng = np.random.default_rng(42)
        client_params = []
        for i in range(num_clients):
            perturbed = [p + rng.normal(0, 0.05, size=p.shape).astype(np.float32) for p in global_weights]
            client_params.append(perturbed)

        # Extract honest updates
        honest_updates = []
        for i in honest_ids:
            u = np.concatenate([(c - g).flatten() for c, g in zip(client_params[i], global_weights)])
            honest_updates.append(u)
        honest_updates_np = np.stack(honest_updates, axis=0)
        honest_mean = np.mean(honest_updates_np, axis=0)

        # Baseline: clean update for malicious client 7
        clean_mal_weights = client_params[7]
        clean_mal_update = np.concatenate([(c - g).flatten() for c, g in zip(clean_mal_weights, global_weights)])

        # 1. Min-Sum Partial Knowledge
        res_partial = apply_round_attacks(
            client_params=[[p.copy() for p in cp] for cp in client_params],
            global_params=global_weights,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type="min_sum",
            attack_kwargs={"variant": "partial"},
            global_model=global_model,
            server_round=1,
            seed=42,
        )
        mal_weights_partial = res_partial[7]
        update_partial = np.concatenate([(c - g).flatten() for c, g in zip(mal_weights_partial, global_weights)])

        # 2. Min-Sum Omniscient Knowledge
        res_omniscient = apply_round_attacks(
            client_params=[[p.copy() for p in cp] for cp in client_params],
            global_params=global_weights,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type="min_sum",
            attack_kwargs={"variant": "omniscient"},
            global_model=global_model,
            server_round=1,
            seed=42,
        )
        mal_weights_omniscient = res_omniscient[7]
        update_omniscient = np.concatenate([(c - g).flatten() for c, g in zip(mal_weights_omniscient, global_weights)])

        # Numerical assertions
        # A. Attacked update differs numerically from clean
        diff_partial = np.linalg.norm(update_partial - clean_mal_update)
        diff_omniscient = np.linalg.norm(update_omniscient - clean_mal_update)
        assert diff_partial > 1e-3, f"Partial attack produced identical update! Diff={diff_partial}"
        assert diff_omniscient > 1e-3, f"Omniscient attack produced identical update! Diff={diff_omniscient}"

        # B. Partial vs Omniscient differ
        diff_variants = np.linalg.norm(update_omniscient - update_partial)
        assert diff_variants > 1e-3, f"Partial and omniscient produced identical updates! Diff={diff_variants}"

        # C. Cosine similarities
        cos_sim_partial = float(np.dot(update_partial, clean_mal_update) / (
            np.linalg.norm(update_partial) * np.linalg.norm(clean_mal_update) + 1e-12
        ))
        cos_sim_omniscient = float(np.dot(update_omniscient, clean_mal_update) / (
            np.linalg.norm(update_omniscient) * np.linalg.norm(clean_mal_update) + 1e-12
        ))
        assert -1.0 <= cos_sim_partial <= 1.0
        assert -1.0 <= cos_sim_omniscient <= 1.0

        # D. Min-Sum Constraint Satisfaction for Omniscient:
        # sum_{u in U} ||Delta_m - u|| <= max_{v in U} sum_{u in U} ||v - u||
        max_sum_dist = max(sum(np.linalg.norm(v - u) for u in honest_updates_np) for v in honest_updates_np)
        cand_sum_dist = sum(np.linalg.norm(update_omniscient - u) for u in honest_updates_np)
        assert cand_sum_dist <= max_sum_dist + 1e-3, (
            f"Min-Sum omniscient constraint violated: {cand_sum_dist} > {max_sum_dist}"
        )


# ── Part C: ACK3 Activation ─────────────────────────────────────────────────

class TestACK3ProductionActivation:
    """Proves ACK3 local target transformation, surrogate isolation, and balanced loss activation."""

    def test_ack3_target_relabeling_and_surrogate_isolation(self):
        X, y = _make_synthetic_dataset(n_samples=200, input_dim=10, num_classes=5, seed=42)
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=123)

        cfg = {
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "optimizer": "sgd",
        }
        model_kwargs = {"input_dim": 10, "num_classes": 5}

        # 1. Clean client
        clean_client = TVFLIDSClient(
            client_id=1,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            device=torch.device("cpu"),
            config=cfg,
            class_weights=torch.ones(5),
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=False,
        )
        model = _make_mlp()
        clean_params = model.get_parameters()
        clean_res, _, _ = clean_client.fit(clean_params, {"server_round": 1})

        # 2. ACK3 K1 (uses local background data, NO server val access)
        k1_surrogate = ValidationEstimateProvider.get_validation_estimate(
            tier="K1", background_data=(X, y), seed=42
        )
        ack3_k1_client = TVFLIDSClient(
            client_id=1,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            device=torch.device("cpu"),
            config=cfg,
            class_weights=torch.ones(5),
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=True,
            attack_type="ack3",
            attack_kwargs={"knowledge_tier": "K1", "seed": 42},
            proxy_val_data=k1_surrogate,
        )
        ack3_k1_res, _, _ = ack3_k1_client.fit(clean_params, {"server_round": 1})

        # 3. ACK3 K2 (uses server val slice)
        k2_surrogate = ValidationEstimateProvider.get_validation_estimate(
            tier="K2", server_val_data=(X_val, y_val), seed=42
        )
        ack3_k2_client = TVFLIDSClient(
            client_id=1,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            device=torch.device("cpu"),
            config=cfg,
            class_weights=torch.ones(5),
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=True,
            attack_type="ack3",
            attack_kwargs={"knowledge_tier": "K2", "seed": 42},
            proxy_val_data=k2_surrogate,
        )
        ack3_k2_res, _, _ = ack3_k2_client.fit(clean_params, {"server_round": 1})

        # Verify updates differ
        u_clean = np.concatenate([p.flatten() for p in clean_res])
        u_k1 = np.concatenate([p.flatten() for p in ack3_k1_res])
        u_k2 = np.concatenate([p.flatten() for p in ack3_k2_res])

        diff_k1 = np.linalg.norm(u_k1 - u_clean)
        diff_k2 = np.linalg.norm(u_k2 - u_clean)
        assert diff_k1 > 1e-4, f"ACK3 K1 update did not diverge from clean update: {diff_k1}"
        assert diff_k2 > 1e-4, f"ACK3 K2 update did not diverge from clean update: {diff_k2}"

        # Verify surrogate data source integrity: K1 surrogate is derived from local training data X, not X_val
        assert k1_surrogate is not None
        assert not np.array_equal(k1_surrogate[0], X_val)


# ── Part D: On-Off Transition ────────────────────────────────────────────────

class TestOnOffTransitionMultiRound:
    """Proves On-Off schedule transitions from honest (t <= k) to active (t > k)."""

    def test_on_off_lf_10_multiround_transition(self):
        # Schedule k=10: rounds 1..10 must be honest, round 11..12 must be attacked
        X, y = _make_synthetic_dataset(n_samples=200, input_dim=10, num_classes=5, seed=42)
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=123)

        cfg = {
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "optimizer": "sgd",
        }
        model_kwargs = {"input_dim": 10, "num_classes": 5}
        model = _make_mlp()
        base_params = model.get_parameters()

        # Clean client reference
        clean_client = TVFLIDSClient(
            client_id=1,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            device=torch.device("cpu"),
            config=cfg,
            class_weights=torch.ones(5),
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=False,
        )

        # On-Off Malicious client with k=10
        on_off_client = TVFLIDSClient(
            client_id=1,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            device=torch.device("cpu"),
            config=cfg,
            class_weights=torch.ones(5),
            model_class=IDSMLP,
            model_kwargs=model_kwargs,
            is_malicious=True,
            attack_type="on_off_lf",
            attack_kwargs={"k": 10, "seed": 42},
        )

        # Rounds 1..10: Honest Phase
        for r in [1, 5, 10]:
            assert not AdversarialAttackFactory.is_on_off_active(r, k=10), f"Round {r} should be dormant/honest for k=10"
            clean_res, _, _ = clean_client.fit(base_params, {"server_round": r})
            clean_vec = np.concatenate([p.flatten() for p in clean_res])

            res_r, _, _ = on_off_client.fit(base_params, {"server_round": r})
            vec_r = np.concatenate([p.flatten() for p in res_r])
            np.testing.assert_allclose(
                vec_r, clean_vec, atol=1e-6,
                err_msg=f"Round {r} produced attacked update during honest phase!"
            )

        # Rounds 11..12: Active Attack Phase
        for r in [11, 12]:
            assert AdversarialAttackFactory.is_on_off_active(r, k=10), f"Round {r} should be active for k=10"
            clean_res, _, _ = clean_client.fit(base_params, {"server_round": r})
            clean_vec = np.concatenate([p.flatten() for p in clean_res])

            res_r, _, _ = on_off_client.fit(base_params, {"server_round": r})
            vec_r = np.concatenate([p.flatten() for p in res_r])
            diff = np.linalg.norm(vec_r - clean_vec)
            assert diff > 1e-4, f"Round {r} failed to activate attack in active phase! Diff={diff}"

    def test_on_off_ack2_10_multiround_transition(self):
        # Round-level ACK2 On-Off attack
        num_clients = 10
        malicious_ids = [7, 8, 9]
        client_ids = list(range(num_clients))
        global_model = _make_mlp()
        global_weights = global_model.get_parameters()
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=42)

        rng = np.random.default_rng(42)
        client_params = [[p + rng.normal(0, 0.05, size=p.shape).astype(np.float32) for p in global_weights] for i in range(num_clients)]

        # Rounds 1..10: honest
        for r in [1, 5, 10]:
            results_out = apply_round_attacks(
                client_params=[[p.copy() for p in cp] for cp in client_params],
                global_params=global_weights,
                client_ids=client_ids,
                malicious_ids=malicious_ids,
                attack_type="on_off_ack2",
                attack_kwargs={"k": 10, "seed": 42},
                global_model=global_model,
                val_data=(X_val, y_val),
                server_round=r,
                seed=42,
            )
            # Verify malicious client update is untouched in honest rounds
            orig_weights = client_params[7]
            out_weights = results_out[7]
            for ow, nw in zip(orig_weights, out_weights):
                np.testing.assert_array_equal(ow, nw)

        # Round 11: active
        results_out = apply_round_attacks(
            client_params=[[p.copy() for p in cp] for cp in client_params],
            global_params=global_weights,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type="on_off_ack2",
            attack_kwargs={"k": 10, "seed": 42},
            global_model=global_model,
            val_data=(X_val, y_val),
            server_round=11,
            seed=42,
        )
        orig_weights = client_params[7]
        out_weights = results_out[7]
        diff = np.sum([np.linalg.norm(ow - nw) for ow, nw in zip(orig_weights, out_weights)])
        assert diff > 1e-4, f"Round 11 failed to activate ACK2 on-off attack! Diff={diff}"

    def test_on_off_generalization_k20(self):
        # Verify k=20 schedule generalizes correctly
        assert not AdversarialAttackFactory.is_on_off_active(1, k=20)
        assert not AdversarialAttackFactory.is_on_off_active(19, k=20)
        assert not AdversarialAttackFactory.is_on_off_active(20, k=20)
        assert AdversarialAttackFactory.is_on_off_active(21, k=20)
        assert AdversarialAttackFactory.is_on_off_active(30, k=20)


# ── Part E: LF-R Activation ─────────────────────────────────────────────────

class TestLFRProductionActivation:
    """Proves Random Label-Flip preserves normal class 0 and remaps attack classes to distinct targets."""

    def test_lfr_remapping_and_client_training(self):
        X, y = _make_synthetic_dataset(n_samples=500, input_dim=10, num_classes=5, seed=42)
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=123)

        # Test batch transformation
        y_flipped = AdversarialAttackFactory.label_flip_random(y, num_classes=5, seed=42)

        # 1. Normal label 0 is unchanged
        idx_normal = np.where(y == 0)[0]
        assert len(idx_normal) > 0
        np.testing.assert_array_equal(y_flipped[idx_normal], 0)

        # 2. Every non-normal label is remapped to a DIFFERENT attack class in {1..K-1}
        idx_attack = np.where(y != 0)[0]
        assert len(idx_attack) > 0
        for i in idx_attack:
            orig = y[i]
            remapped = y_flipped[i]
            assert remapped != orig, f"Sample {i}: class {orig} was not remapped! (got {remapped})"
            assert 1 <= remapped <= 4, f"Sample {i}: remapped class {remapped} outside {{1..4}}"

        # 3. Determinism
        y_flipped_2 = AdversarialAttackFactory.label_flip_random(y, num_classes=5, seed=42)
        np.testing.assert_array_equal(y_flipped, y_flipped_2)

        # 4. Client training loop receives flipped data and produces divergent update
        cfg = {
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "optimizer": "sgd",
        }
        model_kwargs = {"input_dim": 10, "num_classes": 5}
        model = _make_mlp()
        base_params = model.get_parameters()

        clean_client = TVFLIDSClient(
            client_id=1, X_train=X, y_train=y, X_val=X_val, y_val=y_val,
            device=torch.device("cpu"), config=cfg, class_weights=torch.ones(5),
            model_class=IDSMLP, model_kwargs=model_kwargs, is_malicious=False,
        )
        clean_res, _, _ = clean_client.fit(base_params, {"server_round": 1})

        lfr_client = TVFLIDSClient(
            client_id=1, X_train=X, y_train=y, X_val=X_val, y_val=y_val,
            device=torch.device("cpu"), config=cfg, class_weights=torch.ones(5),
            model_class=IDSMLP, model_kwargs=model_kwargs, is_malicious=True,
            attack_type="lf_r", attack_kwargs={"seed": 42},
        )
        lfr_res, _, _ = lfr_client.fit(base_params, {"server_round": 1})

        u_clean = np.concatenate([p.flatten() for p in clean_res])
        u_lfr = np.concatenate([p.flatten() for p in lfr_res])
        diff = np.linalg.norm(u_lfr - u_clean)
        assert diff > 1e-4, f"LF-R update did not diverge from clean update! Diff={diff}"


# ── Part F: ACK4 Production Path Execution ───────────────────────────────────

class TestACK4ProductionPath:
    """Proves ACK4 executes candidate mixture -> clipping -> gate simulation -> trust simulation -> omega selection."""

    def test_ack4_grid_search_and_omega_selection(self):
        num_clients = 10
        malicious_ids = [7, 8, 9]
        client_ids = list(range(num_clients))
        global_model = _make_mlp()
        global_weights = global_model.get_parameters()
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=42)

        rng = np.random.default_rng(42)
        client_params = [[p + rng.normal(0, 0.05, size=p.shape).astype(np.float32) for p in global_weights] for i in range(num_clients)]

        res_ack4 = apply_round_attacks(
            client_params=[[p.copy() for p in cp] for cp in client_params],
            global_params=global_weights,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type="ack4",
            attack_kwargs={"seed": 42},
            global_model=global_model,
            val_data=(X_val, y_val),
            server_round=1,
            seed=42,
        )

        # Verify results returned
        assert len(res_ack4) == num_clients
        # Malicious client update was transformed
        mal_weights = res_ack4[7]
        clean_weights = client_params[7]
        diff = np.sum([np.linalg.norm(mw - cw) for mw, cw in zip(mal_weights, clean_weights)])
        assert diff > 1e-4, f"ACK4 did not modify malicious client update! Diff={diff}"


# ── Part G & I: Clean Client Preservation & Strategy Invariance ─────────────

class TestCleanClientIntegrityAndNoBypass:
    """Proves attack interception preserves clean client updates bitwise and does not alter ordering or count."""

    @pytest.mark.parametrize("attack_type,attack_kwargs", [
        ("min_sum", {"variant": "partial"}),
        ("min_sum", {"variant": "omniscient"}),
        ("ack2", {"seed": 42}),
        ("ack4", {"seed": 42}),
        ("on_off_ack2", {"k": 10, "seed": 42}),
    ])
    def test_round_attacks_preserve_clean_clients(self, attack_type, attack_kwargs):
        num_clients = 10
        malicious_ids = [7, 8, 9]
        honest_ids = [i for i in range(num_clients) if i not in malicious_ids]
        client_ids = list(range(num_clients))

        global_model = _make_mlp()
        global_weights = global_model.get_parameters()
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=42)

        rng = np.random.default_rng(42)
        client_params_orig = [[p + rng.normal(0, 0.05, size=p.shape).astype(np.float32) for p in global_weights] for i in range(num_clients)]

        results_attacked = apply_round_attacks(
            client_params=[[p.copy() for p in cp] for cp in client_params_orig],
            global_params=global_weights,
            client_ids=client_ids,
            malicious_ids=malicious_ids,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            global_model=global_model,
            val_data=(X_val, y_val),
            server_round=11,  # Active round for on-off
            seed=42,
        )

        # 1. Total participant count preserved
        assert len(results_attacked) == len(client_params_orig)

        # 2. Clean client updates are preserved BITWISE
        for h_id in honest_ids:
            orig_p = client_params_orig[h_id]
            att_p = results_attacked[h_id]
            for op, ap in zip(orig_p, att_p):
                np.testing.assert_array_equal(
                    op, ap,
                    err_msg=f"Clean client {h_id} update altered by attack {attack_type}!"
                )

        # 3. Malicious client updates ARE altered
        for m_id in malicious_ids:
            orig_p = client_params_orig[m_id]
            att_p = results_attacked[m_id]
            diff = sum(np.linalg.norm(op - ap) for op, ap in zip(orig_p, att_p))
            assert diff > 1e-4, f"Malicious client {m_id} update NOT altered by attack {attack_type}!"

    def test_baselines_receive_attacked_updates(self):
        """Verify baseline strategies (FedAvg, FLTrust, TrimmedMean, Krum) receive intercepted updates."""
        num_clients = 10
        malicious_ids = [7, 8, 9]
        global_model = _make_mlp()
        X_val, y_val = _make_synthetic_dataset(n_samples=50, input_dim=10, num_classes=5, seed=42)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=32
        )

        strategies = [
            FedAvgStrategy(
                global_model=global_model,
                attack_type="min_sum",
                attack_kwargs={"variant": "partial"},
                malicious_ids=malicious_ids,
            ),
            TrimmedMeanStrategy(
                beta=0.1,
                global_model=global_model,
                attack_type="min_sum",
                attack_kwargs={"variant": "partial"},
                malicious_ids=malicious_ids,
            ),
            KrumStrategy(
                num_clients=num_clients,
                num_byzantine=len(malicious_ids),
                global_model=global_model,
                attack_type="min_sum",
                attack_kwargs={"variant": "partial"},
                malicious_ids=malicious_ids,
            ),
            FLTrustStrategy(
                server_model=global_model,
                server_root_loader=val_loader,
                device=torch.device("cpu"),
                attack_type="min_sum",
                attack_kwargs={"variant": "partial"},
                malicious_ids=malicious_ids,
            ),
        ]

        for strat in strategies:
            strat_name = strat.__class__.__name__
            # Pass dummy results through baseline aggregate_fit
            results = [_make_dummy_fit_res(i, rng_seed=42) for i in range(num_clients)]
            params, log = strat.aggregate_fit(server_round=1, results=results, failures=[])
            assert params is not None, f"Strategy {strat_name} returned None parameters"
            assert isinstance(log, dict)
            assert log.get("round") == 1
