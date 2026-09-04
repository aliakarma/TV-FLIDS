"""
tests/test_ack1_ack2_attacks.py

In-process, Ray-free smoke test for the ACK1 ("Check-1 evasion") and ACK2
("Check-2 coalition evasion") adaptive attacks added to attacks/adversarial.py
and wired into fl/client.py (ACK1, per-client training-time) and
fl/strategy.py (ACK2, strategy-level post-training coalition attack).

Follows the exact in-process idiom already used by tests/test_pipeline_smoke.py
in this repo (_FakeProxy/_FakeFitRes duck-typed stand-ins around
flwr.common.ndarrays_to_parameters, direct calls to
TVFLIDSStrategy.aggregate_fit()) to avoid Flower's Ray-based
fl.simulation.start_simulation(), which crashes in this Windows/Python-3.11
environment with an unrelated Ray-actor/PyTorch DLL load failure (WinError
1114 loading torch/lib/c10.dll) -- an environment issue, not a code issue.

Run: .venv\\Scripts\\python.exe -m pytest tests\\test_ack1_ack2_attacks.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


def make_model(input_dim=20, num_classes=5):
    from models.mlp import IDSMLP
    return IDSMLP(input_dim=input_dim, num_classes=num_classes)


def make_val_loader(n=100, input_dim=20, num_classes=5, seed=0):
    """Synthetic val set with genuine (if simple) learnable structure: the
    label is a deterministic function of feature 0, binned into
    `num_classes` buckets (same construction as test_pipeline_smoke.py)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, input_dim)).astype(np.float32)
    ranks = np.argsort(np.argsort(X[:, 0]))
    y = (ranks * num_classes // n).clip(0, num_classes - 1)
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=32)


def make_client_data(n=100, input_dim=20, num_classes=5, seed=1):
    """Synthetic per-client data with the same learnable structure as the
    val set (feature 0 -> class bucket), so 'clean' training genuinely
    reduces loss and ACK1's auxiliary loss term has real learnable signal
    to exploit, rather than being uninformative i.i.d. noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, input_dim)).astype(np.float32)
    ranks = np.argsort(np.argsort(X[:, 0]))
    y = (ranks * num_classes // n).clip(0, num_classes - 1).astype(np.int64)
    return X, y


class _FakeProxy:
    def __init__(self, cid):
        self.cid = str(cid)


class _FakeFitRes:
    def __init__(self, parameters):
        self.parameters = parameters


# ─────────────────────────────────────────────────────────────────────────
# ACK1: per-client, training-time Check-1 evasion attack
# ─────────────────────────────────────────────────────────────────────────

class TestACK1Evasion(unittest.TestCase):
    def setUp(self):
        self.input_dim, self.num_classes = 20, 5
        self.device = torch.device("cpu")
        self.config = {"local_lr": 0.01, "local_epochs": 3, "local_batch_size": 32}

    def _make_client(self, is_malicious, attack_type, attack_kwargs=None, seed=1):
        from fl.client import TVFLIDSClient
        X, y = make_client_data(n=100, input_dim=self.input_dim,
                                num_classes=self.num_classes, seed=seed)
        # 80/20 split, matching make_client_fn's convention in run_experiment.py
        n_train = int(0.8 * len(X))
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_lv, y_lv = X[n_train:], y[n_train:]
        return TVFLIDSClient(
            client_id=0, X_train=X_tr, y_train=y_tr, X_val=X_lv, y_val=y_lv,
            device=self.device, config=self.config,
            model_kwargs={"input_dim": self.input_dim, "num_classes": self.num_classes},
            is_malicious=is_malicious, attack_type=attack_type,
            attack_kwargs=attack_kwargs or {},
        )

    def test_ack1_runs_and_changes_training_vs_honest(self):
        """ACK1 client trains without error and, because it optimizes a
        combined poisoning + auxiliary-loss objective (not just relabeled
        data), ends up with a measurably different val_loss than a matched
        honest client trained from the identical initial parameters."""
        torch.manual_seed(0)
        honest_client = self._make_client(is_malicious=False, attack_type=None)
        init_params = honest_client.get_parameters(config={})

        torch.manual_seed(0)
        ack1_client = self._make_client(
            is_malicious=True, attack_type="ack1_evasion",
            attack_kwargs={"flip_ratio": 1.0, "target_class": 0,
                            "proxy_val_ratio": 0.15, "aux_loss_weight": 0.5,
                            "seed": 42},
        )

        torch.manual_seed(0)
        honest_params, honest_n, honest_metrics = honest_client.fit(init_params, config={})

        torch.manual_seed(0)
        ack1_params, ack1_n, ack1_metrics = ack1_client.fit(init_params, config={})

        # Runs without error and returns well-formed FitRes-like output.
        self.assertEqual(len(honest_params), len(ack1_params))
        for p in ack1_params:
            self.assertTrue(np.all(np.isfinite(p)))

        # The auxiliary-loss combined objective must actually change
        # training: val_loss differs from the matched honest client's.
        self.assertNotEqual(honest_metrics['val_loss'], ack1_metrics['val_loss'])

        # And the resulting parameters themselves must differ (not a no-op).
        self.assertFalse(all(np.allclose(a, b) for a, b in zip(honest_params, ack1_params)))

    def test_ack1_differs_from_plain_label_flip(self):
        """ACK1 must not be a renamed copy of label_flip: given the same
        flip_ratio/target_class/seed, the trained output must differ
        because ACK1 also carves out a proxy-val slice (shrinking the
        poisoned pool) and adds a real auxiliary-loss training term."""
        torch.manual_seed(0)
        init_client = self._make_client(is_malicious=False, attack_type=None)
        init_params = init_client.get_parameters(config={})

        torch.manual_seed(0)
        label_flip_client = self._make_client(
            is_malicious=True, attack_type="label_flip",
            attack_kwargs={"flip_ratio": 1.0, "target_class": 0, "seed": 42},
        )
        torch.manual_seed(0)
        lf_params, _, lf_metrics = label_flip_client.fit(init_params, config={})

        torch.manual_seed(0)
        ack1_client = self._make_client(
            is_malicious=True, attack_type="ack1_evasion",
            attack_kwargs={"flip_ratio": 1.0, "target_class": 0,
                            "proxy_val_ratio": 0.15, "aux_loss_weight": 0.5,
                            "seed": 42},
        )
        torch.manual_seed(0)
        ack1_params, _, ack1_metrics = ack1_client.fit(init_params, config={})

        self.assertFalse(all(np.allclose(a, b) for a, b in zip(lf_params, ack1_params)))


# ─────────────────────────────────────────────────────────────────────────
# ACK2: strategy-level, post-training coalition Check-2 evasion attack
# ─────────────────────────────────────────────────────────────────────────

class TestACK2Coalition(unittest.TestCase):
    def setUp(self):
        self.input_dim, self.num_classes, self.num_clients = 20, 5, 8
        self.malicious_ids = [5, 6, 7]  # ~3/8 coalition
        self.device = torch.device("cpu")

    def _base_config(self):
        return {
            "trust": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "memory_decay": 0.9,
                      "min_trust": 0.01, "meta_lr": 0.05},
            "verification": {"loss_threshold": -1e9, "cosine_threshold": 0.0,
                              "zscore_threshold": 2.5, "warmup_rounds": 20},
        }

    def _make_strategy(self, attack_type, attack_kwargs):
        from fl.strategy import TVFLIDSStrategy
        model = make_model(self.input_dim, self.num_classes)
        val_loader = make_val_loader(input_dim=self.input_dim, num_classes=self.num_classes)
        return TVFLIDSStrategy(
            num_clients=self.num_clients, config=self._base_config(), val_loader=val_loader,
            model=model, device=self.device, adaptive=True, use_adaptive_thresholds=False,
            known_malicious=self.malicious_ids, attack_type=attack_type,
            attack_kwargs=attack_kwargs, seed=42,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=2, min_evaluate_clients=1, min_available_clients=self.num_clients,
        ), model

    def _build_results(self, global_params, rng, malicious_scale=1.0):
        """All clients 'honestly' train small perturbations around the
        global params (ACK2 dispatch happens entirely at strategy level,
        same pattern as min_max)."""
        results = []
        for cid in range(self.num_clients):
            delta_scale = 0.02 if cid in self.malicious_ids else 0.01
            update = [p + rng.normal(size=p.shape).astype(np.float32) * delta_scale
                     for p in global_params]
            results.append((_FakeProxy(cid), _FakeFitRes(ndarrays_to_parameters(update))))
        return results

    def test_ack2_shifts_malicious_updates_and_produces_valid_aggregation(self):
        from attacks.adversarial import apply_ack2_attack_to_params

        strategy, model = self._make_strategy(
            "ack2_coalition", {"poison_strength": 1.0, "shift_scale": 1.0})
        global_params = model.get_parameters()
        rng = np.random.default_rng(7)
        results = self._build_results(global_params, rng)

        # Unattacked malicious client params (as submitted).
        client_params_unattacked = [parameters_to_ndarrays(r.parameters) for _, r in results]
        client_ids = [int(p.cid) for p, _ in results]

        # Directly exercise apply_ack2_attack_to_params to prove the
        # malicious clients' updates are measurably shifted.
        client_params_attacked = [p.copy() if False else [x.copy() for x in p]
                                  for p in client_params_unattacked]
        client_params_attacked = apply_ack2_attack_to_params(
            client_params_attacked, global_params, client_ids,
            self.malicious_ids, poison_strength=1.0, shift_scale=1.0,
        )

        for idx, cid in enumerate(client_ids):
            if cid in self.malicious_ids:
                unattacked = client_params_unattacked[idx]
                attacked = client_params_attacked[idx]
                self.assertFalse(
                    all(np.allclose(a, b) for a, b in zip(unattacked, attacked)),
                    f"ACK2 did not shift malicious client {cid}'s update",
                )
                for p in attacked:
                    self.assertTrue(np.all(np.isfinite(p)))
            else:
                # Honest clients must be untouched by the coalition attack.
                self.assertTrue(all(np.allclose(a, b) for a, b in zip(
                    client_params_unattacked[idx], client_params_attacked[idx])))

        # Malicious members must now be mutually MORE aligned (a direct
        # signature of sharing the same coalition shift term) than honest
        # clients are with each other -- honest clients are independent
        # random perturbations, so their pairwise cosine similarity in this
        # ~53k-dim parameter space should sit near zero, while the
        # coalition's shared shift term should pull their pairwise cosine
        # up substantially above that baseline.
        def flat_delta(p):
            return np.concatenate([(c - g).flatten() for c, g in zip(p, global_params)])

        def mean_pairwise_cosine(deltas):
            pairs = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    na, nb = np.linalg.norm(deltas[i]), np.linalg.norm(deltas[j])
                    pairs.append(float(np.dot(deltas[i], deltas[j]) / (na * nb)))
            return float(np.mean(pairs))

        mal_deltas = [flat_delta(client_params_attacked[i])
                     for i, cid in enumerate(client_ids) if cid in self.malicious_ids]
        honest_deltas = [flat_delta(client_params_attacked[i])
                         for i, cid in enumerate(client_ids) if cid not in self.malicious_ids]

        mal_cos = mean_pairwise_cosine(mal_deltas)
        honest_cos = mean_pairwise_cosine(honest_deltas)
        self.assertGreater(
            mal_cos, honest_cos + 0.2,
            "Coalition members should be measurably more mutually cosine-aligned "
            f"(mal={mal_cos:.3f}) than the independent honest clients "
            f"(honest={honest_cos:.3f}) after sharing the coalition shift term",
        )

        # ── Now run the SAME attack through the real strategy pipeline ──
        agg_params, log = strategy.aggregate_fit(server_round=1, results=results, failures=[])
        self.assertIsNotNone(agg_params)
        agg_np = parameters_to_ndarrays(agg_params)
        self.assertEqual(len(agg_np), len(global_params))
        for p in agg_np:
            self.assertTrue(np.all(np.isfinite(p)))
        self.assertIn('num_verified', log)
        self.assertIn('num_flagged', log)
        self.assertIn('num_rejected', log)

    def test_ack2_end_to_end_via_aggregate_fit_differs_from_no_attack(self):
        """Running aggregate_fit with attack_type='ack2_coalition' must
        produce a different aggregated model than an identical round with
        no attack at all, proving the coalition attack actually reaches
        and perturbs the aggregation output through the real strategy path."""
        strategy_atk, model_atk = self._make_strategy(
            "ack2_coalition", {"poison_strength": 1.0, "shift_scale": 1.0})
        strategy_none, model_none = self._make_strategy(None, {})

        gp_atk = model_atk.get_parameters()
        gp_none = model_none.get_parameters()
        for a, b in zip(gp_atk, gp_none):
            np.copyto(b, a)  # ensure identical starting params

        rng1 = np.random.default_rng(11)
        results_atk = self._build_results(gp_atk, rng1)
        rng2 = np.random.default_rng(11)
        results_none = self._build_results(gp_none, rng2)

        agg_atk, _ = strategy_atk.aggregate_fit(server_round=1, results=results_atk, failures=[])
        agg_none, _ = strategy_none.aggregate_fit(server_round=1, results=results_none, failures=[])

        agg_atk_np = parameters_to_ndarrays(agg_atk)
        agg_none_np = parameters_to_ndarrays(agg_none)
        self.assertFalse(all(np.allclose(a, b) for a, b in zip(agg_atk_np, agg_none_np)))


# ─────────────────────────────────────────────────────────────────────────
# Bonus: qualitative check that the adaptive attacks move the verification
# gate's behavior in the intended (more-evasive) direction vs. a naive
# attack on the same tiny synthetic setup. Not a statistical claim -- just
# a one-case demonstration of the mechanism.
# ─────────────────────────────────────────────────────────────────────────

class TestAdaptiveVsNaiveGateBehavior(unittest.TestCase):
    def test_ack2_coalition_shift_beats_uncoordinated_sign_flip_on_check2(self):
        """Isolate the actual scientific contribution of ACK2 -- the shared
        coalition shift term -- by comparing it against the 'naive'
        equivalent attack with coordination switched off (shift_scale=0.0,
        i.e. each malicious client independently sign-flips its own delta
        with no coalition coordination, a standard uncoordinated
        gradient-ascent Byzantine attack). Both attacks use the exact same
        base per-client updates and the same poisoning primitive; the ONLY
        difference is whether the coalition's shared shift is added. This
        directly demonstrates that the coalition-coordination mechanism
        (not just 'is this attack scaled differently') is what raises
        cosine similarity to the round's pseudo-gradient -- the actual
        thing Check 2 evaluates."""
        from attacks.adversarial import apply_ack2_attack_to_params

        input_dim, num_classes, num_clients = 20, 5, 8
        malicious_ids = [5, 6, 7]
        model = make_model(input_dim, num_classes)
        global_params = model.get_parameters()

        def build_base_updates(seed):
            rng = np.random.default_rng(seed)
            return [
                [p + rng.normal(size=p.shape).astype(np.float32) * 0.01 for p in global_params]
                for _ in range(num_clients)
            ]

        client_ids = list(range(num_clients))

        # Naive baseline: same sign-flip poisoning primitive, NO coalition
        # coordination (shift_scale=0.0 -> each malicious client acts alone).
        naive_base = build_base_updates(seed=3)
        naive_params = [[x.copy() for x in u] for u in naive_base]
        naive_params = apply_ack2_attack_to_params(
            naive_params, global_params, client_ids, malicious_ids,
            poison_strength=0.3, shift_scale=0.0,
        )

        # Full ACK2: identical base updates and poisoning strength, WITH
        # coalition coordination (shift_scale=1.0).
        ack2_base = build_base_updates(seed=3)
        ack2_params = [[x.copy() for x in u] for u in ack2_base]
        ack2_params = apply_ack2_attack_to_params(
            ack2_params, global_params, client_ids, malicious_ids,
            poison_strength=0.3, shift_scale=1.0,
        )

        def cosine_to_pseudo(all_params):
            deltas = [[c - g for c, g in zip(p, global_params)] for p in all_params]
            pseudo = [np.mean([d[i] for d in deltas], axis=0) for i in range(len(global_params))]
            pflat = np.concatenate([x.flatten() for x in pseudo])
            sims = []
            for d in deltas:
                dflat = np.concatenate([x.flatten() for x in d])
                na, nb = np.linalg.norm(dflat), np.linalg.norm(pflat)
                sims.append(float(np.dot(dflat, pflat) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0)
            return sims

        naive_sims = cosine_to_pseudo(naive_params)
        ack2_sims = cosine_to_pseudo(ack2_params)

        naive_mal_mean = np.mean([naive_sims[i] for i in malicious_ids])
        ack2_mal_mean = np.mean([ack2_sims[i] for i in malicious_ids])

        # Demonstrate the mechanism moves in the intended (more evasive)
        # direction: adding the coalition's shared shift term raises mean
        # cosine similarity to the round's pseudo-gradient relative to the
        # otherwise-identical uncoordinated attack.
        self.assertGreater(ack2_mal_mean, naive_mal_mean)

        # And, as a direct check-mechanics demonstration, run both variants
        # through the real VerificationModule.verify_all() and confirm the
        # coordinated attack's Check-2 flag rate is no worse.
        from trust.verification import VerificationModule

        def flag_count(all_params):
            verifier = VerificationModule(loss_threshold=-1e9, cosine_threshold=0.5,
                                          zscore_threshold=1e9)  # isolate Check 2
            deltas = [[c - g for c, g in zip(p, global_params)] for p in all_params]
            val_loader = make_val_loader(input_dim=input_dim, num_classes=num_classes)
            device = torch.device("cpu")
            criterion = torch.nn.CrossEntropyLoss()
            model.eval()
            with torch.no_grad():
                total, n = 0.0, 0
                for X, y in val_loader:
                    total += criterion(model(X), y).item()
                    n += 1
            global_loss = total / max(n, 1)
            result = verifier.verify_all(deltas, client_ids, global_loss, global_params,
                                         model, device, val_loader, eval_cache={})
            flagged_or_rejected = {cid for cid, _ in result['flagged'] + result['rejected']}
            return sum(1 for cid in malicious_ids if cid in flagged_or_rejected)

        naive_flagged = flag_count(naive_params)
        ack2_flagged = flag_count(ack2_params)
        self.assertLessEqual(
            ack2_flagged, naive_flagged,
            "Coalition-coordinated ACK2 should trigger Check 2 for no more "
            "malicious clients than the uncoordinated naive equivalent",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
