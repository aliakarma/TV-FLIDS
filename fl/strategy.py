"""
fl/strategy.py — TVFLIDSStrategy: Verification → Trust → Weighted Aggregation.
Reference: Guide §14.1
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from trust.trust_scorer import TrustScorer
from trust.adaptive_trust_scorer import AdaptiveTrustScorer
from trust.verification import VerificationModule
from attacks.adversarial import apply_min_max_attack_to_params, apply_ack2_attack_to_params
from evaluation.overhead import OverheadTracker
from utils.ste import clip_ste
import hashlib
import pickle
from fl.ordering import order_results


class TVFLIDSStrategy(FedAvg):
    """
    Custom Flower strategy: three-criteria verification → dynamic trust scoring
    → trust-weighted aggregation.
    """
    def __init__(self, num_clients: int, config: dict, val_loader: DataLoader,
                 model: nn.Module, device: torch.device, adaptive: bool = True,
                 use_adaptive_thresholds: Optional[bool] = None,
                 known_malicious: Optional[List[int]] = None,
                 attack_type: Optional[str] = None,
                 attack_kwargs: Optional[dict] = None,
                 seed: int = 42,
                 evaluate_fn=None,
                 **kwargs):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.num_clients = num_clients
        self.config = config
        self.val_loader = val_loader
        self.model = model
        self.device = device
        self.adaptive = adaptive
        self._known_malicious = set(known_malicious or [])
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.seed = seed
        self._last_round_data: Optional[Dict] = None
        # per-round evaluation cache: param-hash -> loss
        self._eval_cache: Dict[str, float] = {}

        t = config.get('trust', {})
        v = config.get('verification', {})

        self.trust_scorer = (
            AdaptiveTrustScorer(num_clients=num_clients,
                                memory_decay=t.get('memory_decay', 0.9),
                                min_trust=t.get('min_trust', 0.01),
                                meta_lr=t.get('meta_lr', 0.01))
            if adaptive else
            # Fallback weights are 1/3 each, matching paper Table IV
            # ("Initial alpha,beta,gamma = 1/3 each") and ablation A6. They are
            # only reached if `trust.alpha/beta/gamma` are absent from the
            # config; config/fl_config.yaml sets them explicitly.
            TrustScorer(num_clients=num_clients,
                        alpha=t.get('alpha', 1/3), beta=t.get('beta', 1/3),
                        gamma=t.get('gamma', 1/3),
                        memory_decay=t.get('memory_decay', 0.9),
                        min_trust=t.get('min_trust', 0.01))
        )

        self.verifier = VerificationModule(
            loss_threshold=v.get('loss_threshold', 0.0),
            cosine_threshold=v.get('cosine_threshold', 0.0),
            zscore_threshold=v.get('zscore_threshold', 2.5),
        )
        # T_warm (paper Table IV) - the single authoritative knob driving BOTH
        # warmup schedules (tau_L and tau_z, Section IV-A / Eq. (5)).
        self.warmup_rounds = v.get('warmup_rounds', 20)
        # Nominal (post-warmup) thresholds, kept separate from the live
        # per-round values on self.verifier that the annealing overwrites.
        self._nominal_zscore_threshold = v.get('zscore_threshold', 2.5)
        self._nominal_loss_threshold = v.get('loss_threshold', 0.0)
        self._warmup_loss_threshold = v.get('warmup_loss_threshold', -0.1)
        self._warmup_zscore_offset = v.get('warmup_zscore_offset', 0.5)

        # Warmup annealing of the gate thresholds is part of the method as
        # described in Section IV-A and the Table IV footnote, so it is ON by
        # default. It was previously hardcoded to False at the only
        # production call site (experiments/run_experiment.py), which meant
        # no reported experiment ever exercised the schedule the paper
        # describes - and made `verification.warmup_rounds` a dead knob for
        # the Supp. Table S2 "gate warmup schedule" sweep. Set
        # `verification.adaptive_thresholds: false` in the config (or pass
        # use_adaptive_thresholds=False) to restore the fixed-threshold gate.
        if use_adaptive_thresholds is None:
            use_adaptive_thresholds = bool(v.get('adaptive_thresholds', True))
        self.use_adaptive_thresholds = use_adaptive_thresholds

        # Per-stage wall-clock instrumentation backing paper Table XII.
        # Always constructed; `enable_overhead_tracking: false` in the config
        # turns the (sub-millisecond) timing off.
        self.track_overhead = bool(config.get('enable_overhead_tracking', True))
        self.overhead_tracker = OverheadTracker()

        self.round_logs: List[Dict] = []

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Deterministic aggregation order (see fl/ordering.py):
        # Ray yields results in completion order, and float32
        # summation is not associative, so an unsorted round made
        # the same seed drift run to run.
        results = order_results(results)

        # Clear per-round eval cache to avoid unbounded growth and ensure freshness
        self._eval_cache.clear()

        _ohead = self.overhead_tracker if self.track_overhead else _NullTracker()
        _round_timer = _ohead.time_phase("total")
        _round_timer.__enter__()

        _client_timer = _ohead.time_phase("client_processing")
        _client_timer.__enter__()

        client_params = [parameters_to_ndarrays(r.parameters) for _, r in results]
        client_ids    = [int(p.cid) for p, _ in results]
        global_params = self.model.get_parameters()

        if self.attack_type == "min_max" and self._known_malicious:
            client_params = apply_min_max_attack_to_params(
                client_params,
                global_params,
                client_ids,
                list(self._known_malicious),
                gamma=self.attack_kwargs.get("gamma", 2.0),
            )
        elif self.attack_type == "ack2_coalition" and self._known_malicious:
            # ACK2 ("Check-2 coalition evasion"): strategy-level, post-
            # training coalition attack. Needs visibility across all
            # colluding malicious clients' submitted updates this round,
            # so it (like min_max above) must be applied here, before
            # verify_all() computes the round's pseudo-gradient. See
            # attacks/adversarial.py::apply_ack2_attack_to_params.
            client_params = apply_ack2_attack_to_params(
                client_params,
                global_params,
                client_ids,
                list(self._known_malicious),
                poison_strength=self.attack_kwargs.get("poison_strength", 1.0),
                shift_scale=self.attack_kwargs.get("shift_scale", 1.0),
            )

        # Δw_i = w_i^trained − w_global
        updates = [[c - g for c, g in zip(cp, global_params)] for cp in client_params]

        _client_timer.__exit__(None, None, None)

        # Warmup threshold annealing (paper Section IV-A). Both schedules are
        # driven by the SAME T_warm = self.warmup_rounds; neither carries an
        # independent transition length.
        if self.use_adaptive_thresholds:
            self.verifier.zscore_threshold = VerificationModule.adaptive_zscore_threshold(
                self._nominal_zscore_threshold, server_round,
                warmup_rounds=self.warmup_rounds,
                warmup_offset=self._warmup_zscore_offset)
            self.verifier.loss_threshold = VerificationModule.adaptive_loss_threshold(
                server_round,
                initial=self._warmup_loss_threshold,
                final=self._nominal_loss_threshold,
                warmup_rounds=self.warmup_rounds)

        # ── STEP 1: Verify ────────────────────────────────────────────
        global_loss = self._eval_model(global_params)
        with _ohead.time_phase("verification"):
            vr = self.verifier.verify_all(
                updates, client_ids, global_loss, global_params,
                self.model, self.device, self.val_loader,
                eval_cache=self._eval_cache)

        active = vr['verified'] + vr['flagged']
        if not active:
            _round_timer.__exit__(None, None, None)
            # Same key schema as the normal path (minus the trust summary, which
            # is undefined when nothing was aggregated), so a consumer parsing
            # round logs does not have to special-case this branch.
            log = {'round': server_round, 'all_rejected': 1,
                   'global_loss': float(global_loss),
                   'num_verified': 0, 'num_flagged': 0,
                   'num_rejected': len(vr['rejected']),
                   'tau_z': float(self.verifier.zscore_threshold),
                   'tau_L': float(self.verifier.loss_threshold)}
            if self.track_overhead:
                for _phase in ("client_processing", "verification", "total"):
                    _times = self.overhead_tracker.timings.get(_phase)
                    if _times:
                        log[f'time_{_phase}_ms'] = float(_times[-1] * 1000.0)
            self.round_logs.append(log)
            return ndarrays_to_parameters(global_params), log

        a_ids  = [cid for cid, _ in active]
        a_upds = [upd for _, upd in active]
        a_pars = [[g + u for g, u in zip(global_params, upd)] for upd in a_upds]

        if self.config.get("log_client_params", False):
            honest_ids = [cid for cid in a_ids if cid not in self._known_malicious]
            byzantine_ids = [cid for cid in a_ids if cid in self._known_malicious]
            self._last_round_data = {
                "honest_ids": honest_ids,
                "byzantine_ids": byzantine_ids,
                "trust_scores": self.trust_scorer.trust_scores.copy(),
                "client_params": {cid: a_pars[i] for i, cid in enumerate(a_ids)},
            }

        # ── STEP 2: Trust signals ─────────────────────────────────────
        with _ohead.time_phase("trust_scoring"):
            mean_upd = [np.mean([u[i] for u in a_upds], axis=0) for i in range(len(global_params))]
            sim  = self.trust_scorer.compute_similarity_scores(a_upds, mean_upd)
            client_val_losses = [self._eval_model(p) for p in a_pars]  # compute ONCE (cached)
            acc  = self.trust_scorer.compute_accuracy_scores(global_loss, client_val_losses)
            # eq:anom, O_i = 1 - exp(-z_i / tau_z), must use the CURRENT tau_z,
            # which the warmup schedule of eq:tau_anneal may have annealed this
            # round.
            anom = self.trust_scorer.compute_anomaly_scores(
                a_upds, tau_z=self.verifier.zscore_threshold)

            # STEP 3: Update trust (meta-gradient step follows below)
            self.trust_scorer.update_trust(a_ids, sim, acc, anom)

        adaptive_snap = None
        if self.adaptive and isinstance(self.trust_scorer, AdaptiveTrustScorer):
            _sim, _acc, _anom = sim.copy(), acc.copy(), anom.copy()
            _cached_losses = client_val_losses[:]   # snapshot in closure

            def _val_fn(alpha, beta, gamma):
                """
                Differentiable trust-weighted aggregation loss.
                Connects alpha/beta/gamma to validation loss via per-client val losses.
                """
                per_client_losses = torch.tensor(_cached_losses, dtype=torch.float32)
                sim_t = torch.tensor(_sim, dtype=torch.float32)
                acc_t = torch.tensor(_acc, dtype=torch.float32)
                anom_t = torch.tensor(_anom, dtype=torch.float32)

                # eq:hatw: clip_[0,1] with a STRAIGHT-THROUGH gradient, per
                # Section IV-C ("we employ the straight-through estimator").
                # torch.clamp zeroes the gradient for any client whose raw
                # signal is saturated, so those clients could not influence
                # dL_meta/dv at all - the opposite of what the paper states.
                # Forward values are identical to torch.clamp.
                raw_scores = clip_ste(
                    alpha * sim_t + beta * acc_t - gamma * anom_t, 0.0, 1.0
                )
                total = raw_scores.sum()
                weights = raw_scores / (total + 1e-8)

                weighted_val_loss = (weights * per_client_losses).sum()
                return weighted_val_loss

            with _ohead.time_phase("meta_gradient"):
                try:
                    adaptive_snap = self.trust_scorer.meta_update(_val_fn)
                except Exception:
                    pass

        # ── STEP 4: Weighted aggregation ──────────────────────────────
        with _ohead.time_phase("aggregation"):
            weights = self.trust_scorer.get_aggregation_weights(a_ids)
            aggregated = [
                np.sum([weights[i] * a_pars[i][l] for i in range(len(a_ids))], axis=0)
                for l in range(len(global_params))
            ]
        self.model.set_parameters(aggregated)

        _round_timer.__exit__(None, None, None)

        ts = self.trust_scorer.get_summary()
        log = {
            'round': server_round, 'global_loss': float(global_loss),
            'num_verified': len(vr['verified']), 'num_flagged': len(vr['flagged']),
            'num_rejected': len(vr['rejected']), 'all_rejected': 0,
            'tau_z': float(self.verifier.zscore_threshold),
            'tau_L': float(self.verifier.loss_threshold),
            **{f'trust_{k}': float(v) for k, v in ts.items()},
        }
        if self.track_overhead:
            for _phase in ("client_processing", "verification", "trust_scoring",
                           "meta_gradient", "aggregation", "total"):
                _times = self.overhead_tracker.timings.get(_phase)
                if _times:
                    log[f'time_{_phase}_ms'] = float(_times[-1] * 1000.0)
        if adaptive_snap is not None:
            log['adaptive_alpha'] = float(adaptive_snap['alpha'])
            log['adaptive_beta'] = float(adaptive_snap['beta'])
            log['adaptive_gamma'] = float(adaptive_snap['gamma'])
        self.round_logs.append(log)
        return ndarrays_to_parameters(aggregated), log

    def _eval_model(self, params: List[np.ndarray]) -> float:
        # Compute a deterministic hash for this parameter set and consult cache
        try:
            key = hashlib.sha256(pickle.dumps(params)).hexdigest()
        except Exception:
            # Fallback: no caching if hashing fails
            key = None

        if key is not None and key in self._eval_cache:
            return self._eval_cache[key]

        orig = self.model.get_parameters()
        self.model.set_parameters(params)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total, n = 0.0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                total += criterion(self.model(X.to(self.device)), y.to(self.device)).item()
                n += 1
        self.model.set_parameters(orig)
        self.model.train()

        loss = total / max(n, 1)
        if key is not None:
            self._eval_cache[key] = loss
        return loss

    def get_trust_history(self) -> Dict[int, List[float]]:
        return self.trust_scorer.trust_history

    def reset_trust(self) -> None:
        self.trust_scorer.reset()
        self.round_logs.clear()
        # Clear eval cache when resetting trust history
        self._eval_cache.clear()
        self.overhead_tracker = OverheadTracker()

    def get_overhead_summary(self) -> Dict[str, float]:
        """Per-stage timing summary backing paper Table XII.

        Keys are {client_processing,verification,trust_scoring,meta_gradient,
        aggregation,total}_{mean,std}_ms. Empty dict when tracking is off.
        """
        return self.overhead_tracker.get_summary() if self.track_overhead else {}


class _NullTracker:
    """No-op stand-in used when `enable_overhead_tracking` is false."""

    class _NullTimer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def time_phase(self, phase: str):
        return self._NullTimer()
