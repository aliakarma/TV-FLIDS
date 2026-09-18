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
from trust.verification import VerificationModule, compute_class_balanced_loss
from attacks.adversarial import (
    apply_round_attacks,
    apply_min_max_attack_to_params,
    apply_ack2_attack_to_params,
)
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
        self._eval_bal_cache: Dict[str, float] = {}

        self.val_data = None
        if val_loader is not None and self.attack_type in ("ack2", "ack4", "on_off_ack2"):
            if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "tensors"):
                tensors = val_loader.dataset.tensors
                self.val_data = (tensors[0].cpu().numpy(), tensors[1].cpu().numpy())
            elif hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "__len__"):
                val_X_list, val_y_list = [], []
                for i in range(len(val_loader.dataset)):
                    x, y = val_loader.dataset[i]
                    val_X_list.append(x.numpy() if hasattr(x, "numpy") else np.array(x))
                    val_y_list.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
                if val_X_list:
                    self.val_data = (np.stack(val_X_list, axis=0), np.stack(val_y_list, axis=0))

        t = config.get('trust', {})
        v = config.get('verification', {})

        self.trust_scorer = (
            AdaptiveTrustScorer(num_clients=num_clients,
                                lambda_up=t.get('lambda_up', 0.9),
                                lambda_down=t.get('lambda_down', 0.7),
                                min_trust=t.get('min_trust', 0.01),
                                initial_trust=t.get('initial_trust', 0.5),
                                meta_lr=t.get('meta_lr', 0.01),
                                memory_decay=t.get('memory_decay', None))
            if adaptive else
            # Fallback weights are 1/3 each, matching paper Table IV
            # ("Initial alpha,beta,gamma = 1/3 each") and ablation A6. They are
            # only reached if `trust.alpha/beta/gamma` are absent from the
            # config; config/fl_config.yaml sets them explicitly.
            TrustScorer(num_clients=num_clients,
                        alpha=t.get('alpha', 1/3), beta=t.get('beta', 1/3),
                        gamma=t.get('gamma', 1/3),
                        lambda_up=t.get('lambda_up', 0.9),
                        lambda_down=t.get('lambda_down', 0.7),
                        min_trust=t.get('min_trust', 0.01),
                        initial_trust=t.get('initial_trust', 0.5),
                        memory_decay=t.get('memory_decay', None))
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

        # Record the round's mean per-client local-training time (paper
        # Table XII, first row). Clients report it in their FitRes metrics.
        if self.track_overhead:
            _ct = [float(fr.metrics.get("train_time_ms", 0.0))
                   for _, fr in results
                   if getattr(fr, "metrics", None)
                   and "train_time_ms" in fr.metrics]
            if _ct:
                self.overhead_tracker.timings["client_training"].append(
                    float(np.mean(_ct)) / 1000.0)

        # Clear per-round eval cache to avoid unbounded growth and ensure freshness
        self._eval_cache.clear()
        self._eval_bal_cache.clear()

        _ohead = self.overhead_tracker if self.track_overhead else _NullTracker()
        _round_timer = _ohead.time_phase("total")
        _round_timer.__enter__()

        _client_timer = _ohead.time_phase("client_processing")
        _client_timer.__enter__()

        client_params = [parameters_to_ndarrays(r.parameters) for _, r in results]
        client_ids    = [int(p.cid) for p, _ in results]
        global_params = self.model.get_parameters()

        client_params = apply_round_attacks(
            client_params=client_params,
            global_params=global_params,
            client_ids=client_ids,
            malicious_ids=list(self._known_malicious) if self._known_malicious else None,
            attack_type=self.attack_type,
            attack_kwargs=self.attack_kwargs,
            global_model=self.model,
            val_data=getattr(self, "val_data", None),
            server_round=server_round,
            seed=self.seed,
        )

        # Δw_i = w_i^trained − w_global (raw client updates)
        raw_updates = [[c - g for c, g in zip(cp, global_params)] for cp in client_params]

        _client_timer.__exit__(None, None, None)

        # Warmup threshold annealing (paper Section IV-A, Eq. (4)). Driven by
        # T_warm = self.warmup_rounds.
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

        # ── STAGE 1: Median-Radius Norm Clipping (Paper §IV, Eq. (3)) ──
        clipped_updates, clipping_radius, raw_norms = self.verifier.clip_updates(raw_updates)

        # ── STAGE 2: Single Validation-Loss Gate (Paper §IV, Eq. (4)) ──
        global_val_loss, global_bal_loss = self._eval_model_both(global_params)
        global_loss = global_val_loss
        with _ohead.time_phase("verification"):
            vr = self.verifier.evaluate_validation_gate(
                clipped_updates, client_ids, global_val_loss, global_params,
                self.model, self.device, self.val_loader,
                eval_cache=self._eval_cache,
                eval_bal_cache=self._eval_bal_cache,
                loss_threshold=self.verifier.loss_threshold)

        active = vr['accepted']
        rejected = vr['rejected']
        rej_ids = [cid for cid, _ in rejected]
        a_ids  = [cid for cid, _ in active]

        # ── STAGE 3: Trust signals & Memory (Paper §IV, Eq. (5)–(7), Alg. 1) ──
        with _ohead.time_phase("trust_scoring"):
            # Compute norm outlier scores over ALL participants P using unclipped updates
            anom_all = self.trust_scorer.compute_anomaly_scores(
                raw_updates, tau_z=self.verifier.zscore_threshold, cohort_norms=raw_norms
            )
            anom_map = {cid: float(anom_all[idx]) for idx, cid in enumerate(client_ids)}

            if not active:
                # All participants rejected! Algorithm 1 lines 6-8:
                # s_i = 0 for all i in P \ A (which is all participants).
                # Update trust with penalty branch for all participants.
                self.trust_scorer.update_trust(
                    client_ids=[], similarity_scores=np.array([]),
                    accuracy_scores=np.array([]), anomaly_scores=np.array([]),
                    participant_ids=client_ids, rejected_ids=client_ids
                )
                _round_timer.__exit__(None, None, None)
                log = {'round': server_round, 'all_rejected': 1,
                       'global_loss': float(global_val_loss),
                       'global_bal_loss': float(global_bal_loss),
                       'num_accepted': 0, 'num_verified': 0, 'num_flagged': 0,
                       'num_rejected': len(client_ids),
                       'clipping_radius': float(clipping_radius),
                       'tau_z': float(self.verifier.zscore_threshold),
                       'tau_L': float(self.verifier.loss_threshold)}
                if self.track_overhead:
                    for _phase in ("client_processing", "verification", "trust_scoring", "total"):
                        _times = self.overhead_tracker.timings.get(_phase)
                        if _times:
                            log[f'time_{_phase}_ms'] = float(_times[-1] * 1000.0)
                self.round_logs.append(log)
                return ndarrays_to_parameters(global_params), log

            a_upds = [upd for _, upd in active]  # CLIPPED updates
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

            # Direction signal S_i: relative to mean clipped update of accepted cohort
            mean_upd = [np.mean([u[i] for u in a_upds], axis=0) for i in range(len(global_params))]
            sim  = self.trust_scorer.compute_similarity_scores(a_upds, mean_upd)

            # Accuracy signal A_i: class-balanced validation loss improvement
            client_bal_losses = [vr['bal_losses'][cid] for cid in a_ids]
            client_val_losses = [vr['val_losses'][cid] for cid in a_ids]
            acc  = self.trust_scorer.compute_accuracy_scores(global_bal_loss, client_bal_losses)
            anom = np.array([anom_map[cid] for cid in a_ids], dtype=np.float64)

            # Update trust for all participants P (both accepted and rejected)
            self.trust_scorer.update_trust(
                client_ids=a_ids,
                similarity_scores=sim,
                accuracy_scores=acc,
                anomaly_scores=anom,
                participant_ids=client_ids,
                rejected_ids=rej_ids,
            )

        adaptive_snap = None
        if self.adaptive and isinstance(self.trust_scorer, AdaptiveTrustScorer):
            with _ohead.time_phase("meta_gradient"):
                adaptive_snap = self.trust_scorer.adapt_weights(
                    similarity_scores=sim,
                    accuracy_scores=acc,
                    anomaly_scores=anom,
                    val_losses=client_val_losses,
                )

        # ── STAGE 5: Final Model Aggregation (Paper §IV, Eq. (11), Alg. 1) ──
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
            'global_bal_loss': float(global_bal_loss),
            'num_accepted': len(vr['accepted']),
            'num_verified': len(vr['verified']), 'num_flagged': len(vr['flagged']),
            'num_rejected': len(vr['rejected']), 'all_rejected': 0,
            'clipping_radius': float(clipping_radius),
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
            if 'loss' in adaptive_snap:
                log['meta_loss'] = float(adaptive_snap['loss'])
            if 'saturated' in adaptive_snap:
                log['clip_saturated'] = int(adaptive_snap['saturated'])
        self.round_logs.append(log)
        return ndarrays_to_parameters(aggregated), log

    def _eval_model_both(self, params: List[np.ndarray]) -> Tuple[float, float]:
        try:
            key = hashlib.sha256(pickle.dumps(params)).hexdigest()
        except Exception:
            key = None

        if key is not None and key in self._eval_cache and key in self._eval_bal_cache:
            return self._eval_cache[key], self._eval_bal_cache[key]

        val_loss, bal_loss, _, _ = compute_class_balanced_loss(
            self.model, params, self.val_loader, self.device,
            eval_cache=self._eval_cache, eval_bal_cache=self._eval_bal_cache
        )
        if key is not None:
            self._eval_cache[key] = val_loss
            self._eval_bal_cache[key] = bal_loss
        return val_loss, bal_loss

    def _eval_model(self, params: List[np.ndarray]) -> float:
        val_loss, _ = self._eval_model_both(params)
        return val_loss

    def get_trust_history(self) -> Dict[int, List[float]]:
        return self.trust_scorer.trust_history

    def reset_trust(self) -> None:
        self.trust_scorer.reset()
        self.round_logs.clear()
        # Clear eval cache when resetting trust history
        self._eval_cache.clear()
        self._eval_bal_cache.clear()
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
