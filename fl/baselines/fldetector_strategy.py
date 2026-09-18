"""
fl/baselines/fldetector_strategy.py
FLDetector: Defending against Model Poisoning Attacks in Federated Learning.
Reference: Zhang et al., KDD 2022. https://arxiv.org/abs/2110.08482
Supplementary Material Section S2 & Table S2.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from attacks.adversarial import apply_round_attacks, apply_min_max_attack_to_params
from fl.ordering import order_results


def _lbfgs_hessian_vector_product(
    s_list: List[np.ndarray],
    y_list: List[np.ndarray],
    v: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute Hessian-vector product H * v using L-BFGS two-loop recursion
    over stored history pairs (s_k, y_k).
    """
    m = len(s_list)
    if m == 0:
        return v.copy()

    q = v.copy()
    alphas = [0.0] * m
    rhos = [0.0] * m

    for k in range(m - 1, -1, -1):
        s_k = s_list[k]
        y_k = y_list[k]
        denom = float(np.dot(s_k, y_k))
        rho_k = 1.0 / (denom + eps) if abs(denom) > eps else 0.0
        rhos[k] = rho_k
        alpha_k = rho_k * float(np.dot(s_k, q))
        alphas[k] = alpha_k
        q = q - alpha_k * y_k

    # Initial Hessian scaling gamma_0 = (s_{m-1}^T y_{m-1}) / (s_{m-1}^T s_{m-1})
    s_last = s_list[-1]
    y_last = y_list[-1]
    denom_scale = float(np.dot(s_last, s_last))
    gamma_0 = float(np.dot(s_last, y_last)) / (denom_scale + eps) if denom_scale > eps else 1.0
    r = gamma_0 * q

    for k in range(m):
        s_k = s_list[k]
        y_k = y_list[k]
        beta_k = rhos[k] * float(np.dot(y_k, r))
        r = r + s_k * (alphas[k] - beta_k)

    return r


class FLDetectorStrategy(FedAvg):
    """
    FLDetector strategy (Zhang et al., KDD 2022).

    Predicts client updates from history using an L-BFGS Hessian approximation
    over a sliding window of rounds, scores prediction error, detects and excludes
    anomalous clients, and aggregates the remaining clients via Coordinate-wise Trimmed Mean.

    Args:
        window:          History window size W for L-BFGS (default: 10).
        start_round:     Round to begin anomaly detection T_start (default: 10).
        beta:            Coordinate-wise trimming parameter for accepted aggregation (default: 0.1).
        num_clients:     Total number of clients in federation (default: 20).
    """

    def __init__(
        self,
        window: int = 10,
        start_round: int = 10,
        beta: float = 0.1,
        num_clients: int = 20,
        global_model: Optional[object] = None,
        attack_type: Optional[str] = None,
        attack_kwargs: Optional[dict] = None,
        malicious_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = int(window)
        self.start_round = int(start_round)
        self.beta = float(beta)
        self.num_clients = num_clients
        self.global_model = global_model
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs or {}
        self.malicious_ids = malicious_ids or []
        self.round_logs: List[Dict] = []

        # FLDetector State
        self.s_history: List[np.ndarray] = []           # Model step history s^(k) = w^(k) - w^(k-1)
        self.y_history: List[np.ndarray] = []           # Gradient diff history y^(k) = g^(k) - g^(k-1)
        self.client_history: Dict[int, np.ndarray] = {} # Client last observed update
        self.last_global_params: Optional[np.ndarray] = None
        self.last_avg_gradient: Optional[np.ndarray] = None
        self.detected_clients: Set[int] = set()

    def reset(self):
        """Reset historical state across simulation runs."""
        self.s_history.clear()
        self.y_history.clear()
        self.client_history.clear()
        self.last_global_params = None
        self.last_avg_gradient = None
        self.detected_clients.clear()
        self.round_logs.clear()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Deterministic aggregation order
        results = order_results(results)

        client_ids = [int(proxy.cid) for proxy, _ in results]
        params_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if self.attack_type and self.global_model is not None:
            global_params = self.global_model.get_parameters()
            params_list = apply_round_attacks(
                client_params=params_list,
                global_params=global_params,
                client_ids=client_ids,
                malicious_ids=self.malicious_ids,
                attack_type=self.attack_type,
                attack_kwargs=self.attack_kwargs,
                global_model=self.global_model,
                server_round=server_round,
            )

        n = len(params_list)
        n_layers = len(params_list[0])

        if self.global_model is not None:
            global_params = self.global_model.get_parameters()
            if len(global_params) != n_layers:
                global_params = global_params[:n_layers]
        else:
            global_params = [
                np.mean([p[l] for p in params_list], axis=0)
                for l in range(n_layers)
            ]

        flat_global = np.concatenate([g.flatten() for g in global_params]).astype(np.float64)

        # ── Step 1: Compute flat client pseudo-gradients g_i = w^(t) - w_i ───
        flat_gradients = []
        for p in params_list:
            upd = [g - c for g, c in zip(global_params, p)]
            flat_g = np.concatenate([u.flatten() for u in upd]).astype(np.float64)
            flat_gradients.append(flat_g)

        # ── Step 2: Prediction error scoring using L-BFGS ────────────────────
        scores: Dict[int, float] = {}
        eps = 1e-10

        if server_round >= self.start_round and len(self.s_history) > 0 and self.last_global_params is not None:
            s_current = flat_global - self.last_global_params
            for cid, g_i in zip(client_ids, flat_gradients):
                if cid in self.client_history:
                    g_prev = self.client_history[cid]
                    # Predicted gradient change via L-BFGS Hessian-vector product
                    delta_g = _lbfgs_hessian_vector_product(self.s_history, self.y_history, s_current)
                    g_pred = g_prev + delta_g
                    # Prediction inconsistency: 1 - cosine similarity
                    norm_i = float(np.linalg.norm(g_i))
                    norm_pred = float(np.linalg.norm(g_pred))
                    if norm_i > eps and norm_pred > eps:
                        cos_sim = float(np.dot(g_i, g_pred)) / (norm_i * norm_pred)
                        score = float(1.0 - cos_sim)
                    else:
                        score = 0.0
                else:
                    score = 0.0
                scores[cid] = score

        # ── Step 3: Detection & Filtering ────────────────────────────────────
        flagged_in_round: Set[int] = set()
        if scores and server_round >= self.start_round:
            score_vals = np.array(list(scores.values()))
            if len(score_vals) >= 3 and np.std(score_vals) > eps:
                med = float(np.median(score_vals))
                iqr = float(np.percentile(score_vals, 75) - np.percentile(score_vals, 25))
                # Outlier threshold: scores significantly above median
                thresh = med + 1.5 * max(iqr, 0.1)
                for cid, sc in scores.items():
                    if sc > thresh:
                        flagged_in_round.add(cid)
                        self.detected_clients.add(cid)

        # Keep non-flagged clients for aggregation
        accepted_indices = [
            i for i, cid in enumerate(client_ids)
            if cid not in flagged_in_round
        ]
        # Safety fallback: if all flagged, keep all
        if not accepted_indices:
            accepted_indices = list(range(n))

        accepted_params = [params_list[i] for i in accepted_indices]
        n_acc = len(accepted_params)

        # ── Step 4: Aggregate accepted clients via Trimmed Mean ───────────────
        k_trim = int(np.floor(self.beta * n_acc))
        if 2 * k_trim >= n_acc:
            k_trim = 0

        aggregated = []
        for l in range(n_layers):
            stacked = np.stack([accepted_params[i][l] for i in range(n_acc)], axis=0)
            sorted_stacked = np.sort(stacked, axis=0)
            if k_trim > 0:
                trimmed = sorted_stacked[k_trim:-k_trim]
            else:
                trimmed = sorted_stacked
            aggregated.append(np.mean(trimmed, axis=0))

        # ── Step 5: Update historical buffers ────────────────────────────────
        flat_agg = np.concatenate([a.flatten() for a in aggregated]).astype(np.float64)
        avg_gradient = flat_global - flat_agg

        if self.last_global_params is not None and self.last_avg_gradient is not None:
            s_k = flat_global - self.last_global_params
            y_k = avg_gradient - self.last_avg_gradient
            self.s_history.append(s_k)
            self.y_history.append(y_k)
            if len(self.s_history) > self.window:
                self.s_history.pop(0)
                self.y_history.pop(0)

        self.last_global_params = flat_global.copy()
        self.last_avg_gradient = avg_gradient.copy()

        # Update per-client last observed gradients
        for cid, g_i in zip(client_ids, flat_gradients):
            self.client_history[cid] = g_i.copy()

        log = {
            "round":               server_round,
            "fldetector_flagged":  list(flagged_in_round),
            "num_accepted":        int(n_acc),
            "total_detected":      len(self.detected_clients),
        }
        self.round_logs.append(log)

        if self.global_model is not None:
            self.global_model.set_parameters(aggregated)

        return ndarrays_to_parameters(aggregated), log
