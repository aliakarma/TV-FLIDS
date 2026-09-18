"""
attacks/adversarial.py
Adversarial attack implementations for FL poisoning experiments.
Reference: IEEE TIFS Manuscript §III, §VII, and Supplementary §S4.

This module implements:
  - Baseline Primitives:
      - Label Flipping (LF)
      - Gradient Scaling (GS)
      - Noise Injection (NI)
      - Backdoor (BD)
      - Random Attack-to-Attack Label Flipping (LF-R)
  - Optimization-based Attacks (Shejwalkar & Houmansadr, NDSS 2021):
      - Min-Max (Partial & Omniscient)
      - Min-Sum (Partial & Omniscient)
  - Adaptive Evasion Attacks:
      - ACK1 (Acceptance test evasion: surrogate validation loss hinge)
      - ACK2 (Trust floor manipulation: benign-ward gradient with ACK1 hinge line search)
      - ACK3 (Class-balanced evasion: DoS/Probe only relabeling + dual loss & balanced hinges)
      - ACK4 (Meta-weight evasion: Algorithm 1 simulation over mixture grid)
      - On-off attack schedules (k in {10, 20, 30, 50} for LF and ACK2)
  - Knowledge Tier Parameterization (K0, K1, K2).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attacks.knowledge import KnowledgeTier, ValidationEstimateProvider, NSLKDD_VAL_QUOTAS


class AdversarialAttackFactory:
    """
    Factory class for creating adversarial perturbations.

    Injection points:
      - Data-level (before local training): label_flip, label_flip_random, backdoor
      - Local training-level: ack1, ack3
      - Gradient/Model-level (after local training): gradient_scale, noise_injection
      - Round/Strategy-level (coalition coordination): min_max, min_sum, ack2, ack4
    """

    # ── 1. Baseline Poisoning Primitives ──────────────────────────────────────

    @staticmethod
    def label_flip(
        y: np.ndarray,
        source_class: Optional[int] = None,
        target_class: int = 0,
        flip_ratio: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Label Flipping Attack (Data Poisoning).
        Flips non-benign / source_class labels -> target_class (0 = normal).
        """
        rng = np.random.default_rng(seed)
        y_corrupt = y.copy()

        if source_class is None:
            eligible = np.where(y_corrupt != target_class)[0]
        else:
            eligible = np.where(y_corrupt == source_class)[0]

        if len(eligible) == 0:
            return y_corrupt

        n_flip = max(1, int(len(eligible) * flip_ratio))
        flip_idx = rng.choice(eligible, size=n_flip, replace=False)
        y_corrupt[flip_idx] = target_class
        return y_corrupt

    @staticmethod
    def label_flip_random(
        y: np.ndarray,
        num_classes: int = 5,
        flip_ratio: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        LF-R Attack (Benign-Routing Intervention, Paper §VII-C / Table V).
        Relabels attack records (y != 0) to a randomly chosen attack class (y' in {1, ..., K-1}).
        Benign records (y == 0) remain untouched.

        Args:
            y: Label array.
            num_classes: Total number of classes K (default 5 for NSL-KDD).
            flip_ratio: Fraction of attack samples to relabel.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        y_corrupt = y.copy()
        attack_classes = list(range(1, num_classes))
        if len(attack_classes) == 0:
            return y_corrupt

        eligible = np.where(y_corrupt != 0)[0]
        if len(eligible) == 0:
            return y_corrupt

        n_flip = max(1, int(len(eligible) * flip_ratio))
        flip_idx = rng.choice(eligible, size=n_flip, replace=False)

        for idx in flip_idx:
            orig_label = y_corrupt[idx]
            other_attacks = [c for c in attack_classes if c != orig_label]
            if other_attacks:
                y_corrupt[idx] = rng.choice(other_attacks)
            else:
                y_corrupt[idx] = attack_classes[0]

        return y_corrupt

    @staticmethod
    def gradient_scale(
        client_params: List[np.ndarray],
        global_params: List[np.ndarray],
        scale_factor: float = 10.0,
    ) -> List[np.ndarray]:
        """Gradient Scaling Attack (Model Poisoning)."""
        delta = [c - g for c, g in zip(client_params, global_params)]
        scaled_delta = [scale_factor * d for d in delta]
        return [g + sd for g, sd in zip(global_params, scaled_delta)]

    @staticmethod
    def noise_injection(
        client_params: List[np.ndarray],
        noise_std: float = 0.5,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """Noise Injection Attack (Model Poisoning)."""
        rng = np.random.default_rng(seed)
        return [
            p + rng.normal(0, noise_std, p.shape).astype(np.float32)
            for p in client_params
        ]

    @staticmethod
    def backdoor_attack(
        X: np.ndarray,
        y: np.ndarray,
        trigger_feature_idx: int = 0,
        trigger_value: float = 1.0,
        target_class: int = 0,
        poison_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backdoor Attack (Advanced Data Poisoning)."""
        rng = np.random.default_rng(seed)
        X_poison = X.copy()
        y_poison = y.copy()

        n_poison = max(1, int(len(X) * poison_ratio))
        poison_idx = rng.choice(len(X), size=n_poison, replace=False)

        X_poison[poison_idx, trigger_feature_idx] = trigger_value
        y_poison[poison_idx] = target_class
        return X_poison, y_poison

    # ── 2. Optimization-based Attacks (Min-Max & Min-Sum) ─────────────────────

    @staticmethod
    def _flatten_params(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    @staticmethod
    def _unflatten_params(flat: np.ndarray, templates: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        offset = 0
        for t in templates:
            size = t.size
            out.append(flat[offset:offset + size].reshape(t.shape).astype(t.dtype))
            offset += size
        return out

    @classmethod
    def min_max_attack(
        cls,
        reference_updates: List[List[np.ndarray]],
        global_params: List[np.ndarray],
        variant: str = "partial",
        gamma_init: float = 2.0,
        max_iter: int = 25,
    ) -> List[np.ndarray]:
        """
        Min-Max Attack (Shejwalkar & Houmansadr, NDSS 2021).
        Paper §III-C line 145: Inverse standard-deviation perturbation direction.

        Constraint: max_{u in U} ||Delta_m - u||_2 <= max_{u, v in U} ||u - v||_2

        Args:
            reference_updates: List of model parameter deltas from reference cohort U
                               (coalition updates for 'partial', all honest for 'omniscient').
            global_params: Global model parameters w^(t).
            variant: 'partial' or 'omniscient'.
            gamma_init: Initial scaling bound.
            max_iter: Binary search iterations.

        Returns:
            Poisoned parameters list matching global_params shape.
        """
        if len(reference_updates) == 0:
            return [g.copy() for g in global_params]

        flat_updates = np.array([cls._flatten_params(u) for u in reference_updates])  # (M, P)
        M, P = flat_updates.shape

        mean_update = np.mean(flat_updates, axis=0)  # (P,)
        std_update = np.std(flat_updates, axis=0)    # (P,)

        # Inverse standard-deviation direction: -sgn(mean) / (std + eps)
        inv_std_dir = -np.sign(mean_update) / (std_update + 1e-8)
        norm_dir = np.linalg.norm(inv_std_dir)
        if norm_dir > 1e-12:
            v_p = inv_std_dir / norm_dir
        else:
            v_p = -np.sign(mean_update) / (np.linalg.norm(np.sign(mean_update)) + 1e-8)

        if M <= 1:
            # Fallback if only 1 reference update available
            scale = np.linalg.norm(mean_update) * 1.5
            mal_delta = mean_update + scale * v_p
            return [g + d for g, d in zip(global_params, cls._unflatten_params(mal_delta, global_params))]

        # Maximum pairwise distance between reference updates
        dists = []
        for i in range(M):
            for j in range(i + 1, M):
                dists.append(np.linalg.norm(flat_updates[i] - flat_updates[j]))
        max_pairwise_dist = max(dists) if dists else np.linalg.norm(mean_update)

        # Binary search for gamma: max_{u in U} ||(mean + gamma * v_p) - u|| <= max_pairwise_dist
        low = 0.0
        high = float(max_pairwise_dist * 5.0 + 10.0)

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            candidate_delta = mean_update + mid * v_p
            cand_max_dist = max(np.linalg.norm(candidate_delta - u) for u in flat_updates)
            if cand_max_dist <= max_pairwise_dist:
                low = mid
            else:
                high = mid

        best_gamma = low
        best_delta = mean_update + best_gamma * v_p
        return [g + d for g, d in zip(global_params, cls._unflatten_params(best_delta, global_params))]

    @classmethod
    def min_sum_attack(
        cls,
        reference_updates: List[List[np.ndarray]],
        global_params: List[np.ndarray],
        variant: str = "partial",
        max_iter: int = 25,
    ) -> List[np.ndarray]:
        """
        Min-Sum Attack (Shejwalkar & Houmansadr, NDSS 2021).
        Paper §III-C line 145: Inverse standard-deviation perturbation direction.

        Constraint: sum_{u in U} ||Delta_m - u||_2 <= max_{v in U} sum_{u in U} ||v - u||_2

        Args:
            reference_updates: List of model parameter deltas from reference cohort U
                               (coalition updates for 'partial', all honest for 'omniscient').
            global_params: Global model parameters w^(t).
            variant: 'partial' or 'omniscient'.
            max_iter: Binary search iterations.

        Returns:
            Poisoned parameters list matching global_params shape.
        """
        if len(reference_updates) == 0:
            return [g.copy() for g in global_params]

        flat_updates = np.array([cls._flatten_params(u) for u in reference_updates])  # (M, P)
        M, P = flat_updates.shape

        mean_update = np.mean(flat_updates, axis=0)  # (P,)
        std_update = np.std(flat_updates, axis=0)    # (P,)

        # Inverse standard-deviation direction: -sgn(mean) / (std + eps)
        inv_std_dir = -np.sign(mean_update) / (std_update + 1e-8)
        norm_dir = np.linalg.norm(inv_std_dir)
        if norm_dir > 1e-12:
            v_p = inv_std_dir / norm_dir
        else:
            v_p = -np.sign(mean_update) / (np.linalg.norm(np.sign(mean_update)) + 1e-8)

        if M <= 1:
            scale = np.linalg.norm(mean_update) * 1.5
            mal_delta = mean_update + scale * v_p
            return [g + d for g, d in zip(global_params, cls._unflatten_params(mal_delta, global_params))]

        # Maximum sum of distances from any reference update to all others
        max_sum_dist = max(sum(np.linalg.norm(v - u) for u in flat_updates) for v in flat_updates)

        # Binary search for gamma: sum_{u in U} ||(mean + gamma * v_p) - u|| <= max_sum_dist
        low = 0.0
        high = float(max_sum_dist * 5.0 + 10.0)

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            candidate_delta = mean_update + mid * v_p
            cand_sum_dist = sum(np.linalg.norm(candidate_delta - u) for u in flat_updates)
            if cand_sum_dist <= max_sum_dist:
                low = mid
            else:
                high = mid

        best_gamma = low
        best_delta = mean_update + best_gamma * v_p
        return [g + d for g, d in zip(global_params, cls._unflatten_params(best_delta, global_params))]

    # ── 3. Adaptive Attacks (ACK1, ACK2, ACK3, ACK4, On-Off) ─────────────────

    @staticmethod
    def ack1_prepare_proxy_val(
        X: np.ndarray,
        y: np.ndarray,
        proxy_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ACK1 proxy-validation split for K0 fallback."""
        rng = np.random.default_rng(seed)
        n = len(X)
        if n <= 1:
            return X.copy(), y.copy(), X.copy(), y.copy()

        n_proxy = max(1, min(n - 1, int(n * proxy_ratio)))
        idx = rng.permutation(n)
        proxy_idx, pool_idx = idx[:n_proxy], idx[n_proxy:]
        return X[pool_idx].copy(), y[pool_idx].copy(), X[proxy_idx].copy(), y[proxy_idx].copy()

    @classmethod
    def ack2_generate_updates(
        cls,
        global_model: nn.Module,
        global_params: List[np.ndarray],
        X_val_hat: np.ndarray,
        y_val_hat: np.ndarray,
        estimated_clipping_radius: float,
        num_malicious: int,
        psi: float = 0.85,
        m: float = 0.0,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> List[List[np.ndarray]]:
        """
        ACK2 (Trust Floor Manipulation Attack, Paper §III-C, Supplementary §S4).
        Computes benign-ward gradient u = -grad(l_flip) / ||grad(l_flip)|| on validation estimate.
        Generates updates s * C_hat * unit(psi * u + sqrt(1 - psi^2) * xi_j) with line search
        over s in {1.0, 0.5, 0.25} keeping ACK1 hinge satisfied.

        Args:
            global_model: Model instance to compute gradients and evaluate losses.
            global_params: Global model parameters w^(t).
            X_val_hat: Validation estimate features (K0, K1, or K2).
            y_val_hat: Validation estimate labels (K0, K1, or K2).
            estimated_clipping_radius: C_hat estimated by coalition.
            num_malicious: Number of malicious clients in the coalition.
            psi: Alignment parameter in {0.7, 0.85, 0.95}.
            m: ACK1 hinge margin in {0, 0.02}.
            device: Torch execution device.
            seed: Random seed.

        Returns:
            List of M parameter updates (each a list of np.ndarray).
        """
        dev = device or torch.device("cpu")
        rng = np.random.default_rng(seed)

        # Set model to global_params
        global_model.set_parameters(global_params)
        global_model.eval()

        # Step 1: Relabel all attack samples in validation estimate to benign (0)
        y_flip = y_val_hat.copy()
        y_flip[y_flip != 0] = 0

        X_t = torch.tensor(X_val_hat, dtype=torch.float32).to(dev)
        y_flip_t = torch.tensor(y_flip, dtype=torch.long).to(dev)
        y_orig_t = torch.tensor(y_val_hat, dtype=torch.long).to(dev)

        # Compute gradient w.r.t model parameters on flipped validation estimate
        criterion = nn.CrossEntropyLoss()
        global_model.zero_grad()
        out_flip = global_model(X_t)
        loss_flip = criterion(out_flip, y_flip_t)
        loss_flip.backward()

        grads = [p.grad.detach().cpu().numpy() for p in global_model.parameters()]
        flat_grad = cls._flatten_params(grads)
        grad_norm = np.linalg.norm(flat_grad)

        if grad_norm > 1e-12:
            u_dir = -flat_grad / grad_norm
        else:
            u_dir = -np.ones_like(flat_grad) / np.sqrt(len(flat_grad))

        # Baseline validation loss on unflipped validation estimate
        with torch.no_grad():
            orig_val_loss = criterion(global_model(X_t), y_orig_t).item()

        C_hat = max(estimated_clipping_radius, 1e-6)
        malicious_deltas = []

        # Step 2: For each malicious client, generate randomized perturbation and line search
        scale_candidates = [1.0, 0.5, 0.25]

        for j in range(num_malicious):
            xi = rng.normal(size=len(u_dir)).astype(np.float32)
            xi_norm = np.linalg.norm(xi)
            xi_unit = xi / (xi_norm + 1e-12)

            d_j = psi * u_dir + np.sqrt(max(0.0, 1.0 - psi**2)) * xi_unit
            d_j_norm = np.linalg.norm(d_j)
            d_j_unit = d_j / (d_j_norm + 1e-12)

            chosen_scale = 0.25
            for scale in scale_candidates:
                cand_delta_flat = scale * C_hat * d_j_unit
                cand_delta_params = cls._unflatten_params(cand_delta_flat, global_params)
                cand_params = [g + d for g, d in zip(global_params, cand_delta_params)]

                global_model.set_parameters(cand_params)
                with torch.no_grad():
                    cand_val_loss = criterion(global_model(X_t), y_orig_t).item()

                # ACK1 hinge check: l_val(w_cand) - l_val(w_global) + m <= 0
                if cand_val_loss - orig_val_loss + m <= 0:
                    chosen_scale = scale
                    break

            final_delta_flat = chosen_scale * C_hat * d_j_unit
            malicious_deltas.append(cls._unflatten_params(final_delta_flat, global_params))

        # Reset global model
        global_model.set_parameters(global_params)
        return malicious_deltas

    @classmethod
    def ack4_generate_updates(
        cls,
        global_model: nn.Module,
        global_params: List[np.ndarray],
        X_val_hat: np.ndarray,
        y_val_hat: np.ndarray,
        coalition_honest_updates: List[List[np.ndarray]],
        ack2_updates: List[List[np.ndarray]],
        estimated_clipping_radius: float,
        num_malicious: int,
        server_round: int = 1,
        initial_trust: float = 0.5,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> List[List[np.ndarray]]:
        """
        ACK4 (Meta-Weight Evasion Attack, Paper §III-C, Supplementary §S4).
        Simulates Algorithm 1 on validation estimate over mixture grid omega in {0, 0.25, 0.5, 0.75, 1}:
          Delta(omega) = C_hat * unit((1 - omega) * Delta_hon + omega * Delta_ACK2)
        Submits omega* maximizing predicted rho_B * <Delta_tilde(omega), u>.

        Args:
            global_model: Model instance.
            global_params: Current global parameters w^(t).
            X_val_hat: Validation estimate features.
            y_val_hat: Validation estimate labels.
            coalition_honest_updates: Honestly trained updates for simulation.
            ack2_updates: ACK2 updates computed for this round.
            estimated_clipping_radius: C_hat.
            num_malicious: Number of malicious clients.
            server_round: Round number t.
            initial_trust: Initial trust score T^(0).
            device: Torch execution device.
            seed: Random seed.

        Returns:
            List of malicious parameter updates for the coalition.
        """
        from trust.trust_scorer import TrustScorer
        from trust.verification import compute_class_balanced_loss, compute_class_balanced_loss_from_tensors

        dev = device or torch.device("cpu")
        C_hat = max(estimated_clipping_radius, 1e-6)

        # Compute benign-ward direction u = -grad(l_flip) / ||grad(l_flip)||
        global_model.set_parameters(global_params)
        global_model.eval()

        y_flip = y_val_hat.copy()
        y_flip[y_flip != 0] = 0

        X_t = torch.tensor(X_val_hat, dtype=torch.float32).to(dev)
        y_flip_t = torch.tensor(y_flip, dtype=torch.long).to(dev)
        y_orig_t = torch.tensor(y_val_hat, dtype=torch.long).to(dev)

        criterion = nn.CrossEntropyLoss()
        global_model.zero_grad()
        loss_flip = criterion(global_model(X_t), y_flip_t)
        loss_flip.backward()

        grads = [p.grad.detach().cpu().numpy() for p in global_model.parameters()]
        flat_u = cls._flatten_params(grads)
        norm_u = np.linalg.norm(flat_u)
        u_unit = -flat_u / (norm_u + 1e-12) if norm_u > 1e-12 else -np.ones_like(flat_u) / np.sqrt(len(flat_u))

        # Mean honest delta and mean ACK2 delta
        if coalition_honest_updates:
            flat_hon_mean = np.mean([cls._flatten_params(u) for u in coalition_honest_updates], axis=0)
        else:
            flat_hon_mean = np.zeros_like(u_unit)

        if ack2_updates:
            flat_ack2_mean = np.mean([cls._flatten_params(u) for u in ack2_updates], axis=0)
        else:
            flat_ack2_mean = C_hat * u_unit

        with torch.no_grad():
            base_logits = global_model(X_t)
            base_loss = criterion(base_logits, y_orig_t).item()
            _, base_bal_loss, _, _ = compute_class_balanced_loss_from_tensors(base_logits, y_orig_t)

        # Linear warmup threshold tau_L(t) = -0.1 + 0.1 * min(1, t / 20)
        tau_L = -0.1 + 0.1 * min(1.0, server_round / 20.0)

        omega_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_omega = 0.5
        best_objective = -float("inf")

        for omega in omega_grid:
            mix_dir = (1.0 - omega) * flat_hon_mean + omega * flat_ack2_mean
            norm_mix = np.linalg.norm(mix_dir)
            mix_unit = mix_dir / (norm_mix + 1e-12)
            cand_delta_flat = C_hat * mix_unit

            # Simulate Algorithm 1 for this candidate update
            cand_delta_params = cls._unflatten_params(cand_delta_flat, global_params)
            cand_model_params = [g + d for g, d in zip(global_params, cand_delta_params)]

            global_model.set_parameters(cand_model_params)
            with torch.no_grad():
                cand_logits = global_model(X_t)
                cand_loss = criterion(cand_logits, y_orig_t).item()
                _, cand_bal_loss, _, _ = compute_class_balanced_loss_from_tensors(cand_logits, y_orig_t)

            loss_diff = base_loss - cand_loss
            is_accepted = (loss_diff >= tau_L)

            if is_accepted:
                # Malicious client accepted: receives high trust update
                # Signal calculation
                bal_diff = (base_bal_loss - cand_bal_loss) / (base_bal_loss + 1e-8)
                A_i = float(np.clip(bal_diff, -1.0, 1.0))
                S_i = 0.8  # Estimated directional alignment
                O_i = 0.0  # Clipped norm outlier signal
                u_i = (1.0 / 3.0) * S_i + (1.0 / 3.0) * A_i - (1.0 / 3.0) * O_i
                s_i = float(np.clip(u_i, 0.0, 1.0))
                # Asymmetric trust update
                lambda_i = 0.9 if s_i >= initial_trust else 0.7
                T_mal = max(0.01, lambda_i * initial_trust + (1.0 - lambda_i) * s_i)
            else:
                # Rejected client: s_i = 0, penalty decay
                T_mal = max(0.01, 0.7 * initial_trust)

            # Predicted weight share rho_B
            # Assuming honest clients retain trust ~ initial_trust
            honest_count = max(1, len(coalition_honest_updates))
            T_hon_total = honest_count * initial_trust
            rho_B = (num_malicious * T_mal) / (num_malicious * T_mal + T_hon_total)

            # Objective: rho_B * <Delta_tilde(omega), u>
            displacement_proj = float(np.dot(cand_delta_flat, u_unit))
            objective = rho_B * displacement_proj

            if objective > best_objective:
                best_objective = objective
                best_omega = omega

        # Generate updates for all coalition members using best_omega
        mix_dir = (1.0 - best_omega) * flat_hon_mean + best_omega * flat_ack2_mean
        mix_unit = mix_dir / (np.linalg.norm(mix_dir) + 1e-12)

        rng = np.random.default_rng(seed)
        malicious_deltas = []
        for j in range(num_malicious):
            # Add small random perturbation to avoid identical updates
            xi = rng.normal(size=len(mix_unit)).astype(np.float32) * 0.05
            d_j = mix_unit + xi
            d_j_unit = d_j / (np.linalg.norm(d_j) + 1e-12)
            delta_flat = C_hat * d_j_unit
            malicious_deltas.append(cls._unflatten_params(delta_flat, global_params))

        # Reset global model
        global_model.set_parameters(global_params)
        return malicious_deltas

    @staticmethod
    def is_on_off_active(server_round: int, k: int = 30, mode: str = "switch") -> bool:
        """
        On-off attack schedule check (Paper §III-C line 152, Supplementary §S4).
        Byzantine clients behave honestly for k rounds (server_round <= k) and then attack (server_round > k).

        Args:
            server_round: Current round number t (1-indexed).
            k: Honest phase duration in {10, 20, 30, 50}.
            mode: 'switch' (paper definition: honest for k rounds, then attack)
                  or 'periodic' (alternate honest for k rounds, attack for k rounds).

        Returns:
            True if attack is active in this round, False if honest.
        """
        if mode == "periodic":
            return ((server_round - 1) // k) % 2 == 1
        return server_round > k


# Pre-defined attack configurations matching Guide §9.2 and IEEE TIFS Experimental Matrix
ATTACK_CONFIGS = {
    "no_attack":            {"ratio": 0.0,  "type": None},
    "label_flip_10":        {"ratio": 0.10, "type": "label_flip"},
    "label_flip_20":        {"ratio": 0.20, "type": "label_flip"},
    "label_flip_30":        {"ratio": 0.30, "type": "label_flip"},
    "label_flip_40":        {"ratio": 0.40, "type": "label_flip"},
    "gradient_scale_10":    {"ratio": 0.10, "type": "gradient_scale", "factor": 10.0},
    "gradient_scale_30":    {"ratio": 0.30, "type": "gradient_scale", "factor": 10.0},
    "gradient_scale_100":   {"ratio": 0.30, "type": "gradient_scale", "factor": 100.0},
    "gradient_scale_1000":  {"ratio": 0.30, "type": "gradient_scale", "factor": 1000.0},
    "gradient_scale_10000": {"ratio": 0.30, "type": "gradient_scale", "factor": 10000.0},
    "noise_30":             {"ratio": 0.30, "type": "noise", "std": 0.5},
    "backdoor_20":          {"ratio": 0.20, "type": "backdoor", "poison_ratio": 0.1},
    "lf_r_30":              {"ratio": 0.30, "type": "lf_r"},
    # Min-Max (Partial and Omniscient)
    "min_max_30":           {"ratio": 0.30, "type": "min_max", "variant": "partial"},
    "min_max_p_30":         {"ratio": 0.30, "type": "min_max", "variant": "partial"},
    "min_max_o_30":         {"ratio": 0.30, "type": "min_max", "variant": "omniscient"},
    # Min-Sum (Partial and Omniscient)
    "min_sum_30":           {"ratio": 0.30, "type": "min_sum", "variant": "partial"},
    "min_sum_p_30":         {"ratio": 0.30, "type": "min_sum", "variant": "partial"},
    "min_sum_o_30":         {"ratio": 0.30, "type": "min_sum", "variant": "omniscient"},
    # Adaptive Attacks with Knowledge Tiers
    "ack1_30":              {"ratio": 0.30, "type": "ack1", "knowledge_tier": "K1"},
    "ack1_k0_30":           {"ratio": 0.30, "type": "ack1", "knowledge_tier": "K0"},
    "ack1_k1_30":           {"ratio": 0.30, "type": "ack1", "knowledge_tier": "K1"},
    "ack1_k2_30":           {"ratio": 0.30, "type": "ack1", "knowledge_tier": "K2"},
    "ack2_30":              {"ratio": 0.30, "type": "ack2", "knowledge_tier": "K1", "psi": 0.85},
    "ack2_k1_30":           {"ratio": 0.30, "type": "ack2", "knowledge_tier": "K1", "psi": 0.85},
    "ack2_k2_30":           {"ratio": 0.30, "type": "ack2", "knowledge_tier": "K2", "psi": 0.85},
    "ack3_30":              {"ratio": 0.30, "type": "ack3", "knowledge_tier": "K1"},
    "ack3_k1_30":           {"ratio": 0.30, "type": "ack3", "knowledge_tier": "K1"},
    "ack3_k2_30":           {"ratio": 0.30, "type": "ack3", "knowledge_tier": "K2"},
    "ack4_30":              {"ratio": 0.30, "type": "ack4", "knowledge_tier": "K1"},
    "ack4_k1_30":           {"ratio": 0.30, "type": "ack4", "knowledge_tier": "K1"},
    "ack4_k2_30":           {"ratio": 0.30, "type": "ack4", "knowledge_tier": "K2"},
    # On-Off Attacks
    "on_off_lf_10":         {"ratio": 0.30, "type": "on_off_lf", "k": 10},
    "on_off_lf_20":         {"ratio": 0.30, "type": "on_off_lf", "k": 20},
    "on_off_lf_30":         {"ratio": 0.30, "type": "on_off_lf", "k": 30},
    "on_off_lf_50":         {"ratio": 0.30, "type": "on_off_lf", "k": 50},
    "on_off_ack2_10":       {"ratio": 0.30, "type": "on_off_ack2", "k": 10, "knowledge_tier": "K1"},
    "on_off_ack2_20":       {"ratio": 0.30, "type": "on_off_ack2", "k": 20, "knowledge_tier": "K1"},
    "on_off_ack2_30":       {"ratio": 0.30, "type": "on_off_ack2", "k": 30, "knowledge_tier": "K1"},
    "on_off_ack2_50":       {"ratio": 0.30, "type": "on_off_ack2", "k": 50, "knowledge_tier": "K1"},
}


# ── Unified Strategy-Level Attack Interception ─────────────────────────────────

def apply_min_max_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    variant: str = "partial",
    gamma: float = 2.0,
) -> List[List[np.ndarray]]:
    """Apply Min-Max attack to malicious client parameters in-place."""
    if not malicious_ids:
        return client_params
    malicious_set = set(malicious_ids)

    mal_positions = [i for i, cid in enumerate(client_ids) if cid in malicious_set]
    honest_positions = [i for i, cid in enumerate(client_ids) if cid not in malicious_set]

    if not mal_positions:
        return client_params

    # Select reference updates based on partial vs omniscient knowledge
    if variant == "omniscient" and honest_positions:
        ref_updates = [
            [c - g for c, g in zip(client_params[i], global_params)]
            for i in honest_positions
        ]
    else:
        ref_updates = [
            [c - g for c, g in zip(client_params[i], global_params)]
            for i in mal_positions
        ]

    poisoned_params = AdversarialAttackFactory.min_max_attack(
        ref_updates, global_params, variant=variant, gamma_init=gamma
    )

    for idx in mal_positions:
        client_params[idx] = [p.copy() for p in poisoned_params]
    return client_params


def apply_min_sum_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    variant: str = "partial",
) -> List[List[np.ndarray]]:
    """Apply Min-Sum attack to malicious client parameters in-place."""
    if not malicious_ids:
        return client_params
    malicious_set = set(malicious_ids)

    mal_positions = [i for i, cid in enumerate(client_ids) if cid in malicious_set]
    honest_positions = [i for i, cid in enumerate(client_ids) if cid not in malicious_set]

    if not mal_positions:
        return client_params

    if variant == "omniscient" and honest_positions:
        ref_updates = [
            [c - g for c, g in zip(client_params[i], global_params)]
            for i in honest_positions
        ]
    else:
        ref_updates = [
            [c - g for c, g in zip(client_params[i], global_params)]
            for i in mal_positions
        ]

    poisoned_params = AdversarialAttackFactory.min_sum_attack(
        ref_updates, global_params, variant=variant
    )

    for idx in mal_positions:
        client_params[idx] = [p.copy() for p in poisoned_params]
    return client_params


def apply_ack2_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    poison_strength: float = 1.0,
    shift_scale: float = 1.0,
    global_model: Optional[nn.Module] = None,
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    psi: float = 0.85,
    seed: int = 42,
) -> List[List[np.ndarray]]:
    """Apply ACK2 attack to malicious client parameters in-place."""
    if not malicious_ids:
        return client_params
    malicious_set = set(malicious_ids)
    mal_positions = [i for i, cid in enumerate(client_ids) if cid in malicious_set]
    if not mal_positions:
        return client_params

    mal_deltas = [
        [c - g for c, g in zip(client_params[i], global_params)]
        for i in mal_positions
    ]

    # If global_model and val_data are provided, use rigorous ACK2 generation
    if global_model is not None and val_data is not None:
        X_val_hat, y_val_hat = val_data
        mal_norms = [np.linalg.norm(AdversarialAttackFactory._flatten_params(d)) for d in mal_deltas]
        c_hat = float(np.median(mal_norms)) if mal_norms else 1.0
        ack2_deltas = AdversarialAttackFactory.ack2_generate_updates(
            global_model=global_model,
            global_params=global_params,
            X_val_hat=X_val_hat,
            y_val_hat=y_val_hat,
            estimated_clipping_radius=c_hat,
            num_malicious=len(mal_positions),
            psi=psi,
            seed=seed,
        )
        for idx, d_ack2 in zip(mal_positions, ack2_deltas):
            client_params[idx] = [g + d for g, d in zip(global_params, d_ack2)]
        return client_params

    # Fallback to shared coalition shift if no val_data is passed
    poisoned_deltas = [
        [-poison_strength * d for d in delta] for delta in mal_deltas
    ]
    coalition_mean = [
        np.mean([pd[l] for pd in poisoned_deltas], axis=0)
        for l in range(len(global_params))
    ]
    shift = [shift_scale * c for c in coalition_mean]

    for pos, delta in zip(mal_positions, mal_deltas):
        p_delta = [-poison_strength * d + s for d, s in zip(delta, shift)]
        client_params[pos] = [g + d for g, d in zip(global_params, p_delta)]
    return client_params


def apply_round_attacks(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: Optional[List[int]],
    attack_type: Optional[str],
    attack_kwargs: Optional[dict] = None,
    global_model: Optional[nn.Module] = None,
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    server_round: int = 1,
    seed: int = 42,
) -> List[List[np.ndarray]]:
    """
    Unified strategy-level attack interceptor called before aggregation.
    Dispatches min_max, min_sum, ack2, ack4, on_off_ack2 across all strategies.
    """
    if not attack_type or not malicious_ids:
        return client_params

    kwargs = attack_kwargs or {}

    # Handle on-off scheduling
    if attack_type == "on_off_ack2":
        k = kwargs.get("k", 30)
        if not AdversarialAttackFactory.is_on_off_active(server_round, k=k):
            return client_params  # Honest phase: do not perturb
        effective_attack = "ack2"
    elif attack_type == "on_off_lf":
        # Handled at client fit level
        return client_params
    else:
        effective_attack = attack_type

    if effective_attack in ("min_max", "min_max_p", "min_max_o"):
        variant = kwargs.get("variant", "omniscient" if effective_attack == "min_max_o" else "partial")
        return apply_min_max_attack_to_params(
            client_params, global_params, client_ids, malicious_ids,
            variant=variant, gamma=kwargs.get("gamma", 2.0)
        )

    elif effective_attack in ("min_sum", "min_sum_p", "min_sum_o"):
        variant = kwargs.get("variant", "omniscient" if effective_attack == "min_sum_o" else "partial")
        return apply_min_sum_attack_to_params(
            client_params, global_params, client_ids, malicious_ids,
            variant=variant
        )

    elif effective_attack in ("ack2", "ack2_coalition"):
        return apply_ack2_attack_to_params(
            client_params, global_params, client_ids, malicious_ids,
            poison_strength=kwargs.get("poison_strength", 1.0),
            shift_scale=kwargs.get("shift_scale", 1.0),
            global_model=global_model,
            val_data=val_data,
            psi=kwargs.get("psi", 0.85),
            seed=seed + server_round,
        )

    elif effective_attack == "ack4":
        if global_model is None or val_data is None:
            return client_params
        X_val_hat, y_val_hat = val_data
        mal_positions = [i for i, cid in enumerate(client_ids) if cid in set(malicious_ids)]
        mal_deltas = [[c - g for c, g in zip(client_params[i], global_params)] for i in mal_positions]
        mal_norms = [np.linalg.norm(AdversarialAttackFactory._flatten_params(d)) for d in mal_deltas]
        c_hat = float(np.median(mal_norms)) if mal_norms else 1.0

        # Compute ACK2 updates as component
        ack2_deltas = AdversarialAttackFactory.ack2_generate_updates(
            global_model=global_model,
            global_params=global_params,
            X_val_hat=X_val_hat,
            y_val_hat=y_val_hat,
            estimated_clipping_radius=c_hat,
            num_malicious=len(mal_positions),
            psi=kwargs.get("psi", 0.85),
            seed=seed + server_round,
        )

        ack4_deltas = AdversarialAttackFactory.ack4_generate_updates(
            global_model=global_model,
            global_params=global_params,
            X_val_hat=X_val_hat,
            y_val_hat=y_val_hat,
            coalition_honest_updates=mal_deltas,
            ack2_updates=ack2_deltas,
            estimated_clipping_radius=c_hat,
            num_malicious=len(mal_positions),
            server_round=server_round,
            seed=seed + server_round,
        )

        for idx, d_ack4 in zip(mal_positions, ack4_deltas):
            client_params[idx] = [g + d for g, d in zip(global_params, d_ack4)]
        return client_params

    return client_params


def get_malicious_client_ids(
    num_clients: int,
    attack_ratio: float,
    seed: int = 42,
) -> List[int]:
    """Deterministically select which clients are adversarial."""
    rng = np.random.default_rng(seed)
    n_malicious = max(0, int(num_clients * attack_ratio))
    if n_malicious == 0:
        return []
    all_ids = np.arange(num_clients)
    malicious = rng.choice(all_ids, n_malicious, replace=False).tolist()
    return sorted(int(x) for x in malicious)


# Module-level aliases for convenience and direct imports
min_max_attack = AdversarialAttackFactory.min_max_attack
min_sum_attack = AdversarialAttackFactory.min_sum_attack
label_flip = AdversarialAttackFactory.label_flip
label_flip_random = AdversarialAttackFactory.label_flip_random
gradient_scale = AdversarialAttackFactory.gradient_scale
noise_injection = AdversarialAttackFactory.noise_injection
backdoor_attack = AdversarialAttackFactory.backdoor_attack
ack2_generate_updates = AdversarialAttackFactory.ack2_generate_updates
ack4_generate_updates = AdversarialAttackFactory.ack4_generate_updates
is_on_off_active = AdversarialAttackFactory.is_on_off_active

