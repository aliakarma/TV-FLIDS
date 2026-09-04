"""
attacks/adversarial.py
Adversarial attack implementations for FL poisoning experiments.
Reference: Guide §9

In addition to the five baseline attacks (label_flip, gradient_scale,
noise_injection, min_max, backdoor), this module implements two white-box
*adaptive* attacks purpose-built to evade trust/verification.py's
three-check verification gate (Guide §8 / paper Table XI):

  - ACK1 ("Check-1 evasion"): a per-client, training-time attack. See
    `AdversarialAttackFactory.ack1_prepare_proxy_val` and its wiring in
    `fl/client.py::TVFLIDSClient.fit()`.
  - ACK2 ("Check-2 coalition"): a strategy-level, post-training coalition
    attack. See `apply_ack2_attack_to_params` / `AdversarialAttackFactory
    .ack2_coalition_shift` and its wiring in
    `fl/strategy.py::TVFLIDSStrategy.aggregate_fit()`.

Both are documented in detail at their definitions below, including the
explicit threat-model assumptions each one relies on.
"""

import numpy as np
from typing import List, Optional, Tuple


class AdversarialAttackFactory:
    """
    Factory class for creating adversarial perturbations.

    All attacks are implemented as static methods.
    Injection points:
      - Data-level  (before local training): label_flip, backdoor
      - Gradient-level (after local training): gradient_scale, noise_injection
    """

    @staticmethod
    def label_flip(y: np.ndarray, source_class: Optional[int] = None,
                   target_class: int = 0, flip_ratio: float = 1.0,
                   seed: int = 42) -> np.ndarray:
        """
        Label Flipping Attack (Data Poisoning).
        Flips attack labels -> 'normal' (0) to evade IDS detection.

        Args:
            y:            Label array.
            source_class: Which class to flip (None = all non-target classes).
            target_class: Class to flip to (0 = normal).
            flip_ratio:   Fraction of eligible labels to flip.
            seed:         Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
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
    def gradient_scale(client_params: List[np.ndarray],
                        global_params: List[np.ndarray],
                        scale_factor: float = 10.0) -> List[np.ndarray]:
        """
        Gradient Scaling Attack (Model Poisoning).
        Amplifies gradient update by scale_factor to dominate aggregation.

        Injection point: after local training (gradient-level).
        """
        delta = [c - g for c, g in zip(client_params, global_params)]
        scaled_delta = [scale_factor * d for d in delta]
        return [g + sd for g, sd in zip(global_params, scaled_delta)]

    @staticmethod
    def noise_injection(client_params: List[np.ndarray],
                        noise_std: float = 0.5,
                        seed: int = 42) -> List[np.ndarray]:
        """
        Noise Injection Attack.
        Adds large Gaussian noise to all parameters.

        Injection point: after local training (gradient-level).
        """
        rng = np.random.RandomState(seed)
        return [
            p + rng.normal(0, noise_std, p.shape).astype(np.float32)
            for p in client_params
        ]

    @staticmethod
    def min_max_attack(client_params: List[np.ndarray],
                       global_params: List[np.ndarray],
                       all_updates: List[List[np.ndarray]],
                       gamma: float = 2.0) -> List[np.ndarray]:
        """
        Min-Max Attack (Shejwalkar & Houmansadr, NDSS 2021).
        Maximizes deviation from honest aggregate while staying within the
        norm ball of honest updates, minimizing detectability.

        Reference: https://arxiv.org/abs/2103.06820
        """
        honest_norms = [
            np.linalg.norm(np.concatenate([p.flatten() for p in u]))
            for u in all_updates
        ]
        bound = np.mean(honest_norms) + gamma * np.std(honest_norms)

        delta = [c - g for c, g in zip(client_params, global_params)]
        flat = np.concatenate([d.flatten() for d in delta])
        scale = min(bound / (np.linalg.norm(flat) + 1e-8), gamma)
        return [g + scale * d for g, d in zip(global_params, delta)]

    @staticmethod
    def backdoor_attack(X: np.ndarray, y: np.ndarray,
                        trigger_feature_idx: int = 0,
                        trigger_value: float = 1.0,
                        target_class: int = 0,
                        poison_ratio: float = 0.1,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backdoor Attack (Advanced Data Poisoning).
        Inserts a trigger pattern into training samples, labels them target_class.

        In IoT IDS context: specific feature value acts as trigger causing the
        model to predict 'normal' regardless of actual traffic.

        Injection point: before local training (data-level).
        """
        rng = np.random.RandomState(seed)
        X_poison = X.copy()
        y_poison = y.copy()

        n_poison = max(1, int(len(X) * poison_ratio))
        poison_idx = rng.choice(len(X), size=n_poison, replace=False)

        X_poison[poison_idx, trigger_feature_idx] = trigger_value
        y_poison[poison_idx] = target_class

        return X_poison, y_poison

    # ── ACK1: Check-1 (loss-consistency) evasion ────────────────────────

    @staticmethod
    def ack1_prepare_proxy_val(X: np.ndarray, y: np.ndarray,
                               proxy_ratio: float = 0.15,
                               seed: int = 42
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ACK1 ("Check-1 evasion") — proxy-validation split.

        Threat model (white-box-but-not-omniscient, per the paper's Section
        XI adaptive-attack description): the adversary knows *how* Check 1
        works (trust/verification.py rejects a tentative update whenever
        `global_loss - loss_after < loss_threshold`, i.e. whenever the
        update fails to improve loss on the server's held-out D_val), but it
        cannot read the server's actual D_val samples — it does not have
        network access to the server's validation set, only to its own
        local shard. It therefore reconstructs a *proxy* for D_val out of a
        clean (unflipped, unpoisoned) slice of its own local data, on the
        assumption its own traffic distribution is not radically dissimilar
        from the server's. This proxy is used purely to compute an
        auxiliary loss term during local training (see
        `fl/client.py::TVFLIDSClient.fit()`, attack_type == 'ack1_evasion')
        — it is never sent anywhere and never touches the server.

        Args:
            X, y:        Client's local training data.
            proxy_ratio: Fraction of the client's data reserved (unmodified)
                         as the D_val proxy.
            seed:        Random seed for reproducibility.

        Returns:
            (X_pool, y_pool, X_proxy, y_proxy) — X_pool/y_pool is the
            remainder available for poisoning + normal training; X_proxy/
            y_proxy is the clean slice used only for the auxiliary loss.
        """
        rng = np.random.RandomState(seed)
        n = len(X)
        if n <= 1:
            return X.copy(), y.copy(), X.copy(), y.copy()

        n_proxy = max(1, min(n - 1, int(n * proxy_ratio)))
        idx = rng.permutation(n)
        proxy_idx, pool_idx = idx[:n_proxy], idx[n_proxy:]
        return X[pool_idx].copy(), y[pool_idx].copy(), X[proxy_idx].copy(), y[proxy_idx].copy()

    # ── ACK2: Check-2 (cosine-similarity) coalition evasion ─────────────

    @staticmethod
    def ack2_coalition_shift(delta: List[np.ndarray],
                             coalition_shift: List[np.ndarray],
                             poison_strength: float = 1.0) -> List[np.ndarray]:
        """
        ACK2 ("Check-2 coalition evasion") — per-client half of the
        coalition attack.

        Threat model: `malicious_ids` clients collude within a round (e.g.
        via an out-of-band side-channel — a Sybil coordinator, shared
        infrastructure, etc.) but do NOT observe honest clients' updates
        before submitting their own (the coalition is white-box on the
        *defense mechanism*, i.e. it knows Check 2 in trust/verification.py
        flags an update whenever its cosine similarity to
        `pseudo = mean(ALL submitted updates that round)` falls below
        `cosine_threshold`, but it is not an omniscient man-in-the-middle
        over its honest peers).

        Step 1 (poisoning): the client's own honestly-trained delta is
        sign-flipped (gradient-ascent-style — a well-known Byzantine
        poisoning primitive: pushing directly away from the loss-improving
        direction rather than a random or magnitude-only perturbation)
        scaled by `poison_strength`.

        Step 2 (coalition coordination, done by the caller,
        `apply_ack2_attack_to_params`): every colluding member's poisoned
        delta is combined with the SAME shared `coalition_shift` term. This
        keeps colluding members mutually cosine-aligned with each other
        (so, relative to the true honest majority, they look like a
        coherent block rather than n independent outliers) and — because
        the coalition is a nontrivial fraction of the round's submissions —
        drags the round's own `pseudo` mean toward the same poisoned
        direction, raising each malicious member's post-hoc cosine
        similarity to `pseudo` above what an uncoordinated (e.g. naive
        gradient-ascent) attacker would achieve.

        Args:
            delta:           This client's own (pre-attack) parameter delta.
            coalition_shift: Shared shift term, common to every colluding
                              member this round (see
                              `apply_ack2_attack_to_params`).
            poison_strength: Magnitude of the sign-flip poisoning step.
        """
        poisoned = [-poison_strength * d for d in delta]
        return [p + s for p, s in zip(poisoned, coalition_shift)]


# Pre-defined attack configurations matching Guide §9.2
ATTACK_CONFIGS = {
    "no_attack":          {"ratio": 0.0,  "type": None},
    "label_flip_10":      {"ratio": 0.10, "type": "label_flip"},
    "label_flip_20":      {"ratio": 0.20, "type": "label_flip"},
    "label_flip_30":      {"ratio": 0.30, "type": "label_flip"},
    "gradient_scale_10":  {"ratio": 0.10, "type": "gradient_scale", "factor": 10.0},
    "gradient_scale_30":  {"ratio": 0.30, "type": "gradient_scale", "factor": 10.0},
    "noise_30":           {"ratio": 0.30, "type": "noise", "std": 0.5},
    "backdoor_20":        {"ratio": 0.20, "type": "backdoor", "poison_ratio": 0.1},
    "min_max_30":         {"ratio": 0.30, "type": "min_max", "gamma": 2.0},
    # Adaptive attacks purpose-built to evade the 3-check verification gate
    # (trust/verification.py). Paper §XI / Table XI, 30% Byzantine ratio
    # (matching the convention of the other "_30"-suffixed configs above).
    "ack1_evasion_30":    {"ratio": 0.30, "type": "ack1_evasion",
                            "flip_ratio": 1.0, "proxy_val_ratio": 0.15,
                            "aux_loss_weight": 0.5},
    "ack2_coalition_30":  {"ratio": 0.30, "type": "ack2_coalition",
                            "poison_strength": 1.0, "shift_scale": 1.0},
}


def apply_min_max_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    gamma: float = 2.0,
) -> List[List[np.ndarray]]:
    """Apply Min-Max attack to malicious client parameters in-place."""
    if not malicious_ids:
        return client_params
    malicious_set = set(malicious_ids)
    if not malicious_set:
        return client_params

    all_updates = [
        [c - g for c, g in zip(params, global_params)]
        for params in client_params
    ]

    for idx, cid in enumerate(client_ids):
        if cid in malicious_set:
            client_params[idx] = AdversarialAttackFactory.min_max_attack(
                client_params[idx], global_params, all_updates, gamma=gamma
            )
    return client_params


def apply_ack2_attack_to_params(
    client_params: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    client_ids: List[int],
    malicious_ids: List[int],
    poison_strength: float = 1.0,
    shift_scale: float = 1.0,
) -> List[List[np.ndarray]]:
    """
    ACK2 ("Check-2 coalition evasion") — strategy-level orchestration.

    Applied post-training, before `trust/verification.py::verify_all()`,
    exactly like `apply_min_max_attack_to_params` (this is the required
    integration point since Check 2 — cosine similarity to the round's
    pseudo-gradient — only exists in `TVFLIDSStrategy`). Needs visibility
    across all colluding malicious clients' submitted updates within the
    round, which only the strategy (not an individual client) has.

    Algorithm:
      1. For every malicious client i, compute its raw trained delta
         d_i = client_params[i] - global_params.
      2. Sign-flip (gradient-ascent-style) each d_i to build a genuinely
         poisoned per-client direction: d_i^poison = -poison_strength * d_i.
      3. Compute the coalition's shared bias term as the MEAN of the
         coalition's own poisoned directions (i.e. "computed across all
         malicious clients' updates that round"):
             shift = shift_scale * mean_i(d_i^poison)
      4. Every colluding member's final delta is
             d_i' = d_i^poison + shift
         so all malicious members carry the identical `shift` term, making
         them mutually cosine-aligned and jointly dragging the round's
         pseudo-gradient (mean of ALL submitted updates, computed later in
         `VerificationModule.verify_all`) toward the same poisoned
         direction — the "colluding-coalition pseudo-gradient shifting"
         mechanism.

    See `AdversarialAttackFactory.ack2_coalition_shift` for the per-client
    half of this computation and its full threat-model docstring.
    """
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
    poisoned_deltas = [
        [-poison_strength * d for d in delta] for delta in mal_deltas
    ]
    coalition_mean = [
        np.mean([pd[l] for pd in poisoned_deltas], axis=0)
        for l in range(len(global_params))
    ]
    shift = [shift_scale * c for c in coalition_mean]

    for pos, delta in zip(mal_positions, mal_deltas):
        new_delta = AdversarialAttackFactory.ack2_coalition_shift(
            delta, shift, poison_strength=poison_strength
        )
        client_params[pos] = [g + d for g, d in zip(global_params, new_delta)]
    return client_params


def get_malicious_client_ids(
    num_clients: int,
    attack_ratio: float,
    seed: int = 42,
) -> List[int]:
    """
    Deterministically select which clients are adversarial.

    Args:
        num_clients:  Total number of FL clients.
        attack_ratio: Fraction to designate as malicious (e.g. 0.30).
        seed:         Random seed for reproducibility.

    Returns:
        Sorted list of adversarial client IDs.
    """
    rng = np.random.default_rng(seed)
    n_malicious = max(0, int(num_clients * attack_ratio))
    if n_malicious == 0:
        return []
    all_ids = np.arange(num_clients)
    malicious = rng.choice(all_ids, n_malicious, replace=False).tolist()
    return sorted(int(x) for x in malicious)
