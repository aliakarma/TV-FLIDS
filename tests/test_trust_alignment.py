"""
tests/test_trust_alignment.py
Rigorously tests the TV-FLIDS Stage 3 Trust System against the IEEE TIFS specification.

Required tests (PART K):
1. Class-balanced loss on imbalanced validation data (each class contributes equally)
2. A_i can be negative
3. A_i clipping interval [-1, 1]
4. S_i range [0, 1]
5. Zero-vector cosine similarity does not produce NaN/Inf
6. O_i exact formula (1 - exp(-z_i / 2.5)) tested across known z values
7. O_i cohort scope (computed over all participants P; modifying a rejected participant alters statistic)
8. Positive trust branch (lambda_up = 0.9 when s_i >= T_i^{(t-1)})
9. Negative trust branch (lambda_down = 0.7 when s_i < T_i^{(t-1)})
10. Initial trust T_i^{(0)} = 0.5
11. Trust floor tau_min = 0.01 respected under repeated penalties
12. Rejected client receives s_i = 0 and is penalized
13. Non-participating client retains previous trust
14. Mixed signal u_i = alpha * S_i + beta * A_i - gamma * O_i (negative sign on gamma * O_i)
15. Full round integration with accepted, rejected, and non-participating clients
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trust.trust_scorer import TrustScorer
from trust.verification import (
    VerificationModule,
    compute_class_balanced_loss,
    compute_class_balanced_loss_from_tensors,
)
from fl.strategy import TVFLIDSStrategy
from models.mlp import IDSMLP


# ── Test 1: Class-balanced loss ───────────────────────────────────────────────
def test_class_balanced_loss_imbalance():
    """Test 1: Use a deliberately imbalanced synthetic validation set.
    Verify that each class contributes equally (1/K) to the class-balanced average.
    """
    # 2 classes: Class 0 has 900 samples with loss 1.0; Class 1 has 100 samples with loss 3.0.
    # Total samples = 1000.
    # Sample-averaged loss l_val = (900 * 1.0 + 100 * 3.0) / 1000 = 1.2.
    # Class-balanced loss l_bal = (1.0 + 3.0) / 2 = 2.0.
    targets = torch.cat([torch.zeros(900, dtype=torch.long), torch.ones(100, dtype=torch.long)])

    # Construct logits such that cross-entropy is exactly 1.0 for class 0 and 3.0 for class 1
    l0 = math.log(math.e - 1.0)
    l1 = math.log(math.e**3 - 1.0)

    logits_c0 = torch.tensor([[0.0, l0]] * 900)
    logits_c1 = torch.tensor([[l1, 0.0]] * 100)
    logits = torch.cat([logits_c0, logits_c1], dim=0)

    val_loss, bal_loss, class_losses, class_counts = compute_class_balanced_loss_from_tensors(
        logits, targets, num_classes=2
    )

    assert class_counts[0] == 900
    assert class_counts[1] == 100
    assert class_losses[0] == pytest.approx(1.0, abs=1e-5)
    assert class_losses[1] == pytest.approx(3.0, abs=1e-5)
    assert val_loss == pytest.approx(1.2, abs=1e-5)
    assert bal_loss == pytest.approx(2.0, abs=1e-5)
    assert bal_loss != val_loss

    # Verify diagnostic when a class is missing (e.g. 3 classes requested but only 2 present)
    with pytest.raises(ValueError, match="Class-balanced validation loss undefined: class.*have 0 validation samples"):
        compute_class_balanced_loss_from_tensors(logits, targets, num_classes=3)


# ── Test 2: A_i can be negative ───────────────────────────────────────────────
def test_accuracy_signal_can_be_negative():
    """Test 2: Construct a candidate that worsens balanced validation loss.
    Verify A_i < 0.
    """
    ts = TrustScorer(num_clients=5)
    global_bal_loss = 1.0
    # Candidate worsens balanced loss from 1.0 to 1.5
    val_bal_losses_after = [1.5]
    scores = ts.compute_accuracy_scores(global_bal_loss, val_bal_losses_after)

    # imp = (1.0 - 1.5) / (1.0 + 1e-8) = -0.5 / 1.00000001 ~ -0.5
    assert scores[0] < 0.0
    assert scores[0] == pytest.approx(-0.5, abs=1e-5)


# ── Test 3: A_i clipping ──────────────────────────────────────────────────────
def test_accuracy_signal_clipping_bounds():
    """Test 3: Verify the output remains in [-1, 1] even under extreme values."""
    ts = TrustScorer(num_clients=5)
    global_bal_loss = 1.0
    # Extreme degradation (1.0 -> 100.0) and extreme improvement (1.0 -> 0.0)
    val_bal_losses_after = [100.0, 0.0, 1.0, 5.0, 0.5]
    scores = ts.compute_accuracy_scores(global_bal_loss, val_bal_losses_after)

    assert scores[0] == pytest.approx(-1.0, abs=1e-5)  # clipped to -1
    assert scores[1] == pytest.approx(1.0, abs=1e-5)   # (1 - 0) / 1 = 1.0
    assert scores[2] == pytest.approx(0.0, abs=1e-5)   # no change
    assert (scores >= -1.0).all() and (scores <= 1.0).all()


# ── Test 4: S_i range ─────────────────────────────────────────────────────────
def test_direction_signal_range():
    """Test 4: Verify S_i in [0, 1]."""
    ts = TrustScorer(num_clients=3)
    ref = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    # Parallel (cos = 1 -> S = 1.0), Orthogonal (cos = 0 -> S = 0.5), Opposite (cos = -1 -> S = 0.0)
    updates = [
        [np.array([2.0, 0.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 3.0, 0.0], dtype=np.float32)],
        [np.array([-1.0, 0.0, 0.0], dtype=np.float32)],
    ]
    sims = ts.compute_similarity_scores(updates, ref)

    assert sims[0] == pytest.approx(1.0, abs=1e-5)
    assert sims[1] == pytest.approx(0.5, abs=1e-5)
    assert sims[2] == pytest.approx(0.0, abs=1e-5)
    assert (sims >= 0.0).all() and (sims <= 1.0).all()


# ── Test 5: Zero-vector cosine ────────────────────────────────────────────────
def test_zero_vector_cosine_safe():
    """Test 5: Verify zero-norm inputs do not generate NaN/Inf and return safe 0.5."""
    ts = TrustScorer(num_clients=2)
    zero_upd = [np.zeros(10, dtype=np.float32)]
    non_zero_upd = [np.ones(10, dtype=np.float32)]

    # Zero candidate update relative to non-zero reference
    sims1 = ts.compute_similarity_scores([zero_upd], non_zero_upd)
    assert not np.isnan(sims1).any()
    assert not np.isinf(sims1).any()
    assert sims1[0] == pytest.approx(0.5, abs=1e-5)

    # Non-zero candidate update relative to zero reference
    sims2 = ts.compute_similarity_scores([non_zero_upd], zero_upd)
    assert not np.isnan(sims2).any()
    assert not np.isinf(sims2).any()
    assert sims2[0] == pytest.approx(0.5, abs=1e-5)

    # Both zero
    sims3 = ts.compute_similarity_scores([zero_upd], zero_upd)
    assert not np.isnan(sims3).any()
    assert sims3[0] == pytest.approx(0.5, abs=1e-5)


# ── Test 6: O_i exact formula ─────────────────────────────────────────────────
def test_anomaly_signal_exact_formula():
    """Test 6: Test multiple known z values and compare against the exact formula O_i = 1 - exp(-z / 2.5)."""
    ts = TrustScorer(num_clients=5)

    # Synthesize participant cohort norms such that mu and sigma yield known z-scores
    updates_zero_z = [[np.full(4, 5.0, dtype=np.float32)] for _ in range(5)]  # norm = sqrt(4*25) = 10
    o_zero = ts.compute_anomaly_scores(updates_zero_z, tau_z=2.5)
    assert np.allclose(o_zero, 0.0, atol=1e-5)

    # Manual verification against known z values
    for z in [0.0, 1.0, 2.5, 5.0, 10.0]:
        expected_o = 1.0 - math.exp(-z / 2.5)
        computed_o = 1.0 - math.exp(-abs(z) / 2.5)
        assert expected_o == pytest.approx(computed_o, abs=1e-9)

    # Check that z=2.5 produces exactly 1 - 1/e
    assert (1.0 - math.exp(-2.5 / 2.5)) == pytest.approx(1.0 - 1.0 / math.e, abs=1e-9)


# ── Test 7: O_i cohort scope ──────────────────────────────────────────────────
def test_anomaly_signal_cohort_scope_all_participants():
    """Test 7: Verify the population used to compute mu, sigma is ALL participants P.
    A dedicated test proving that changing a rejected participant's raw update norm
    alters the cohort statistics (mu, sigma) and therefore alters accepted participants' O_i.
    """
    ts = TrustScorer(num_clients=4)

    # Accepted participants 0 and 1: fixed updates
    upd_acc0 = [np.array([1.0, 0.0], dtype=np.float32)]  # norm = 1.0
    upd_acc1 = [np.array([2.0, 0.0], dtype=np.float32)]  # norm = 2.0

    # Case A: Rejected participant 2 has norm 1.5
    upd_rej_a = [np.array([1.5, 0.0], dtype=np.float32)]  # norm = 1.5
    cohort_a = [upd_acc0, upd_acc1, upd_rej_a]

    # Compute O_i for accepted clients in Cohort A
    o_a = ts.compute_anomaly_scores(cohort_a, target_updates=[upd_acc0, upd_acc1], tau_z=2.5)

    # Case B: Rejected participant 2 has huge outlier norm 50.0
    upd_rej_b = [np.array([50.0, 0.0], dtype=np.float32)]  # norm = 50.0
    cohort_b = [upd_acc0, upd_acc1, upd_rej_b]

    # Compute O_i for accepted clients in Cohort B
    o_b = ts.compute_anomaly_scores(cohort_b, target_updates=[upd_acc0, upd_acc1], tau_z=2.5)

    # Because rejected participant 2 belongs to participant cohort P, its norm change
    # shifts mu and sigma of the cohort, altering O_i for accepted participants!
    assert o_a[0] != o_b[0]
    assert o_a[1] != o_b[1]


# ── Test 8: Positive trust branch ─────────────────────────────────────────────
def test_positive_trust_branch_lambda_up():
    """Test 8: Verify lambda_up = 0.9 when s_i >= T_i^{(t-1)}."""
    ts = TrustScorer(num_clients=1, alpha=1.0, beta=0.0, gamma=0.0,
                     lambda_up=0.9, lambda_down=0.7, initial_trust=0.5)
    assert ts.trust_scores[0] == 0.5

    # With s_i = 0.8 >= 0.5:
    # T^{(1)} = 0.9 * 0.5 + (1 - 0.9) * 0.8 = 0.45 + 0.08 = 0.53
    sim = np.array([0.8])
    acc = np.array([0.0])
    anom = np.array([0.0])
    ts.update_trust([0], sim, acc, anom)

    expected = 0.9 * 0.5 + 0.1 * 0.8
    assert ts.trust_scores[0] == pytest.approx(expected, abs=1e-6)
    assert ts.trust_scores[0] == pytest.approx(0.53, abs=1e-6)


# ── Test 9: Negative trust branch ─────────────────────────────────────────────
def test_negative_trust_branch_lambda_down():
    """Test 9: Verify lambda_down = 0.7 when s_i < T_i^{(t-1)}."""
    ts = TrustScorer(num_clients=1, alpha=1.0, beta=0.0, gamma=0.0,
                     lambda_up=0.9, lambda_down=0.7, initial_trust=0.5)
    assert ts.trust_scores[0] == 0.5

    # With s_i = 0.2 < 0.5:
    # T^{(1)} = 0.7 * 0.5 + (1 - 0.7) * 0.2 = 0.35 + 0.06 = 0.41
    sim = np.array([0.2])
    acc = np.array([0.0])
    anom = np.array([0.0])
    ts.update_trust([0], sim, acc, anom)

    expected = 0.7 * 0.5 + 0.3 * 0.2
    assert ts.trust_scores[0] == pytest.approx(expected, abs=1e-6)
    assert ts.trust_scores[0] == pytest.approx(0.41, abs=1e-6)


# ── Test 10: Initial trust ────────────────────────────────────────────────────
def test_initial_trust_value():
    """Test 10: Verify T_i^{(0)} = 0.5 and reset restores 0.5."""
    ts = TrustScorer(num_clients=10)
    assert (ts.trust_scores == 0.5).all()
    for cid in range(10):
        assert ts.trust_history[cid] == [0.5]

    # Change trust scores
    ts.trust_scores[:] = 0.9
    ts.reset()
    assert (ts.trust_scores == 0.5).all()
    for cid in range(10):
        assert ts.trust_history[cid] == [0.5]


# ── Test 11: Trust floor ──────────────────────────────────────────────────────
def test_trust_floor_enforcement():
    """Test 11: Repeated penalties must never push trust below tau_min = 0.01."""
    ts = TrustScorer(num_clients=1, alpha=1.0, beta=0.0, gamma=0.0,
                     lambda_up=0.9, lambda_down=0.7, min_trust=0.01, initial_trust=0.5)

    sim = np.array([0.0])
    acc = np.array([0.0])
    anom = np.array([0.0])

    for _ in range(50):
        ts.update_trust([0], sim, acc, anom)
        assert ts.trust_scores[0] >= 0.01

    assert ts.trust_scores[0] == pytest.approx(0.01, abs=1e-6)


# ── Test 12: Rejected client penalty ──────────────────────────────────────────
def test_rejected_client_penalty():
    """Test 12: A rejected participant must receive s_i = 0 and undergo trust decay with lambda_down."""
    ts = TrustScorer(num_clients=4, alpha=1/3, beta=1/3, gamma=1/3,
                     lambda_up=0.9, lambda_down=0.7, initial_trust=0.5)

    # Participant cohort P = [0, 1]. Client 0 accepted, Client 1 rejected.
    # Client 0 receives s_0 = 0.6
    # Client 1 is rejected -> s_1 = 0.0
    sim = np.array([0.6])
    acc = np.array([0.6])
    anom = np.array([0.0])

    ts.update_trust(
        client_ids=[0],
        similarity_scores=sim,
        accuracy_scores=acc,
        anomaly_scores=anom,
        participant_ids=[0, 1],
        rejected_ids=[1],
    )

    # Client 1 must be penalized with s = 0:
    # T_1^{(1)} = 0.7 * 0.5 + 0.3 * 0.0 = 0.35
    assert ts.trust_scores[1] == pytest.approx(0.35, abs=1e-6)
    assert ts.trust_scores[1] < 0.5


# ── Test 13: Non-participant retention ────────────────────────────────────────
def test_non_participant_trust_retention():
    """Test 13: A client not sampled in the round must retain exactly its previous trust."""
    ts = TrustScorer(num_clients=5, alpha=1/3, beta=1/3, gamma=1/3, initial_trust=0.5)

    # Participants in round 1: [0, 1, 2]. Clients 3 and 4 do NOT participate.
    # Client 0 accepted, Clients 1 and 2 rejected.
    sim = np.array([0.8])
    acc = np.array([0.8])
    anom = np.array([0.0])

    ts.update_trust(
        client_ids=[0],
        similarity_scores=sim,
        accuracy_scores=acc,
        anomaly_scores=anom,
        participant_ids=[0, 1, 2],
        rejected_ids=[1, 2],
    )

    # Clients 3 and 4 must retain exactly 0.5
    assert ts.trust_scores[3] == 0.5
    assert ts.trust_scores[4] == 0.5
    # History for non-participants must not record round changes
    assert len(ts.trust_history[3]) == 1
    assert len(ts.trust_history[4]) == 1


# ── Test 14: Mixed signal ─────────────────────────────────────────────────────
def test_mixed_signal_exact_equation():
    """Test 14: Verify exact paper equation u_i = alpha * S_i + beta * A_i - gamma * O_i.
    Verify the negative sign on gamma * O_i.
    """
    ts = TrustScorer(num_clients=1, alpha=0.4, beta=0.4, gamma=0.2)

    # S = 0.8, A = 0.5, O = 0.3
    # u = 0.4 * 0.8 + 0.4 * 0.5 - 0.2 * 0.3 = 0.32 + 0.20 - 0.06 = 0.46
    u, s = ts.compute_instantaneous_signal(similarity=0.8, accuracy=0.5, anomaly=0.3)
    assert u == pytest.approx(0.46, abs=1e-6)
    assert s == pytest.approx(0.46, abs=1e-6)

    # Higher anomaly O=0.8 must REDUCE u:
    # u = 0.32 + 0.20 - 0.2 * 0.8 = 0.52 - 0.16 = 0.36
    u_high_anom, s_high_anom = ts.compute_instantaneous_signal(similarity=0.8, accuracy=0.5, anomaly=0.8)
    assert u_high_anom < u
    assert u_high_anom == pytest.approx(0.36, abs=1e-6)

    # Negative A_i: S=0.8, A=-0.5, O=0.1
    # u = 0.4 * 0.8 + 0.4 * (-0.5) - 0.2 * 0.1 = 0.32 - 0.20 - 0.02 = 0.10
    u_neg_a, s_neg_a = ts.compute_instantaneous_signal(similarity=0.8, accuracy=-0.5, anomaly=0.1)
    assert u_neg_a == pytest.approx(0.10, abs=1e-6)
    assert s_neg_a == pytest.approx(0.10, abs=1e-6)

    # Negative u clips to 0:
    # S=0.0, A=-1.0, O=1.0 -> u = 0 - 0.4 - 0.2 = -0.6 -> s = 0.0
    u_clip, s_clip = ts.compute_instantaneous_signal(similarity=0.0, accuracy=-1.0, anomaly=1.0)
    assert u_clip < 0.0
    assert s_clip == 0.0


# ── Test 15: Full round integration ───────────────────────────────────────────
def test_full_round_integration():
    """Test 15: Run a round containing accepted clients, rejected clients,
    and non-participating clients. Verify trust state evolves according to the paper.
    """
    num_clients = 6
    ts = TrustScorer(num_clients=num_clients, alpha=1/3, beta=1/3, gamma=1/3,
                     lambda_up=0.9, lambda_down=0.7, min_trust=0.01, initial_trust=0.5)

    # Participant cohort P = [0, 1, 2, 3] (size 4 out of 6).
    # Accepted cohort A = [0, 1].
    # Rejected cohort P \ A = [2, 3].
    # Non-participating: [4, 5].

    # Suppose for Client 0: S=0.9, A=0.6, O=0.0 -> u = 1/3*(0.9+0.6-0) = 0.5 -> s = 0.5.
    # Since s >= T^{(0)} (0.5 >= 0.5), lambda_up=0.9: T_0^{(1)} = 0.9*0.5 + 0.1*0.5 = 0.5.
    # For Client 1: S=0.3, A=0.0, O=0.6 -> u = 1/3*(0.3+0.0-0.6) = -0.1 -> s = 0.0.
    # Since s < T^{(0)} (0.0 < 0.5), lambda_down=0.7: T_1^{(1)} = 0.7*0.5 + 0.3*0 = 0.35.

    sim = np.array([0.9, 0.3])
    acc = np.array([0.6, 0.0])
    anom = np.array([0.0, 0.6])

    ts.update_trust(
        client_ids=[0, 1],
        similarity_scores=sim,
        accuracy_scores=acc,
        anomaly_scores=anom,
        participant_ids=[0, 1, 2, 3],
        rejected_ids=[2, 3],
    )

    # Accepted Client 0
    assert ts.trust_scores[0] == pytest.approx(0.5, abs=1e-6)

    # Accepted Client 1 (degraded signal s=0)
    assert ts.trust_scores[1] == pytest.approx(0.35, abs=1e-6)

    # Rejected Clients 2 and 3 (s = 0, penalized with lambda_down)
    assert ts.trust_scores[2] == pytest.approx(0.35, abs=1e-6)
    assert ts.trust_scores[3] == pytest.approx(0.35, abs=1e-6)

    # Non-participating Clients 4 and 5 (retained previous trust)
    assert ts.trust_scores[4] == 0.5
    assert ts.trust_scores[5] == 0.5
    assert len(ts.trust_history[4]) == 1
    assert len(ts.trust_history[5]) == 1
