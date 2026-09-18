import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from attacks.adversarial import is_on_off_active


@pytest.mark.parametrize("k", [10, 20, 30, 50])
def test_on_off_paper_schedule(k):
    """Verify paper schedule: clients behave honestly for round <= k, then attack for round > k."""
    # Rounds 1..k should be inactive (honest)
    for r in range(1, k + 1):
        assert is_on_off_active(r, k=k, mode="switch") is False, f"Round {r} should be honest for k={k}"

    # Rounds (k+1)..2k should be active (attacking)
    for r in range(k + 1, 2 * k + 1):
        assert is_on_off_active(r, k=k, mode="switch") is True, f"Round {r} should be attacking for k={k}"


@pytest.mark.parametrize("k", [10, 20, 30, 50])
def test_on_off_periodic_schedule(k):
    """Verify periodic schedule mode."""
    total_rounds = 4 * k

    # Cycle 1: 1..k honest (False), k+1..2k attack (True)
    for r in range(1, k + 1):
        assert is_on_off_active(r, k=k, mode="periodic") is False
    for r in range(k + 1, 2 * k + 1):
        assert is_on_off_active(r, k=k, mode="periodic") is True

    # Cycle 2: 2k+1..3k honest (False), 3k+1..4k attack (True)
    for r in range(2 * k + 1, 3 * k + 1):
        assert is_on_off_active(r, k=k, mode="periodic") is False
    for r in range(3 * k + 1, 4 * k + 1):
        assert is_on_off_active(r, k=k, mode="periodic") is True


def test_on_off_boundaries():
    """Verify boundary transitions for k=30."""
    k = 30
    assert is_on_off_active(30, k=k, mode="switch") is False
    assert is_on_off_active(31, k=k, mode="switch") is True
