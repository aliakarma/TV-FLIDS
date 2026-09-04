"""
fl/ordering.py - deterministic ordering of a round's client results.

Ray returns each round's client results in *completion* order, which depends on
worker scheduling rather than on the run seed. Summation of float32 arrays is
not associative, so two runs of the same seed aggregated the same updates in
different orders and drifted apart round by round - small at first (~1e-3
accuracy after two rounds), compounding over a 100-round run.

Sorting by client id before aggregation removes that dependency. It changes no
mathematics: every aggregation rule in this repository is a sum, mean, median
or selection over the set of submitted updates, all of which are order
invariant in exact arithmetic. Only the floating-point rounding sequence is
pinned, and pinned to something reproducible.
"""

from typing import List, Tuple


def order_results(results: List[Tuple]) -> List[Tuple]:
    """Return this round's (ClientProxy, FitRes) pairs sorted by client id.

    Falls back to the given order if a cid is not an integer, so a non-numeric
    client id can never make aggregation raise.
    """
    try:
        return sorted(results, key=lambda pair: int(pair[0].cid))
    except (TypeError, ValueError):
        return list(results)
