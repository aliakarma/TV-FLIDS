"""
utils/ste.py — Straight-through estimator (STE) for the clipped trust signal.

Paper reference: Section IV-C (Stage 3: Meta-Gradient Weight Adaptation),
the sentence following the aggregation-weight equation for w_hat_i
(labelled eq:hatw; Eq. (13) in the current numbering):

    "Wherever the clip is active -- which, by the range argument of (10),
     means only where the argument falls below 0 -- we employ the
     straight-through estimator: the forward pass uses the clipped value,
     while the backward pass treats the clip as the identity and passes the
     incoming gradient through unchanged [bengio2013ste]."

Note the wording: the estimator applies over the whole SATURATED REGION
(x < lo, or x > hi), not merely "at boundary points" where the argument
exactly equals a bound. That is both what the paper now says and what
autograd actually does here -- `torch.clamp` zeroes the gradient throughout
the saturated region, and the boundary points themselves are a measure-zero
set that no gradient step would ever land on exactly. An earlier revision of
this docstring quoted the superseded "at boundary points" phrasing, which
described a strictly weaker (and, in floating point, effectively vacuous)
intervention than the one implemented below.

Why this module exists
----------------------
That equation defines the meta-gradient aggregation weight

    w_hat_i = clip_[0,1](alpha*S_i + beta*A_i - gamma*O_i) / (sum_j ... + eps)

and the meta-loss (eq:meta_loss) backpropagates through it to (alpha, beta, gamma).
``torch.clamp`` implements the *hard* clip: its backward pass zeroes the
gradient wherever the argument is outside [0, 1]. Under that behaviour, a
client whose raw trust signal is saturated contributes exactly nothing to
d L_meta / d v, so the meta-gradient cannot learn its way out of a
configuration in which many clients sit in the saturated region — precisely
the "binds transiently in early rounds" regime the paper describes.

The straight-through estimator keeps the forward value clipped (so the
computed weights are identical to the hard-clip forward pass, bit for bit)
while letting the backward pass treat the clip as the identity, so gradient
flows for saturated clients too. This module supplies exactly that, and
nothing else: no other optimization behaviour is altered.

Range note (paper Section IV-B, eq:signal_range): with alpha + beta + gamma = 1,
alpha,beta,gamma > 0, S_i, A_i in [0,1] and O_i in [0,1), the raw signal
alpha*S_i + beta*A_i - gamma*O_i lies in (-gamma, 1-gamma]. Its upper end is
strictly below 1, so the UPPER clip at 1 can never bind and only the lower
boundary at 0 is ever active in practice. The estimator is implemented for
both boundaries regardless, so it remains correct if the signal definitions
change.
"""

from typing import Optional

import torch


class _ClipSTE(torch.autograd.Function):
    """clip_[lo,hi](x) forward; identity backward (straight-through)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        # No saved tensors: the backward pass is the identity and therefore
        # does not depend on x, lo or hi.
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass the incoming gradient unchanged, i.e. treat
        # clip as the identity map everywhere (including the saturated
        # region), per Bengio et al. (2013).
        return grad_output, None, None


def clip_ste(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """Clip ``x`` to [lo, hi] with a straight-through gradient.

    Forward:  identical to ``torch.clamp(x, lo, hi)``.
    Backward: d(clip_ste(x))/dx = 1 everywhere, including where x < lo or
              x > hi (where ``torch.clamp`` would give 0).

    Args:
        x:  Input tensor (must be floating point to carry gradient).
        lo: Lower clip bound.
        hi: Upper clip bound.

    Returns:
        Tensor with the clipped forward value and identity backward.
    """
    if hi < lo:
        raise ValueError(f"clip_ste requires lo <= hi, got lo={lo}, hi={hi}")
    return _ClipSTE.apply(x, float(lo), float(hi))


__all__ = ["clip_ste"]
