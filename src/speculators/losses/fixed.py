"""Numerically fixed softmax / log-softmax / cross-entropy.

Implements the rewrite from Barron, Gnanapragasam & Tschannen, "A Fix for
Softmax and Cross-Entropy" (arXiv:2610.31642). Standard implementations shift
by max and then form ``1 + sum(exp(z_j - z_max))``; once the margin exceeds the
format's precision cliff (~5.5 bf16, ~16.6 fp32) that denominator rounds to 1
and the target-logit gradient becomes exactly 0 while non-target gradients stay
positive, so the gradient no longer sums to zero. Here the max/target slot is
set to 0 *outside* the log-sum-exp and the two pieces are combined with
softplus, keeping the target gradient accurate down to exponent underflow.
"""

from __future__ import annotations

import torch


def log_softmax(z: torch.Tensor) -> torch.Tensor:
    """log_softmax with a zero-sum gradient (paper Eq. 5)."""
    i_max = z.argmax(dim=-1, keepdim=True)
    z_max = z.gather(-1, i_max)
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, i_max, True)
    shifted = torch.where(is_max, torch.zeros_like(z), z - z_max)
    others = torch.where(is_max, torch.full_like(z, float("-inf")), shifted)
    m = -torch.logsumexp(others, dim=-1, keepdim=True)
    return shifted - torch.nn.functional.softplus(-m)


def softmax(z: torch.Tensor) -> torch.Tensor:
    return log_softmax(z).exp()


def cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Hard-target CE (paper Eq. 6); ``target_ids`` may contain -100."""
    valid = target_ids != -100
    safe = target_ids.clamp_min(0)
    z_t = logits.gather(-1, safe.unsqueeze(-1))
    is_t = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, safe.unsqueeze(-1), True)
    others = torch.where(is_t, torch.full_like(logits, float("-inf")), logits - z_t)
    m = -torch.logsumexp(others, dim=-1)
    loss = torch.nn.functional.softplus(-m)
    return torch.where(valid, loss, torch.zeros_like(loss))


def soft_target_cross_entropy(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Soft-target CE with the exact ``q - p`` gradient in log space (paper Eq. 9)."""
    log_q = log_softmax(z_s)
    q = log_q.exp()
    p = softmax(z_t)
    delta = z_s - z_t
    d = delta - (p * delta).sum(-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))
    u = p * torch.expm1(s)
    g = u - q * u.sum(-1, keepdim=True)
    loss = -(p * log_q.detach()).sum(-1)
    return loss + (g.detach() * (z_s - z_s.detach())).sum(-1)
