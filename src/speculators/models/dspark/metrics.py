"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).

Two acceptance lengths are reported, both counting the verifier's bonus token:
``accept_len`` from that analytical rate (acceptance under rejection sampling)
and ``eal`` from greedy argmax matches (acceptance at temperature 0, and the
value comparable across the DFlash family).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    tv_loss,
)
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

__all__ = [
    "compute_candidate_metrics",
    "compute_metrics",
    "compute_selector_loss",
    "selector_training_candidates",
    "target_log_normalizer",
]

_EPS = 1e-8


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(
            pos_idx.to(weighted.dtype), elementwise_loss=elementwise
        )
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
    )

    # Analytical per-position acceptance rate = distributional overlap
    # = 1 - TV; the fused kernel avoids the two full-vocab fp32 softmaxes.
    with torch.no_grad():
        accept_rate = 1.0 - tv_loss_fn(logits, targets)  # [1, T]
        # Per-block cumulative acceptance product over the draft slots (slot 0
        # is the anchor), shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        # D-PACE expects token CE, so keep the confidence BCE on fixed decay.
        confidence_decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )
        conf_loss = _masked_decayed_mean(bce, loss_mask, pos_idx, confidence_decay_fn)
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted bonus.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # Greedy counterpart to accept_len, on the same per-block/bonus-token
    # convention, so DFlash-family runs compare on one number.
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(num_blocks, block_size)[:, start_pos:],
        draft_mask.to(torch.bool),
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total

    return loss, metrics


@torch.no_grad()
def target_log_normalizer(targets: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """logsumexp over the vocab in fp32, chunked over positions ([1, T, 1]).

    Avoids materializing a full [T, V] fp32 copy of the verifier logits.
    """
    out = torch.empty(
        (*targets.shape[:-1], 1), dtype=torch.float32, device=targets.device
    )
    for s in range(0, targets.shape[1], chunk):
        out[:, s : s + chunk] = torch.logsumexp(
            targets[:, s : s + chunk].float(), dim=-1, keepdim=True
        )
    return out


def _candidate_distributions(
    candidate_logits: torch.Tensor,  # [1, T, K]
    candidate_ids: torch.Tensor,  # [1, T, K]
    targets: torch.Tensor,  # [1, T, V]
    log_z: torch.Tensor | None = None,  # [1, T, 1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (q over K, p at the K candidates, p mass outside the K), fp32."""
    q = torch.softmax(candidate_logits.float(), dim=-1)
    if log_z is None:
        log_z = target_log_normalizer(targets)
    p_k = torch.exp(targets.gather(-1, candidate_ids).float() - log_z)
    p_out = (1.0 - p_k.sum(dim=-1)).clamp_min(0.0)
    return q, p_k, p_out


def compute_selector_loss(
    candidate_logits: torch.Tensor,  # [1, T, K]
    candidate_ids: torch.Tensor,  # [1, T, K] (target injected at K-1 if missing)
    target_positions: torch.Tensor,  # [1, T]
    targets: torch.Tensor,  # [1, T, V] verifier logits
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    *,
    gamma: float,
    ce_alpha: float,
    tv_alpha: float,
    sample_from_anchor: bool = True,
    log_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-forced K-way selector objective (z-lab #171 shape).

    ``ce``: hard CE of the target's argmax over the K candidates (the weakest
    candidate is replaced by the target when missing). ``tv``: total variation
    between the selector's K-way proposal ``q`` (zero outside the candidates)
    and the verifier distribution ``p``: ``0.5 * (sum_K |q - p| + p(outside))``.
    That is 1 - acceptance rate of the proposal the server actually samples.
    """
    seq_len = candidate_logits.shape[1]
    pos_idx = (
        torch.arange(seq_len, device=candidate_logits.device) % block_size
    ).unsqueeze(0)
    decay_fn = partial(
        dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
    )
    ce = torch.nn.functional.cross_entropy(
        candidate_logits.float().flatten(0, 1),
        target_positions.flatten(),
        reduction="none",
    ).view_as(target_positions)
    q, p_k, p_out = _candidate_distributions(
        candidate_logits, candidate_ids, targets, log_z
    )
    tv = 0.5 * ((q - p_k).abs().sum(dim=-1) + p_out)
    ce_loss = _masked_decayed_mean(ce, loss_mask, pos_idx, decay_fn)
    tv_loss_ = _masked_decayed_mean(tv, loss_mask, pos_idx, decay_fn)
    loss = ce_alpha * ce_loss + tv_alpha * tv_loss_
    one = torch.ones((), device=candidate_logits.device)
    return loss, {
        "selector_ce_sum": ce_loss.detach().clone(),
        "selector_ce_total": one,
        "selector_tv_sum": tv_loss_.detach().clone(),
        "selector_tv_total": one.clone(),
    }


@torch.no_grad()
def compute_candidate_metrics(
    candidate_logits: torch.Tensor,  # [1, T, K] serving-time candidate scores
    candidate_ids: torch.Tensor,  # [1, T, K] (target injected at K-1 if missing)
    contains_target: torch.Tensor,  # [1, T] target in the ORIGINAL unary top-K
    target_ids: torch.Tensor,  # [1, T]
    targets: torch.Tensor,  # [1, T, V]
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    *,
    top_k: int,
    sample_from_anchor: bool = True,
    corrected_contains: torch.Tensor | None = None,  # [1, T]
    unary_top1: torch.Tensor | None = None,  # [1, T] unary argmax == target
    log_z: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Top-K headroom metrics, per position (teacher-forced predecessors).

    * ``cand_recall_at_K_pos{i}``: target argmax inside the unary top-K.
    * ``cand_top1_pos{i}``: the candidate-score argmax is the target argmax
      (only counted if the target was in the unary top-K).
    * ``cand_accept_rate_pos{i}``: analytic acceptance of the K-way proposal,
      sum_K min(q, p) with the injected target masked out when it was missing.
    * ``cand_oracle_al`` / ``cand_greedy_al``: block accepted length (with bonus)
      for an oracle picking the target whenever it is among the K, vs the
      greedy argmax of the candidate scores; ``cand_expected_al``: the
      cumulative product of the K-way acceptance rates.
    """
    valid = loss_mask.to(torch.bool)
    # Remove the injected target from serving-side quantities.
    serve_logits = candidate_logits.float().clone()
    inject = ~contains_target
    serve_logits[..., -1] = torch.where(
        inject, torch.full_like(serve_logits[..., -1], float("-inf")), serve_logits[..., -1]
    )
    q, p_k, _ = _candidate_distributions(serve_logits, candidate_ids, targets, log_z)
    accept = torch.minimum(q, p_k).sum(dim=-1)  # [1, T]
    pick = candidate_ids.gather(-1, serve_logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
    top1 = pick.eq(target_ids) & contains_target

    num_blocks = loss_mask.shape[1] // block_size
    start = 0 if sample_from_anchor else 1
    vb = valid.view(num_blocks, block_size)[:, start:]
    rec_b = contains_target.view(num_blocks, block_size)[:, start:]
    top1_b = top1.view(num_blocks, block_size)[:, start:]
    acc_b = accept.view(num_blocks, block_size)[:, start:]
    vf = vb.float()

    m: dict[str, torch.Tensor] = {}
    for i in range(vb.shape[1]):
        tot = vf[:, i].sum()
        pos = i + start
        m[f"cand_recall_at_{top_k}_pos{pos}_sum"] = (rec_b[:, i].float() * vf[:, i]).sum()
        m[f"cand_recall_at_{top_k}_pos{pos}_total"] = tot
        m[f"cand_top1_pos{pos}_sum"] = (top1_b[:, i].float() * vf[:, i]).sum()
        m[f"cand_top1_pos{pos}_total"] = tot.clone()
        m[f"cand_accept_rate_pos{pos}_sum"] = (acc_b[:, i] * vf[:, i]).sum()
        m[f"cand_accept_rate_pos{pos}_total"] = tot.clone()
    # Full block and the k=4 serving horizon.
    pairs = [("oracle", rec_b), ("greedy", top1_b)]
    if unary_top1 is not None:
        u_b = unary_top1.view(num_blocks, block_size)[:, start:]
        pairs.append(("unary", u_b))
        for i in range(vb.shape[1]):
            m[f"cand_unary_top1_pos{i + start}_sum"] = (u_b[:, i].float() * vf[:, i]).sum()
            m[f"cand_unary_top1_pos{i + start}_total"] = vf[:, i].sum()
    if corrected_contains is not None:
        crec_b = corrected_contains.view(num_blocks, block_size)[:, start:]
        pairs.append(("corr_oracle", crec_b))
        for i in range(vb.shape[1]):
            m[f"cand_corr_recall_at_{top_k}_pos{i + start}_sum"] = (
                crec_b[:, i].float() * vf[:, i]
            ).sum()
            m[f"cand_corr_recall_at_{top_k}_pos{i + start}_total"] = vf[:, i].sum()
    for horizon, sfx in ((vb.shape[1], ""), (min(4, vb.shape[1]), "4")):
        for name, corr in pairs:
            s_, t_ = compute_accepted_length_counts(
                corr[:, :horizon], vb[:, :horizon]
            )
            m[f"cand_{name}_al{sfx}_sum"] = s_
            m[f"cand_{name}_al{sfx}_total"] = t_
        v_h = vf[:, :horizon]
        block_valid = (v_h.sum(dim=-1) > 0).float()
        exp_al = (acc_b[:, :horizon] * v_h).cumprod(dim=-1).sum(dim=-1) + 1.0
        m[f"cand_expected_al{sfx}_sum"] = (exp_al * block_valid).sum()
        m[f"cand_expected_al{sfx}_total"] = block_valid.sum().clamp_min(1.0)
    return m


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, K]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unary top-K with the target injected at slot K-1 when missing.

    Same contract as ``dflash2.metrics.selector_training_candidates`` (kept here
    to avoid a dspark <-> dflash2 import cycle).
    """
    top_k = candidate_ids.shape[-1]
    matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains = matches.any(dim=-1)
    positions = torch.where(contains, matches.to(torch.int64).argmax(dim=-1), top_k - 1)
    out = candidate_ids.clone()
    out[..., -1] = torch.where(contains, out[..., -1], target_ids)
    return out, positions, contains
