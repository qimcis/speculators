"""Carryover training data (arXiv 2609.14717): shifted anchors + rejected-suffix states.

Enabled only by ``--carryover-data DIR`` (DSpark). ``DIR`` holds one transitions file
per dataset row, ``tr_<file_idx>.safetensors``, built by ``scripts/carryover``:

    anchors        int64 [n]          first-draft anchors a_i (window-local positions)
    accepted       int64 [n]          greedy accept length m_i of the first draft (0..K)
    carry_states   fp8   [R, L, H]    rejected-suffix target states, anchor-major;
                                      anchor i owns K - m_i rows, at positions
                                      a_i + m_i + 1 + j (j = 0..K-m_i-1)
    carry_scales   fp32  [R, L, 1]    per-(row, layer) dequantization scales
    K              int64 [1]          first-draft length the transitions were made at

The second (trained) draft runs at the shifted anchors a'_i = a_i + m_i + 1 with the
recorded prefix, and its owning block also reads the carried rows (plus the learned
``carry_embedding``) -- exactly the context vLLM's carryover path builds at serving.
Fully accepted anchors (m_i = K) carry nothing and are plain shifted-anchor training.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from speculators.train.data import BaseDataset, CollateFn, SampleUnavailable

BatchType = dict[str, Any]

CARRY_KEYS = ("carry_anchors", "carry_states", "carry_positions", "carry_counts")


def load_transitions(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        return None
    data = load_file(str(path))
    states = data["carry_states"].float() * data["carry_scales"].float()
    return {
        "anchors": data["anchors"].long(),
        "accepted": data["accepted"].long(),
        "carry_states": states.flatten(1),  # [R, L*H]
        "K": int(data["K"].view(-1)[0]),
    }


def shifted_transitions(
    tr: dict[str, torch.Tensor], seq_len: int, block_size: int, loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample shifted anchors and their carried rows, dropping anchors that fall
    off the end of the sample or onto a non-trained token.

    Returns (anchors [n'], states [R', L*H], positions [R'], counts [n'])."""
    k = tr["K"]
    anchors, accepted = tr["anchors"], tr["accepted"]
    counts = (k - accepted).clamp(min=0)
    starts = torch.cumsum(counts, 0) - counts
    shifted = anchors + accepted + 1
    keep = (shifted + block_size <= seq_len) & (shifted >= 0)
    keep &= loss_mask[shifted.clamp(max=seq_len - 1)].bool()
    rows, positions, kept_counts = [], [], []
    for i in torch.nonzero(keep).view(-1).tolist():
        c = int(counts[i])
        s = int(starts[i])
        rows.append(tr["carry_states"][s : s + c])
        positions.append(torch.arange(c, dtype=torch.long) + int(shifted[i]))
        kept_counts.append(c)
    width = tr["carry_states"].shape[1]
    states = torch.cat(rows) if rows else torch.empty(0, width)
    pos = torch.cat(positions) if positions else torch.empty(0, dtype=torch.long)
    return shifted[keep], states, pos, torch.tensor(kept_counts, dtype=torch.long)


class CarryoverDataset(BaseDataset):
    """Wraps a row dataset; adds each row's shifted anchors and carried rows."""

    def __init__(self, base: BaseDataset, carryover_data: str | Path, block_size: int):
        self.base = base
        self.dir = Path(carryover_data)
        self.block_size = block_size
        self.max_len = base.max_len
        self.transform = None
        self.hidden_states_dtype = base.hidden_states_dtype
        self.approx_lengths = base.approx_lengths

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index) -> BatchType | SampleUnavailable:
        data = self.base[index]
        if isinstance(data, SampleUnavailable):
            return data
        file_idx = self.base._map_to_file_idx(index)  # noqa: SLF001
        tr = load_transitions(self.dir / f"tr_{file_idx}.safetensors")
        if tr is None:
            return SampleUnavailable(reason=f"no carryover transitions for row {file_idx}")
        seq_len = data["input_ids"].shape[0]
        anchors, states, positions, counts = shifted_transitions(
            tr, seq_len, self.block_size, torch.as_tensor(data["loss_mask"])
        )
        data["carry_anchors"] = anchors
        data["carry_states"] = states
        data["carry_positions"] = positions
        data["carry_counts"] = counts
        return data


class CarryoverCollateFn(CollateFn):
    """Packs rows like :class:`CollateFn`, then the carry tensors at fixed sizes
    (``max_anchors`` anchors, ``max_anchors * max_carry`` carried rows; -1 pads) so
    the compiled forward sees one shape."""

    def __init__(self, *args, max_anchors: int, block_size: int, max_carry: int, **kw):
        super().__init__(*args, **kw)
        self.max_anchors = max_anchors
        self.block_size = block_size
        self.max_carry = max_carry

    def __call__(self, batch: Sequence[BatchType | SampleUnavailable | None]) -> BatchType:
        carry = []
        stripped = []
        for item in batch:
            if isinstance(item, dict) and "carry_anchors" in item:
                carry.append({k: item.pop(k) for k in CARRY_KEYS})
                carry[-1]["seq_len"] = int(item["input_ids"].shape[0])
            stripped.append(item)
        out = super().__call__(stripped)
        width = self.hidden_size * self.num_target_layers
        n_rows = self.max_anchors * self.max_carry
        anchors = torch.full((self.max_anchors,), -1, dtype=torch.long)
        owner = torch.full((n_rows,), -1, dtype=torch.long)
        positions = torch.zeros(n_rows, dtype=torch.long)
        states = torch.zeros(n_rows, width, dtype=self.dtype)
        offset = n_anchor = n_carry = 0
        for c in carry:
            room = self.max_len - offset
            for i, a in enumerate(c["carry_anchors"].tolist()):
                cnt = int(c["carry_counts"][i])
                start = int(c["carry_counts"][:i].sum())
                if a + self.block_size > room or n_anchor >= self.max_anchors:
                    continue
                if n_carry + cnt > n_rows or cnt > self.max_carry:
                    continue
                anchors[n_anchor] = offset + a
                if cnt:
                    owner[n_carry : n_carry + cnt] = n_anchor
                    positions[n_carry : n_carry + cnt] = c["carry_positions"][start : start + cnt]
                    states[n_carry : n_carry + cnt] = c["carry_states"][start : start + cnt]
                n_carry += cnt
                n_anchor += 1
            offset += c["seq_len"]
            if offset >= self.max_len:
                break
        out["carry_anchors"] = anchors.unsqueeze(0)
        out["carry_owner"] = owner.unsqueeze(0)
        out["carry_positions"] = positions.unsqueeze(0)
        out["carry_hidden_states"] = states.unsqueeze(0)
        return out
