import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn

    Returns:
        mask_mod, q_len, kv_len
    """
    # Always use non_causal for full attn
    non_causal = sliding_window is None or sliding_window_non_causal

    device = document_ids.device
    anchor_positions = anchor_positions.to(device=device, dtype=torch.long).contiguous()

    if anchor_positions.ndim != 1:
        raise ValueError(
            f"anchor_positions must be 1D, got shape {tuple(anchor_positions.shape)}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend to tokens in their own synthetic block.
        Non-causal unless non_causal=False,
        in which case only prior positions are attended.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def create_anchor_block_carry_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    carry_owner: torch.Tensor,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    """Anchor-block mask with a Carryover segment (arXiv 2609.14717).

    KV side: ``[ packed sequence (T) | carried rows (C) | anchor blocks ]``. Carried
    rows are rejected-suffix target states that only the query block that owns them
    may read (``carry_owner[c]`` = anchor index, -1 for padding). Base and in-block
    rules are those of :func:`create_anchor_block_mask_mod`.
    """
    base_mod, q_len, _ = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    carry_owner = carry_owner.to(device=document_ids.device, dtype=torch.long)
    num_carry = carry_owner.numel()
    kv_len = total_seq_len + num_carry + q_len

    def shifted_mod(b, h, q_idx, kv_idx):
        # Remove the carry segment so the base mask sees its usual KV layout.
        is_block = kv_idx >= total_seq_len + num_carry
        kv_base = torch.where(is_block, kv_idx - num_carry, kv_idx)
        is_carry = (kv_idx >= total_seq_len) & ~is_block
        return base_mod(b, h, q_idx, kv_base) & ~is_carry

    def carry_mod(_b, _h, q_idx, kv_idx):
        is_carry = (kv_idx >= total_seq_len) & (kv_idx < total_seq_len + num_carry)
        c = torch.clamp(kv_idx - total_seq_len, 0, max(num_carry - 1, 0))
        return is_carry & (carry_owner[c] == q_idx // block_size)

    return or_masks(shifted_mod, carry_mod), q_len, kv_len
