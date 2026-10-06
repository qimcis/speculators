"""Export hidden states for the tail of a request, after its prefix.

A request that sets ``kv_transfer_params["hidden_states_start"]`` is prefilled
in full, so every position attends to the whole context, but only positions
from that offset on are copied out of the KV cache. Saves start at the last
cache-block boundary at or before the offset; callers trim the few positions
before it (see ``speculators.data_generation.offline.window_hidden_states``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, TypeVar


class _PendingSave(Protocol):
    token_ids: Any
    block_ids: list[int]


P = TypeVar("P", bound=_PendingSave)


def skip_prefix_blocks(
    pending: P, kv_transfer_params: dict[str, Any] | None, block_size: int
) -> P:
    start = int((kv_transfer_params or {}).get("hidden_states_start") or 0)
    skip = min(start, len(pending.token_ids)) // block_size
    if skip == 0:
        return pending
    return dataclasses.replace(
        pending,
        token_ids=pending.token_ids[skip * block_size :],
        block_ids=pending.block_ids[skip:],
    )
