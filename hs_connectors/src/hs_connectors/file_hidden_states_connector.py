"""vLLM's ``ExampleHiddenStatesConnector``, saving only after a request's prefix.

A request that sets ``kv_transfer_params["hidden_states_start"]`` saves its
hidden states from that offset on (see :mod:`hs_connectors.prefix`); any other
request saves every position, as upstream does. Tensors are written unchanged.
"""

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as _eh_mod,
)

from hs_connectors.prefix import skip_prefix_blocks


class FileHiddenStatesConnector(_eh_mod.ExampleHiddenStatesConnector):
    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        result = super().request_finished(request, block_ids)
        pending = self._pending_saves.get(request.request_id)
        if pending is not None:
            self._pending_saves[request.request_id] = skip_prefix_blocks(
                pending, request.kv_transfer_params, self._block_size
            )
        return result
