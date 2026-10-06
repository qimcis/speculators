"""FP8-quantizing sibling of vLLM's ``ExampleHiddenStatesConnector``.

Quantizes hidden states to ``float8_e4m3fn`` (per-token scaling) before
writing them to disk, roughly halving the on-disk/on-wire size of extracted
hidden states relative to the bf16 baseline. Everything else
(async DtoH copy, thread-pool disk writes, file locking, scheduler-side
bookkeeping) is inherited unchanged from ``ExampleHiddenStatesConnector`` --
only the final tensor-write step is overridden.

Loaded out-of-tree via ``kv_connector_module_path``; must be used with the
``extract_hidden_states`` speculative method. Paired on the read side with
:class:`hs_connectors.transfer.FP8Transfer`, which dequantizes transparently
so consumers (e.g. ``speculators.train.data.ArrowDataset``) never see FP8
tensors.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from safetensors.torch import save_file
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as _eh_mod,
)

from hs_connectors.fp8_utils import SCALES_KEY, quantize_tensor_to_fp8
from hs_connectors.prefix import skip_prefix_blocks


class FP8HiddenStatesConnector(_eh_mod.ExampleHiddenStatesConnector):
    """Quantizes hidden states to float8_e4m3fn before saving to safetensors.

    The output file contains three tensors::

        hidden_states        - fp8 [seq_len, num_layers, hidden_size]
        hidden_states_scales - fp32 [seq_len, 1, 1]
        token_ids            - int64 [seq_len]
    """

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

    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        try:
            event.synchronize()
            hidden_states = tensors["hidden_states"]
            fp8_hs, scales = quantize_tensor_to_fp8(hidden_states)
            quantized_tensors = {
                **tensors,
                "hidden_states": fp8_hs,
                SCALES_KEY: scales,
            }
            save_file(quantized_tensors, filename)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
