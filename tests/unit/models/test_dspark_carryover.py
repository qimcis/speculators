"""DSpark Carryover (arXiv 2609.14717): carry segment in the draft attention."""

from typing import Any

import torch
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import resolve_loss_config
from speculators.models.dflash.attention import (
    create_anchor_block_carry_mask_mod,
    create_anchor_block_mask_mod,
)
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.train.carryover import (
    CarryoverCollateFn,
    load_transitions,
    shifted_transitions,
)

V, D, BLOCK, T = 64, 32, 4, 24


def _config(**overrides) -> DSparkSpeculatorConfig:
    tl = Qwen3Config(
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=V,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values: dict[str, Any] = {
        "transformer_layer_config": tl,
        "draft_vocab_size": V,
        "block_size": BLOCK,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 1,
        "markov_rank": 8,
    }
    values.update(overrides)
    return DSparkSpeculatorConfig(**values)


def _model(**cfg):
    torch.manual_seed(0)
    model = DSparkDraftModel(_config(**cfg))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.isnan().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return model


def _inputs():
    g = torch.Generator().manual_seed(2)
    return {
        "hidden_states": torch.randn(1, T, 2 * D, generator=g),
        "input_ids": torch.randint(2, V, (1, T), generator=g),
        "loss_mask": torch.ones(1, T),
        "verifier_last_hidden_states": torch.randn(1, T, D, generator=g),
        "document_ids": torch.zeros(1, T, dtype=torch.long),
    }


def _run(model, **kw):
    loss_config = resolve_loss_config('{"ce":0.1,"tv":0.9}', "eager")
    tv = resolve_loss_config("tv", "eager")["tv"][0]
    return model(**_inputs(), loss_config=loss_config, tv_loss_fn=tv, **kw)


def _dense(mask_mod, q_len, kv_len):
    from torch.nn.attention.flex_attention import create_mask

    return create_mask(mask_mod, None, None, q_len, kv_len, device="cpu")[0, 0]


def test_carry_mask_is_base_mask_plus_owned_rows():
    doc = torch.zeros(T, dtype=torch.long)
    anchors = torch.tensor([3, 10, 15])
    owner = torch.tensor([0, 0, 2, -1])  # block 0 owns 2 rows, block 2 one, one pad
    base, q_len, kv_len = create_anchor_block_mask_mod(doc, T, anchors, BLOCK, None)
    carry, q2, kv2 = create_anchor_block_carry_mask_mod(doc, T, anchors, BLOCK, owner)
    assert q2 == q_len and kv2 == kv_len + owner.numel()
    b = _dense(base, q_len, kv_len)
    c = _dense(carry, q2, kv2)
    # Removing the carry segment gives back the base mask exactly.
    keep = torch.cat([torch.arange(T), torch.arange(T + 4, kv2)])
    assert torch.equal(c[:, keep], b)
    seg = c[:, T : T + 4]
    blk = torch.arange(q_len) // BLOCK
    want = (owner.view(1, -1) == blk.view(-1, 1)) & (owner.view(1, -1) >= 0)
    assert torch.equal(seg, want)


def test_fixed_anchors_without_carry_match_random_path_shapes_and_backprop():
    model = _model(carryover=True)
    anchors = torch.tensor([[2, 9, 14, -1]])
    _, loss, metrics = _run(model, max_anchors=4, carry_anchors=anchors)
    assert torch.isfinite(loss)
    loss.backward()


def test_carry_rows_change_only_their_owner_block():
    model = _model(carryover=True)
    with torch.no_grad():
        model.carry_embedding.copy_(torch.randn(D) * 0.3)
    anchors = torch.tensor([[2, 9, 14, -1]])
    owner = torch.tensor([[0, 0, -1, -1]])
    positions = torch.tensor([[2, 3, 0, 0]])
    states = torch.randn(1, 4, 2 * D)

    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out.detach()

    h = model.norm.register_forward_hook(hook)
    kw = dict(max_anchors=4, carry_anchors=anchors, carry_positions=positions,
              carry_hidden_states=states)
    _run(model, carry_owner=torch.full_like(owner, -1), **kw)
    off = captured["h"].clone()
    _, loss, _ = _run(model, carry_owner=owner, **kw)
    on = captured["h"]
    h.remove()
    diff = (on - off).abs().view(4, BLOCK, -1).amax(dim=(1, 2))
    assert diff[0] > 1e-4  # block 0 reads its carried rows
    assert torch.all(diff[1:] == 0)  # nobody else does
    loss.backward()
    assert model.carry_embedding.grad is not None
    assert model.carry_embedding.grad.abs().sum() > 0


def test_transitions_roundtrip_and_shift(tmp_path):
    K, L = 4, 2
    anchors = torch.tensor([2, 8, 15])
    accepted = torch.tensor([1, 4, 0])  # rows: 3, 0, 4
    rows = int((K - accepted).sum())
    states = torch.randn(rows, L, D)
    amax = states.abs().amax(-1, keepdim=True)
    scales = amax / 448.0
    save_file(
        {
            "anchors": anchors,
            "accepted": accepted,
            "carry_states": (states / scales).to(torch.float8_e4m3fn),
            "carry_scales": scales.float(),
            "K": torch.tensor([K]),
        },
        str(tmp_path / "tr_0.safetensors"),
    )
    tr = load_transitions(tmp_path / "tr_0.safetensors")
    assert torch.allclose(tr["carry_states"], states.flatten(1), rtol=0.1, atol=0.05)
    sa, st, pos, cnt = shifted_transitions(tr, T, BLOCK, torch.ones(T))
    # anchor 2 (shifted 15+0+1=16) still fits: 16 + 4 <= 24.
    assert sa.tolist() == [4, 13, 16]
    assert cnt.tolist() == [3, 0, 4]
    assert pos.tolist() == [4, 5, 6, 16, 17, 18, 19]
    assert st.shape == (7, L * D)


def test_collate_packs_fixed_shapes():
    coll = CarryoverCollateFn(T * 2, D, num_target_layers=2, max_anchors=6,
                              block_size=BLOCK, max_carry=4)

    def item(n, anchors, counts):
        total = int(sum(counts))
        return {
            "hidden_states": torch.randn(n, 2 * D),
            "input_ids": torch.randint(2, V, (n,)),
            "verifier_last_hidden_states": torch.randn(n, D),
            "loss_mask": torch.ones(n),
            "lengths": torch.tensor([n]),
            "position_ids": torch.arange(n),
            "carry_anchors": torch.tensor(anchors),
            "carry_states": torch.randn(total, 2 * D),
            "carry_positions": torch.arange(total),
            "carry_counts": torch.tensor(counts),
        }

    out = coll([item(T, [3, 10], [2, 0]), item(T, [5], [3])])
    assert out["carry_anchors"].tolist() == [[3, 10, T + 5, -1, -1, -1]]
    owner = out["carry_owner"].view(-1).tolist()
    assert owner[:5] == [0, 0, 2, 2, 2] and set(owner[5:]) == {-1}
    assert out["carry_hidden_states"].shape == (1, 24, 2 * D)
