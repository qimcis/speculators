"""DSpark + DFlash2-style conv and top-K candidate selector (warm-start path)."""

from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import resolve_loss_config
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_selector_loss
from speculators.models.dspark.model_definitions import EmbeddingCandidateSelector
from speculators.train.optimizers import split_named_params_for_muon

V, D, BLOCK = 64, 32, 4


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
        "dspark_conv_group_size": 4,
        "dspark_selector_rank": 8,
        "dspark_selector_top_k": 4,
    }
    values.update(overrides)
    return DSparkSpeculatorConfig(**values)


def _fill_nan(model):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.isnan().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def _inputs(seq_len=24):
    g = torch.Generator().manual_seed(2)
    return {
        "hidden_states": torch.randn(1, seq_len, 2 * D, generator=g),
        "input_ids": torch.randint(2, V, (1, seq_len), generator=g),
        "loss_mask": torch.ones(1, seq_len),
        "verifier_last_hidden_states": torch.randn(1, seq_len, D, generator=g),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long),
    }


def _run(model, **kw):
    loss_config = resolve_loss_config('{"ce":0.1,"tv":0.9}', "eager")
    tv = resolve_loss_config("tv", "eager")["tv"][0]
    torch.manual_seed(0)
    return model(**_inputs(), max_anchors=3, loss_config=loss_config, tv_loss_fn=tv, **kw)


def _base_and_warm(tmp_path, **arch):
    torch.manual_seed(0)
    base = DSparkDraftModel(_config())
    _fill_nan(base)
    with torch.no_grad():  # make the base non-trivial
        for p in base.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    sd = {k: v.contiguous() for k, v in base.state_dict().items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    warm = DSparkDraftModel(_config(**arch))
    _fill_nan(warm)
    missing, unexpected = warm.load_state_dict(sd, strict=False)
    assert not unexpected
    new, other = DSparkDraftModel.missing_new_module_keys(warm, set(sd))
    assert set(missing) == set(new) and not other
    return base, warm, new


def test_identity_conv_reproduces_base_exactly(tmp_path):
    base, warm, new = _base_and_warm(tmp_path, dspark_conv=True)
    assert any("attention_conv" in k for k in new)
    _, l0, m0 = _run(base)
    _, l1, m1 = _run(warm)
    assert torch.equal(l0, l1)
    for k in ("accept_len_sum", "eal_sum", "position_3_acc_sum"):
        assert torch.equal(m0[k], m1[k]), k


def test_hf_init_weights_restores_identity_conv():
    model = DSparkDraftModel(_config(dspark_conv=True))
    conv = model.layers[0].attention_conv
    with torch.no_grad():
        conv.base_kernel.normal_()
        conv.kernel_projection.weight.normal_()
    model._init_weights(conv)  # noqa: SLF001
    assert torch.equal(conv.kernel_projection.weight, torch.zeros_like(conv.kernel_projection.weight))
    assert torch.equal(conv.base_kernel[:, 0], torch.ones_like(conv.base_kernel[:, 0]))
    assert torch.equal(conv.base_kernel[:, 1], torch.zeros_like(conv.base_kernel[:, 1]))


def test_zero_selector_keeps_backbone_loss_and_trains(tmp_path):
    base, warm, new = _base_and_warm(tmp_path, dspark_conv=True, dspark_selector=True)
    _, l0, m0 = _run(base)
    _, l1, m1 = _run(warm)
    # The DSpark part of the loss is unchanged; the selector adds its own term.
    sel = 0.1 * m1["selector_ce_sum"] + 0.9 * m1["selector_tv_sum"]
    torch.testing.assert_close(l1 - sel, l0)
    assert torch.equal(m0["accept_len_sum"], m1["accept_len_sum"])
    l1.backward()
    for name, p in warm.named_parameters():
        if any(f in name for f in ("attention_conv", "mlp_conv", "candidate_selector")):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # succ_proj starts at zero -> its gradient must be nonzero for the selector to learn.
    assert torch.count_nonzero(warm.candidate_selector.succ_proj.weight.grad)
    assert torch.count_nonzero(warm.layers[0].mlp_conv.kernel_projection.weight.grad)


def test_candidate_metrics_without_selector(tmp_path):
    base = DSparkDraftModel(_config())
    _fill_nan(base)
    _, _, m = _run(base, candidate_metrics_top_k=4)
    for k in ("cand_oracle_al_sum", "cand_greedy_al_sum", "cand_recall_at_4_pos0_sum"):
        assert k in m
    assert m["cand_oracle_al_sum"] >= m["cand_greedy_al_sum"]


def test_selector_tv_matches_definition():
    torch.manual_seed(0)
    cand_logits = torch.randn(1, 4, 3)
    targets = torch.randn(1, 4, 10)
    ids = torch.stack([torch.randperm(10)[:3] for _ in range(4)]).unsqueeze(0)
    pos = torch.zeros(1, 4, dtype=torch.long)
    _, m = compute_selector_loss(
        cand_logits, ids, pos, targets, torch.ones(1, 4), 4,
        gamma=1e9, ce_alpha=0.0, tv_alpha=1.0,
    )
    q = torch.zeros(1, 4, 10).scatter(-1, ids, cand_logits.softmax(-1))
    tv = 0.5 * (q - targets.softmax(-1)).abs().sum(-1)
    torch.testing.assert_close(m["selector_tv_sum"], tv.mean(), rtol=1e-5, atol=1e-6)


def test_selector_params_are_small_and_muon_split():
    sel = EmbeddingCandidateSelector(hidden_size=6144, rank=256, top_k=16)
    n = sum(p.numel() for p in sel.parameters())
    assert n < 5_000_000
    model = DSparkDraftModel(_config(dspark_conv=True, dspark_selector=True))
    muon, _ = split_named_params_for_muon(model)
    names = {n for n, _ in muon}
    assert "candidate_selector.pred_proj.weight" in names
    assert not any("base_kernel" in n for n in names)


def test_config_serializes_new_fields():
    d = _config(dspark_conv=True, dspark_selector=True).to_dict()
    assert d["dspark_conv"] is True and d["dspark_selector"] is True
    assert d["dspark_selector_top_k"] == 4


def test_arch_kwargs_only_passes_given_flags():
    assert DSparkDraftModel.arch_kwargs_from_args({"dspark_conv": None}) == {}
    assert DSparkDraftModel.arch_kwargs_from_args({"dspark_conv": True}) == {
        "dspark_conv": True
    }


@pytest.mark.parametrize("flag", ["dspark_conv", "dspark_selector"])
def test_flags_default_off(flag):
    assert getattr(_config(), flag) is False
