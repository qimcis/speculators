from collections.abc import Callable
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import (
    LossConfig,
    kl_div_loss,
    resolve_loss_config,
    tv_loss,
)
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash2.model_definitions import (
    GroupedDynamicCausalConv,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.dspark.metrics import (
    compute_candidate_metrics,
    compute_metrics,
    compute_selector_loss,
    selector_training_candidates,
    target_log_normalizer,
)
from speculators.models.dspark.model_definitions import (
    ConfidenceHead,
    EmbeddingCandidateSelector,
    MarkovHead,
)
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "Qwen3DFlash2DecoderLayer"]

    # New modules added on top of a trained DSpark checkpoint. Warm starts may
    # miss exactly these keys and nothing else (see missing_new_module_keys).
    NEW_MODULE_KEY_PATTERNS: ClassVar[tuple[str, ...]] = (
        ".attention_conv.",
        ".mlp_conv.",
        "candidate_selector.",
        "carry_embedding",
    )

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> torch.nn.Module:
        if not getattr(config, "dspark_conv", False):
            return super()._make_decoder_layer(config, layer_idx)
        assert isinstance(config, DSparkSpeculatorConfig)  # noqa: S101
        return Qwen3DFlash2DecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.dspark_conv_kernel_size,
            conv_group_size=config.dspark_conv_group_size,
        )

    def _init_weights(self, module):
        # HF initializes params missing from a checkpoint through _init_weights,
        # which would draw the conv's kernel projection from N(0, 0.02) and leave
        # base_kernel uninitialized; restore the identity init instead.
        if isinstance(module, GroupedDynamicCausalConv):
            module.reset_parameters()
            return
        if isinstance(module, EmbeddingCandidateSelector):
            module.reset_parameters(
                getattr(self.config.transformer_layer_config, "initializer_range", 0.02)
                or 0.02
            )
            return
        super()._init_weights(module)

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size
        if config.dspark_conv:
            for layer_ in self.layers:
                layer_.reset_convolutions()  # type: ignore[operator]

        self.candidate_selector: EmbeddingCandidateSelector | None = None
        if config.dspark_selector:
            if self.use_draft_vocab:
                raise ValueError("dspark_selector requires the full verifier vocab.")
            self.candidate_selector = EmbeddingCandidateSelector(
                hidden_size=hidden_size,
                rank=config.dspark_selector_rank,
                top_k=config.dspark_selector_top_k,
            )

        # Carryover embedding, in fused-feature space (after fc, before hidden_norm);
        # zero-initialized so a warm start begins as "no explicit discrimination".
        self.carry_embedding: torch.nn.Parameter | None = None
        if getattr(config, "carryover", False):
            self.carry_embedding = torch.nn.Parameter(torch.zeros(hidden_size))

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            **cls.arch_kwargs_from_args(kwargs),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def arch_kwargs_from_args(kwargs: dict) -> dict:
        """Conv/selector config fields from training args (only those given)."""
        out = {}
        for key in (
            "dspark_conv",
            "dspark_conv_kernel_size",
            "dspark_conv_group_size",
            "dspark_selector",
            "dspark_selector_rank",
            "dspark_selector_top_k",
            "dspark_selector_succ_basis",
            "carryover",
        ):
            if kwargs.get(key) is not None:
                out[key] = kwargs[key]
        return out

    @classmethod
    def missing_new_module_keys(
        cls, model: "DSparkDraftModel", checkpoint_keys: set[str]
    ) -> tuple[list[str], list[str]]:
        """Split the model keys absent from a checkpoint into (new, unexpected).

        ``new`` are conv/selector keys that a warm start legitimately lacks;
        ``unexpected`` is anything else that is missing (must be empty).
        """
        ignored = set(model._keys_to_ignore_on_load_missing)  # noqa: SLF001
        new, other = [], []
        for key in model.state_dict():
            if key in checkpoint_keys or key in ignored:
                continue
            if any(p in f".{key}" for p in cls.NEW_MODULE_KEY_PATTERNS):
                new.append(key)
            else:
                other.append(key)
        return new, other

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
            "selector_ce_alpha": kwargs.get("dspark_selector_ce_alpha", 0.1),
            "selector_tv_alpha": kwargs.get("dspark_selector_tv_alpha", 0.9),
            "candidate_metrics_top_k": kwargs.get("dspark_candidate_metrics_top_k", 0),
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        selector_ce_alpha: float = 0.1,
        selector_tv_alpha: float = 0.9,
        candidate_metrics_top_k: int = 0,
        carry_anchors: torch.Tensor | None = None,  # [1, max_anchors], -1 = pad
        carry_hidden_states: torch.Tensor | None = None,  # [1, C, L*H]
        carry_positions: torch.Tensor | None = None,  # [1, C]
        carry_owner: torch.Tensor | None = None,  # [1, C], -1 = pad
        **kwargs,
    ):
        fixed_anchors = None
        carry = None
        if carry_anchors is not None:
            # Carryover batch (speculators.train.carryover): fixed shifted anchors,
            # plus each block's rejected-suffix states of the first draft.
            fixed_anchors = carry_anchors.view(-1)
            max_anchors = fixed_anchors.numel()
            if self.carry_embedding is not None and carry_owner is not None:
                carry = {
                    "hidden_states": carry_hidden_states,
                    "positions": carry_positions.view(-1),
                    "owner": carry_owner.view(-1),
                    "embedding": self.carry_embedding,
                }
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                fixed_anchors=fixed_anchors,
                carry=carry,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        unary_logits = logits  # backbone-only logits, before the Markov bias
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                    dim=-1,
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )

        top_k = (
            self.candidate_selector.top_k
            if self.candidate_selector is not None
            else candidate_metrics_top_k
        )
        if top_k > 0:
            # Candidates are the backbone's unary top-K (the set vLLM's top-k
            # DSpark path and DFlash2 both rerank); scores add the Markov bias
            # and, with the selector, the low-rank transition term.
            with torch.no_grad():
                target_ids = targets.argmax(dim=-1)
                candidate_ids = unary_logits.detach().topk(top_k, dim=-1).indices
                train_ids, target_positions, contains_target = (
                    selector_training_candidates(candidate_ids, target_ids)
                )
                log_z = target_log_normalizer(targets)
            candidate_logits = logits.gather(-1, train_ids)
            if self.candidate_selector is not None:
                pred_table, succ_table = self.candidate_selector.tables(
                    self.embed_tokens.weight,
                    self.lm_head.weight
                    if self.config.dspark_selector_succ_basis == "lm_head"
                    else None,
                )
                transition = self.candidate_selector.transition_scores(
                    pred_table=pred_table,
                    succ_table=succ_table,
                    hidden_states=hidden,
                    predecessor_ids=prev_token_ids.reshape(1, -1),
                    candidate_ids=train_ids,
                )
                candidate_logits = candidate_logits + transition.to(
                    candidate_logits.dtype
                )
                sel_loss, sel_metrics = compute_selector_loss(
                    candidate_logits,
                    train_ids,
                    target_positions,
                    targets,
                    aligned_loss_mask,
                    self.block_size,
                    gamma=gamma,
                    ce_alpha=selector_ce_alpha,
                    tv_alpha=selector_tv_alpha,
                    sample_from_anchor=self.config.sample_from_anchor,
                    log_z=log_z,
                )
                loss = loss + sel_loss
                metrics.update(sel_metrics)
                metrics["loss_sum"] = loss.detach().clone()
            metrics.update(
                compute_candidate_metrics(
                    candidate_logits.detach(),
                    train_ids,
                    contains_target,
                    target_ids,
                    targets,
                    aligned_loss_mask,
                    self.block_size,
                    top_k=top_k,
                    sample_from_anchor=self.config.sample_from_anchor,
                    corrected_contains=logits.detach()
                    .topk(top_k, dim=-1)
                    .indices.eq(target_ids.unsqueeze(-1))
                    .any(dim=-1),
                    unary_top1=candidate_ids[..., 0].eq(target_ids),
                    log_z=log_z,
                )
            )
        return None, loss, metrics
