from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus a Markov logit-bias head and a confidence head.

    The Markov head lets each draft position condition on previously sampled
    tokens within the block; the confidence head predicts the per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the Markov previous-token embedding with the backbone "
            "hidden state as the confidence-head input."
        ),
    )

    # DFlash2-style intra-block convolution in every draft layer.
    dspark_conv: bool = Field(
        default=False,
        description=(
            "Wrap attention and MLP of every draft layer in DFlash2's grouped "
            "dynamic causal convolution (identity-initialized)."
        ),
    )
    dspark_conv_kernel_size: int = Field(default=2, ge=1)
    dspark_conv_group_size: int = Field(default=16, ge=1)

    # Top-K candidate selector on top of the Markov-corrected logits.
    dspark_selector: bool = Field(
        default=False,
        description=(
            "Keep the top-K of the Markov-corrected logits at each position and "
            "add a low-rank transition score <A(prev) * H(h), B(cand)>, with "
            "A linear in the frozen target input embedding and B in the frozen target lm_head. "
            "The draft samples from softmax over the K rescored candidates."
        ),
    )
    # Carryover (arXiv 2609.14717): rejected-suffix target states as one-round
    # extra draft context, marked by a learned embedding added after ``fc``.
    carryover: bool = Field(
        default=False,
        description=(
            "Train/serve with Carryover: rejected-suffix target states of the "
            "previous verification are extra context for the next draft, marked "
            "by the learned carry_embedding (vLLM: draft config 'carryover')."
        ),
    )
    dspark_selector_rank: int = Field(default=256, ge=1)
    dspark_selector_top_k: int = Field(default=16, ge=1)
    dspark_selector_mode: Literal["reweight", "truncate"] = Field(
        default="reweight",
        description=(
            "'reweight': add the transition score to the top-K candidates' "
            "logits and keep every other token at its corrected logit (full "
            "support; a zero selector is exactly plain DSpark; trained through "
            "the DSpark loss on the full distribution). 'truncate': sample only "
            "among the K candidates (DFlash2-style; extra K-way CE+TV loss)."
        ),
    )
    dspark_selector_candidates: Literal["corrected", "unary"] = Field(
        default="corrected",
        description=(
            "Candidate set the selector reranks: top-K of the Markov-corrected "
            "logits (avg50 held-out Recall@16 at pos 3: 0.85) or of the backbone "
            "unary logits (0.70; DFlash2 / vLLM dspark_draft_topk style)."
        ),
    )
    dspark_selector_succ_basis: Literal["lm_head", "embed"] = Field(
        default="lm_head",
        description=(
            "Frozen target matrix the successor factor B is linear in. inco's "
            "GLM-5.3-DFlash2 codebooks: predecessor ~ input embedding (R^2 0.988), "
            "successor ~ lm_head (R^2 0.987; ~ input embedding only 0.60)."
        ),
    )
