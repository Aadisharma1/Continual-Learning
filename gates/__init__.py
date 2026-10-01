"""Gates G7 / G8 -- probe-and-steer instrumentation for the Conti pivot.

Lane A question (G7): are the three repair states actually distinguishable, or
is the third one a detector artifact?

Lane B question (G8): does trained personalization move the readout further
than a prompted persona does, at matched behavioural displacement?

Conventions that are load-bearing and must not drift:

  * Prose layer numbers are 1-based ("layer 16 of 32"). Code indices are 0-based
    (:func:`gates.acts.LayerTap` yields ``resid_in[i]`` / ``resid_post[i]`` for
    ``model.model.layers[i]``). Every figure caption states the mapping.
  * Generative evaluation LEFT-PADS and reads ``h[:, -1, :]``. Offline feature
    collection RIGHT-PADS and indexes ``attn_mask.sum(1) - 1``. Mixing these
    silently changes the probe target.
  * ``residual_scale`` fixes the steering dose unit u = median ||resid_post||.
    All alphas are reported in units of u so they are comparable across layers,
    models, and checkpoints.
  * TF32 on for training, OFF for probe and steering measurements.
"""

from __future__ import annotations

from .acts import (
    LayerTap,
    blocks,
    collect,
    gather_pos,
    load,
    mask_mean,
    mask_tail_mean,
    pos_last_idx,
    residual_scale,
    rewrap,
    unwrap,
)
from .judge import (
    LexicalJudge,
    LLMJudge,
    cohens_kappa,
    get_judge,
    judge_validation_report,
)
from .probes import (
    LayerProbe,
    leak_audit,
    logit_diff,
    nc3_token_shuffle,
    permutation_null_auc,
    train_layer_probes,
)
from .prompts import (
    PERSONAS,
    PromptItem,
    PromptSet,
    assert_disjoint,
    build_personalised_prompts,
    load_prompts,
)
from .steer import ALPHA_GRID, Steering

__version__ = "0.1.0"

__all__ = [
    "ALPHA_GRID",
    "LayerTap",
    "LexicalJudge",
    "LLMJudge",
    "PERSONAS",
    "PromptItem",
    "PromptSet",
    "Steering",
    "LayerProbe",
    "assert_disjoint",
    "blocks",
    "build_personalised_prompts",
    "cohens_kappa",
    "collect",
        "gather_pos",
        "get_judge",
        "judge_validation_report",
        "leak_audit",
        "load",
        "load_prompts",
        "logit_diff",
        "mask_mean",
        "mask_tail_mean",
        "nc3_token_shuffle",
        "permutation_null_auc",
        "pos_last_idx",
        "residual_scale",
        "rewrap",
        "train_layer_probes",
        "unwrap",
        "__version__",
]