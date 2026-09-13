
from dataclasses import dataclass, field
from typing import List


@dataclass
class SEALConfig:
    """All hyperparameters for the 3-stage SEAL baseline experiment."""

    # ── Model ──────────────────────────────────────────────────────
    # CRITICAL: Must be the BASE model, NOT the -Instruct variant.
    # Instruct guardrails poison the zero-shot baseline numbers.
    model_id: str = "Qwen/Qwen2.5-7B"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"          # shard across all available GPUs
    trust_remote_code: bool = True

    # ── LoRA (SEAL paper exact params) ─────────────────────────────
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Training (per-passage inner loop) ──────────────────────────
    lr: float = 1e-3
    epochs_per_passage: int = 10
    max_seq_length: int = 1024
    micro_batch_size: int = 2         # passage datasets are tiny
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # ── Evaluation ─────────────────────────────────────────────────
    squad_split: str = "validation"
    max_eval_samples_stage1: int = 0  # 0 = full validation set
    max_passages_stage2_3: int = 100  # per-passage inner-loop budget
    eval_max_new_tokens: int = 32
    eval_batch_size: int = 32

    # ── Synthetic Q&A Generation ───────────────────────────────────
    num_synthetic_pairs: int = 5
    gen_max_new_tokens: int = 512
    gen_batch_size: int = 8           # passages batched together
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9

    # ── Misc ───────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"


# ═══════════════════════════════════════════════════════════════════
# Prompt Templates — EXACT format from the SEAL paper.
# DO NOT change whitespace, casing, or punctuation.
# ═══════════════════════════════════════════════════════════════════

# Stage 1 eval AND Stage 2/3 eval (closed-book QA — NO context!)
PROMPT_CLOSEDBOOK_EVAL = (
    "Answer the following question.\n"
    "Question: {question}\n"
    "Answer:"
)

# Stage 2 training: the model sees ONLY the passage text.
PROMPT_PASSAGE_TRAIN = (
    "Read the following passage and memorize the facts.\n"
    "Passage: {context}"
)

# Stage 3 training: synthetic Q&A pair (question + ground truth answer).
PROMPT_QA_TRAIN = (
    "Answer the following question.\n"
    "Question: {question}\n"
    "Answer: {answer}"
)

# Stage 3 generation: prompt fed to the frozen base model to
# produce synthetic Q&A pairs from a passage.
PROMPT_GENERATE_QA = (
    "Passage: {context}\n"
    "Generate {n} factual Question and Answer pairs based on the passage above:\n"
    "1. Question:"
)
