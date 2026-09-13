
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List

from config import SEALConfig


# ═══════════════════════════════════════════════════════════════════
# LoRA configuration
# ═══════════════════════════════════════════════════════════════════

def create_lora_config(cfg: SEALConfig) -> LoraConfig:
    """Create the SEAL paper's LoRA configuration."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )


# ═══════════════════════════════════════════════════════════════════
# Weight reset — gives us a "fresh" adapter without reloading
# ═══════════════════════════════════════════════════════════════════

def reset_lora_weights(model) -> None:
    """
    Re-initialise every LoRA adapter matrix to its default state.

    lora_A  →  Kaiming uniform (as per nn.Linear default)
    lora_B  →  Zeros

    When B = 0 the adapter output is exactly zero, so the model
    behaves identically to the frozen base model.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            nn.init.kaiming_uniform_(param, a=5 ** 0.5)
        elif "lora_B" in name:
            nn.init.zeros_(param)


# ═══════════════════════════════════════════════════════════════════
# Collation — dynamic padding for variable-length samples
# ═══════════════════════════════════════════════════════════════════

def _make_collate_fn(pad_token_id: int):
    """Return a collate function that pads to the longest item in the batch."""

    def collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list = []
        attn_mask_list = []
        labels_list = []

        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids_list.append(
                torch.cat([
                    item["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
            )
            attn_mask_list.append(
                torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )
            labels_list.append(
                torch.cat([
                    item["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels": torch.stack(labels_list),
        }

    return collate


# ═══════════════════════════════════════════════════════════════════
# Training loop (per-passage inner loop)
# ═══════════════════════════════════════════════════════════════════

def train_lora_on_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    cfg: SEALConfig,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train LoRA adapter on a small dataset (per-passage inner loop).

    Steps:
        1. Reset LoRA weights to fresh random init
        2. Create a new AdamW optimiser (lr = 1e-3)
        3. Train for `cfg.epochs_per_passage` epochs
        4. Return average loss

    The base model weights are FROZEN — only LoRA params update.
    """
    # ── 1. Fresh init ──────────────────────────────────────────────
    reset_lora_weights(model)

    device = model.get_input_embeddings().weight.device

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = _make_collate_fn(pad_id)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── 2. Optimiser (no scheduler — 10 epochs is too short) ──────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── 3. Train ──────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    num_steps = 0

    for epoch in range(cfg.epochs_per_passage):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_steps += 1

    avg_loss = total_loss / max(num_steps, 1)

    if verbose:
        print(f"    avg_loss={avg_loss:.4f}  steps={num_steps}")

    model.eval()
    return {"avg_loss": avg_loss, "num_steps": num_steps}
