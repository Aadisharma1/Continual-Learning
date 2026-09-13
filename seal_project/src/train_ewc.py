"""
SFT training with Elastic Weight Consolidation (EWC) + Replay Buffer.

Standard bfloat16 LoRA fine-tuning with:
  - EWC penalty computed on SAFETY data (not training data)
  - Replay buffer that ACTUALLY interleaves safety examples into train batches
  - Optional trajectory verifier that filters unsafe self-edits before training

Single GPU (RTX 6000 96GB). Gradient checkpointing enabled.
"""

import getpass
import os
import json
import re
import copy
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ─── Trajectory Verifier ──────────────────────────────────────

UNSAFE_KEYWORDS = [
    "hack", "exploit", "bomb", "weapon", "kill", "murder", "attack",
    "steal", "fraud", "phishing", "malware", "ransomware", "ddos",
    "cocaine", "heroin", "meth", "fentanyl", "synthesize drugs",
    "child abuse", "trafficking", "terrorism", "suicide method",
    "how to hurt", "how to harm", "bypass security", "break into",
]


def verify_trajectory(sample: Dict) -> bool:
    """
    Filter out potentially harmful self-edit trajectories.
    Returns True if the sample is safe to train on, False if it should be dropped.
    """
    text = json.dumps(sample).lower()

    # check if the response contains harmful content
    response = sample.get("response", sample.get("self_edit", "")).lower()
    instruction = sample.get("instruction", "").lower()

    # if the instruction is harmful AND the response doesn't refuse -> drop it
    instr_harmful = any(kw in instruction for kw in UNSAFE_KEYWORDS)
    resp_has_refusal = any(
        phrase in response for phrase in [
            "i cannot", "i can't", "i won't", "i will not",
            "i'm sorry", "i apologize", "not appropriate",
            "cannot assist", "unethical", "illegal", "harmful",
        ]
    )

    if instr_harmful and not resp_has_refusal:
        return False  # harmful instruction with compliant response = poison

    # also drop if the response itself teaches harmful content
    resp_harmful = sum(1 for kw in UNSAFE_KEYWORDS if kw in response)
    if resp_harmful >= 2:
        return False

    return True


# ─── Dataset ──────────────────────────────────────────────────

class SelfEditDataset(Dataset):
    """Loads JSONL self-edit trajectories for SFT, with optional safety filtering."""

    def __init__(self, path: str, tokenizer, max_length: int = 1024,
                 verify: bool = False, tag: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.tag = tag

        dropped = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if verify and not verify_trajectory(sample):
                        dropped += 1
                        continue
                    self.samples.append(sample)

        msg = f"  [{tag}] loaded {len(self.samples)} samples from {Path(path).name}"
        if dropped > 0:
            msg += f" (dropped {dropped} unsafe)"
        print(msg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # handle both formats:
        # old: {"instruction": "...", "response": "..."}
        # new: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        if "messages" in item:
            msgs = item["messages"]
            user_parts = [m["content"] for m in msgs if m["role"] == "user"]
            asst_parts = [m["content"] for m in msgs if m["role"] == "assistant"]
            instruction = "\n".join(user_parts)
            response = "\n".join(asst_parts)
        else:
            instruction = item.get("instruction", "")
            response = item.get("response", item.get("self_edit", ""))

        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─── EWC Implementation ──────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.

    Fisher is computed on SAFETY data — this tells the model which weights
    are important for safe behavior, so those weights resist change during SFT.
    """

    def __init__(self, model, dataloader, device, n_samples: int = 200):
        self.params = {}
        self.fisher = {}
        self._compute_fisher(model, dataloader, device, n_samples)

    def _compute_fisher(self, model, dataloader, device, n_samples: int):
        print("  computing Fisher Information Matrix on safety data...")
        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
                self.fisher[name] = torch.zeros_like(param.data)

        count = 0
        for batch in tqdm(dataloader, desc="  fisher", total=min(n_samples, len(dataloader))):
            if count >= n_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2

            count += 1

        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

        model.zero_grad()
        print(f"  Fisher computed over {count} batches of safety data")

    def penalty(self, model) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher:
                fisher = self.fisher[name].to(param.device)
                old_param = self.params[name].to(param.device)
                loss += (fisher * (param - old_param) ** 2).sum()
        return loss


# ─── Training Loop ────────────────────────────────────────────

def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    fisher_dataset: Optional[SelfEditDataset],
    replay_dataset: Optional[SelfEditDataset],
    args,
):
    device = next(model.parameters()).device

    # ── Build the actual training dataset ─────────────────
    # If replay data exists, CONCATENATE it with the training data
    # so safety examples are interleaved into every epoch
    if replay_dataset is not None and len(replay_dataset) > 0:
        combined = ConcatDataset([train_dataset, replay_dataset])
        n_train = len(train_dataset)
        n_replay = len(replay_dataset)
        replay_ratio = n_replay / (n_train + n_replay)
        print(f"\n  replay buffer active: {n_train} train + {n_replay} replay "
              f"= {len(combined)} total ({replay_ratio:.1%} safety)")
    else:
        combined = train_dataset

    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,  # never drop — small datasets would lose data
    )

    # ── Compute EWC on SAFETY data (not training data!) ───
    ewc = None
    if args.ewc_lambda > 0:
        if fisher_dataset is not None and len(fisher_dataset) > 0:
            fisher_loader = DataLoader(
                fisher_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
            ewc = EWC(model, fisher_loader, device, n_samples=args.fisher_samples)
        else:
            print("  WARNING: ewc_lambda > 0 but no --fisher-data provided!")
            print("  EWC will NOT be applied. Pass --fisher-data with safety anchor data.")

    torch.cuda.empty_cache()  # free Fisher computation memory before training

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n  training config:")
    print(f"    epochs:     {args.epochs} ({total_steps} steps)")
    print(f"    ewc_lambda: {args.ewc_lambda}")
    print(f"    lr:         {args.lr}")
    print(f"    batch_size: {args.batch_size}")
    print(f"    replay:     {'yes' if replay_dataset else 'no'}")
    print(f"    verifier:   {'yes' if args.verify else 'no'}")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss, epoch_ewc_loss = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"  epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            sft_loss = outputs.loss

            ewc_loss = torch.tensor(0.0, device=device)
            if ewc is not None:
                ewc_loss = args.ewc_lambda * ewc.penalty(model)

            total_loss = sft_loss + ewc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += sft_loss.item()
            epoch_ewc_loss += ewc_loss.item()
            global_step += 1

            pbar.set_postfix({
                "sft": f"{sft_loss.item():.4f}",
                "ewc": f"{ewc_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_ewc = epoch_ewc_loss / len(train_loader)
        print(f"  epoch {epoch+1}: sft_loss={avg_loss:.4f}, ewc_loss={avg_ewc:.4f}")

    return model


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT + EWC + Replay training")
    parser.add_argument("--base-model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--train-data", type=str, required=True,
                        help="path to JSONL training data (task: SQuAD/GSM8K)")
    parser.add_argument("--fisher-data", type=str, default=None,
                        help="path to SAFETY JSONL for Fisher computation. "
                             "must be safety_anchor.jsonl, NOT training data")
    parser.add_argument("--replay-data", type=str, default=None,
                        help="path to SAFETY JSONL to mix into training batches. "
                             "these samples are concatenated with train-data and shuffled")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ewc_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ewc-lambda", type=float, default=0.5,
                        help="EWC penalty weight. 0 = no EWC (naive SFT)")
    parser.add_argument("--fisher-samples", type=int, default=200,
                        help="number of batches for Fisher computation")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="path to existing LoRA adapter to continue training")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--verify", action="store_true",
                        help="run safety verifier on training data before SFT")
    args = parser.parse_args()

    # ── API key prompt ────────────────────────────────────────
    if not os.environ.get("HF_TOKEN"):
        token = getpass.getpass("Enter HF_TOKEN: ").strip()
        if token:
            os.environ["HF_TOKEN"] = token

    # load model in bfloat16
    print(f"loading {args.base_model} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path:
        # Load existing adapter and continue training
        from peft import PeftModel
        print(f"  [Resume] Loading existing LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        # wrap in fresh LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,  # SEAL paper: no dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    # gradient checkpointing MUST come after get_peft_model() wrapping
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required when using grad checkpointing with LoRA
    
    model.print_trainable_parameters()

    # load datasets
    train_dataset = SelfEditDataset(
        args.train_data, tokenizer, args.max_length,
        verify=args.verify, tag="train"
    )

    fisher_dataset = None
    if args.fisher_data:
        fisher_dataset = SelfEditDataset(
            args.fisher_data, tokenizer, args.max_length,
            verify=False, tag="fisher/safety"
        )

    replay_dataset = None
    if args.replay_data:
        replay_dataset = SelfEditDataset(
            args.replay_data, tokenizer, args.max_length,
            verify=False, tag="replay/safety"
        )

    # train
    model = train(model, tokenizer, train_dataset, fisher_dataset, replay_dataset, args)

    # save
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"\n  model saved to {out_path}")


if __name__ == "__main__":
    main()
