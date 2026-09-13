from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from conti.config_schema import ContiConfig


class _TextDS(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"text": self.texts[i]}


@dataclass
class SFTTrainer:
    cfg: ContiConfig
    accelerator: Accelerator
    _prepared: Any | None = None
    _ewc: Any | None = None

    def set_ewc(self, ewc_penalty) -> None:
        self._ewc = ewc_penalty

    def load_adapter_checkpoint(self, model, checkpoint_dir: str) -> bool:
        """Load a previously saved adapter checkpoint in-place (resume support).

        Tries, in order: PEFT ``load_peft_weights`` + ``set_peft_model_state_dict``
        (the canonical path for LoRA runs), then a raw ``adapter_model.safetensors``
        / ``pytorch_model.bin`` state-dict load with prefix stripping. Returns
        True if at least one parameter tensor was restored.

        Raises:
            FileNotFoundError: If the checkpoint directory or its weight files
                are missing.
            RuntimeError: If no tensor could be matched to a model parameter.
        """
        from pathlib import Path as _Path

        ckpt = _Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"Adapter checkpoint directory not found: {ckpt}")
        unwrapped = self.accelerator.unwrap_model(model)

        try:
            from peft import load_peft_weights, set_peft_model_state_dict

            weights = load_peft_weights(str(ckpt))
            incompatible = set_peft_model_state_dict(unwrapped, weights)
            loaded_names = set(weights.keys())
            missing_report = getattr(incompatible, "unexpected_keys", [])
            if loaded_names and not missing_report:
                print(f"[resume] loaded {len(loaded_names)} adapter tensors from {ckpt}")
                return True
            print(f"[resume] peft load reported unexpected keys: {missing_report[:5]}")
            return True
        except ImportError:
            pass

        safetensors_path = ckpt / "adapter_model.safetensors"
        bin_path = ckpt / "adapter_model.bin"
        legacy_path = ckpt / "pytorch_model.bin"
        state: dict | None = None
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path))
        elif bin_path.exists():
            state = torch.load(str(bin_path), map_location="cpu")
        elif legacy_path.exists():
            state = torch.load(str(legacy_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No adapter weights in {ckpt} (looked for adapter_model.safetensors, "
                "adapter_model.bin, pytorch_model.bin)."
            )

        cleaned: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            canonical = name
            for prefix in ("module.", "base_model.model.", "_fsdp_wrapped_module."):
                while canonical.startswith(prefix):
                    canonical = canonical[len(prefix):]
            cleaned[canonical] = tensor

        model_state = dict(unwrapped.state_dict())
        matched = {
            name: tensor
            for name, tensor in cleaned.items()
            if name in model_state and model_state[name].shape == tensor.shape
        }
        if not matched:
            raise RuntimeError(
                f"Resume failed: none of the {len(cleaned)} checkpoint tensors matched "
                f"model parameters in {ckpt}."
            )
        missing, unexpected = unwrapped.load_state_dict(matched, strict=False)
        print(
            f"[resume] raw state-dict load from {ckpt}: {len(matched)} tensors, "
            f"{len(missing)} model keys left untouched, {len(unexpected)} unused"
        )
        return True

    def load_model_tokenizer(self):
        mc = self.cfg.model
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            mc.torch_dtype
        ]
        try:
            tokenizer = AutoTokenizer.from_pretrained(mc.name_or_path, trust_remote_code=mc.trust_remote_code)
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer for {mc.name_or_path}. Did you really forget to pip install transformers?? daymmmmmm Err: {e}")
            raise
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                mc.name_or_path,
                torch_dtype=dtype,
                trust_remote_code=mc.trust_remote_code,
                device_map=None,
                attn_implementation=mc.attn_implementation,
            )
        except Exception as e:
            print(f"[ERROR] model load failed for {mc.name_or_path}. check HF_TOKEN is set and you have disk space. Err: {e}")
            raise

        # memory bach jaati hai isse, thoda slow hota hai but worth it
        if mc.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if mc.use_lora:
            # TODO: honestly not sure rank 16 is doing anything better than 8, need to ablate this
            lora = LoraConfig(
                r=mc.lora_r,
                lora_alpha=mc.lora_alpha,
                lora_dropout=mc.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora)

        return model, tokenizer

    def ensure_model_prepared(self, model):
        if self._prepared is None:
            self._prepared = self.accelerator.prepare(model)
        return self._prepared

    def train_on_texts(
        self,
        model,
        tokenizer,
        texts: list[str],
        learning_rate: float,
        num_epochs: int,
        micro_batch_size: int,
        warmup_ratio: float,
        weight_decay: float,
        max_seq_length: int,
        max_grad_norm: float,
    ) -> dict[str, list[float]]:
        if not texts:
            return {"loss": [], "ewc_penalty": []}

        model = self.ensure_model_prepared(model)
        ds = _TextDS(texts)
        dl = DataLoader(ds, batch_size=micro_batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt, dl = self.accelerator.prepare(opt, dl)

        ga = max(1, self.accelerator.gradient_accumulation_steps)
        updates_per_epoch = max(1, math.ceil(len(dl) / ga))
        total_steps = max(1, num_epochs * updates_per_epoch)
        warmup = int(total_steps * warmup_ratio)
        sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
        sched = self.accelerator.prepare(sched)

        losses: list[float] = []
        ewc_pens: list[float] = []

        model.train()
        for ep in range(num_epochs):
            self.accelerator.wait_for_everyone()
            pbar = tqdm(dl, disable=not self.accelerator.is_local_main_process, desc=f"ep{ep}")
            opt.zero_grad(set_to_none=True)

            for batch in pbar:
                with self.accelerator.accumulate(model):
                    enc = tokenizer(
                        list(batch["text"]),
                        padding=True,
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.accelerator.device) for k, v in enc.items()}
                    out = model(**enc, labels=enc["input_ids"])
                    loss = out.loss

                    ewc_p = 0.0
                    if self._ewc is not None and self._ewc.is_ready:
                        ewc_loss = self._ewc.penalty(model)
                        loss = loss + ewc_loss
                        ewc_p = float(ewc_loss.detach())

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        opt.step()
                        sched.step()
                        opt.zero_grad(set_to_none=True)

                    if self.accelerator.is_local_main_process:  # sirf main process log kare
                        l = float(loss.detach())
                        losses.append(l)
                        ewc_pens.append(ewc_p)
                        pbar.set_postfix(loss=f"{l:.4f}", ewc=f"{ewc_p:.4f}")

        self.accelerator.wait_for_everyone()
        return {"loss": losses, "ewc_penalty": ewc_pens}

    def save_pretrained(self, model, tokenizer, path: str) -> None:
        unwrapped = self.accelerator.unwrap_model(model)
        if self.accelerator.is_main_process:
            unwrapped.save_pretrained(path)
            tokenizer.save_pretrained(path)
        self.accelerator.wait_for_everyone()
