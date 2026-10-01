"""Residual-stream activation capture for the G7/G8 gate stage.

CONVENTIONS (non-negotiable; inter-reviewer deaths are caused by ambiguity):

  For decoder layer ``i`` in ``[0, L-1]``:
    ``resid_in[i]``   = input to  ``model.model.layers[i]``  (residual after layers 0..i-1)
    ``resid_post[i]`` = output of ``model.model.layers[i]``  (post-MLP, pre-next-norm)

  Prose uses 1-based layer numbers: "layer 16 of 32" == ``resid_post[15]`` in code.

  Token position: refusal is a first-token-of-response decision, so the
  residual state that makes it is the one at the FINAL PROMPT TOKEN.
    - generative evaluation -> LEFT-PAD, read h[:, -1, :]
    - offline collection w/ ragged lengths -> RIGHT-PAD, index attn_mask.sum(1) - 1

See docs/PROBE_CONVENTIONS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "unwrap",
    "is_adapter",
    "adapter_base",
    "rewrap",
    "load",
    "blocks",
    "LayerTap",
    "pos_last_idx",
    "gather_pos",
    "mask_mean",
    "mask_tail_mean",
    "collect",
    "residual_scale",
]


# ----------------------------------------------------------------- HF compat
def unwrap(out):
    """HF decoder layers return a bare tensor (transformers>=4.54) or a tuple."""
    return out[0] if isinstance(out, (tuple, list)) else out


def rewrap(out, new):
    """Inverse of :func:`unwrap`, preserving any auxiliary entries."""
    if isinstance(out, (tuple, list)):
        return type(out)([new, *list(out[1:])])
    return new


def is_adapter(path: str) -> bool:
    """True when ``path`` is a PEFT adapter directory, not a model directory.

    A merged export has ``config.json``; a LoRA adapter has
    ``adapter_config.json`` and no ``config.json``. Confusing the two used to
    make discovery return an empty list and the gates run with no Arm B.
    """
    import os

    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "adapter_config.json"))
    )


def adapter_base(path: str) -> str | None:
    """The base model id an adapter was trained on, or None."""
    import json
    import os

    cfg = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    except (ValueError, OSError):
        return None


def load(
    model_id: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    merge_adapters: bool = True,
):
    """Load a causal LM frozen for inference, plus its tokenizer.

    ``model_id`` may be a merged model directory, a hub id, or a PEFT adapter
    directory. Adapters are resolved to their base and (by default) MERGED, so
    the returned object is a plain ``AutoModelForCausalLM`` and the
    ``model.model.layers[i]`` convention in this module holds unchanged. The
    merge is a no-op numerically only in the limit of zero adapter rank, which
    is why it is the faithful choice: a merged checkpoint export is exactly
    what the merge reproduces, and Arm B of G8 must be that model.

    Returns ``(model, tokenizer, provenance)``. ``provenance`` is a dict
    recording what was actually loaded, so a results file can never be
    ambiguous about which weights produced it.
    """
    provenance: dict = {"requested": model_id, "kind": "model", "merged": False}
    base_id = model_id

    if is_adapter(model_id):
        base_id = adapter_base(model_id)
        if not base_id:
            raise ValueError(
                f"{model_id} is an adapter with no base_model_name_or_path; "
                "pass the base model explicitly."
            )
        provenance.update(kind="peft_adapter", base=base_id,
                          merged=bool(merge_adapters))

    tok_src = model_id if is_adapter(model_id) is False else base_id
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, attn_implementation="sdpa"
    )

    if is_adapter(model_id):
        try:
            from peft import PeftModel
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{model_id} is a LoRA adapter and needs `peft` installed.\n"
                f"  {Path(sys.executable).name} -m pip install -U peft"
            ) from exc
        model = PeftModel.from_pretrained(model, model_id,
                                          torch_dtype=dtype)
        if merge_adapters:
            model = model.merge_and_unload()
            provenance["merged"] = True
        else:
            provenance["merged"] = False
            provenance["warning"] = (
                "adapter left unmerged: the model is a PeftModel, so "
                "gates.acts.blocks() (model.model.layers) does not apply."
            )

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    provenance["loaded"] = base_id
    return model, tok, provenance


def blocks(model):
    """Llama* and Qwen2* both expose ``model.model.layers``."""
    return model.model.layers


# ------------------------------------------------------------ layer tap
class LayerTap:
    """Capture ``resid_post[l]`` for ``l`` in ``layers``, keeping only those.

    Memory-safe: only the requested layers are retained.

        with LayerTap(model, [10, 14, 18]) as tap:
            model(**batch)
            h = tap.get(14)            # [B, T, D]
    """

    def __init__(self, model, layers: Sequence[int]):
        self.model = model
        self.layers = list(layers)
        self._h: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _mk(self, layer_idx: int):
        def fn(_mod, _args, out):
            self._h[layer_idx] = unwrap(out).detach()

        return fn

    def __enter__(self) -> "LayerTap":
        blks = blocks(self.model)
        for layer_idx in self.layers:
            self._handles.append(
                blks[layer_idx].register_forward_hook(self._mk(layer_idx))
            )
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._h.clear()

    def get(self, layer_idx: int) -> torch.Tensor:
        return self._h[layer_idx]

    def stack(self) -> torch.Tensor:
        """[n_layers, B, T, D]"""
        return torch.stack([self._h[i] for i in self.layers], 0)


# ------------------------------------------------------- position gather
def pos_last_idx(attn_mask: torch.Tensor) -> torch.Tensor:
    """Right-padded batch -> index of the last real token. [B]"""
    return attn_mask.sum(1) - 1


def gather_pos(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """h [B,T,D], idx [B] -> [B,D]"""
    return h[torch.arange(h.size(0), device=h.device), idx]


def mask_mean(h: torch.Tensor, attn_mask: torch.Tensor, drop_special: bool = True):
    """Mean over real tokens. h [B,T,D], attn_mask [B,T] with 1=real."""
    m = attn_mask.to(h.dtype)
    if drop_special:
        m = m.clone()
        m[:, 0] = 0
    return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def mask_tail_mean(h: torch.Tensor, attn_mask: torch.Tensor, k: int = 8):
    """Mean over the last k real prompt tokens. k=1 == pos_last for left-padded input."""
    m = attn_mask.to(h.dtype)
    T = h.size(1)
    idx = torch.arange(T, device=h.device)[None, :].expand_as(m)
    last = (attn_mask.sum(1) - 1)[:, None]
    keep = ((idx <= last) & (idx > last - k)).to(h.dtype)
    return (h * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)


# ---------------------------------------------------- offline extraction
@torch.no_grad()
def collect(
    model,
    tok,
    prompts: Sequence[str],
    layers: Sequence[int],
    pos: Literal["last", "tail8", "meanprompt"] = "last",
    batch_size: int = 8,
    max_len: int = 512,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Collect activations -> [n_layers, N, D].

    This is the ONLY sanctioned way to obtain probe features. Deterministic:
    greedy, ``use_cache=False``, identical ``batch_size`` for every checkpoint
    being compared (padding is irrelevant for right-padded ``pos_last`` but is
    NOT for ``meanprompt``).
    """
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"  # pre-flight runs happen on CPU-only laptops

    per_layer: list[list[torch.Tensor]] = [[] for _ in layers]

    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        tok.padding_side = "right"  # required for pos_last_idx
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).to(device)

        with LayerTap(model, layers) as tap:
            model(**enc, use_cache=False)
            for slot, layer_idx in enumerate(layers):
                h = tap.get(layer_idx).to(dtype)
                if pos == "last":
                    f = gather_pos(h, pos_last_idx(enc["attention_mask"]))
                elif pos == "tail8":
                    f = mask_tail_mean(h, enc["attention_mask"], 8)
                elif pos == "meanprompt":
                    f = mask_mean(h, enc["attention_mask"])
                else:
                    raise ValueError(f"unknown pos={pos!r}")
                per_layer[slot].append(f.cpu())

    return torch.stack([torch.cat(chunks) for chunks in per_layer], 0)


# ------------------------------------------------------- dose calibration
@torch.no_grad()
def residual_scale(
    model,
    tok,
    calib_prompts: Sequence[str],
    layers: Sequence[int],
    device: str = "cuda:0",
) -> float:
    """u = median ||resid_post[l]|| over (calibration prompts x injected layers).

    Every alpha is expressed in units of ``u``, because raw alpha on a unit-norm
    direction is not comparable across layers (residual norms grow with depth).
    """
    H = collect(model, tok, calib_prompts, layers, pos="meanprompt", device=device)
    return float(H.norm(dim=-1).median().item())
