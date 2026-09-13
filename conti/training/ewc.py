"""Exact empirical diagonal Fisher matrix engine and EWC regularization utilities.

This module implements:

* :class:`EmpiricalFisherEstimator` -- exact empirical diagonal Fisher Information
  Matrix over trainable (LoRA) parameters using per-sample gradient backward
  passes:  F_i = (1/N) * sum_n (d/dtheta_i log p(y_n | x_n))^2.
* :class:`EWCPenaltyLoss` -- the quadratic Elastic Weight Consolidation penalty
  L_EWC(theta) = (lambda / 2) * sum_i F_0,i * (theta_i - theta_0,i)^2.
* :class:`FisherDivergence` -- divergence metrics between two diagonal Fisher
  states (Frobenius parameter divergence, relative trace shift, Jensen-Shannon
  divergence over normalized parameter-variance distributions).
* Safety-refusal DataLoader builders that consume ``conti/data/safety_replay.jsonl``.

Design constraints honored here (see docs/ENGINEERING_PLAN_subliminal_code.md):

* Only parameters with ``requires_grad=True`` are touched (LoRA adapters);
  frozen base weights never receive Fisher entries.
* Fisher diagonals are accumulated in float32 on the parameter's device and
  offloaded to a storage device (CPU by default) after accumulation to keep
  VRAM free during long runs.
* Per-sample gradients are computed exactly (one backward pass per row) so the
  uncentered second moment is exact under any batch size. Approximate
  batch-sum squared gradients are deliberately NOT offered: they silently
  conflate cross-sample gradient variance with the mean gradient.
* Every cross-device transfer is explicit. A batch placed on the wrong device
  raises :class:`DeviceMismatchError` instead of silently computing on CPU.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceMismatchError",
    "NonFiniteLossError",
    "canonical_param_name",
    "unwrap_data_parallel",
    "trainable_named_parameters",
    "compute_per_sample_grads",
    "EmpiricalFisherEstimator",
    "EWCPenaltyLoss",
    "FisherDivergence",
    "SafetyTextDataset",
    "make_fisher_collate",
    "make_safety_fisher_dataloader",
]

_EPS_DEFAULT = 1e-8


class DeviceMismatchError(RuntimeError):
    """Raised when a tensor is on a different device than the model parameters."""


class NonFiniteLossError(RuntimeError):
    """Raised when a training/Fisher loss is NaN or Inf and cannot be used."""


def canonical_param_name(name: str) -> str:
    """Normalize a parameter name across DDP / FSDP / PEFT wrapper prefixes.

    Distributed and PEFT wrappers prepend prefixes such as ``module.``,
    ``base_model.model.`` or ``_fsdp_wrapped_module.`` to parameter names. The
    Fisher state must be addressable identically no matter which wrapper view
    produced it, so all known wrapper prefixes are stripped here.
    """
    prefixes = ("module.", "base_model.model.", "_fsdp_wrapped_module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Peel ``DistributedDataParallel``-style ``.module`` wrappers off a model.

    Only ``.module`` chains are unwrapped here. PEFT models are intentionally
    kept intact because their trainable parameters are registered on the PEFT
    wrapper itself.
    """
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module  # type: ignore[assignment]
    return model


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return ``(canonical_name, parameter)`` pairs for all trainable parameters.

    Raises:
        ValueError: If the model has no trainable parameters (e.g. LoRA was
            never attached), which would make Fisher computation meaningless.
    """
    pairs = [
        (canonical_param_name(name), param)
        for name, param in model.named_parameters(remove_duplicate=False)
        if param.requires_grad
    ]
    if not pairs:
        raise ValueError(
            "Model has no trainable parameters (requires_grad=True). "
            "Attach LoRA adapters before Fisher/EWC computations."
        )
    seen: set[str] = set()
    for name, _ in pairs:
        if name in seen:
            raise ValueError(f"Canonical parameter name collision after unwrapping: '{name}'")
        seen.add(name)
    return pairs


def _batch_device(batch: Mapping[str, Any]) -> torch.device:
    tensors = [v for v in batch.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError("Batch contains no torch.Tensor entries.")
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        raise DeviceMismatchError(
            f"Batch tensors span multiple devices: {sorted(str(d) for d in devices)}"
        )
    return tensors[0].device


def _assert_batch_on_device(batch: Mapping[str, Any], reference: torch.device) -> None:
    device = _batch_device(batch)
    if device != reference:
        raise DeviceMismatchError(
            f"Batch is on device '{device}' but model parameters are on '{reference}'. "
            "Move the batch to the model device explicitly (batch.to(device))."
        )


def _row_slice(batch: Mapping[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    row: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.ndim < 1:
                raise ValueError(f"Batch tensor '{key}' must have a batch dimension.")
            row[key] = value[index: index + 1]
        else:
            row[key] = value
    return row


def compute_per_sample_grads(
    model: nn.Module,
    row_batch: Mapping[str, torch.Tensor],
    *,
    grad_clip: float | None = None,
) -> dict[str, torch.Tensor] | None:
    """Compute exact per-sample gradients of the CE loss over trainable params.

    The model is switched to eval mode for the duration of the call (dropout
    disabled) and its previous mode is restored afterwards. Gradients are
    returned as float32 clones keyed by canonical parameter name; the model's
    ``.grad`` buffers are always cleared before returning.

    Args:
        model: Any module whose forward accepts ``input_ids``/``attention_mask``
            (and optionally ``labels``) and returns an object with a ``loss``
            attribute when labels are supplied.
        row_batch: A single-row batch (leading batch dimension of size 1).
        grad_clip: Optional per-sample gradient-norm clipping applied *before*
            the gradient is read, for numerical stability of the second moment.

    Returns:
        Mapping of canonical parameter name to gradient tensor, or ``None`` if
        the loss was non-finite (the caller decides whether to skip the sample).
    """
    if "input_ids" not in row_batch:
        raise KeyError("row_batch must contain 'input_ids'.")
    first_param_device = next(p.device for _, p in trainable_named_parameters(model))
    _assert_batch_on_device(row_batch, first_param_device)

    labels = row_batch.get("labels")
    if labels is None:
        raise ValueError(
            "row_batch must contain 'labels' (use -100 padding for ignored positions); "
            "the Fisher engine requires explicit supervised targets."
        )

    was_training = model.training
    model.eval()
    try:
        outputs = model(**dict(row_batch))
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError(
                "Model forward did not return a 'loss' attribute; pass labels and "
                "ensure the model computes the CE loss internally."
            )
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss encountered (%s); sample skipped.", loss.item())
            return None
        loss.backward()
        params = trainable_named_parameters(model)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in params], max_norm=float(grad_clip)
            )
        grads: dict[str, torch.Tensor] = {}
        for name, param in params:
            if param.grad is None:
                continue
            grads[name] = param.grad.detach().to(torch.float32).clone()
        return grads
    finally:
        for _, param in model.named_parameters():
            param.grad = None
        model.train(was_training)


def _default_labels_from_input_ids(
    input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return labels


class EmpiricalFisherEstimator:
    """Exact empirical diagonal Fisher estimator over trainable parameters.

    The estimator iterates over a dataloader of supervised batches, splits every
    batch into single rows, runs one exact backward pass per row, and accumulates
    the uncentered second moment of the per-sample gradients::

        F_i = (1/N) * sum_n (grad_i log p(y_n | x_n))^2 + eps

    Notes:
        * Gradients are clipped per-sample (``grad_clip``) before squaring so a
          single outlier cannot dominate the diagonal.
        * Accumulation happens in float32 on the parameter's device; after the
          pass the diagonals are moved to ``store_device`` (CPU by default) so
          long-running jobs do not hold adapter-sized buffers on the GPU.
        * The estimator never mutates model weights; it toggles eval mode during
          computation and restores the previous mode afterwards.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        grad_clip: float = 1.0,
        eps: float = _EPS_DEFAULT,
        store_device: str | torch.device = "cpu",
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0, got {grad_clip}.")
        self._model = unwrap_data_parallel(model)
        self._params = trainable_named_parameters(self._model)
        self._grad_clip = float(grad_clip)
        self._eps = float(eps)
        self._store_device = torch.device(store_device)
        self._fisher: dict[str, torch.Tensor] = {}
        self._n_samples = 0
        self._n_skipped_nonfinite = 0
        self._ready = False

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_skipped_nonfinite(self) -> int:
        return self._n_skipped_nonfinite

    @property
    def is_ready(self) -> bool:
        return self._ready and len(self._fisher) > 0

    @property
    def fisher(self) -> dict[str, torch.Tensor]:
        """Copy of the current Fisher diagonal (keyed by canonical names)."""
        return {name: tensor.clone() for name, tensor in self._fisher.items()}

    def compute_fisher(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        max_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the exact per-sample Fisher pass over ``batches``.

        Args:
            batches: Iterable of dicts with ``input_ids``, ``attention_mask`` and
                ``labels`` tensors (any extra tensor entries are forwarded to the
                model). May be a ``torch.utils.data.DataLoader``.
            max_samples: Optional cap on the number of *rows* consumed. Rows are
                counted individually because every row gets its own backward pass.

        Returns:
            A fresh dict mapping canonical parameter names to float32 Fisher
            diagonals placed on ``store_device``.

        Raises:
            DeviceMismatchError: If a batch is on the wrong device.
            RuntimeError: If fewer than one usable sample was processed.
        """
        reference_device = self._params[0][1].device
        accumulators: dict[str, torch.Tensor] = {}
        n_rows = 0
        n_skipped = 0
        row_cap = max_samples if (max_samples is not None and max_samples > 0) else None

        was_training = self._model.training
        self._model.eval()
        try:
            for batch in batches:
                if not isinstance(batch, Mapping):
                    raise TypeError(
                        f"Fisher batches must be mappings of tensors, got {type(batch)!r}."
                    )
                if "input_ids" not in batch:
                    raise KeyError("Fisher batch is missing 'input_ids'.")
                if "labels" not in batch:
                    batch = dict(batch)
                    batch["labels"] = _default_labels_from_input_ids(
                        batch["input_ids"], batch.get("attention_mask")
                    )
                _assert_batch_on_device(batch, reference_device)

                batch_size = int(batch["input_ids"].shape[0])
                for row_index in range(batch_size):
                    if row_cap is not None and n_rows >= row_cap:
                        break
                    row = _row_slice(batch, row_index)
                    grads = compute_per_sample_grads(
                        self._model, row, grad_clip=self._grad_clip
                    )
                    if grads is None:
                        n_skipped += 1
                        continue
                    for name, grad in grads.items():
                        squared = grad.pow(2)
                        if name not in accumulators:
                            accumulators[name] = squared.clone()
                        else:
                            accumulators[name] += squared
                    n_rows += 1
                if row_cap is not None and n_rows >= row_cap:
                    break
        finally:
            self._model.train(was_training)
            for _, param in self._model.named_parameters():
                param.grad = None

        if n_rows == 0:
            raise RuntimeError(
                "EmpiricalFisherEstimator processed zero usable samples "
                f"({n_skipped} skipped as non-finite). Check the dataloader and labels."
            )

        fisher: dict[str, torch.Tensor] = {}
        for name, accumulator in accumulators.items():
            diagonal = accumulator / float(n_rows) + self._eps
            if not torch.isfinite(diagonal).all():
                raise NonFiniteLossError(f"Fisher diagonal for '{name}' contains non-finite values.")
            fisher[name] = diagonal.to(self._store_device)

        self._fisher = fisher
        self._n_samples = n_rows
        self._n_skipped_nonfinite = n_skipped
        self._ready = True
        logger.info(
            "Empirical Fisher computed over %d samples (%d skipped); %d parameter tensors.",
            n_rows,
            n_skipped,
            len(fisher),
        )
        return {name: tensor.clone() for name, tensor in fisher.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "fisher": {name: tensor.detach().cpu() for name, tensor in self._fisher.items()},
            "n_samples": self._n_samples,
            "n_skipped_nonfinite": self._n_skipped_nonfinite,
            "eps": self._eps,
            "grad_clip": self._grad_clip,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        fisher = state.get("fisher")
        if not isinstance(fisher, Mapping) or not fisher:
            raise ValueError("state_dict['fisher'] must be a non-empty mapping of tensors.")
        loaded: dict[str, torch.Tensor] = {}
        for name, tensor in fisher.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Fisher entry '{name}' is not a tensor (got {type(tensor)!r}).")
            if tensor.numel() == 0:
                raise ValueError(f"Fisher entry '{name}' is empty.")
            if not torch.isfinite(tensor).all():
                raise NonFiniteLossError(f"Fisher entry '{name}' contains non-finite values.")
            if (tensor < 0).any():
                raise ValueError(f"Fisher entry '{name}' contains negative values.")
            loaded[str(name)] = tensor.detach().to(self._store_device, dtype=torch.float32)
        self._fisher = loaded
        self._n_samples = int(state.get("n_samples", 0))
        self._n_skipped_nonfinite = int(state.get("n_skipped_nonfinite", 0))
        self._ready = True

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fisher state file not found: {path}")
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)


class EWCPenaltyLoss:
    """Quadratic EWC penalty against frozen Round-0 parameters.

    Computes ``L = (lambda / 2) * sum_i F_0,i * (theta_i - theta_0,i)^2`` over
    the intersection of the Fisher state and the model's trainable parameters.
    Fisher and reference tensors may live on CPU; per-name cached copies are
    moved lazily to the parameter's device on first use and reused afterwards.
    """

    def __init__(
        self,
        fisher: Mapping[str, torch.Tensor],
        ref_params: Mapping[str, torch.Tensor],
        lambda_ewc: float = 5000.0,
    ) -> None:
        if lambda_ewc < 0:
            raise ValueError(f"lambda_ewc must be >= 0, got {lambda_ewc}.")
        if not fisher or not ref_params:
            raise ValueError("Both 'fisher' and 'ref_params' must be non-empty mappings.")
        self._lambda = float(lambda_ewc)
        self._fisher_src: dict[str, torch.Tensor] = {}
        self._ref_src: dict[str, torch.Tensor] = {}
        for name, tensor in fisher.items():
            if name not in ref_params:
                continue
            if not isinstance(tensor, torch.Tensor) or not isinstance(ref_params[name], torch.Tensor):
                raise TypeError(f"Fisher/reference entry '{name}' must be tensors.")
            if tensor.shape != ref_params[name].shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': fisher {tuple(tensor.shape)} vs "
                    f"reference {tuple(ref_params[name].shape)}."
                )
            if not torch.isfinite(tensor).all():
                raise NonFiniteLossError(f"Fisher entry '{name}' contains non-finite values.")
            if (tensor < 0).any():
                raise ValueError(f"Fisher entry '{name}' contains negative values.")
            self._fisher_src[name] = tensor.detach().to(torch.float32).cpu()
            self._ref_src[name] = ref_params[name].detach().to(torch.float32).cpu()
        if not self._fisher_src:
            raise RuntimeError(
                "EWCPenaltyLoss: Fisher state and reference parameters share no common names. "
                f"fisher names (first 5): {sorted(fisher)[:5]}; "
                f"ref names (first 5): {sorted(ref_params)[:5]}."
            )
        self._fisher_cache: dict[str, torch.Tensor] = {}
        self._ref_cache: dict[str, torch.Tensor] = {}

    @property
    def lambda_ewc(self) -> float:
        return self._lambda

    @property
    def is_ready(self) -> bool:
        return len(self._fisher_src) > 0

    def shared_names(self) -> list[str]:
        return sorted(self._fisher_src.keys())

    def _to_device(self, cache: dict[str, torch.Tensor], source: Mapping[str, torch.Tensor],
                   name: str, device: torch.device) -> torch.Tensor:
        tensor = cache.get(name)
        if tensor is None or tensor.device != device:
            tensor = source[name].to(device, non_blocking=True)
            cache[name] = tensor
        return tensor

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Differentiable EWC penalty for the current parameter values."""
        total: torch.Tensor | None = None
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            canonical = canonical_param_name(name)
            if canonical not in self._fisher_src:
                continue
            fisher_t = self._to_device(self._fisher_cache, self._fisher_src, canonical, param.device)
            ref_t = self._to_device(self._ref_cache, self._ref_src, canonical, param.device)
            delta = param - ref_t
            term = (fisher_t * delta * delta).sum()
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError(
                "EWCPenaltyLoss.penalty found no trainable parameters overlapping the Fisher "
                "state. The model was probably re-wrapped (LoRA re-attached) after the "
                "penalty was constructed."
            )
        penalty_value = 0.5 * self._lambda * total
        if not torch.isfinite(penalty_value):
            raise NonFiniteLossError("EWC penalty is non-finite.")
        return penalty_value

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """Alias of :meth:`penalty` so the loss object can be invoked directly."""
        return self.penalty(model)


class FisherDivergence:
    """Divergence metrics between two diagonal Fisher states."""

    @staticmethod
    def _validate_pair(
        fisher_a: Mapping[str, torch.Tensor], fisher_b: Mapping[str, torch.Tensor]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if not fisher_a or not fisher_b:
            raise ValueError("Both Fisher states must be non-empty mappings.")
        shared = sorted(set(fisher_a.keys()) & set(fisher_b.keys()))
        if not shared:
            raise RuntimeError(
                "Fisher states share no common parameter names; cannot compute divergence."
            )
        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in shared:
            a = fisher_a[name].detach().to(torch.float32).cpu().flatten()
            b = fisher_b[name].detach().to(torch.float32).cpu().flatten()
            if a.shape != b.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': {tuple(a.shape)} vs {tuple(b.shape)}."
                )
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise NonFiniteLossError(f"Fisher entry '{name}' contains non-finite values.")
            pairs[name] = (a, b)
        return pairs

    @staticmethod
    def frobenius_parameter_divergence(
        fisher_baseline: Mapping[str, torch.Tensor],
        fisher_current: Mapping[str, torch.Tensor],
    ) -> float:
        """||F_current - F_baseline||_F over the shared parameter names."""
        pairs = FisherDivergence._validate_pair(fisher_baseline, fisher_current)
        total = torch.zeros((), dtype=torch.float32)
        for _, (a, b) in pairs.items():
            diff = a - b
            total = total + diff.pow(2).sum()
        return float(torch.sqrt(total).item())

    @staticmethod
    def relative_trace_shift(
        fisher_baseline: Mapping[str, torch.Tensor],
        fisher_current: Mapping[str, torch.Tensor],
        *,
        eps: float = _EPS_DEFAULT,
    ) -> float:
        """(tr(F_current) - tr(F_baseline)) / (tr(F_baseline) + eps)."""
        pairs = FisherDivergence._validate_pair(fisher_baseline, fisher_current)
        trace_a = float(sum(a.sum().item() for a, _ in pairs.values()))
        trace_b = float(sum(b.sum().item() for _, b in pairs.values()))
        return (trace_b - trace_a) / (trace_a + eps)

    @staticmethod
    def jensen_shannon_divergence(
        fisher_baseline: Mapping[str, torch.Tensor],
        fisher_current: Mapping[str, torch.Tensor],
        *,
        eps: float = _EPS_DEFAULT,
    ) -> float:
        """Jensen-Shannon divergence (nats) over normalized Fisher mass distributions.

        Each Fisher diagonal is flattened over all shared parameters and
        normalized into a probability distribution over parameter coordinates;
        the JSD between the two distributions is returned (base e, so the
        maximum is ln(2)).
        """
        pairs = FisherDivergence._validate_pair(fisher_baseline, fisher_current)
        a_parts = [a for a, _ in pairs.values()]
        b_parts = [b for _, b in pairs.values()]
        a_flat = torch.cat(a_parts) if a_parts else torch.zeros(1)
        b_flat = torch.cat(b_parts) if b_parts else torch.zeros(1)
        a_prob = a_flat / (a_flat.sum() + eps)
        b_prob = b_flat / (b_flat.sum() + eps)
        m_prob = 0.5 * (a_prob + b_prob)

        def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            mask = p > 0
            return (p[mask] * torch.log((p[mask] + eps) / (q[mask] + eps))).sum()

        jsd = 0.5 * _kl(a_prob, m_prob) + 0.5 * _kl(b_prob, m_prob)
        return float(max(0.0, min(float(jsd.item()), math.log(2.0))))

    @staticmethod
    def compute_all(
        fisher_baseline: Mapping[str, torch.Tensor],
        fisher_current: Mapping[str, torch.Tensor],
    ) -> dict[str, float | int]:
        """Convenience bundle of every divergence metric."""
        pairs = FisherDivergence._validate_pair(fisher_baseline, fisher_current)
        n_shared = int(sum(a.numel() for a, _ in pairs.values()))
        jsd_nats = FisherDivergence.jensen_shannon_divergence(fisher_baseline, fisher_current)
        return {
            "frobenius": FisherDivergence.frobenius_parameter_divergence(
                fisher_baseline, fisher_current
            ),
            "relative_trace_shift": FisherDivergence.relative_trace_shift(
                fisher_baseline, fisher_current
            ),
            "jensen_shannon_nats": jsd_nats,
            "jensen_shannon_bits": jsd_nats / math.log(2.0),
            "n_shared_params": n_shared,
        }


class SafetyTextDataset(Dataset):
    """Dataset over a JSONL file of safety refusal examples.

    Accepted row schemas (validated lazily at ``__getitem__`` time):

    * ``{"prompt": str, "response": str}`` -- preferred; the Fisher loss is
      computed as log p(response | prompt) by masking prompt tokens in labels.
    * ``{"text": str}`` -- plain language-modeling over the full text.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Safety JSONL not found: {self._path}")
        rows: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {self._path}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} of {self._path} is not a JSON object.")
                rows.append(row)
        if not rows:
            raise ValueError(f"Safety JSONL at {self._path} contains no rows.")
        for index, row in enumerate(rows):
            has_pair = "prompt" in row and "response" in row
            has_text = "text" in row
            if not (has_pair or has_text):
                raise ValueError(
                    f"Row {index} of {self._path} needs ('prompt'+'response') or 'text'; "
                    f"got keys {sorted(row.keys())}."
                )
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]


def make_fisher_collate(
    tokenizer: Any,
    *,
    max_len: int = 512,
):
    """Build a collate function producing Fisher-ready tensor batches.

    The collate tokenizes ``prompt``/``response`` pairs (masking the prompt out
    of the labels so the Fisher measures log p(refusal | prompt)) or plain
    ``text`` rows. Batches are padded to the longest sequence in the batch.
    """

    def _encode_ids(text: str, add_specials: bool) -> list[int]:
        encoded = tokenizer(
            text,
            add_special_tokens=add_specials,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        ids = encoded.get("input_ids")
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        if not ids:
            raise ValueError(f"Tokenizer produced zero tokens for text: {text[:80]!r}")
        return [int(t) for t in ids]

    pad_id: int | None = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        pad_id = int(eos_id)

    def collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for row in rows:
            if "prompt" in row and "response" in row:
                prompt_ids = _encode_ids(str(row["prompt"]), add_specials=True)
                response_ids = _encode_ids(str(row["response"]), add_specials=False)
                allowed_response = max(1, max_len - len(prompt_ids))
                response_ids = response_ids[:allowed_response]
                input_rows.append(prompt_ids + response_ids)
                label_rows.append([-100] * len(prompt_ids) + response_ids)
            else:
                ids = _encode_ids(str(row["text"]), add_specials=True)[:max_len]
                input_rows.append(ids)
                label_rows.append(list(ids))

        max_length = max(len(ids) for ids in input_rows)
        batch_input: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_mask: list[list[int]] = []
        for ids, labels in zip(input_rows, label_rows):
            padding = max_length - len(ids)
            batch_input.append(ids + [int(pad_id)] * padding)
            batch_labels.append(labels + [-100] * padding)
            batch_mask.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def make_safety_fisher_dataloader(
    jsonl_path: str | Path,
    tokenizer: Any,
    *,
    batch_size: int = 2,
    max_len: int = 512,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """DataLoader over a safety refusal JSONL for Fisher computation.

    Batch size defaults to a small value because the estimator splits batches
    into single rows internally; the batch only controls tokenization padding
    efficiency.
    """
    dataset = SafetyTextDataset(jsonl_path)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=make_fisher_collate(tokenizer, max_len=max_len),
        num_workers=0,
    )
