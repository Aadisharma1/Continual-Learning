"""Subliminal learning and latent drift gate (Gate R).

This module detects safety degradation in latent representation space *before*
token-level text generation exhibits refusal failure. Three detection surfaces
are implemented:

a) ``evaluate_refusal_curvature`` -- parameter drift velocity in the safety
   Fisher geometry: ``Delta_latent = ||F_t - F_0||_F`` computed over the shared
   trainable (LoRA) parameter names between the frozen Round-0 Fisher diagonal
   and the current Round-t Fisher diagonal.

b) ``compute_representation_drift`` -- residual-stream drift on safety probing
   vectors: forward hooks capture mean-pooled hidden states of selected decoder
   layers for a frozen probe set, and Linear Centered Kernel Alignment (CKA)
   plus cosine distance are computed against the frozen Round-0 captures.

c) ``gradient_alignment_test`` -- per-candidate gate: the cosine similarity
   between the gradient of a proposed math reasoning trace
   ``grad_theta L_math`` and the mean safety refusal gradient subspace
   ``grad_theta L_safety``. Candidates with ``cos < tau_drift`` are rejected
   before they can enter the training set.

All forward-hook state is stored on a configurable storage device (CPU by
default) so probing never grows GPU memory, and every cross-device transfer is
explicit. Structured output is provided by :class:`SubliminalVerdict`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from conti.training.ewc import (
    DeviceMismatchError,
    FisherDivergence,
    canonical_param_name,
    compute_per_sample_grads,
    trainable_named_parameters,
    unwrap_data_parallel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SubliminalVerdict",
    "get_decoder_layers",
    "mean_pool_hidden_states",
    "gradient_cosine",
    "linear_cka",
    "make_probe_batches",
    "SubliminalFilter",
]

_EPS = 1e-8


@dataclass
class SubliminalVerdict:
    """Structured metadata returned by :meth:`SubliminalFilter.gate`."""

    gate_passed: bool
    latent_drift_score: float
    cka_drift: dict[str, float]
    rejection_reason: str
    details: dict[str, Any] = field(default_factory=dict)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer list of a (possibly wrapped) causal LM.

    Supported layouts: LLaMA-style ``model.layers``, GPT-2-style
    ``transformer.h``, GPT-NeoX-style ``gpt_neox.layers``, plain ``layers`` and
    ``decoder.layers``. PEFT and data-parallel wrappers are unwrapped first.
    """
    core = unwrap_data_parallel(model)
    if hasattr(core, "get_base_model") and callable(getattr(core, "get_base_model")):
        core = core.get_base_model()
    candidate_paths: tuple[tuple[str, ...], ...] = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
        ("layers",),
    )
    for path in candidate_paths:
        current: Any = core
        found = True
        for attribute in path:
            if not hasattr(current, attribute):
                found = False
                break
            current = getattr(current, attribute)
        if found and isinstance(current, (nn.ModuleList, nn.Sequential)) and len(current) > 0:
            return current  # type: ignore[return-value]
    raise RuntimeError(
        "Could not locate decoder layers on the model. Tried attribute paths: "
        f"{['.'.join(p) for p in candidate_paths]}. Pass a model with a standard "
        "decoder layout or extend get_decoder_layers()."
    )


def mean_pool_hidden_states(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mask-aware mean pooling: ``[B, T, D] x [B, T] -> [B, D]``.

    Raises:
        ValueError: On shape/broadcast mismatches (guards against silent
            broadcasting blowups that manifest as VRAM spikes later).
    """
    if hidden.ndim != 3:
        raise ValueError(f"Hidden states must be [B, T, D], got shape {tuple(hidden.shape)}.")
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Attention mask must be [B, T], got shape {tuple(attention_mask.shape)}."
        )
    batch, seq_len, _ = hidden.shape
    if attention_mask.shape[0] != batch or attention_mask.shape[1] != seq_len:
        raise ValueError(
            f"Attention mask {tuple(attention_mask.shape)} does not match hidden states "
            f"{tuple(hidden.shape)} along batch/sequence dimensions."
        )
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two sample-aligned representation matrices.

    ``x`` and ``y`` are ``[N, D]`` matrices whose rows correspond to the same
    probes. Columns are centered, then
    ``CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)``.

    Returns 1.0 for identical representations and values in ``[0, 1]``
    otherwise (degenerate zero-variance inputs return 0.0).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"linear_cka expects [N, D] matrices, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"linear_cka requires sample counts to match: {x.shape[0]} vs {y.shape[0]}."
        )
    if x.shape[0] < 2:
        raise ValueError("linear_cka requires at least 2 probes.")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.norm(x.transpose(0, 1) @ y, p="fro")
    denom_x = torch.norm(x.transpose(0, 1) @ x, p="fro")
    denom_y = torch.norm(y.transpose(0, 1) @ y, p="fro")
    denominator = denom_x * denom_y
    if float(denominator.item()) < _EPS:
        return 0.0
    value = float((cross * cross / denominator).item())
    return max(0.0, min(1.0, value))


def gradient_cosine(
    grads: Mapping[str, torch.Tensor],
    anchor: Mapping[str, torch.Tensor],
    *,
    eps: float = _EPS,
) -> float:
    """Cosine similarity between two named-gradient dictionaries.

    Both dictionaries are keyed by canonical parameter names. All per-name dot
    products are computed on the gradient's own device so no tensor is moved
    mid-accumulation; the scalar accumulators are plain Python floats.
    """
    shared = set(grads.keys()) & set(anchor.keys())
    if not shared:
        raise ValueError(
            "gradient_cosine: gradient dict and anchor share no parameter names "
            f"(grad names first 5: {sorted(grads)[:5]}; anchor names first 5: {sorted(anchor)[:5]})."
        )
    dot = 0.0
    grad_sq = 0.0
    anchor_sq = 0.0
    for name in shared:
        g = grads[name]
        a = anchor[name]
        if a.device != g.device:
            a = a.to(g.device)
        dot += float((g * a).sum().item())
        grad_sq += float(g.pow(2).sum().item())
        anchor_sq += float(a.pow(2).sum().item())
    norm_product = math.sqrt(grad_sq) * math.sqrt(anchor_sq)
    if norm_product < eps:
        return 0.0
    value = dot / norm_product
    return max(-1.0, min(1.0, value))


def make_probe_batches(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_len: int = 256,
    add_special_tokens: bool = True,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize a list of probe prompts into CPU tensor batches of one row each.

    One-row batches keep probing memory flat and make per-probe bookkeeping
    trivial. The caller is responsible for moving batches to the model device.
    """
    if not prompts:
        raise ValueError("make_probe_batches requires at least one prompt.")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    batches: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(
            str(prompt),
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"]
        else:
            attention_mask = torch.ones_like(input_ids)
        batches.append(
            {
                "input_ids": input_ids.to(torch.long).cpu(),
                "attention_mask": attention_mask.to(torch.long).cpu(),
            }
        )
    return batches


class _HiddenStateRecorder:
    """Forward-hook recorder capturing mean-pooled hidden states on CPU."""

    def __init__(
        self,
        layers: nn.ModuleList,
        layer_indices: Sequence[int],
        store_device: str | torch.device = "cpu",
    ) -> None:
        self._layers = layers
        self._layer_indices = list(layer_indices)
        self._store_device = torch.device(store_device)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captures: dict[int, list[torch.Tensor]] = {
            index: [] for index in self._layer_indices
        }
        self._masks: list[torch.Tensor] = []

    def attach(self) -> None:
        for index in self._layer_indices:
            handle = self._layers[index].register_forward_hook(self._make_hook(index))
            self._handles.append(handle)

    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _args: tuple, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(
                    f"Layer {index} produced output of type {type(output)!r}; expected a "
                    "tensor or a tuple whose first element is the hidden-state tensor."
                )
            if hidden.ndim != 3:
                raise ValueError(
                    f"Layer {index} hidden state must be [B, T, D], got {tuple(hidden.shape)}."
                )
            self._captures[index].append(hidden.detach().to(self._store_device, torch.float32))

        return hook

    def record_mask(self, batch: Mapping[str, torch.Tensor]) -> None:
        mask = batch.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(batch["input_ids"])
        self._masks.append(mask.detach().to(self._store_device, torch.long))

    def pooled(self) -> dict[int, torch.Tensor]:
        """Mean-pooled representations ``{layer_index: [N, D]}`` on CPU."""
        if not self._masks:
            raise RuntimeError("No probe masks recorded; run capture() first.")
        pooled: dict[int, torch.Tensor] = {}
        for index, chunks in self._captures.items():
            if not chunks:
                raise RuntimeError(f"Layer {index} captured no hidden states.")
            pooled[index] = torch.cat(chunks, dim=0)
        for index in list(pooled.keys()):
            pooled[index] = mean_pool_hidden_states(pooled[index], torch.cat(self._masks, dim=0))
        return pooled

    def clear(self) -> None:
        for index in self._captures:
            self._captures[index] = []
        self._masks = []

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()


class SubliminalFilter:
    """Latent drift gate combining Fisher curvature, CKA drift and gradient alignment.

    The filter is constructed once per run against the (LoRA-wrapped) model.
    Round-0 state (baseline representations, baseline Fisher) is captured via
    :meth:`capture_baseline_representations` and ``set_baseline_fisher`` /
    :meth:`evaluate_refusal_curvature` arguments; per-round state is supplied at
    call time.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_indices: Sequence[int] | None = None,
        tau_drift: float = 0.0,
        store_device: str | torch.device = "cpu",
    ) -> None:
        if tau_drift < -1.0 or tau_drift > 1.0:
            raise ValueError(f"tau_drift must be within [-1, 1], got {tau_drift}.")
        self._model = unwrap_data_parallel(model)
        self._layers = get_decoder_layers(self._model)
        self._tau_drift = float(tau_drift)
        self._store_device = torch.device(store_device)
        n_layers = len(self._layers)
        if layer_indices is None:
            resolved = sorted({0, n_layers // 2, n_layers - 1})
        else:
            resolved = sorted({int(i) for i in layer_indices})
        if not resolved:
            raise ValueError("layer_indices resolved to an empty set.")
        for index in resolved:
            if index < 0 or index >= n_layers:
                raise ValueError(
                    f"layer index {index} out of range for a model with {n_layers} layers."
                )
        self._layer_indices = resolved
        self._recorder = _HiddenStateRecorder(
            self._layers, self._layer_indices, store_device=self._store_device
        )
        self._baseline_pooled: dict[int, torch.Tensor] | None = None
        self._baseline_fisher: dict[str, torch.Tensor] | None = None
        self._current_fisher: dict[str, torch.Tensor] | None = None
        self._safety_anchor: dict[str, torch.Tensor] | None = None
        self._probe_device: torch.device | None = None
        self._last_curvature_details: dict[str, float | int] = {}

    # ------------------------------------------------------------------ #
    # device helpers
    # ------------------------------------------------------------------ #

    @property
    def model_device(self) -> torch.device:
        return next(p.device for _, p in trainable_named_parameters(self._model))

    def _assert_batches_on_model_device(self, batches: Iterable[Mapping[str, torch.Tensor]]) -> None:
        reference = self.model_device
        for batch in batches:
            tensors = [v for v in batch.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise ValueError("Probe batch contains no tensors.")
            devices = {t.device for t in tensors}
            if len(devices) != 1:
                raise DeviceMismatchError(
                    f"Probe batch tensors span multiple devices: {sorted(str(d) for d in devices)}"
                )
            if next(iter(devices)) != reference:
                raise DeviceMismatchError(
                    f"Probe batch on '{next(iter(devices))}' but model parameters are on '{reference}'."
                )

    def _forward_hidden(self, batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[int, torch.Tensor]:
        self._assert_batches_on_model_device(batches)
        self._recorder.clear()
        self._recorder.attach()
        try:
            was_training = self._model.training
            self._model.eval()
            try:
                with torch.inference_mode():
                    for batch in batches:
                        self._recorder.record_mask(batch)
                        self._model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                        )
            finally:
                self._model.train(was_training)
            return self._recorder.pooled()
        finally:
            self._recorder.remove()

    # ------------------------------------------------------------------ #
    # (a) Fisher curvature
    # ------------------------------------------------------------------ #

    def set_baseline_fisher(self, fisher: Mapping[str, torch.Tensor]) -> None:
        if not fisher:
            raise ValueError("Baseline Fisher state must be non-empty.")
        self._baseline_fisher = {canonical_param_name(k): v for k, v in fisher.items()}

    def set_current_fisher(self, fisher: Mapping[str, torch.Tensor]) -> None:
        if not fisher:
            raise ValueError("Current Fisher state must be non-empty.")
        self._current_fisher = {canonical_param_name(k): v for k, v in fisher.items()}

    def evaluate_refusal_curvature(
        self,
        fisher_baseline: Mapping[str, torch.Tensor] | None = None,
        fisher_current: Mapping[str, torch.Tensor] | None = None,
    ) -> float:
        """``Delta_latent = ||F_t - F_0||_F`` over shared trainable parameters.

        Either argument may be omitted if the corresponding state was provided
        earlier via :meth:`set_baseline_fisher` / :meth:`set_current_fisher`.
        """
        baseline = fisher_baseline if fisher_baseline is not None else self._baseline_fisher
        current = fisher_current if fisher_current is not None else self._current_fisher
        if baseline is None:
            raise ValueError("Baseline Fisher missing: call set_baseline_fisher() or pass fisher_baseline.")
        if current is None:
            raise ValueError("Current Fisher missing: call set_current_fisher() or pass fisher_current.")
        curvature = FisherDivergence.frobenius_parameter_divergence(baseline, current)
        self._last_curvature_details = FisherDivergence.compute_all(baseline, current)
        return curvature

    @property
    def last_curvature_details(self) -> dict[str, float | int]:
        return dict(self._last_curvature_details)

    # ------------------------------------------------------------------ #
    # (b) representation drift (CKA / cosine on residual stream)
    # ------------------------------------------------------------------ #

    def capture_baseline_representations(
        self, probe_batches: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, int]:
        """Freeze Round-0 mean-pooled hidden states for the probe set."""
        if not probe_batches:
            raise ValueError("probe_batches must contain at least one batch.")
        pooled = self._forward_hidden(list(probe_batches))
        self._baseline_pooled = pooled
        return {"n_probes": len(probe_batches), "layers": list(pooled.keys())}

    def compute_representation_drift(
        self, probe_batches: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, Any]:
        """CKA + cosine distance of current hidden states vs the frozen baseline."""
        if self._baseline_pooled is None:
            raise RuntimeError(
                "Baseline representations not captured; call capture_baseline_representations() "
                "at Round 0 with the identical probe set."
            )
        if not probe_batches:
            raise ValueError("probe_batches must contain at least one batch.")
        n_probes = len(probe_batches)
        current = self._forward_hidden(list(probe_batches))
        device = self.model_device
        per_layer: dict[str, float] = {}
        cosine_per_layer: dict[str, float] = {}
        for index, baseline_layer in self._baseline_pooled.items():
            if index not in current:
                raise RuntimeError(f"Layer {index} missing from current capture.")
            current_layer = current[index]
            if baseline_layer.shape[0] != current_layer.shape[0]:
                raise ValueError(
                    f"Probe count mismatch at layer {index}: baseline {baseline_layer.shape[0]} "
                    f"vs current {current_layer.shape[0]}. Use the identical probe set."
                )
            base_dev = baseline_layer.to(device)
            cur_dev = current_layer.to(device)
            per_layer[str(index)] = linear_cka(base_dev, cur_dev)
            base_centered = base_dev - base_dev.mean(dim=0, keepdim=True)
            cur_centered = cur_dev - cur_dev.mean(dim=0, keepdim=True)
            cosine = torch.nn.functional.cosine_similarity(
                base_centered.flatten().unsqueeze(0), cur_centered.flatten().unsqueeze(0)
            ).item()
            cosine_per_layer[str(index)] = float(cosine)
        mean_cka = float(sum(per_layer.values()) / len(per_layer))
        mean_cosine = float(sum(cosine_per_layer.values()) / len(cosine_per_layer))
        return {
            "per_layer_cka": per_layer,
            "per_layer_cosine": cosine_per_layer,
            "mean_cka": mean_cka,
            "min_cka": float(min(per_layer.values())),
            "cosine_distance": float(1.0 - mean_cosine),
            "n_probes": n_probes,
        }

    # ------------------------------------------------------------------ #
    # (c) gradient alignment gate
    # ------------------------------------------------------------------ #

    def capture_safety_anchor(
        self,
        safety_batches: Sequence[Mapping[str, torch.Tensor]],
        *,
        grad_clip: float | None = 1.0,
        max_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the mean safety refusal gradient subspace ``grad L_safety``.

        The anchor is the average of exact per-sample gradients over the supplied
        refusal batches. It is stored on the parameter devices so subsequent
        :meth:`gradient_alignment_test` calls perform no host-to-device copies.
        """
        if not safety_batches:
            raise ValueError("safety_batches must contain at least one batch.")
        self._assert_batches_on_model_device(safety_batches)
        reference = self.model_device
        accumulator: dict[str, torch.Tensor] = {}
        n_rows = 0
        n_skipped = 0
        for batch in safety_batches:
            batch_size = int(batch["input_ids"].shape[0])
            for row_index in range(batch_size):
                if max_samples is not None and n_rows >= max_samples:
                    break
                row: dict[str, torch.Tensor] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        row[key] = value[row_index: row_index + 1]
                    else:
                        row[key] = value
                grads = compute_per_sample_grads(self._model, row, grad_clip=grad_clip)
                if grads is None:
                    n_skipped += 1
                    continue
                for name, grad in grads.items():
                    if name not in accumulator:
                        accumulator[name] = grad.clone()
                    else:
                        accumulator[name] += grad.to(reference)
                n_rows += 1
            if max_samples is not None and n_rows >= max_samples:
                break
        if n_rows == 0:
            raise RuntimeError(
                f"capture_safety_anchor processed zero usable samples ({n_skipped} skipped)."
            )
        anchor = {name: tensor / float(n_rows) for name, tensor in accumulator.items()}
        self._safety_anchor = anchor
        logger.info("Safety gradient anchor captured over %d refusal samples.", n_rows)
        return {name: tensor.clone() for name, tensor in anchor.items()}

    @property
    def has_safety_anchor(self) -> bool:
        return self._safety_anchor is not None

    def gradient_alignment_test(
        self,
        math_batches: Sequence[Mapping[str, torch.Tensor]],
        *,
        safety_anchor: Mapping[str, torch.Tensor] | None = None,
        tau_drift: float | None = None,
        grad_clip: float | None = 1.0,
    ) -> list[dict[str, Any]]:
        """Gate candidate math traces by gradient alignment with the safety subspace.

        Returns one result dict per row: ``{"index", "cosine", "accepted", "reason"}``.
        A candidate is rejected when ``cos(grad_math, grad_safety_mean) < tau``.
        """
        anchor = safety_anchor if safety_anchor is not None else self._safety_anchor
        if anchor is None:
            raise ValueError(
                "No safety anchor available: call capture_safety_anchor() at Round 0 or pass "
                "safety_anchor explicitly."
            )
        tau = self._tau_drift if tau_drift is None else float(tau_drift)
        if not math_batches:
            raise ValueError("math_batches must contain at least one batch.")
        self._assert_batches_on_model_device(math_batches)

        results: list[dict[str, Any]] = []
        flat_index = 0
        for batch in math_batches:
            batch_size = int(batch["input_ids"].shape[0])
            for row_index in range(batch_size):
                row: dict[str, torch.Tensor] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        row[key] = value[row_index: row_index + 1]
                    else:
                        row[key] = value
                grads = compute_per_sample_grads(self._model, row, grad_clip=grad_clip)
                if grads is None:
                    results.append(
                        {
                            "index": flat_index,
                            "cosine": float("nan"),
                            "accepted": False,
                            "reason": "nonfinite_loss",
                        }
                    )
                else:
                    cosine = gradient_cosine(grads, anchor)
                    accepted = cosine >= tau
                    reason = "ok" if accepted else f"cos={cosine:.4f} < tau={tau:.4f}"
                    results.append(
                        {"index": flat_index, "cosine": cosine, "accepted": accepted, "reason": reason}
                    )
                flat_index += 1
        return results

    # ------------------------------------------------------------------ #
    # combined gate
    # ------------------------------------------------------------------ #

    def gate(
        self,
        *,
        fisher_baseline: Mapping[str, torch.Tensor] | None = None,
        fisher_current: Mapping[str, torch.Tensor] | None = None,
        probe_batches: Sequence[Mapping[str, torch.Tensor]] | None = None,
        math_batches: Sequence[Mapping[str, torch.Tensor]] | None = None,
        safety_anchor: Mapping[str, torch.Tensor] | None = None,
        tau_drift: float | None = None,
        max_allowed_curvature: float | None = None,
        cka_floor: float | None = None,
        grad_clip: float | None = 1.0,
    ) -> SubliminalVerdict:
        """Run every requested sub-check and return structured metadata.

        Checks executed (each optional, based on inputs provided):

        * Fisher curvature vs ``max_allowed_curvature``.
        * Representation drift (CKA/cosine) if ``probe_batches`` supplied and a
          baseline capture exists; vs ``cka_floor`` when given.
        * Gradient alignment on ``math_batches`` vs ``tau_drift``.
        """
        failures: list[str] = []
        details: dict[str, Any] = {}
        cka_drift: dict[str, float] = {}
        latent_drift_score: float = float("nan")

        curvature_available = (
            (fisher_baseline is not None or self._baseline_fisher is not None)
            and (fisher_current is not None or self._current_fisher is not None)
        )
        if curvature_available:
            curvature = self.evaluate_refusal_curvature(
                fisher_baseline if fisher_baseline is not None else self._baseline_fisher,
                fisher_current if fisher_current is not None else self._current_fisher,
            )
            latent_drift_score = curvature
            details["fisher_curvature"] = curvature
            details["fisher_divergence"] = dict(self._last_curvature_details)
            if max_allowed_curvature is not None and curvature > float(max_allowed_curvature):
                failures.append(
                    f"fisher_curvature {curvature:.4f} exceeds max_allowed {max_allowed_curvature:.4f}"
                )

        if probe_batches is not None and self._baseline_pooled is not None:
            rep = self.compute_representation_drift(probe_batches)
            cka_drift = {
                "mean_cka": rep["mean_cka"],
                "min_cka": rep["min_cka"],
                "cosine_distance": rep["cosine_distance"],
            }
            details["representation_drift"] = rep
            if latent_drift_score != latent_drift_score:  # NaN check
                latent_drift_score = float(1.0 - rep["mean_cka"])
            if cka_floor is not None and rep["mean_cka"] < float(cka_floor):
                failures.append(
                    f"mean_cka {rep['mean_cka']:.4f} below floor {cka_floor:.4f}"
                )

        if math_batches is not None:
            grad_results = self.gradient_alignment_test(
                math_batches,
                safety_anchor=safety_anchor,
                tau_drift=tau_drift,
                grad_clip=grad_clip,
            )
            finite_cos = [r["cosine"] for r in grad_results if r["cosine"] == r["cosine"]]
            rejected = [r for r in grad_results if not r["accepted"]]
            details["gradient_alignment"] = {
                "n_candidates": len(grad_results),
                "n_rejected": len(rejected),
                "min_cosine": min(finite_cos) if finite_cos else float("nan"),
                "mean_cosine": (
                    sum(finite_cos) / len(finite_cos) if finite_cos else float("nan")
                ),
                "rejection_reasons": [r["reason"] for r in rejected[:16]],
            }
            if grad_results and finite_cos and min(finite_cos) < (
                self._tau_drift if tau_drift is None else float(tau_drift)
            ):
                failures.append(
                    f"gradient_alignment: {len(rejected)}/{len(grad_results)} candidates below tau"
                )

        if (
            latent_drift_score != latent_drift_score
            and not cka_drift
            and not details
        ):
            failures.append("insufficient_inputs: no check produced a drift score")

        return SubliminalVerdict(
            gate_passed=len(failures) == 0,
            latent_drift_score=latent_drift_score,
            cka_drift=cka_drift,
            rejection_reason="; ".join(failures) if failures else "ok",
            details=details,
        )

    def remove_hooks(self) -> None:
        self._recorder.remove()

    def __enter__(self) -> "SubliminalFilter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove_hooks()
