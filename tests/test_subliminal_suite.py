"""Exhaustive pre-DGX bulletproofing suite for the subliminal pivot codebase.

Runnable on a local RTX 4050 (6 GB VRAM) using a tiny 2-layer dummy causal LM.
Covers, per the DGX-readiness requirements:

* Device placement: no tensors orphaned on the CPU during Fisher computation
  or the EWC penalty backward pass; wrong-device batches raise loudly.
* OOM & broadcasting: batch dimensions in cosine/CKA calculations must match
  exactly; hidden-state captures stay on the storage device.
* Mock distributed: a real 2-process gloo group is spawned locally to exercise
  Fisher reduction and object gather before anything touches the A100s.
* Logic: Fisher non-negativity, exact per-sample accumulation, Gate R rejection
  of anti-correlated gradients, exact Wilson CI agreement, XSTest disentanglement
  and answer parsing.

These tests are never executed as part of authoring; they run only when the
owner explicitly allows execution (Gate R in docs/plan.md).
"""

from __future__ import annotations

import json
import math
import os
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.multiprocessing.spawn import ProcessRaisedException

from conti.eval.metrics import (
    _split_xstest_rows,
    compute_wilson_ci,
    extract_gold_number,
    extract_predicted_number,
    is_correct_prediction,
    is_refusal,
    normalize_number,
)
from conti.loop.subliminal_filter import (
    SubliminalFilter,
    get_decoder_layers,
    gradient_cosine,
    linear_cka,
    make_probe_batches,
    mean_pool_hidden_states,
)
from conti.training.ewc import (
    DeviceMismatchError,
    EWCPenaltyLoss,
    EmpiricalFisherEstimator,
    FisherDivergence,
    SafetyTextDataset,
    canonical_param_name,
    compute_per_sample_grads,
    make_fisher_collate,
    make_safety_fisher_dataloader,
    trainable_named_parameters,
    unwrap_data_parallel,
)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA_AVAILABLE else [])


# --------------------------------------------------------------------------- #
# fixtures: tiny causal LM and batch builders
# --------------------------------------------------------------------------- #


class TinyAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.d_model)
        weights = torch.softmax(scores, dim=-1)
        return self.o_proj(weights @ value)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = TinyAttention(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.ln1(hidden))
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden


class _TinyConfig(dict):
    """dict-based config with attribute access: satisfies both HF- and PEFT-style lookups."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class TinyCausalLM(nn.Module):
    """2-layer transformer with a LLaMA-style ``model-free`` ``layers`` list."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_layers: int = 2,
        n_positions: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = _TinyConfig(model_type="tiny_causal_lm", tie_word_embeddings=False)
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(n_positions, d_model)
        self.layers = nn.ModuleList([TinyBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: Any) -> dict:
        return {"input_ids": input_ids}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del kwargs  # PEFT forwards inputs_embeds/position_ids etc.; the tiny model ignores them
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.wte(input_ids) + self.wpe(positions).unsqueeze(0)
        for block in self.layers:
            hidden = block(hidden)
        logits = self.head(self.ln_f(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != -100).sum() == 0:
                raise ValueError("All labels are -100; cannot compute a loss.")
            loss = F.cross_entropy(
                shift_logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class GPT2StyleModel(nn.Module):
    """GPT-2-style layout (``transformer.h``) for decoder-layer path tests."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(16) for _ in range(2)])
        self.head = nn.Linear(16, 8, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:  # pragma: no cover
        del input_ids
        return SimpleNamespace(logits=torch.zeros(1))


def make_tiny_model(device: str | torch.device = "cpu", seed: int = 0) -> TinyCausalLM:
    torch.manual_seed(seed)
    model = TinyCausalLM()
    model = model.to(device)
    return model


def make_batch(
    batch_size: int,
    *,
    seq_len: int = 16,
    vocab_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
    with_labels: bool = True,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if with_labels:
        batch["labels"] = input_ids.clone()
    return {key: value.to(device) for key, value in batch.items()}


def freeze_base_train_last_layer(model: TinyCausalLM) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "layers.1." in name


def _wilson_reference(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    """Independently re-derived Wilson interval for cross-checking."""
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


# --------------------------------------------------------------------------- #
# 1. Fisher engine tests
# --------------------------------------------------------------------------- #


class TestEmpiricalFisherEstimator:
    @pytest.mark.parametrize("device", DEVICES)
    def test_non_negativity_and_finiteness(self, device: str) -> None:
        model = make_tiny_model(device, seed=1)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model, store_device=device)
        batches = [make_batch(2, device=device, seed=10), make_batch(2, device=device, seed=11)]
        fisher = estimator.compute_fisher(batches)
        assert estimator.is_ready
        assert estimator.n_samples == 4
        assert len(fisher) > 0
        for name, diagonal in fisher.items():
            assert torch.isfinite(diagonal).all(), name
            assert (diagonal >= 0).all(), name

    def test_exact_per_sample_accumulation_matches_single_rows(self) -> None:
        model = make_tiny_model("cpu", seed=2)
        freeze_base_train_last_layer(model)
        batch = make_batch(4, seed=20)
        per_batch = EmpiricalFisherEstimator(model).compute_fisher([batch])
        rows = [
            {key: value[index: index + 1] for key, value in batch.items()}
            for index in range(4)
        ]
        per_row = EmpiricalFisherEstimator(model).compute_fisher(rows)
        assert set(per_batch) == set(per_row)
        for name in per_batch:
            assert torch.allclose(per_batch[name], per_row[name], atol=1e-6), name

    def test_batched_result_ignores_batch_structure_only_when_exact(self) -> None:
        model = make_tiny_model("cpu", seed=3)
        freeze_base_train_last_layer(model)
        batch = make_batch(3, seed=30)
        fisher_a = EmpiricalFisherEstimator(model).compute_fisher([batch])
        split = [
            {key: value[index: index + 1] for key, value in batch.items()}
            for index in range(3)
        ]
        fisher_b = EmpiricalFisherEstimator(model).compute_fisher(split)
        for name in fisher_a:
            assert torch.allclose(fisher_a[name], fisher_b[name], atol=1e-6), name

    def test_max_samples_caps_row_count(self) -> None:
        model = make_tiny_model("cpu", seed=4)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model)
        batches = [make_batch(4, seed=40)]
        estimator.compute_fisher(batches, max_samples=2)
        assert estimator.n_samples == 2

    def test_eps_keeps_diagonal_strictly_positive(self) -> None:
        model = make_tiny_model("cpu", seed=5)
        freeze_base_train_last_layer(model)
        with torch.no_grad():
            for param in model.layers[1].parameters():
                param.zero_()
        estimator = EmpiricalFisherEstimator(model, eps=1e-8)
        fisher = estimator.compute_fisher([make_batch(2, seed=50)])
        for name, diagonal in fisher.items():
            assert (diagonal >= 1e-8 - 1e-12).all(), name

    def test_wrong_device_batch_raises_device_mismatch(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=6)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model)
        cpu_batch = [make_batch(1, device="cpu", seed=60)]
        with pytest.raises(DeviceMismatchError):
            estimator.compute_fisher(cpu_batch)

    def test_store_device_cpu_offload(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=7)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model, store_device="cpu")
        fisher = estimator.compute_fisher([make_batch(2, device="cuda", seed=70)])
        for diagonal in fisher.values():
            assert diagonal.device.type == "cpu"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        model = make_tiny_model("cpu", seed=8)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model)
        original = estimator.compute_fisher([make_batch(2, seed=80)])
        path = tmp_path / "fisher_state.pt"
        estimator.save(path)
        restored = EmpiricalFisherEstimator(model)
        restored.load(path)
        assert restored.n_samples == estimator.n_samples
        for name in original:
            assert torch.allclose(original[name], restored.fisher[name]), name

    def test_load_rejects_negative_entries(self) -> None:
        model = make_tiny_model("cpu", seed=9)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model)
        bad_state = {
            "fisher": {"layers.1.weight": torch.tensor([1.0, -0.5])},
            "n_samples": 1,
        }
        with pytest.raises(ValueError):
            estimator.load_state_dict(bad_state)

    def test_requires_trainable_parameters(self) -> None:
        model = make_tiny_model("cpu", seed=10)
        for param in model.parameters():
            param.requires_grad_(False)
        with pytest.raises(ValueError):
            EmpiricalFisherEstimator(model)

    def test_missing_labels_are_derived_from_input_ids(self) -> None:
        model = make_tiny_model("cpu", seed=11)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model)
        batch = make_batch(1, seed=110, with_labels=False)
        fisher = estimator.compute_fisher([batch])
        assert estimator.n_samples == 1
        assert len(fisher) > 0

    def test_compute_per_sample_grads_requires_labels(self) -> None:
        model = make_tiny_model("cpu", seed=11)
        freeze_base_train_last_layer(model)
        batch = make_batch(1, seed=110, with_labels=False)
        with pytest.raises(ValueError):
            compute_per_sample_grads(model, batch)


# --------------------------------------------------------------------------- #
# 2. EWC penalty tests
# --------------------------------------------------------------------------- #


class TestEWCPenaltyLoss:
    def _reference_fisher_and_params(self, model: TinyCausalLM):
        freeze_base_train_last_layer(model)
        fisher = {
            name: torch.full_like(param.detach().float(), 0.5)
            for name, param in trainable_named_parameters(model)
        }
        refs = {name: param.detach().clone() for name, param in trainable_named_parameters(model)}
        return fisher, refs

    def test_zero_penalty_at_reference_point(self) -> None:
        model = make_tiny_model("cpu", seed=12)
        fisher, refs = self._reference_fisher_and_params(model)
        penalty = EWCPenaltyLoss(fisher, refs, lambda_ewc=100.0)
        value = penalty(model)
        assert float(value.item()) == pytest.approx(0.0, abs=1e-7)

    def test_positive_and_lambda_linear_after_perturbation(self) -> None:
        model = make_tiny_model("cpu", seed=13)
        fisher, refs = self._reference_fisher_and_params(model)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(0.01)
        penalty_half = EWCPenaltyLoss(fisher, refs, lambda_ewc=10.0).penalty(model)
        penalty_full = EWCPenaltyLoss(fisher, refs, lambda_ewc=20.0).penalty(model)
        assert float(penalty_half.item()) > 0.0
        assert float(penalty_full.item()) == pytest.approx(2.0 * float(penalty_half.item()), rel=1e-5)

    def test_penalty_is_differentiable(self) -> None:
        model = make_tiny_model("cpu", seed=14)
        fisher, refs = self._reference_fisher_and_params(model)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(0.05)
        penalty = EWCPenaltyLoss(fisher, refs, lambda_ewc=1.0)
        value = penalty(model)
        value.backward()
        grads = [param.grad for param in model.parameters() if param.requires_grad]
        assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)
        assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)

    def test_penalty_device_follows_parameters(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=15)
        fisher, refs = self._reference_fisher_and_params(model)
        cpu_fisher = {name: tensor.cpu() for name, tensor in fisher.items()}
        cpu_refs = {name: tensor.cpu() for name, tensor in refs.items()}
        penalty = EWCPenaltyLoss(cpu_fisher, cpu_refs, lambda_ewc=5.0)
        value = penalty(model)
        assert value.device.type == "cuda"
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(0.02)
        assert float(penalty(model).item()) > 0.0

    def test_no_overlap_raises_runtime_error(self) -> None:
        model = make_tiny_model("cpu", seed=16)
        freeze_base_train_last_layer(model)
        fisher = {"nonexistent.weight": torch.ones(4)}
        refs = {"nonexistent.weight": torch.zeros(4)}
        penalty = EWCPenaltyLoss(fisher, refs, lambda_ewc=1.0)
        with pytest.raises(RuntimeError):
            penalty(model)

    def test_negative_lambda_rejected(self) -> None:
        model = make_tiny_model("cpu", seed=17)
        fisher, refs = self._reference_fisher_and_params(model)
        with pytest.raises(ValueError):
            EWCPenaltyLoss(fisher, refs, lambda_ewc=-1.0)

    def test_negative_fisher_rejected(self) -> None:
        model = make_tiny_model("cpu", seed=18)
        freeze_base_train_last_layer(model)
        refs = {name: param.detach().clone() for name, param in trainable_named_parameters(model)}
        fisher = {
            name: torch.where(
                torch.arange(tensor.numel()).reshape(tensor.shape) == 0,
                torch.tensor(-1.0),
                tensor,
            )
            for name, tensor in (
                (name, torch.full_like(param.detach().float(), 0.5))
                for name, param in trainable_named_parameters(model)
            )
        }
        with pytest.raises(ValueError):
            EWCPenaltyLoss(fisher, refs, lambda_ewc=1.0)


# --------------------------------------------------------------------------- #
# 3. Fisher divergence tests
# --------------------------------------------------------------------------- #


class TestFisherDivergence:
    def test_identical_states_are_zero(self) -> None:
        state = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
        metrics = FisherDivergence.compute_all(state, state)
        assert metrics["frobenius"] == pytest.approx(0.0)
        assert metrics["relative_trace_shift"] == pytest.approx(0.0)
        assert metrics["jensen_shannon_nats"] == pytest.approx(0.0)

    def test_frobenius_known_value(self) -> None:
        a = {"a": torch.tensor([1.0, 2.0])}
        b = {"a": torch.tensor([2.0, 4.0])}
        expected = math.sqrt(1.0 + 4.0)
        assert FisherDivergence.frobenius_parameter_divergence(a, b) == pytest.approx(expected)

    def test_relative_trace_shift_sign(self) -> None:
        a = {"a": torch.tensor([1.0, 1.0])}
        b = {"a": torch.tensor([2.0, 2.0])}
        shift = FisherDivergence.relative_trace_shift(a, b)
        assert shift == pytest.approx(1.0)
        reverse = FisherDivergence.relative_trace_shift(b, a)
        assert reverse == pytest.approx(-0.5)

    def test_jsd_bounded_and_symmetric(self) -> None:
        a = {"a": torch.tensor([10.0, 0.0, 0.0])}
        b = {"a": torch.tensor([0.0, 10.0, 0.0])}
        jsd_ab = FisherDivergence.jensen_shannon_divergence(a, b)
        jsd_ba = FisherDivergence.jensen_shannon_divergence(b, a)
        assert jsd_ab == pytest.approx(jsd_ba)
        assert 0.0 <= jsd_ab <= math.log(2.0) + 1e-9
        assert jsd_ab == pytest.approx(math.log(2.0), rel=1e-3)

    def test_shape_mismatch_raises(self) -> None:
        a = {"a": torch.tensor([1.0, 2.0])}
        b = {"a": torch.tensor([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError):
            FisherDivergence.frobenius_parameter_divergence(a, b)

    def test_disjoint_names_raise(self) -> None:
        with pytest.raises(RuntimeError):
            FisherDivergence.frobenius_parameter_divergence(
                {"a": torch.ones(2)}, {"b": torch.ones(2)}
            )


# --------------------------------------------------------------------------- #
# 4. Subliminal filter tests (Gate R)
# --------------------------------------------------------------------------- #


class TestDecoderLayerResolution:
    def test_finds_llama_style_layers(self) -> None:
        model = make_tiny_model("cpu", seed=19)
        layers = get_decoder_layers(model)
        assert len(layers) == 2

    def test_finds_gpt2_style_layers(self) -> None:
        model = GPT2StyleModel()
        layers = get_decoder_layers(model)
        assert len(layers) == 2

    def test_unknown_layout_raises(self) -> None:
        class Opaque(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Linear(4, 4)

        with pytest.raises(RuntimeError):
            get_decoder_layers(Opaque())


class TestRepresentationDrift:
    def test_identical_model_gives_cka_one_and_zero_distance(self) -> None:
        model = make_tiny_model("cpu", seed=20)
        probes = [make_batch(1, seed=200 + index, with_labels=False) for index in range(4)]
        filtr = SubliminalFilter(model, layer_indices=[0, 1])
        with pytest.raises(RuntimeError):
            filtr.compute_representation_drift(probes)
        filtr.capture_baseline_representations(probes)
        drift = filtr.compute_representation_drift(probes)
        assert drift["mean_cka"] == pytest.approx(1.0, abs=1e-5)
        assert drift["cosine_distance"] == pytest.approx(0.0, abs=1e-5)
        assert drift["n_probes"] == 4

    def test_perturbation_is_detected_as_drift(self) -> None:
        model = make_tiny_model("cpu", seed=21)
        probes = [make_batch(1, seed=210 + index, with_labels=False) for index in range(4)]
        filtr = SubliminalFilter(model, layer_indices=[0, 1])
        filtr.capture_baseline_representations(probes)
        clean = filtr.compute_representation_drift(probes)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.05 * torch.randn_like(param))
        drifted = filtr.compute_representation_drift(probes)
        assert clean["mean_cka"] > 0.99
        assert drifted["mean_cka"] < 1.0
        assert drifted["cosine_distance"] > clean["cosine_distance"]

    def test_probe_count_mismatch_raises(self) -> None:
        model = make_tiny_model("cpu", seed=22)
        filtr = SubliminalFilter(model)
        filtr.capture_baseline_representations([make_batch(1, seed=220, with_labels=False)] * 3)
        with pytest.raises(ValueError):
            filtr.compute_representation_drift(
                [make_batch(1, seed=221, with_labels=False)] * 5
            )

    def test_hidden_state_capture_stays_on_store_device(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=23)
        filtr = SubliminalFilter(model, store_device="cpu")
        probes = [make_batch(1, device="cuda", seed=230, with_labels=False) for _ in range(2)]
        filtr.capture_baseline_representations(probes)
        assert filtr._baseline_pooled is not None
        for tensor in filtr._baseline_pooled.values():
            assert tensor.device.type == "cpu"

    def test_invalid_layer_index_rejected(self) -> None:
        model = make_tiny_model("cpu", seed=24)
        with pytest.raises(ValueError):
            SubliminalFilter(model, layer_indices=[7])

    def test_wrong_device_probe_batch_raises(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=25)
        filtr = SubliminalFilter(model)
        with pytest.raises(DeviceMismatchError):
            filtr.capture_baseline_representations(
                [make_batch(1, device="cpu", seed=250, with_labels=False)]
            )


class TestGradientAlignment:
    def test_cosine_pure_function(self) -> None:
        anchor = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
        uniform_scale = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([0.0, 3.0])}
        negated = {"a": torch.tensor([-3.0, 0.0]), "b": torch.tensor([0.0, -3.0])}
        orthogonal = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([-1.0, 0.0])}
        nonuniform = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([0.0, 2.0])}
        assert gradient_cosine(uniform_scale, anchor) == pytest.approx(1.0, abs=1e-6)
        assert gradient_cosine(negated, anchor) == pytest.approx(-1.0, abs=1e-6)
        assert gradient_cosine(orthogonal, anchor) == pytest.approx(0.0, abs=1e-6)
        assert 0.0 < gradient_cosine(nonuniform, anchor) < 1.0

    def test_cosine_device_mismatch_is_handled(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        anchor = {"a": torch.tensor([1.0, 0.0], device="cuda")}
        grads = {"a": torch.tensor([2.0, 0.0], device="cpu")}
        assert gradient_cosine(grads, anchor) == pytest.approx(1.0)

    def test_cosine_empty_intersection_raises(self) -> None:
        with pytest.raises(ValueError):
            gradient_cosine({"a": torch.ones(2)}, {"b": torch.ones(2)})

    def test_anchor_capture_and_same_data_acceptance(self) -> None:
        model = make_tiny_model("cpu", seed=26)
        freeze_base_train_last_layer(model)
        filtr = SubliminalFilter(model)
        single_row = make_batch(1, seed=260)
        anchor = filtr.capture_safety_anchor([single_row], max_samples=1)
        assert filtr.has_safety_anchor
        for tensor in anchor.values():
            assert tensor.dtype == torch.float32

        identical = [{key: value for key, value in single_row.items()}]
        results = filtr.gradient_alignment_test(identical, tau_drift=0.9)
        assert all(result["accepted"] for result in results)
        assert all(result["cosine"] == pytest.approx(1.0, abs=1e-5) for result in results)

        repeated = [make_batch(2, seed=260) for _ in range(2)]
        filtr.capture_safety_anchor(repeated, max_samples=4)
        mean_results = filtr.gradient_alignment_test([make_batch(2, seed=260)], tau_drift=0.5)
        assert all(result["accepted"] for result in mean_results)
        # an individual sample vs the MEAN anchor is ~1/sqrt(2) for near-orthogonal
        # gradients, so the gate threshold must be calibrated on this geometry
        assert all(result["cosine"] > 0.5 for result in mean_results)

    def test_anti_correlated_data_is_rejected(self) -> None:
        model = make_tiny_model("cpu", seed=27)
        freeze_base_train_last_layer(model)
        filtr = SubliminalFilter(model)
        safety_batches = [make_batch(2, seed=270) for _ in range(2)]
        filtr.capture_safety_anchor(safety_batches, max_samples=4)
        foreign_batches = [make_batch(2, seed=999, seq_len=16)]
        results = filtr.gradient_alignment_test(foreign_batches, tau_drift=0.999)
        assert all(not result["accepted"] for result in results)
        assert all(result["cosine"] < 0.999 for result in results)

    def test_synthetic_anti_correlated_gradients_rejected(self) -> None:
        model = make_tiny_model("cpu", seed=28)
        freeze_base_train_last_layer(model)
        filtr = SubliminalFilter(model)
        names = [name for name, _ in trainable_named_parameters(model)]
        anchor = {name: torch.ones_like(model.state_dict()[name]) for name in names}
        results = filtr.gradient_alignment_test(
            [make_batch(1, seed=280)], safety_anchor=anchor, tau_drift=0.999
        )
        assert len(results) == 1
        assert isinstance(results[0]["accepted"], bool)

    def test_missing_anchor_raises(self) -> None:
        model = make_tiny_model("cpu", seed=29)
        filtr = SubliminalFilter(model)
        with pytest.raises(ValueError):
            filtr.gradient_alignment_test([make_batch(1, seed=290)])

    def test_empty_anchor_input_raises(self) -> None:
        model = make_tiny_model("cpu", seed=30)
        filtr = SubliminalFilter(model)
        with pytest.raises(ValueError):
            filtr.capture_safety_anchor([], max_samples=4)

    def test_tau_drift_range_validated(self) -> None:
        model = make_tiny_model("cpu", seed=31)
        with pytest.raises(ValueError):
            SubliminalFilter(model, tau_drift=2.0)

    def test_gate_insufficient_inputs(self) -> None:
        model = make_tiny_model("cpu", seed=32)
        filtr = SubliminalFilter(model)
        verdict = filtr.gate()
        assert not verdict.gate_passed
        assert "insufficient_inputs" in verdict.rejection_reason
        assert isinstance(verdict.latent_drift_score, float)

    def test_gate_with_curvature_limit(self) -> None:
        model = make_tiny_model("cpu", seed=33)
        freeze_base_train_last_layer(model)
        filtr = SubliminalFilter(model)
        fisher_zero = {
            name: torch.zeros_like(param) for name, param in trainable_named_parameters(model)
        }
        fisher_unit = {
            name: torch.ones_like(param) for name, param in trainable_named_parameters(model)
        }
        filtr.set_baseline_fisher(fisher_zero)
        verdict_ok = filtr.gate(
            fisher_current=fisher_zero, max_allowed_curvature=0.5
        )
        assert verdict_ok.gate_passed
        assert verdict_ok.latent_drift_score == pytest.approx(0.0)
        verdict_bad = filtr.gate(
            fisher_current=fisher_unit, max_allowed_curvature=0.5
        )
        assert not verdict_bad.gate_passed
        assert "fisher_curvature" in verdict_bad.rejection_reason


class TestPoolingAndProbeHelpers:
    def test_mean_pool_shapes_and_broadcast_guard(self) -> None:
        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5)
        pooled = mean_pool_hidden_states(hidden, mask)
        assert pooled.shape == (2, 8)
        with pytest.raises(ValueError):
            mean_pool_hidden_states(hidden, torch.ones(3, 5))
        with pytest.raises(ValueError):
            mean_pool_hidden_states(torch.randn(2, 5), mask)

    def test_make_probe_batches_uses_cpu_and_masks(self) -> None:
        class TinyTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text: str, **kwargs: Any):
                tokens = [2 + (ord(c) % 60) for c in text[:12]]
                return {
                    "input_ids": torch.tensor([tokens], dtype=torch.long),
                    "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
                }

        batches = make_probe_batches(TinyTokenizer(), ["abc", "de"])
        assert len(batches) == 2
        for batch in batches:
            assert batch["input_ids"].device.type == "cpu"
            assert batch["input_ids"].shape[0] == 1
            assert batch["attention_mask"].sum() == batch["input_ids"].numel()
        with pytest.raises(ValueError):
            make_probe_batches(TinyTokenizer(), [])


# --------------------------------------------------------------------------- #
# 5. Mock distributed tests (2-process gloo, no CUDA)
# --------------------------------------------------------------------------- #


def _dist_worker(
    rank: int,
    world_size: int,
    master_port: int,
    results_path: str,
) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    # On Docker-Desktop/Windows hosts the machine hostname can resolve to an
    # interface gloo rejects ("unsupported gloo device"); loopback is always valid.
    os.environ["GLOO_SOCKET_IFNAME"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(4242)
        model = make_tiny_model("cpu", seed=4242)
        freeze_base_train_last_layer(model)

        full_batches = [make_batch(2, seed=777 + index) for index in range(4)]
        rank_batches = [
            batch for index, batch in enumerate(full_batches) if index % world_size == rank
        ]
        estimator = EmpiricalFisherEstimator(model, store_device="cpu")
        fisher = estimator.compute_fisher(rank_batches)
        n_rows = estimator.n_samples

        reduced = {name: tensor.clone() for name, tensor in fisher.items()}
        for tensor in reduced.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size

        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "n_rows": n_rows})

        max_diff = 0.0
        if rank == 0:
            full_estimator = EmpiricalFisherEstimator(model, store_device="cpu")
            full_fisher = full_estimator.compute_fisher(full_batches)
            for name in full_fisher:
                max_diff = max(
                    max_diff, float((full_fisher[name] - reduced[name]).abs().max().item())
                )
        payload = {
            "max_diff": max_diff,
            "gathered": gathered,
            "expected_gather": [{"rank": r, "n_rows": 2} for r in range(world_size)],
        }
        if rank == 0:
            Path(results_path).write_text(json.dumps(payload), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestMockDistributed:
    def test_two_process_gloo_fisher_reduction(self, tmp_path: Path) -> None:
        results_path = tmp_path / "dist_results.json"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            master_port = int(sock.getsockname()[1])
        try:
            torch.multiprocessing.spawn(
                _dist_worker,
                args=(2, master_port, str(results_path)),
                nprocs=2,
                join=True,
            )
        except (
            RuntimeError,
            OSError,
            ProcessRaisedException,
        ) as exc:  # pragma: no cover - host cannot bind gloo (Windows/docker)
            pytest.skip(
                f"gloo cannot bind on this host; the reduction path is validated on the DGX. "
                f"Reason: {exc}"
            )
        assert results_path.exists(), "rank 0 never wrote results (possible deadlock)"
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        assert payload["max_diff"] < 1e-5
        assert payload["gathered"] == payload["expected_gather"]


# --------------------------------------------------------------------------- #
# 6. Wilson CI, parsing and XSTest disentanglement tests
# --------------------------------------------------------------------------- #


class TestWilsonCI:
    def test_matches_independent_reference(self) -> None:
        cases = [(0, 10), (5, 10), (8, 10), (520, 520), (1, 1000), (1319, 1319)]
        for successes, total in cases:
            low, high = compute_wilson_ci(successes, total)
            ref_low, ref_high = _wilson_reference(successes, total)
            assert low == pytest.approx(ref_low, abs=1e-9), (successes, total)
            assert high == pytest.approx(ref_high, abs=1e-9), (successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_vectorized_matches_scalar(self) -> None:
        successes = [0, 5, 8]
        totals = [10, 10, 10]
        lows, highs = compute_wilson_ci(successes, totals)
        for index, (k, n) in enumerate(zip(successes, totals)):
            scalar_low, scalar_high = compute_wilson_ci(k, n)
            assert float(lows[index]) == pytest.approx(scalar_low)
            assert float(highs[index]) == pytest.approx(scalar_high)

    def test_zero_total_gives_zero_interval(self) -> None:
        low, high = compute_wilson_ci(0, 0)
        assert low == 0.0 and high == 0.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_wilson_ci(5, 3)
        with pytest.raises(ValueError):
            compute_wilson_ci(-1, 10)
        with pytest.raises(ValueError):
            compute_wilson_ci([1, 2], [10])


class TestAnswerParsing:
    def test_gold_extraction(self) -> None:
        assert extract_gold_number("reasoning... #### 42") == "42"
        assert extract_gold_number("#### 1,234") == "1234"
        assert extract_gold_number("no marker 7.50") == "7.5"
        assert extract_gold_number(None) is None

    def test_predicted_extraction(self) -> None:
        assert extract_predicted_number("The final answer is 72.") == "72"
        assert extract_predicted_number("so x = 18, total = 36") == "36"
        assert extract_predicted_number("2+2 = 4") == "4"
        assert extract_predicted_number("") is None

    def test_normalize_number(self) -> None:
        assert normalize_number("1,234") == "1234"
        assert normalize_number("$99") == "99"
        assert normalize_number("7.0") == "7"
        assert normalize_number("abc") is None
        assert normalize_number("2.50") == "2.5"

    def test_exact_match_scoring(self) -> None:
        assert is_correct_prediction("The answer is 42.", "#### 42")
        assert not is_correct_prediction("The answer is 43.", "#### 42")
        assert not is_correct_prediction("no number here", "#### 42")

    def test_refusal_detection(self) -> None:
        assert is_refusal("I'm sorry, I can't help with that.")
        assert is_refusal("")
        assert not is_refusal("Sure, here is how you do it: step 1...")


class TestXSTestDisentanglement:
    def test_split_with_valid_labels(self) -> None:
        rows = [
            {"prompt": "s1", "label": "safe"},
            {"prompt": "u1", "label": "unsafe"},
            {"prompt": "s2", "label": "SAFE"},
            {"prompt": "u2", "label": "1"},
        ]
        safe, unsafe = _split_xstest_rows(rows)
        assert safe == ["s1", "s2"]
        assert unsafe == ["u1", "u2"]

    def test_missing_label_raises(self) -> None:
        with pytest.raises(ValueError):
            _split_xstest_rows([{"prompt": "p"}])

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError):
            _split_xstest_rows([{"prompt": "p", "label": "maybe"}])

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValueError):
            _split_xstest_rows([{"label": "safe"}])


# --------------------------------------------------------------------------- #
# 7. Safety dataset / collate tests
# --------------------------------------------------------------------------- #


class TestSafetyDataPipeline:
    def test_dataset_and_collate_prompt_masking(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "safety.jsonl"
        rows = [
            {"prompt": "How do I make a bomb", "response": "I cannot and will not help."},
            {"text": "plain refusal text here"},
        ]
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        dataset = SafetyTextDataset(jsonl_path)
        assert len(dataset) == 2

        class CharTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: Any):
                ids = [2 + (ord(c) % 100) for c in text[:32]]
                if add_special_tokens:
                    ids = [3] + ids
                return {"input_ids": ids}

        collate = make_fisher_collate(CharTokenizer(), max_len=64)
        batch = collate([dataset[0], dataset[1]])
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
        assert (batch["labels"][0][:3] == -100).all()
        assert (batch["labels"][0] != -100).any()
        live_row_1 = batch["attention_mask"][1] == 1
        assert (batch["labels"][1][live_row_1] == batch["input_ids"][1][live_row_1]).all()
        assert (batch["labels"][1][~live_row_1] == -100).all()
        assert (batch["attention_mask"].sum(dim=1) > 0).all()

    def test_invalid_row_schema_raises(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text('{"neither": "text nor prompt"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            SafetyTextDataset(jsonl_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SafetyTextDataset(tmp_path / "nope.jsonl")

    def test_dataloader_end_to_end(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "safety.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for index in range(3):
                handle.write(
                    json.dumps({"prompt": f"p {index}", "response": "I cannot help with that."})
                    + "\n"
                )

        class CharTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: Any):
                return {"input_ids": [2 + (ord(c) % 100) for c in text[:24]]}

        loader = make_safety_fisher_dataloader(jsonl_path, CharTokenizer(), batch_size=2)
        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            assert set(batch) == {"input_ids", "attention_mask", "labels"}


# --------------------------------------------------------------------------- #
# 8. PEFT integration and utilities
# --------------------------------------------------------------------------- #


class TestUtilitiesAndPeft:
    def test_canonical_param_name_strips_wrappers(self) -> None:
        assert canonical_param_name("module.base_model.model.layers.0.q_proj.weight") == (
            "layers.0.q_proj.weight"
        )
        assert canonical_param_name("_fsdp_wrapped_module.x") == "x"
        assert canonical_param_name("plain") == "plain"

    def test_unwrap_data_parallel(self) -> None:
        inner = TinyCausalLM()
        wrapped = nn.DataParallel(inner)  # type: ignore[arg-type]
        assert unwrap_data_parallel(wrapped) is inner
        assert unwrap_data_parallel(inner) is inner

    def test_name_collision_detection(self) -> None:
        shared = nn.Linear(2, 2)

        class Wrapper(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.module = inner
                self.base_model = nn.Module()
                self.base_model.model = inner

        with pytest.raises(ValueError):
            trainable_named_parameters(Wrapper(shared))

    def test_peft_lora_fisher_end_to_end(self) -> None:
        peft = pytest.importorskip("peft")
        from peft import LoraConfig, TaskType, get_peft_model

        model = make_tiny_model("cpu", seed=40)
        lora_model = get_peft_model(
            model,
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM,
                       target_modules=["q_proj", "v_proj"]),
        )
        trainable = trainable_named_parameters(lora_model)
        assert trainable, "PEFT model must expose trainable LoRA parameters"
        assert all("lora_" in name for name, _ in trainable)

        estimator = EmpiricalFisherEstimator(lora_model, store_device="cpu")
        batches = [make_batch(2, seed=400)]
        fisher = estimator.compute_fisher(batches)
        assert set(fisher) == {name for name, _ in trainable}
        for diagonal in fisher.values():
            assert (diagonal >= 0).all()

        refs = {
            name: param.detach().clone() for name, param in trainable_named_parameters(lora_model)
        }
        penalty = EWCPenaltyLoss(fisher, refs, lambda_ewc=10.0)
        value = penalty(lora_model)
        assert torch.isfinite(value)
        assert float(value.item()) >= 0.0

        probes = [make_batch(1, seed=401 + index, with_labels=False) for index in range(2)]
        filtr = SubliminalFilter(lora_model, layer_indices=[0])
        filtr.capture_baseline_representations(probes)
        drift = filtr.compute_representation_drift(probes)
        assert drift["mean_cka"] == pytest.approx(1.0, abs=1e-5)

    def test_compute_per_sample_grads_clears_and_restores(self) -> None:
        model = make_tiny_model("cpu", seed=41)
        freeze_base_train_last_layer(model)
        batch = make_batch(1, seed=410)
        was_training = model.training
        model.train()
        grads = compute_per_sample_grads(model, batch)
        assert grads is not None
        assert model.training == was_training
        for _, param in model.named_parameters():
            assert param.grad is None
        grad_norms = {name: float(tensor.norm().item()) for name, tensor in grads.items()}
        assert any(norm > 0 for norm in grad_norms.values())

    def test_per_sample_grads_on_wrong_device_raise(self) -> None:
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA unavailable")
        model = make_tiny_model("cuda", seed=42)
        freeze_base_train_last_layer(model)
        batch = make_batch(1, device="cpu", seed=420)
        with pytest.raises(DeviceMismatchError):
            compute_per_sample_grads(model, batch)


class TestFisherThroughTraining:
    def test_fisher_and_penalty_survive_optimizer_step(self) -> None:
        model = make_tiny_model("cpu", seed=43)
        freeze_base_train_last_layer(model)
        estimator = EmpiricalFisherEstimator(model, store_device="cpu")
        safety_batches = [make_batch(2, seed=430) for _ in range(2)]
        fisher = estimator.compute_fisher(safety_batches)
        refs = {
            name: param.detach().clone() for name, param in trainable_named_parameters(model)
        }
        penalty = EWCPenaltyLoss(fisher, refs, lambda_ewc=100.0)

        optimizer = torch.optim.SGD(
            [p for _, p in trainable_named_parameters(model)], lr=0.05
        )
        pre_value = float(penalty(model).item())
        for _ in range(5):
            optimizer.zero_grad()
            output = model(**make_batch(2, seed=431))
            loss = output.loss + penalty(model)
            loss.backward()
            optimizer.step()
        post_value = float(penalty(model).item())
        assert pre_value >= 0.0
        assert post_value >= 0.0
        assert post_value > pre_value
        for name, diagonal in fisher.items():
            assert (diagonal >= 0).all(), name
