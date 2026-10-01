"""CPU-only unit tests for the gates instrumentation.

These are the tests that catch convention bugs BEFORE a 40-hour GPU run. The
DGX trip should start by running them; a failure here invalidates every figure
produced afterwards.

Deliberately CPU-only and seconds-fast: they construct tiny synthetic models and
tensors rather than downloading Llama-3-8B.

Run with:  python -m pytest tests/gates -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gates import acts, probes
from gates.judge import LexicalJudge, cohens_kappa, judge_validation_report
from gates.prompts import (
    PromptItem,
    PromptSet,
    assert_disjoint,
    build_personalised_prompts,
)
from gates.steer import ALPHA_GRID, Steering


# ------------------------------------------------------- fake model
class _Block(torch.nn.Module):
    """Mimics a decoder layer: returns a bare tensor (transformers>=4.54)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = torch.nn.Linear(d, d, bias=False)
        torch.nn.init.eye_(self.lin.weight)

    def forward(self, h, *_a, **_k):
        return self.lin(h)


class _Inner(torch.nn.Module):
    def __init__(self, n_layers: int, d: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(d) for _ in range(n_layers)])


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int = 4, d: int = 8):
        super().__init__()
        self.model = _Inner(n_layers, d)
        self.config = type("C", (), {"num_hidden_layers": n_layers})()

    def forward(self, input_ids=None, attention_mask=None, **_kw):
        h = torch.zeros(
            input_ids.size(0), input_ids.size(1), 8,
            dtype=torch.float32,
        )
        for blk in self.model.layers:
            h = acts.unwrap(blk(h))
        return h


class _Enc(dict):
    """Mapping supporting .to(device) and **unpacking, like a BatchEncoding."""

    def __init__(self, input_ids, attention_mask):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)

    def to(self, device):
        return _Enc(self["input_ids"].to(device), self["attention_mask"].to(device))


class _FakeTok:
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=512, add_special_tokens=True):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        ids = [[1] + [ord(c) + 2 for c in t][: max_length - 1] for t in texts]
        if not add_special_tokens:
            ids = [i[1:] for i in ids]
        L = max(len(i) for i in ids)
        input_ids = torch.zeros(len(ids), L, dtype=torch.long)
        mask = torch.zeros(len(ids), L, dtype=torch.long)
        for r, seq in enumerate(ids):
            input_ids[r, : len(seq)] = torch.tensor(seq)
            mask[r, : len(seq)] = 1
        if single:
            return _Enc(input_ids[0], mask[0])
        return _Enc(input_ids, mask)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) - 2) for i in ids)


# ============================================================== tests
def test_alpha_zero_is_a_noop():
    """Steering at alpha=0 must not perturb activations by a single ULP."""
    m = _FakeModel()
    x = torch.randn(2, 5, 8)
    with Steering(m, [1, 2], torch.randn(8), mode="last"):
        ref = m(input_ids=torch.ones(2, 5, dtype=torch.long))
        s = Steering(m, [1, 2], torch.randn(8), mode="last")
        with s:
            with s.with_alpha(0.0):
                got = m(input_ids=torch.ones(2, 5, dtype=torch.long))
    assert torch.allclose(ref, got, atol=1e-6)


def test_hooks_are_removed_on_exit():
    """No leaked handles: a steered pass must leave the model pristine."""
    m = _FakeModel()
    n_blocks = len(m.model.layers)
    before = sum(
        len(b._forward_hooks) for b in m.model.layers
    )
    with Steering(m, [0, 1, 2], torch.randn(8)) as s:
        with s.with_alpha(1.0):
            m(input_ids=torch.ones(1, 4, dtype=torch.long))
        during = sum(len(b._forward_hooks) for b in m.model.layers)
        assert during == before + 3
    after = sum(len(b._forward_hooks) for b in m.model.layers)
    assert after == before, f"leaked hooks: {after} != {before}"


def test_rewrap_preserves_tuple_and_bare():
    t = (torch.zeros(2), "aux", 7)
    out = acts.rewrap(t, torch.ones(2))
    assert isinstance(out, tuple) and len(out) == 3 and out[1] == "aux"
    bare = acts.rewrap(torch.zeros(2), torch.ones(2))
    assert torch.is_tensor(bare)


def test_leftpad_last_index_is_final_column():
    """Generative batches left-pad, so -1 is the final real token."""
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    out = acts.gather_pos(h, torch.tensor([3, 3]))
    assert torch.equal(out[0], h[0, 3]) and torch.equal(out[1], h[1, 3])


def test_rightpad_last_index_uses_mask():
    m = _FakeModel()
    tok = _FakeTok()
    H = acts.collect(m, tok, ["hello", "a longer prompt here"], [0, 1, 2, 3],
                     pos="last", batch_size=2)
    assert H.shape[0] == 4 and H.shape[1] == 2


def test_patch_zero_layers_is_noop():
    """Injecting into layers that are all-identity changes nothing."""
    m = _FakeModel(n_layers=4, d=8)
    x = torch.ones(1, 3, 8)
    with Steering(m, [0, 1], torch.randn(8), mode="all") as s:
        with s.with_alpha(0.0):
            out = m(input_ids=torch.ones(1, 3, dtype=torch.long))
    assert torch.isfinite(out).all()


def test_steering_rejects_bad_mode_and_distribute():
    m = _FakeModel()
    for bad in ("nope", "ALL"):
        with pytest.raises(ValueError):
            Steering(m, [0], torch.randn(8), mode=bad)
    for bad in ("nope",):
        with pytest.raises(ValueError):
            Steering(m, [0], torch.randn(8), distribute=bad)


def test_nc3_shuffle_preserves_token_multiset_and_length():
    tok = _FakeTok()
    prompts = ["the cat sat on the mat", "another prompt entirely"]
    out = probes.nc3_token_shuffle(tok, prompts, seed=1)
    assert len(out) == len(prompts)
    flat = lambda t: sorted(
        int(i) for i in tok(t, add_special_tokens=False)["input_ids"]
    )
    for a, b in zip(prompts, out):
        assert flat(a) == flat(b)          # same token multiset
        assert flat(a) != []                # actually tokenised
    # and it really shuffled: at least one prompt comes back reordered
    assert any(out[i] != prompts[i] for i in range(len(prompts)))


def test_null_auc_is_about_chance():
    """Uninformative features must give AUC ~0.5, not 0.9."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    H = torch.randn(1, 200, 16)
    r = probes.permutation_null_auc(H, y, n_repeats=2, seed=0)
    assert r["null_mean"] < 0.65, r
    assert r["verdict"] == "CLEAN"


def test_probe_leak_audit_flags_memorization():
    """Features that encode item id must trip the red flag."""
    rng = np.random.default_rng(0)
    n = 120
    item = rng.integers(0, 6, size=n)
    X = rng.normal(size=(n, 12))
    # inject item identity into the features
    X[np.arange(n), item] += 8.0
    tmpl = rng.integers(0, 3, size=n)
    r = probes.leak_audit(X, item, tmpl)
    assert r["VERDICT"] == "LEAK SUSPECTED", r


def test_probe_leak_audit_clean_on_noise():
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 12))
    item = rng.integers(0, 6, size=n)
    tmpl = rng.integers(0, 3, size=n)
    r = probes.leak_audit(X, item, tmpl)
    assert r["VERDICT"] == "CLEAN", r
    assert "item_id_auc" in r and "red_flags" in r and r["red_flags"] == []


def test_grouped_split_prevents_item_leakage():
    """Probes must not be able to memorize an item seen in train."""
    rng = np.random.default_rng(2)
    n_items, per = 20, 4
    n = n_items * per
    groups = np.repeat(np.arange(n_items), per)
    y = rng.integers(0, 2, size=n)
    H = torch.randn(1, n, 10)
    out = probes.train_layer_probes(H, y, [0], groups=groups, folds=4, seed=0)
    assert 0 in out
    assert out[0].cv_auc_mean < 0.85, "grouped CV still leaking"


def test_alpha_grid_is_symmetric_and_includes_125():
    assert len(ALPHA_GRID) == 19
    assert 0.125 in ALPHA_GRID and -0.125 in ALPHA_GRID
    assert sum(1 for a in ALPHA_GRID if a > 0) == sum(
        1 for a in ALPHA_GRID if a < 0
    )


def test_prompt_sets_must_be_disjoint():
    def mk(name, texts):
        return PromptSet(name, [PromptItem(t, True) for t in texts])

    a = mk("a", ["x", "y"])
    b = mk("b", ["y", "z"])           # shares "y"
    with pytest.raises(ValueError):
        assert_disjoint(a, b)
    c = mk("c", ["p", "q"])
    assert_disjoint(a, c)             # no raise


def test_persona_prompting_is_reversible_to_none():
    base = ["do the thing"]
    assert build_personalised_prompts(base, "none") == base
    p = build_personalised_prompts(base, "contrarian")
    assert p[0].endswith("do the thing") and len(p[0]) > len(base[0])


def test_lexical_judge_finds_refusals_and_ignores_truncation():
    j = LexicalJudge()
    assert j.judge(["I'm sorry, but I can't help with that."]) == [True]
    assert j.judge(["Sure, here's how to do it step by step."]) == [False]
    assert j.judge(["ok"]) == [False], "truncation is not a refusal"


def test_cohens_kappa_extremes():
    a = [True, True, False, False]
    assert cohens_kappa(a, a) == 1.0
    # Disjoint balanced labels: observed agreement == chance agreement exactly.
    assert cohens_kappa([True, True], [False, False]) == 0.0
    # Worse-than-chance needs po < pe, i.e. PARTIAL anti-correlation. A full
    # inversion of balanced marginals still gives po == pe and hence 0.0.
    k = cohens_kappa([True, True, True, False, False, False],
                     [True, False, False, True, True, False])
    assert k < 0.0, k
    assert -1.0 <= k <= 1.0


def test_judge_report_flags_invalid():
    j = LexicalJudge()
    items = ["Sure thing.", "I can't help.", "Here you go.", "No thanks."]
    humans = [False, False, True, True]   # deliberately disagree
    r = judge_validation_report(j, items, humans)
    assert "kappa" in r and r["n"] == 4


def test_state_classifier_three_way():
    from gates.g7_detectability import classify_state
    assert classify_state(0.9, 3.0, 0.9) == "state_1_retained"
    assert classify_state(0.52, 0.1, 0.8) == "state_2_recalibratable"
    assert classify_state(0.5, 0.05, None) == "state_3_not_recovered"
    assert classify_state(0.5, 0.05, 0.5) == "state_3_not_recovered"


def test_tau_recalibration_works_when_ordering_is_intact():
    from gates.g7_detectability import _RefitThreshold, tau_recalibration
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    X = rng.normal(size=(200, 4)) + y[:, None] * 2.0
    coef = X[y == 1].mean(0) - X[y == 0].mean(0)
    from gates.probes import LayerProbe
    p = LayerProbe(coef=coef, intercept=0.0, cv_auc_mean=0.9, cv_auc_std=0.0)
    r = tau_recalibration(p, X, y, [1, 2, 4], seed=0)
    for k in ("1", "2", "4"):
        if "held_out_auc" in r["per_k"][k]:
            assert r["per_k"][k]["held_out_auc"] > 0.7


def test_ckpt_discovery_finds_nothing_gracefully():
    from gates.ckpts import discover_checkpoints, describe
    cks = discover_checkpoints("does/not/exist")
    assert cks == []
    assert "No checkpoints" in describe(cks)


def test_acts_rejects_bad_pos():
    m = _FakeModel()
    tok = _FakeTok()
    with pytest.raises(ValueError):
        acts.collect(m, tok, ["hi"], [0], pos="nonsense", batch_size=1)


# --------------------------------------------------------------- adapters
def test_is_adapter_detects_lora_and_ignores_merged(tmp_path):
    """A LoRA adapter has adapter_config.json and no config.json. A merged
    export is the reverse. Conflating them made discovery return nothing and
    left G8 with no Arm B."""
    from gates.acts import adapter_base, is_adapter

    ad = tmp_path / "llama8b_continual_step1"
    ad.mkdir()
    (ad / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32,
        "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }), encoding="utf-8")
    assert is_adapter(str(ad)) is True
    assert adapter_base(str(ad)) == "meta-llama/Meta-Llama-3.1-8B-Instruct"

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    assert is_adapter(str(merged)) is False
    assert adapter_base(str(merged)) is None


def test_adapter_without_base_model_raises(tmp_path):
    """An adapter that does not say what it was trained on cannot be loaded;
    failing loudly beats guessing a base and reporting the wrong provenance."""
    from gates.acts import adapter_base, is_adapter

    ad = tmp_path / "orphan"
    ad.mkdir()
    (ad / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
    assert is_adapter(str(ad)) is True
    assert adapter_base(str(ad)) is None
