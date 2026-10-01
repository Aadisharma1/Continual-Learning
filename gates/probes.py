"""Linear probes with the methodology that keeps them from being worthless.

The eight-step protocol (in order):

  1. Split by prompt ITEM, never by token. Token-level random splits inflate
     AUC by 0.10-0.30 because tokens from one prompt share a label, a template
     and often most of their content.
  2. Also split by template family and topic family, holding out entire
     families -- otherwise the probe learns "which template", not "harmful".
  3. Three disjoint prompt sets at the ITEM level: P_probe trains, P_steer
     steers, P_test only ever reports. The probe never sees its own training
     prompts.
  4. Tune regularisation inside CV only. Never select on test.
  5. Report CV AUC (mean +/- std), held-out AUC, AND per-category AUC.
     An aggregate AUC can be 0.95 while being inverted on one category.
  6. Null and ceiling controls on every run.
  7. Orthogonalise the probe against nuisance directions ("probe the probe").
  8. Report TRANSFER AUC (probe trained on M_A, evaluated on M_F activations),
     not in-sample prediction.

The mid-layer readout uses per-layer probes rather than the raw logit lens,
because a raw lens read mid-network is systematically miscalibrated (different
norm, different basis, missing later-layer contributions) and produces
confidently-wrong results that look interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "LayerProbe",
    "train_layer_probes",
    "leak_audit",
    "nc3_token_shuffle",
    "permutation_null_auc",
    "logit_diff",
]

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _fit(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    )
    clf.fit(X, y)
    return clf


@dataclass
class LayerProbe:
    """A fitted per-layer probe plus the CV evidence that it is trustworthy."""

    coef: np.ndarray                 # [D]
    intercept: float
    cv_auc_mean: float
    cv_auc_std: float
    cv_auc_per_fold: list[float] = field(default_factory=list)
    best_C: float = 0.03
    n_train: int = 0

    def decision(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.coef) + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision(X)
        return 1.0 / (1.0 + np.exp(-z))

    def auc(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(roc_auc_score(y, self.decision(X)))

    def to_dict(self) -> dict:
        return {
            "cv_auc_mean": self.cv_auc_mean,
            "cv_auc_std": self.cv_auc_std,
            "cv_auc_per_fold": self.cv_auc_per_fold,
            "best_C": self.best_C,
            "n_train": self.n_train,
            "intercept": self.intercept,
            "coef_norm": float(np.linalg.norm(self.coef)),
        }


def train_layer_probes(
    H: torch.Tensor,
    y: np.ndarray,
    layers: Sequence[int],
    groups: np.ndarray | None = None,
    folds: int = 5,
    seed: int = 0,
    tune_C: bool = True,
) -> dict[int, LayerProbe]:
    """H [L, N, D] activations -> per-layer :class:`LayerProbe`.

    ``groups`` should be prompt item ids. When supplied, splits are grouped so
    no item appears in both train and validation (step 1); otherwise stratified
    folds are used, which is only safe for genuinely item-level data.
    """
    if H.ndim != 3:
        raise ValueError(f"expected H [L,N,D], got {tuple(H.shape)}")
    y = np.asarray(y).astype(int)
    out: dict[int, LayerProbe] = {}

    for slot, layer_idx in enumerate(layers):
        X = H[slot].numpy().astype(np.float64)

        if groups is not None:
            groups = np.asarray(groups)
            splitter = GroupKFold(n_splits=folds)
            splits = list(splitter.split(X, y, groups))
            # GroupKFold does not shuffle; seed it through a permutation so the
            # fold assignment is reproducible but not order-dependent.
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(splits))
            splits = [splits[i] for i in perm]
        else:
            splitter = StratifiedKFold(
                folds, shuffle=True, random_state=seed
            )
            splits = list(splitter.split(X, y))

        best = (-1.0, C_GRID[0], [], None)
        candidates = C_GRID if tune_C else (0.03,)
        for C in candidates:
            aucs = []
            for tr, va in splits:
                try:
                    clf = _fit(X[tr], y[tr], C)
                    aucs.append(
                        float(roc_auc_score(y[va], clf.decision_function(X[va])))
                    )
                except ValueError:
                    continue
            if not aucs:
                continue
            m = float(np.mean(aucs))
            if m > best[0]:
                best = (m, C, aucs, None)

        mean_auc, best_C, aucs, _ = best
        clf = _fit(X, y, best_C)
        out[layer_idx] = LayerProbe(
            coef=clf[-1].coef_[0].astype(np.float64),
            intercept=float(clf[-1].intercept_[0]),
            cv_auc_mean=mean_auc,
            cv_auc_std=float(np.std(aucs)),
            cv_auc_per_fold=[float(a) for a in aucs],
            best_C=best_C,
            n_train=int(len(y)),
        )

    return out


# ------------------------------------------------------------- controls
def permutation_null_auc(
    H: torch.Tensor,
    y: np.ndarray,
    layer_slot: int = 0,
    folds: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
) -> dict:
    """Label-permutation null. Expect AUC ~0.50 +/- 0.03.

    Anything above 0.55 means the split leaked, and the real probe's AUC is
    optimistic regardless of how high it reads.
    """
    X = H[layer_slot].numpy().astype(np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_repeats):
        yp = rng.permutation(y)
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        fold_aucs = []
        for tr, va in splitter.split(X, yp):
            try:
                clf = _fit(X[tr], yp[tr], 0.03)
                fold_aucs.append(
                    float(roc_auc_score(yp[va], clf.decision_function(X[va])))
                )
            except ValueError:
                continue
        if fold_aucs:
            aucs.append(float(np.mean(fold_aucs)))
    return {
        "null_aucs": aucs,
        "null_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "verdict": (
            "CLEAN"
            if aucs and float(np.mean(aucs)) <= 0.55
            else "LEAK SUSPECTED"
        ),
    }


def nc3_token_shuffle(tok, prompts: Sequence[str], seed: int = 0) -> list[str]:
    """NC-3: token-shuffled harmful prompts.

    The single most decisive control, and it costs almost nothing. A probe
    trained on harmful-vs-benign prompts that scores HIGH on shuffled versions
    never learned harm; it learned the dataset's lexical surface.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            out.append(p)
            continue
        perm = torch.randperm(len(ids), generator=g)
        shuffled = [ids[int(j)] for j in perm]
        out.append(tok.decode(shuffled, skip_special_tokens=True))
    return out


def _multiclass_auc(clf, X, y) -> float:
    """ROC-AUC for a possibly-multiclass problem, out-of-sample.

    sklearn requires *probabilities* for multiclass OVR, not the raw decision
    function, so this always routes through predict_proba. Degrades to the
    binary path when the problem has two classes.
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    if len(classes) == 2:
        return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    return float(roc_auc_score(
        y, clf.predict_proba(X), multi_class="ovr", average="macro",
    ))


def leak_audit(
    X_layer: np.ndarray,
    item_ids: np.ndarray,
    template_labels: np.ndarray | None = None,
    harm_probe: LayerProbe | None = None,
    seed: int = 0,
) -> dict:
    """Red flags on the features a probe is about to be fitted on.

    (a) item identity: can a single forward pass be attributed back to the
        prompt that produced it? Item ids are held out ROW-wise here, so a
        well-separated held-out AUC means item identity is linearly encoded in
        the representation. The harm probe is then partly an item lookup, and
        any item-homogeneous harm label will be read as signal. Trips at 0.90.
    (b) template: are template families separable from the same features? A
        probe cannot be said to read harm if it is reading the surface form of
        the prompt.
    (c) template coupling: does the harm probe's score track the template
        probe's score? |r| >= 0.7 means it is reading template, not harm.

    Fails CLOSED. An audit that could not run its own checks has established
    nothing, so an unavailable required check yields INCONCLUSIVE, never
    CLEAN. Callers must treat anything other than CLEAN as void.
    """
    flags: dict[str, float | str] = {}
    X_layer = np.asarray(X_layer, dtype=np.float64)
    item_ids = np.asarray(item_ids)
    n = len(X_layer)
    if n < 8 or len(np.unique(item_ids)) < 2:
        return {"VERDICT": "INCONCLUSIVE", "reason": "too few items"}

    # ---- (a) can the activations name the item that produced them? -------
    # Row-level split: the classifier never sees these exact activations, so a
    # high held-out AUC is genuine item-identity encoding, not a fit artefact.
    try:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        split = max(2, int(0.7 * n))
        tr, te = order[:split], order[split:]
        if (len(np.unique(item_ids[tr])) < 2
                or len(np.unique(item_ids[te])) < 2):
            raise ValueError("an item id is absent from one split")
        clf = _fit(X_layer[tr], item_ids[tr], 0.1)
        flags["item_id_auc"] = _multiclass_auc(clf, X_layer[te], item_ids[te])
    except Exception as exc:  # noqa: BLE001 - report, never crash the gate
        flags["item_id_auc"] = float("nan")
        flags["item_id_error"] = str(exc)

    # ---- (b) are template families separable? ---------------------------
    if template_labels is not None:
        template_labels = np.asarray(template_labels)
        try:
            rng = np.random.default_rng(seed)
            order = rng.permutation(n)
            split = max(2, int(0.7 * n))
            tr, te = order[:split], order[split:]
            if (len(np.unique(template_labels[tr])) < 2
                    or len(np.unique(template_labels[te])) < 2):
                raise ValueError("a template family is absent from one split")
            clf = _fit(X_layer[tr], template_labels[tr], 0.1)
            flags["template_auc"] = _multiclass_auc(
                clf, X_layer[te], template_labels[te]
            )
        except Exception as exc:  # noqa: BLE001
            flags["template_auc"] = float("nan")
            flags["template_error"] = str(exc)

        # ---- (c) is the harm probe just reading template? ---------------
        if harm_probe is not None:
            try:
                hp = harm_probe.decision(X_layer)
                clf2 = _fit(X_layer, template_labels, 0.1)
                proba = clf2.predict_proba(X_layer)
                # Reduce the template score to 1-D before correlating: max
                # confidence ("how strongly does this look like SOME family").
                tp = proba.max(axis=1)
                if np.std(hp) < 1e-12 or np.std(tp) < 1e-12:
                    raise ValueError("a score is constant, correlation undefined")
                flags["corr_with_template"] = float(np.corrcoef(hp, tp)[0, 1])
            except Exception as exc:  # noqa: BLE001
                flags["corr_with_template"] = float("nan")
                flags["corr_error"] = str(exc)

    # ---- verdict --------------------------------------------------------
    # Only evidence-backed flags go in red_flags. A check that could not run
    # is an error, not evidence of a leak, and must not masquerade as one.
    red: list[str] = []
    item_auc = flags.get("item_id_auc", float("nan"))
    if np.isfinite(item_auc) and item_auc >= 0.90:
        red.append("item_id_auc>=0.90")
    corr = flags.get("corr_with_template", float("nan"))
    if np.isfinite(corr) and abs(corr) >= 0.70:
        red.append("corr_with_template>=0.7")

    required = ["item_id_auc"]
    if template_labels is not None:
        required.append("template_auc")
    if harm_probe is not None:
        required.append("corr_with_template")
    unavailable = [
        k for k in required if not np.isfinite(flags.get(k, float("nan")))
    ]
    errs = sorted(k for k in flags if k.endswith("_error"))

    flags["red_flags"] = red
    flags["unavailable"] = unavailable
    if red:
        flags["VERDICT"] = "LEAK SUSPECTED"
    elif unavailable or errs:
        flags["VERDICT"] = "INCONCLUSIVE"
        flags["errors"] = errs + unavailable
    else:
        flags["VERDICT"] = "CLEAN"
    return flags


def logit_diff(
    last_logits: torch.Tensor,
    refusal_ids: Sequence[int],
    compliance_ids: Sequence[int],
) -> torch.Tensor:
    """Mean refusal-init logit minus mean compliance-init logit. [B]

    The token sets are MODEL-SPECIFIC (tokenizer-dependent). Build them by
    string once and freeze them in the supplementary; assert non-empty.
    """
    if not len(refusal_ids) or not len(compliance_ids):
        raise ValueError("refusal/compliance token id sets must be non-empty")
    dev = last_logits.device
    rid = torch.as_tensor(list(refusal_ids), device=dev)
    cid = torch.as_tensor(list(compliance_ids), device=dev)
    return last_logits[:, rid].mean(-1) - last_logits[:, cid].mean(-1)
