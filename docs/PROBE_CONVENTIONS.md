# Probe conventions

Load-bearing definitions for every G7/G8 measurement. Ambiguity here has
produced more bad conclusions than any statistical mistake, so each rule is
stated once and is binding on all gate code.

## 1. Layer indexing

For decoder layer `i` in `[0, L-1]` on a Llama-3-8B model (L = 32):

| Name | Code | Meaning |
|---|---|---|
| `resid_in[i]` | `model.model.layers[i]` input | residual after layers `0..i-1` |
| `resid_post[i]` | `model.model.layers[i]` output | post-attention + post-MLP, pre-next-norm |

**Prose is 1-based; code is 0-based.** "Layer 16 of 32" is `resid_post[15]`.
Config files (`--layers 15-22`) use **1-based** prose numbers and are converted
once by `gates/g7_detectability.py::_layer_range`. Never hard-code an index.

Sanity check: `resid_in[i+1] == resid_post[i]` up to dtype precision. If this
fails, the hook fired on the wrong tensor.

## 2. Token position

Refusal is a **first-token-of-response decision**. The residual state that
produces it is the one at the **final prompt token** — not the last token of the
padded sequence, and not the mean over positions.

| Collection mode | Padding | Read index |
|---|---|---|
| Generative evaluation | LEFT-pad | `h[:, -1, :]` |
| Offline collection (ragged) | RIGHT-pad | `h[torch.arange(n), attn_mask.sum(1) - 1]` |

Mixing these up yields a plausible number that is silently the wrong number.
`attn_mask.sum(1) - 1` is correct **only** when every row has ≥1 real token.

## 3. Dose unit

Steering doses are expressed in units `u`, not raw norms, so that two
checkpoints with different residual scales are comparable:

```
u = median over the probe set of ||resid_post[L]||
```

An injection of magnitude `alpha * u` into layer set `S` therefore means
"alpha times the typical residual scale at that depth". Raw-norm doses are not
comparable across checkpoints and make Arm A / Arm B ratios meaningless.

Layer set is shared between arms. Changing the layer set between arms changes
the estimand, not the noise.

## 4. Checkpoint provenance

Every results file records a `provenance` block written by
`gates/acts.py::load`, because all checkpoints in `checkpoints/` are **LoRA
adapters**, not full models:

- `kind`: `full` | `adapter`
- `base_model_name_or_path`: from `adapter_config.json`
- `merged`: whether `merge_and_unload()` was applied before probing

A merged adapter reproduces the exported merged checkpoint exactly, so merging
is the faithful path and is the default. A live `PeftModel` wraps every block
and breaks the `model.model.layers[i]` convention above.

## 5. Probe hygiene

- **Splits are by item id**, never by row. Repeated templates of the same item
  must not straddle train/test (`GroupKFold` by item).
- **Leak audit** (`probes.py::leak_audit`) fits the item-id classifier on a
  *row-level permutation split*, so the held-out AUC measures genuine item-identity
  encoding rather than a fit artifact. Leakage null is **0.50 ± 0.03**.
- **NC-3**: a probe that fires on template surface but not on harm has AUC > 0.65
  from surface alone ⇒ `PROBE IS LEXICAL, RESULTS VOID`.
- A measurement that could not run is **INCONCLUSIVE**, never evidence. A check
  that errors must not be reported as a clean control.
