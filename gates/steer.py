"""Causal intervention: residual-stream steering.

    h_l  <-  h_l + alpha * v     at resid_post[l], for l in layers, on chosen positions

Dose is expressed in units of the median residual norm ``u`` (see
:func:`gates.acts.residual_scale`) so that alphas are comparable across layers.

This module is the decisive instrument of Gate G8: it is what lets us ask
whether a *trained* persona and a *prompted* persona act through the same
readout at different doses, or through different readouts.
"""

from __future__ import annotations

from typing import Sequence

import torch

from gates.acts import blocks, rewrap

__all__ = ["Steering", "ALPHA_GRID", "ALPHA_GRID_NEG"]


# The 19-point dose grid. +/-0.125 exists specifically to locate alpha_crit in
# the sub-threshold regime where nothing is visibly broken yet.
ALPHA_GRID: tuple[float, ...] = (
    -3.0, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, -0.125,
    0.0,
    0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0,
)
ALPHA_GRID_NEG: tuple[float, ...] = tuple(a for a in ALPHA_GRID if a < 0)


class Steering:
    """Additive steering on ``resid_post`` of the given layers.

    mode:
      "all"    every sequence position, including generated ones (the hook is
               live for the whole forward pass)
      "prompt" only real prompt positions (mask taken from attention_mask)
      "last"   only the final prompt token -- DEFAULT for refusal work
      "gen"    generated positions only; the "keep steering off the prompt"
               control that blocks the prompt-repetition failure mode

    distribute:
      "each"   add the full alpha*v at every injected layer
      "split"  add alpha*v/|layers| at each -- DEFAULT, because "each" lets the
               effect scale with |layers| and so confounds layer count with dose

    IMPORTANT: a fresh :class:`Steering` instance must not be reused across
    models; the hook captures ``self.v`` by reference.
    """

    def __init__(
        self,
        model,
        layers: Sequence[int],
        v: torch.Tensor,
        mode: str = "last",
        distribute: str = "split",
        normalize: bool = True,
    ):
        if mode not in ("all", "prompt", "last", "gen"):
            raise ValueError(f"bad mode={mode!r}")
        if distribute not in ("each", "split"):
            raise ValueError(f"bad distribute={distribute!r}")
        self.model = model
        self.layers = list(layers)
        self.v = v
        self.mode = mode
        self.distribute = distribute
        self.normalize = normalize
        self.handles: list = []
        self._alpha = 0.0
        self._mask: torch.Tensor | None = None
        self._gen_start: int | None = None

    # -------------------------------------------------------------- hook
    def _mk(self, _layer_idx: int):
        def fn(_mod, _args, out):
            alpha = self._alpha
            if alpha == 0.0:
                return out

            hs = out[0] if isinstance(out, (tuple, list)) else out
            v = self.v
            if self.normalize:
                v = v / v.norm().clamp(min=1e-6)
            step = alpha / len(self.layers) if self.distribute == "split" else alpha
            d = (step * v).to(hs.dtype).to(hs.device)

            mask = self._mode_mask(hs)
            hs = hs + d * mask if mask is not None else hs + d
            return rewrap(out, hs)

        return fn

    def _mode_mask(self, hs: torch.Tensor) -> torch.Tensor | None:
        """Broadcastable [B,T,1] multiplier, or None for 'add everywhere'."""
        if self.mode == "all":
            return None

        B, T, _ = hs.shape
        if self.mode == "last":
            # Left-padded generative batches: the final column is the last real
            # token. Undefined for right-padded input, which is why generative
            # evaluation left-pads (see docs/PROBE_CONVENTIONS.md).
            m = torch.zeros(B, T, 1, dtype=hs.dtype, device=hs.device)
            m[:, -1, :] = 1.0
            return m

        if self.mode == "prompt" and self._mask is not None:
            return self._mask.to(hs.dtype)[:, :, None]

        if self.mode == "gen" and self._gen_start is not None:
            m = torch.zeros(B, T, 1, dtype=hs.dtype, device=hs.device)
            m[:, self._gen_start :, :] = 1.0
            return m

        return None

    # --------------------------------------------------------- lifecycle
    def __enter__(self) -> "Steering":
        blks = blocks(self.model)
        for layer_idx in self.layers:
            self.handles.append(
                blks[layer_idx].register_forward_hook(self._mk(layer_idx))
            )
        return self

    def __exit__(self, *_exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._alpha = 0.0
        self._mask = None
        self._gen_start = None

    # ------------------------------------------------------------ dosing
    def set_alpha(
        self,
        alpha: float,
        attn_mask: torch.Tensor | None = None,
        gen_start: int | None = None,
    ) -> None:
        self._alpha = float(alpha)
        self._mask = attn_mask
        self._gen_start = gen_start

    def with_alpha(
        self,
        alpha: float,
        attn_mask: torch.Tensor | None = None,
        gen_start: int | None = None,
    ) -> "_AlphaCtx":
        return _AlphaCtx(self, alpha, attn_mask, gen_start)

    @property
    def alpha(self) -> float:
        return self._alpha


class _AlphaCtx:
    """Context manager that restores alpha to 0 on exit, even on exception."""

    def __init__(self, s: Steering, a: float, m, g):
        self.s, self.a, self.m, self.g = s, a, m, g

    def __enter__(self) -> Steering:
        self.s.set_alpha(self.a, self.m, self.g)
        return self.s

    def __exit__(self, *_exc) -> None:
        self.s.set_alpha(0.0, None, None)
