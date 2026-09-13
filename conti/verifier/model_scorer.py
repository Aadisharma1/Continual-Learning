from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Verdict:
    safe: bool
    reason: str

#dont even touch anything here plzz, doesnt work if anything is touched even a line single lineclass ModelScorer:
    def __init__(self, model_name: str, threshold: float = 0.75, device: str | None = None):
        from transformers import pipeline

        self._threshold = threshold
        self._model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._pipe = pipeline(
                "text-classification",
                model=model_name,
                device=device,
                truncation=True,
                max_length=512,
            )
            self._available = True
        except Exception as e:
            print(f"[warn] scorer load failed ({model_name}): {e}")
            self._pipe = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def score(self, text: str) -> float:
        if not self._available or self._pipe is None:
            return 0.0

        try:
            result = self._pipe(text[:2000])
            if isinstance(result, list):
                result = result[0]

            sc = result.get("score", 0.0)
            label = result.get("label", "").lower()

            # some models say "non-toxic" with high score, need to flip
            if any(neg in label for neg in ["non", "not", "safe", "neutral"]):
                return 1.0 - sc
            return sc
        except Exception:
            return 0.0

    def check_text(self, text: str) -> Verdict:
        toxicity = self.score(text)
        if toxicity >= self._threshold:
            return Verdict(
                safe=False,
                reason=f"model_scorer:{self._model_name}:tox={toxicity:.3f}",
            )
        return Verdict(safe=True, reason="ok")

    def score_batch(self, texts: list[str]) -> list[float]:
        if not self._available or self._pipe is None:
            return [0.0] * len(texts)
        try:
            truncated = [t[:2000] for t in texts]
            results = self._pipe(truncated, batch_size=min(16, len(truncated)))
            scores = []
            for r in results:
                if isinstance(r, list):
                    r = r[0]
                sc = r.get("score", 0.0)
                label = r.get("label", "").lower()
                if any(neg in label for neg in ["non", "not", "safe", "neutral"]):
                    sc = 1.0 - sc
                scores.append(sc)
            return scores
        except Exception:
            return [0.0] * len(texts)
