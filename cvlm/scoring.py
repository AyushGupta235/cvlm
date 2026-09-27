"""State/candidate scoring: CLM's projection heads, or raw cosine (CLM's `clm-raw` ablation).

The `clm` score is ``exp(logit_scale) * cos(state_head(s), action_head(a))`` over unit-norm
encoder embeddings, exactly as ``clm.engine.Engine.answer`` computes it.
"""
from __future__ import annotations

import numpy as np

from .clm_src import heads
from .encoders import l2

RAW_SCALE = 100.0          # clm.engine.RAW_SCALE


class Scorer:
    def __init__(self, kind: str, ckpt: str | None = None, device: str = "cpu"):
        self.kind = kind
        if kind == "clm":
            path = ckpt or heads.default_checkpoint() or heads.download()
            self.pair = heads.HeadPair("clm", path, device).ensure()
            self.scale = self.pair.scale
            self.width = int(self.pair.cfg.get("hidden_size", heads.HIDDEN))
        elif kind == "raw":
            self.pair, self.scale, self.width = None, RAW_SCALE, None
        else:
            raise ValueError(f"unknown head {kind!r}; expected 'clm' or 'raw'")

    def _check(self, E: np.ndarray) -> np.ndarray:
        if self.width is not None and E.shape[1] != self.width:
            raise ValueError(f"CLM head expects {self.width}-d encoder embeddings, got {E.shape[1]}-d "
                             "(use head: raw with stand-in models)")
        return np.ascontiguousarray(E, dtype=np.float32)

    def states(self, E: np.ndarray) -> np.ndarray:
        E = self._check(E)
        return l2(E) if self.pair is None else self.pair.project_states(E).cpu().numpy()

    def actions(self, E: np.ndarray) -> np.ndarray:
        E = self._check(E)
        return l2(E) if self.pair is None else self.pair.project_actions(E).cpu().numpy()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()
