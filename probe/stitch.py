"""Maps from the VL model's embedding space into the text model's (CLM's) space.

* identity    no map (only defined when the widths match; Qwen3-VL-8B and Qwen3-8B are both 4096)
* ridge       affine least squares with an L2 penalty picked on an inner validation split
* procrustes  centred orthogonal map (rotation + offset), semi-orthogonal if the widths differ

Every mapped vector is L2-normalised again, because the CLM heads were trained on unit-norm
encoder embeddings. Fitting streams the rows in chunks in float64, so the full 40k x 4096 fit
set never has to exist in float64 at once.
"""
from __future__ import annotations

import numpy as np

from .encoders import l2

CHUNK = 4096


def _moments(X: np.ndarray, Y: np.ndarray):
    """Means, X^T X and X^T Y of the centred data, accumulated in float64 chunks."""
    mx = X.mean(0, dtype=np.float64)
    my = Y.mean(0, dtype=np.float64)
    G = np.zeros((X.shape[1], X.shape[1]))
    C = np.zeros((X.shape[1], Y.shape[1]))
    for s in range(0, len(X), CHUNK):
        xc = X[s:s + CHUNK].astype(np.float64) - mx
        yc = Y[s:s + CHUNK].astype(np.float64) - my
        G += xc.T @ xc
        C += xc.T @ yc
    return mx, my, G, C


def mean_cos(A: np.ndarray, B: np.ndarray) -> float:
    return float((l2(A) * l2(B)).sum(1).mean())


class Map:
    def __init__(self, kind: str, W: np.ndarray | None = None, mx=None, my=None, lam: float | None = None):
        self.kind, self.W, self.mx, self.my, self.lam = kind, W, mx, my, lam

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "identity":
            return l2(X.astype(np.float32))
        out = np.empty((len(X), self.W.shape[1]), dtype=np.float32)
        for s in range(0, len(X), CHUNK):
            out[s:s + CHUNK] = (X[s:s + CHUNK].astype(np.float64) - self.mx) @ self.W + self.my
        return l2(out)

    def arrays(self) -> dict:
        if self.kind == "identity":
            return {}
        return {f"{self.kind}_W": self.W.astype(np.float32), f"{self.kind}_mx": self.mx, f"{self.kind}_my": self.my}


def fit_ridge(X: np.ndarray, Y: np.ndarray, lambdas: list[float], val_frac: float = 0.1,
              seed: int = 0) -> tuple[Map, dict[float, float]]:
    """Pick the penalty on an inner split, then refit on everything.

    ``lambdas`` are relative to the mean eigenvalue of X^T X, so the grid means the same thing
    for any width or sample count. One eigendecomposition serves the whole grid.
    """
    rng = np.random.default_rng(seed)
    val = rng.random(len(X)) < val_frac
    mx, my, G, C = _moments(X[~val], Y[~val])
    evals, V = np.linalg.eigh(G)
    VtC = V.T @ C
    scale = max(evals.mean(), 1e-12)
    scores = {}
    for lam in lambdas:
        W = V @ (VtC / (evals + lam * scale)[:, None])
        scores[lam] = mean_cos((X[val].astype(np.float64) - mx) @ W + my, Y[val])
    best = max(scores, key=scores.get)
    mx, my, G, C = _moments(X, Y)
    W = np.linalg.solve(G + best * max(np.trace(G) / len(G), 1e-12) * np.eye(len(G)), C)
    return Map("ridge", W, mx, my, best), scores


def fit_procrustes(X: np.ndarray, Y: np.ndarray) -> Map:
    mx, my, _, C = _moments(X, Y)
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Map("procrustes", U @ Vt, mx, my)


def fit_all(X: np.ndarray, Y: np.ndarray, lambdas: list[float], seed: int = 0) -> tuple[dict[str, Map], dict]:
    maps: dict[str, Map] = {}
    info: dict = {}
    if X.shape[1] == Y.shape[1]:
        maps["identity"] = Map("identity")
    maps["ridge"], scores = fit_ridge(X, Y, lambdas, seed=seed)
    info["ridge_lambda"] = maps["ridge"].lam
    info["ridge_val_cos"] = {str(k): v for k, v in scores.items()}
    maps["procrustes"] = fit_procrustes(X, Y)
    return maps, info


def save(path: str, maps: dict[str, Map]) -> None:
    arrays = {"kinds": np.array(list(maps))}
    for m in maps.values():
        arrays.update(m.arrays())
        if m.lam is not None:
            arrays[f"{m.kind}_lam"] = np.array(m.lam)
    np.savez(path, **arrays)


def load(path: str) -> dict[str, Map]:
    z = np.load(path)
    maps = {}
    for kind in z["kinds"].tolist():
        if kind == "identity":
            maps[kind] = Map("identity")
        else:
            lam = float(z[f"{kind}_lam"]) if f"{kind}_lam" in z else None
            maps[kind] = Map(kind, z[f"{kind}_W"].astype(np.float64), z[f"{kind}_mx"], z[f"{kind}_my"], lam)
    return maps
