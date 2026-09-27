"""Stage `eval`: alignment fidelity, typed-decisions (text) and A-OKVQA (image) decisions.

Naming: E["text/<set>"] are text-model embeddings (CLM's own encoder), E["vl/<set>"] are VL-model
embeddings. ``vl_<map>`` conditions send the VL vector through that map into the text model's
space, then through CLM's state head. Unless the name ends in ``_both``, candidates are
embedded by the text model, as a deployed CLM would have them cached.
"""
from __future__ import annotations

import numpy as np

from .scoring import Scorer, softmax
from .align import Map


def bootstrap_ci(x: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summary(correct: np.ndarray, n_boot: int, seed: int) -> dict:
    lo, hi = bootstrap_ci(correct.astype(np.float64), n_boot, seed)
    return {"acc": float(correct.mean()) if len(correct) else float("nan"), "ci": [lo, hi], "n": int(len(correct))}


def fidelity(maps: dict[str, Map], X: np.ndarray, Y: np.ndarray, scorer: Scorer,
             n_retrieval: int = 2048, seed: int = 0) -> dict:
    """How closely each map reproduces the text model's embedding of held-out texts."""
    rng = np.random.default_rng(seed)
    sub = rng.permutation(len(X))[:n_retrieval]
    head_y = scorer.states(Y)
    out = {}
    for name, m in maps.items():
        mapped = m(X)
        sims = mapped[sub] @ Y[sub].T
        out[name] = {"cos": float((mapped * Y).sum(1).mean()),
                     "head_cos": float((scorer.states(mapped) * head_y).sum(1).mean()),
                     "retrieval_at_1": float((sims.argmax(1) == np.arange(len(sub))).mean()),
                     "n": int(len(X))}
    return out


def _decide(S: np.ndarray, A: np.ndarray, cand_rows: list[list[int]], scale: float) -> list[np.ndarray]:
    return [softmax(scale * (A[rows] @ S[i])) for i, rows in enumerate(cand_rows)]


def typed_decisions(items: list[dict], E: dict[str, np.ndarray], cand_index: dict[str, int],
                    maps: dict[str, Map], scorer: Scorer, n_boot: int, seed: int) -> dict:
    rows = [[cand_index[c] for c in it["candidates"]] for it in items]
    A_ref = scorer.actions(E["text/td_cands"])
    conds = {"ref": (scorer.states(E["text/td_states"]), A_ref)}
    for name, m in maps.items():
        conds[f"vl_{name}"] = (scorer.states(m(E["vl/td_states"])), A_ref)
    if "ridge" in maps:
        conds["vl_ridge_both"] = (conds["vl_ridge"][0], scorer.actions(maps["ridge"](E["vl/td_cands"])))

    labels = np.array([it["label"] for it in items])
    targets = [np.array(it["target"]) for it in items]
    types = np.array([it["type"] for it in items])
    ref_probs = _decide(*conds["ref"], rows, scorer.scale)
    ref_pred = np.array([p.argmax() for p in ref_probs])
    out = {}
    for name, (S, A) in conds.items():
        probs = _decide(S, A, rows, scorer.scale)
        pred = np.array([p.argmax() for p in probs])
        correct = pred == labels
        res = _summary(correct, n_boot, seed)
        res["tv_gold"] = float(np.mean([0.5 * np.abs(p - t).sum() for p, t in zip(probs, targets)]))
        res["agree_ref"] = float((pred == ref_pred).mean())
        res["tv_ref"] = float(np.mean([0.5 * np.abs(p - q).sum() for p, q in zip(probs, ref_probs)]))
        res["by_type"] = {t: float(correct[types == t].mean()) for t in sorted(set(types))}
        out[name] = res
    return out


def aokvqa(items: list[dict], E: dict[str, np.ndarray], choice_index: dict[str, int], maps: dict[str, Map],
           scorer: Scorer, mc_logits: np.ndarray | None, n_boot: int, seed: int) -> dict:
    rows = [[choice_index[c] for c in it["choices"]] for it in items]
    labels = np.array([it["label"] for it in items])
    A_text = scorer.actions(E["text/ok_choices"])
    conds = {
        "question_only": (scorer.states(E["text/ok_q"]), A_text),
        "caption_then_clm": (scorer.states(E["text/ok_cap"]), A_text),
    }
    for name, m in maps.items():
        conds[f"vl_{name}"] = (scorer.states(m(E["vl/ok_img"])), A_text)
    if "ridge" in maps:
        conds["vl_ridge_both"] = (conds["vl_ridge"][0], scorer.actions(maps["ridge"](E["vl/ok_choices"])))

    correct: dict[str, np.ndarray] = {}
    for name, (S, A) in conds.items():
        correct[name] = np.array([p.argmax() for p in _decide(S, A, rows, scorer.scale)]) == labels
    # No heads at all: raw cosine inside the VL model's own space.
    S, A = E["vl/ok_img"], E["vl/ok_choices"]
    correct["vl_raw"] = np.array([p.argmax() for p in _decide(S, A, rows, 100.0)]) == labels
    if mc_logits is not None:
        n_opt = [len(it["choices"]) for it in items]
        correct["vl_generative"] = np.array([mc_logits[i, :k].argmax() for i, k in enumerate(n_opt)]) == labels

    out = {name: _summary(c, n_boot, seed) for name, c in correct.items()}
    out["chance"] = {"acc": float(np.mean([1 / len(it["choices"]) for it in items])), "ci": None, "n": len(items)}
    # Paired differences against the two text-only baselines (same items, so resample jointly).
    for base in ("question_only", "caption_then_clm"):
        for name, c in correct.items():
            if name == base:
                continue
            d = c.astype(np.float64) - correct[base].astype(np.float64)
            out[name].setdefault("delta", {})[base] = {"mean": float(d.mean()), "ci": list(bootstrap_ci(d, n_boot, seed))}
    return out
