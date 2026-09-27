"""metrics.json + stage manifests -> report.md."""
from __future__ import annotations

import json
import os

from .config import Config

AOKVQA_ORDER = ["chance", "question_only", "caption_then_clm", "vl_identity", "vl_procrustes", "vl_ridge",
                "vl_ridge_both", "vl_raw", "vl_generative"]
AOKVQA_DESC = {
    "chance": "uniform guess",
    "question_only": "text model, question alone (no image): the prior",
    "caption_then_clm": "VL caption + question through the text model",
    "vl_identity": "image + question through VL, no map, into the CLM head",
    "vl_procrustes": "image + question through VL, orthogonal map",
    "vl_ridge": "image + question through VL, ridge map",
    "vl_ridge_both": "as vl_ridge, candidates also through VL + ridge",
    "vl_raw": "raw cosine in VL space, no map and no head",
    "vl_generative": "VL answers the letter itself (ceiling)",
}


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(ci) -> str:
    return "" if not ci else f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def _manifest(run_dir: str, sub: str) -> dict:
    p = os.path.join(run_dir, sub, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write(cfg: Config, run_dir: str) -> str:
    mpath = os.path.join(run_dir, "eval", "metrics.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"{mpath} missing; run the eval stage first")
    M = json.load(open(mpath))
    prep, vl, tx, fit = (_manifest(run_dir, s) for s in ("data", "vl", "text", "fit"))
    L = [f"# Probe-1 report: `{cfg.name}`", ""]
    L += [f"- text model (CLM's encoder): `{cfg.text_model}`", f"- VL model: `{cfg.vl_model}`",
          f"- scorer: `{M['head']['kind']}` (logit scale {M['head']['scale']:.1f})",
          f"- data: {json.dumps(prep.get('stats', {}))}", ""]

    L += ["## Sanity checks", "",
          "| check | value |", "|---|---|",
          f"| VL vs text tokenizer, identical ids | {_pct(vl.get('stats', {}).get('tokenizer_parity'))}% |",
          f"| batched vs one-at-a-time, min cos (VL / text) | {vl.get('stats', {}).get('batch_parity_min_cos', float('nan')):.5f} / "
          f"{tx.get('stats', {}).get('batch_parity_min_cos', float('nan')):.5f} |",
          f"| image tokens (first image) | {vl.get('stats', {}).get('image_tokens_first')} |",
          f"| truncated texts (VL / text) | {vl.get('stats', {}).get('truncated')} / {tx.get('stats', {}).get('truncated')} |",
          f"| stage seconds (VL / text) | {vl.get('seconds')} / {tx.get('seconds')} on {vl.get('stats', {}).get('device')} |",
          ""]

    L += ["## Stitch fidelity (held-out texts, VL → text space)", "",
          f"Ridge λ = {fit.get('stats', {}).get('ridge_lambda')} (relative to the mean eigenvalue), "
          f"fitted on {fit.get('stats', {}).get('n_fit')} texts.", "",
          "| map | cos (encoder space) | cos (after state head) | retrieval@1 |", "|---|---|---|---|"]
    for name, r in M["fidelity"].items():
        L.append(f"| {name} | {r['cos']:.4f} | {r['head_cos']:.4f} | {_pct(r['retrieval_at_1'])}% |")
    L.append("")

    td = M["typed_decisions"]
    types = sorted({t for r in td.values() for t in r["by_type"]})
    L += ["## Typed decisions (text only): does the stitch preserve CLM's behaviour?", "",
          "`agree ref` is how often the condition picks the same option as the text-model path; "
          "`TV ref` is the total-variation distance between their distributions.", "",
          "| condition | acc vs gold | 95% CI | agree ref | TV ref | TV gold | " + " | ".join(types) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(types)]
    for name, r in td.items():
        L.append(f"| {name} | {_pct(r['acc'])} | {_ci(r['ci'])} | {_pct(r['agree_ref'])} | {r['tv_ref']:.3f} | "
                 f"{r['tv_gold']:.3f} | " + " | ".join(_pct(r['by_type'].get(t)) for t in types) + " |")
    L.append("")

    ok = M["aokvqa"]
    L += ["## A-OKVQA (validation, multiple choice): decisions over images", "",
          "Δ columns are paired differences in points with bootstrap 95% CIs. A stitched condition earns its keep "
          "only if Δ vs question-only is clearly positive, and it is competitive only if Δ vs caption is near 0 or positive.", "",
          "| condition | acc | 95% CI | Δ vs question-only | Δ vs caption | what it is |", "|---|---|---|---|---|---|"]
    for name in [n for n in AOKVQA_ORDER if n in ok] + [n for n in ok if n not in AOKVQA_ORDER]:
        r = ok[name]
        d = r.get("delta", {})

        def fmt(base):
            if base not in d:
                return ""
            x = d[base]
            return f"{100 * x['mean']:+.1f} {_ci(x['ci'])}"
        L.append(f"| {name} | {_pct(r['acc'])} | {_ci(r['ci'])} | {fmt('question_only')} | {fmt('caption_then_clm')} | "
                 f"{AOKVQA_DESC.get(name, '')} |")
    L += ["", "## Caveats", "",
          "- Embeddings come from Hugging Face transformers (last token, after the final norm, L2-normalised), not from "
          "CLM's vLLM pooling server. Every condition shares this path, so comparisons between conditions are "
          "apples-to-apples. Absolute `ref` numbers can differ slightly from served CLM.",
          "- The maps are fitted on text only. Image states are out of distribution for them by construction; "
          "measuring how far off they are is the point of the probe.",
          "- A-OKVQA answers are short and often guessable from the question; read vl_* against question_only, "
          "not against chance.", ""]
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
