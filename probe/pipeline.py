"""Stages and their on-disk cache.

    <run>/data/    prepare     fit.jsonl, td.jsonl, aokvqa.jsonl, images/
    <run>/vl/      embed-vl    VL embeddings (*.npy), captions.json, ok_mc.npy
    <run>/text/    embed-text  text-model embeddings (*.npy)
    <run>/fit/     fit         maps.npz
    <run>/eval/    eval        metrics.json
    <run>/report.md

Each stage directory holds a manifest.json whose ``key`` hashes the config fields and
upstream keys the stage depends on. A stage whose key matches is skipped, so a run resumes
where it stopped, and fit/eval can be re-run on a laptop from embeddings made on a GPU.
Device is deliberately not part of any key.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time

import numpy as np

from . import data as data_mod
from . import evaluate, report, stitch
from .config import Config
from .encoders import TextEncoder, VLEncoder, free, pick_device, pick_dtype

STAGES = ["prepare", "embed-vl", "embed-text", "fit", "eval", "report"]
DIRS = {"prepare": "data", "embed-vl": "vl", "embed-text": "text", "fit": "fit", "eval": "eval"}


# ---------------------------------------------------------------------------- manifests
def _manifest_path(run_dir: str, stage: str) -> str:
    return os.path.join(run_dir, DIRS[stage], "manifest.json")


def read_manifest(run_dir: str, stage: str) -> dict | None:
    p = _manifest_path(run_dir, stage)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def stage_key(cfg: Config, run_dir: str, stage: str) -> str:
    fields = {
        "prepare": ("fit_limit", "heldout_frac", "td_limit", "img_limit", "seed"),
        "embed-vl": ("vl_model", "text_model", "dtype", "max_len", "max_pixels", "min_pixels",
                     "image_state_template", "caption_prompt", "caption_max_new_tokens"),
        "embed-text": ("text_model", "dtype", "max_len"),
        "fit": ("ridge_lambdas", "seed"),
        "eval": ("head", "head_ckpt", "bootstrap", "seed"),
    }[stage]
    upstream = {"prepare": [], "embed-vl": ["prepare"], "embed-text": ["prepare", "embed-vl"],
                "fit": ["embed-vl", "embed-text"], "eval": ["fit"]}[stage]
    parts = [cfg.key(*fields)] + [stage_key(cfg, run_dir, u) for u in upstream]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def is_done(cfg: Config, run_dir: str, stage: str) -> bool:
    m = read_manifest(run_dir, stage)
    return bool(m) and m.get("key") == stage_key(cfg, run_dir, stage)


def _finish(cfg: Config, run_dir: str, stage: str, t0: float, stats: dict) -> None:
    import torch
    import transformers
    m = {"stage": stage, "key": stage_key(cfg, run_dir, stage), "seconds": round(time.time() - t0, 1),
         "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
         "versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "stats": stats}
    with open(_manifest_path(run_dir, stage), "w") as f:
        json.dump(m, f, indent=1)


# ---------------------------------------------------------------------------- text sets
def unique(xs) -> list[str]:
    return list(dict.fromkeys(xs))


def load_data(run_dir: str) -> dict:
    d = os.path.join(run_dir, "data")
    return {"fit": data_mod.read_jsonl(os.path.join(d, "fit.jsonl")),
            "td": data_mod.read_jsonl(os.path.join(d, "td.jsonl")),
            "ok": data_mod.read_jsonl(os.path.join(d, "aokvqa.jsonl"))}


def text_sets(D: dict) -> dict[str, tuple[list[str], str | list[str]]]:
    """Texts both encoders embed: name -> (texts, keep)."""
    from .clm_src import schema
    return {
        "fit": ([t["text"] for t in D["fit"]], ["tail" if t["kind"] == "state" else "head" for t in D["fit"]]),
        "td_states": ([it["state"] for it in D["td"]], "tail"),
        "td_cands": (unique(c for it in D["td"] for c in it["candidates"]), "head"),
        "ok_choices": (unique(c for it in D["ok"] for c in it["choices"]), "head"),
        "ok_q": ([schema.state_text("", it["question"]) for it in D["ok"]], "tail"),
    }


def _save(dir_: str, name: str, x: np.ndarray) -> None:
    np.save(os.path.join(dir_, f"{name}.npy"), x.astype(np.float16))


def _load(dir_: str, name: str) -> np.ndarray:
    return np.load(os.path.join(dir_, f"{name}.npy")).astype(np.float32)


# ---------------------------------------------------------------------------- stages
def run_prepare(cfg: Config, run_dir: str) -> dict:
    return data_mod.prepare(cfg, os.path.join(run_dir, "data"))


def run_embed_vl(cfg: Config, run_dir: str) -> dict:
    from transformers import AutoTokenizer
    D, out = load_data(run_dir), os.path.join(run_dir, "vl")
    device = pick_device(cfg.device)
    enc = VLEncoder(cfg.vl_model, device, pick_dtype(cfg.dtype, device), cfg.max_len, cfg.token_budget,
                    cfg.max_batch, cfg.min_pixels, cfg.max_pixels)
    stats: dict = {"device": device, "hidden_size": enc.hidden_size, "truncated": {}}

    # Does the VL tokenizer split text exactly like CLM's encoder? (Identity/stitch assume it.)
    ref_tok = AutoTokenizer.from_pretrained(cfg.text_model)
    sample = [t["text"] for t in D["fit"][:2000]]
    same = [ref_tok(s, add_special_tokens=False)["input_ids"] == enc.tok(s, add_special_tokens=False)["input_ids"]
            for s in sample]
    stats["tokenizer_parity"] = float(np.mean(same)) if same else float("nan")
    stats["batch_parity_min_cos"] = enc.batch_parity(sample[:8])

    for name, (texts, keep) in text_sets(D).items():
        if name == "ok_q":
            continue                                   # only the text model scores question-only states
        E, stats["truncated"][name] = enc.embed_texts(texts, keep, label=f"vl/{name}")
        _save(out, name, E)

    images = [os.path.join(run_dir, "data", it["image"]) for it in D["ok"]]
    states = [cfg.image_state_template.format(question=it["question"]) for it in D["ok"]]
    stats["image_tokens_first"] = enc.image_token_count(images[0]) if images else 0
    _save(out, "ok_img", enc.embed_image_states(states, images))
    np.save(os.path.join(out, "ok_mc.npy"),
            enc.mc_letter_logits([it["question"] for it in D["ok"]], [it["choices"] for it in D["ok"]], images))
    captions = enc.caption(images, cfg.caption_prompt, cfg.caption_max_new_tokens, cfg.generate_batch)
    with open(os.path.join(out, "captions.json"), "w") as f:
        json.dump({it["id"]: c for it, c in zip(D["ok"], captions)}, f, indent=1, ensure_ascii=False)
    del enc
    free(device)
    return stats


def run_embed_text(cfg: Config, run_dir: str) -> dict:
    from .clm_src import schema
    D, out = load_data(run_dir), os.path.join(run_dir, "text")
    with open(os.path.join(run_dir, "vl", "captions.json")) as f:
        captions = json.load(f)
    device = pick_device(cfg.device)
    enc = TextEncoder(cfg.text_model, device, pick_dtype(cfg.dtype, device), cfg.max_len, cfg.token_budget,
                      cfg.max_batch)
    stats: dict = {"device": device, "hidden_size": enc.hidden_size, "truncated": {}}
    stats["batch_parity_min_cos"] = enc.batch_parity([t["text"] for t in D["fit"][:8]])
    sets = text_sets(D)
    sets["ok_cap"] = ([schema.state_text(captions[it["id"]], it["question"]) for it in D["ok"]], "tail")
    for name, (texts, keep) in sets.items():
        E, stats["truncated"][name] = enc.embed_texts(texts, keep, label=f"text/{name}")
        _save(out, name, E)
    del enc
    free(device)
    return stats


def load_embeddings(run_dir: str) -> dict[str, np.ndarray]:
    E = {}
    for side in ("vl", "text"):
        d = os.path.join(run_dir, side)
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy") and f != "ok_mc.npy":
                E[f"{side}/{f[:-4]}"] = _load(d, f[:-4])
    return E


def _split(D: dict) -> tuple[np.ndarray, np.ndarray]:
    split = np.array([t["split"] for t in D["fit"]])
    return split == "fit", split == "heldout"


def run_fit(cfg: Config, run_dir: str) -> dict:
    D, E = load_data(run_dir), load_embeddings(run_dir)
    fit_rows, _ = _split(D)
    X, Y = E["vl/fit"][fit_rows], E["text/fit"][fit_rows]
    maps, info = stitch.fit_all(X, Y, cfg.ridge_lambdas, seed=cfg.seed)
    stitch.save(os.path.join(run_dir, "fit", "maps.npz"), maps)
    info["n_fit"] = int(fit_rows.sum())
    return info


def run_eval(cfg: Config, run_dir: str) -> dict:
    from .scoring import Scorer
    D, E = load_data(run_dir), load_embeddings(run_dir)
    maps = stitch.load(os.path.join(run_dir, "fit", "maps.npz"))
    scorer = Scorer(cfg.head, cfg.head_ckpt)
    _, held = _split(D)
    sets = text_sets(D)
    td_index = {c: i for i, c in enumerate(sets["td_cands"][0])}
    ok_index = {c: i for i, c in enumerate(sets["ok_choices"][0])}
    mc = np.load(os.path.join(run_dir, "vl", "ok_mc.npy"))
    metrics = {
        "fidelity": evaluate.fidelity(maps, E["vl/fit"][held], E["text/fit"][held], scorer, seed=cfg.seed),
        "typed_decisions": evaluate.typed_decisions(D["td"], E, td_index, maps, scorer, cfg.bootstrap, cfg.seed),
        "aokvqa": evaluate.aokvqa(D["ok"], E, ok_index, maps, scorer, mc, cfg.bootstrap, cfg.seed),
        "head": {"kind": cfg.head, "scale": scorer.scale},
    }
    with open(os.path.join(run_dir, "eval", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return {"head": cfg.head}


RUNNERS = {"prepare": run_prepare, "embed-vl": run_embed_vl, "embed-text": run_embed_text,
           "fit": run_fit, "eval": run_eval}


def run_stage(cfg: Config, run_dir: str, stage: str, force: bool = False) -> None:
    if stage == "report":
        path = report.write(cfg, run_dir)
        print(f"[report] {path}", flush=True)
        return
    if not force and is_done(cfg, run_dir, stage):
        print(f"[{stage}] cached ({DIRS[stage]}/manifest.json)", flush=True)
        return
    os.makedirs(os.path.join(run_dir, DIRS[stage]), exist_ok=True)
    print(f"[{stage}] start", flush=True)
    t0 = time.time()
    stats = RUNNERS[stage](cfg, run_dir)
    _finish(cfg, run_dir, stage, t0, stats)
    print(f"[{stage}] done in {time.time() - t0:,.0f}s", flush=True)
