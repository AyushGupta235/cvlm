"""Stage `prepare`: the fit corpus and the two evaluation sets, written to <run>/data.

Everything here is CPU-only and runs on the Mac before a pod exists.

* fit.jsonl   unique texts both encoders embed; the alignment maps are fitted on ``split == "fit"``
              and fidelity is measured on ``split == "heldout"``. Sources: typed-decisions
              train (CLM state texts and option texts) and A-OKVQA train text (questions,
              choices, rationale + question). No image bytes are read for train.
* td.jsonl    typed-decisions test questions (text only): state text, option keys and
              texts, gold label and the annotator distribution.
* aokvqa.jsonl + images/   A-OKVQA validation, four-way multiple choice with images.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from .clm_src import adapters, schema
from .config import AOKVQA, TYPED_DECISIONS, Config


def _hash01(text: str, salt: str = "") -> float:
    return int(hashlib.sha1((salt + text).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _typed_decisions(split: str) -> list[dict]:
    repo, rev = TYPED_DECISIONS
    path = hf_hub_download(repo, f"all/{split}-00000-of-00001.parquet", repo_type="dataset", revision=rev)
    return pq.read_table(path).to_pylist()


def _question_types(row: dict) -> dict[str, str]:
    qs = row["questions"]
    qs = json.loads(qs) if isinstance(qs, str) else qs
    return {qid: q["type"] for qid, q in qs.items()}


def typed_decision_items(split: str) -> list[dict]:
    """One item per (row, question) with a gold label, via CLM's own adapter."""
    rows = _typed_decisions(split)
    types = {str(r.get("id", "")): _question_types(r) for r in rows}
    items = []
    for ex in adapters.typed_decision_examples(rows):
        items.append({"id": f"{ex.group}/{ex.qid}", "type": types[ex.group][ex.qid], "workflow": ex.workflow,
                      "state": ex.state_text, "keys": ex.keys, "candidates": ex.candidates,
                      "label": ex.label, "target": ex.target})
    return items


def _aokvqa_text_rows(split: str) -> list[dict]:
    """Text columns only, via column projection over the Hub, so image bytes are never downloaded."""
    repo, rev = AOKVQA
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}@{rev}/data/{split}-*.parquet"))
    rows = []
    for f in files:
        with fs.open(f, "rb") as fh:
            rows += pq.read_table(fh, columns=["question_id", "question", "choices", "rationales"]).to_pylist()
    return rows


def fit_corpus(cfg: Config) -> list[dict]:
    texts: dict[str, dict] = {}

    def add(text: str, kind: str, source: str):
        text = text.strip()
        if text and text not in texts:
            texts[text] = {"text": text, "kind": kind, "source": source}

    for it in typed_decision_items("train"):
        add(it["state"], "state", "td")
        for c in it["candidates"]:
            add(c, "action", "td")
    td_texts = list(texts.values())

    texts.clear()
    for r in _aokvqa_text_rows("train"):
        add(schema.state_text("", r["question"]), "state", "aokvqa_q")
        for c in r["choices"]:
            add(c, "action", "aokvqa_choice")
        if r["rationales"]:
            add(schema.state_text(r["rationales"][0], r["question"]), "state", "aokvqa_rationale")
    ok_texts = list(texts.values())

    # All typed-decisions texts (closest to CLM's own inputs), then A-OKVQA text up to the limit.
    rng = random.Random(cfg.seed)
    rng.shuffle(ok_texts)
    budget = None if cfg.fit_limit is None else max(0, cfg.fit_limit - len(td_texts))
    if cfg.fit_limit is not None and cfg.fit_limit < len(td_texts):
        rng.shuffle(td_texts)
        td_texts = td_texts[:cfg.fit_limit]
    corpus = td_texts + (ok_texts if budget is None else ok_texts[:budget])
    for i, t in enumerate(corpus):
        t["id"] = i
        t["split"] = "heldout" if _hash01(t["text"], "heldout") < cfg.heldout_frac else "fit"
    return corpus


def aokvqa_items(cfg: Config, image_dir: str) -> list[dict]:
    repo, rev = AOKVQA
    fs = HfFileSystem()
    (path,) = fs.glob(f"datasets/{repo}@{rev}/data/validation-*.parquet")
    local = hf_hub_download(repo, path.split(f"{rev}/", 1)[1], repo_type="dataset", revision=rev)
    rows = pq.read_table(local).to_pylist()
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    if cfg.img_limit is not None:
        rows = rows[:cfg.img_limit]
    os.makedirs(image_dir, exist_ok=True)
    items = []
    from PIL import Image
    for r in rows:
        name = f"{r['question_id']}.jpg"
        dest = os.path.join(image_dir, name)
        if not os.path.exists(dest):
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            img.save(dest, quality=95)
        items.append({"id": r["question_id"], "question": r["question"], "choices": list(r["choices"]),
                      "label": int(r["correct_choice_idx"]), "image": f"images/{name}"})
    return items


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(cfg: Config, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    corpus = fit_corpus(cfg)
    write_jsonl(os.path.join(out_dir, "fit.jsonl"), corpus)

    td = typed_decision_items("test")
    if cfg.td_limit is not None:
        random.Random(cfg.seed).shuffle(td)
        td = td[:cfg.td_limit]
    write_jsonl(os.path.join(out_dir, "td.jsonl"), td)

    ok = aokvqa_items(cfg, os.path.join(out_dir, "images"))
    write_jsonl(os.path.join(out_dir, "aokvqa.jsonl"), ok)

    counts: dict[str, int] = {}
    for t in corpus:
        counts[f"{t['source']}/{t['split']}"] = counts.get(f"{t['source']}/{t['split']}", 0) + 1
    return {"fit_texts": len(corpus), "fit_by_source": counts, "td_questions": len(td), "aokvqa_items": len(ok)}
