"""Run configuration: a YAML file plus command-line overrides."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import yaml

# Pinned dataset revisions, so the Mac and the pod read identical data.
TYPED_DECISIONS = ("LocalLLaMA/typed-decisions", "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8")
AOKVQA = ("HuggingFaceM4/A-OKVQA", "d1b0efa3a436e9101dfbde3752db7607da696c35")

IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass
class Config:
    name: str
    text_model: str                      # the encoder CLM's heads were trained on (Qwen3-8B in the real run)
    vl_model: str                        # the vision-language model we stitch from
    head: str = "clm"                    # "clm": CLM projection heads; "raw": cosine in the encoder space (clm-raw)
    head_ckpt: str | None = None         # None: download the CLM-v0.1-8B reference head
    device: str = "auto"                 # auto: cuda > mps > cpu
    dtype: str = "auto"                  # auto: bfloat16 on cuda, float16 on mps, float32 on cpu
    max_len: int = 2048                  # CLM's served context; texts keep max_len - 1 tokens
    max_pixels: int = 640 * 640          # image resolution cap (Qwen3-VL: one token per 32x32 pixels)
    min_pixels: int = 64 * 64
    token_budget: int = 16384            # padded tokens per forward batch
    max_batch: int = 64
    fit_limit: int | None = 40000        # size of the text corpus the maps are fitted on
    heldout_frac: float = 0.1            # of the fit corpus, for fidelity metrics
    td_limit: int | None = None          # typed-decisions test questions (None: all)
    img_limit: int | None = None         # A-OKVQA validation items (None: all 1145)
    image_state_template: str = IMAGE_TOKENS + "\n\n{question}"
    caption_prompt: str = "Describe this image in two or three sentences. Mention the objects, people, text and setting."
    caption_max_new_tokens: int = 96
    generate_batch: int = 8
    ridge_lambdas: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    bootstrap: int = 1000
    seed: int = 0

    def key(self, *fields_: str) -> str:
        """Stable hash of the named fields, used to decide whether a cached stage is still valid."""
        d = {f: getattr(self, f) for f in fields_}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _coerce(value: str, current: Any) -> Any:
    if value.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [float(v) for v in value.split(",")]
    return value


def load(path: str, overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = Config(**raw)
    for item in overrides or []:
        k, _, v = item.partition("=")
        if k not in known:
            raise ValueError(f"--set {item}: unknown key {k!r}")
        setattr(cfg, k, _coerce(v, getattr(cfg, k)))
    return cfg
