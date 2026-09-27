"""python -m cvlm {prepare,embed-vl,embed-text,fit,eval,report,run} --config configs/e1/<name>.yaml"""
from __future__ import annotations

import argparse
import os

from . import config as config_mod
from .pipeline import STAGES, run_stage


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m cvlm", description=__doc__)
    ap.add_argument("stage", choices=STAGES + ["run"], help="one stage, or `run` for all of them in order")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", help="default: runs/<config name>")
    ap.add_argument("--device", help="override the config's device (auto, cuda, mps, cpu)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    ap.add_argument("--force", action="store_true", help="recompute even if the stage is cached")
    ap.add_argument("--until", choices=STAGES, help="with `run`: stop after this stage")
    a = ap.parse_args(argv)

    overrides = list(a.set) + ([f"device={a.device}"] if a.device else [])
    cfg = config_mod.load(a.config, overrides)
    run_dir = a.run_dir or os.path.join("runs", cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    if a.stage == "run":
        stages = STAGES[:STAGES.index(a.until) + 1] if a.until else STAGES
    else:
        stages = [a.stage]
    for s in stages:
        run_stage(cfg, run_dir, s, force=a.force)


if __name__ == "__main__":
    main()
