# CVLM: Contrastive Visual Language Model

Vision for System One models. [CLM-8B](https://github.com/Contrastive-LM/CLM) makes fast decisions over text: it
embeds a state once, keeps action embeddings cached, and scores each candidate with a dot product. CVLM gives it the
same kind of decisions over images: screens, scenes, documents.

The experimental design (E1 linear alignment, E2 vision projector, E3 VLM backbone) is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), and the reading list is in [docs/LITERATURE.md](docs/LITERATURE.md).

## Layout

```
cvlm/            library
  data.py          corpora and eval sets (typed-decisions, A-OKVQA), pinned revisions
  encoders.py      last-token embeddings (Qwen3, Qwen3-VL); VL captions and multiple-choice answers
  align.py         maps from a VLM's space into CLM's encoder space (identity, ridge, Procrustes)
  scoring.py       CLM projection heads, or raw cosine
  evaluate.py      fidelity, typed decisions, A-OKVQA, bootstrap CIs
  pipeline.py      cached, resumable stages;  report.py: report.md
configs/e1/      tiny (random models, CPU) · local (Qwen3-1.7B + Qwen3-VL-2B, Apple Silicon) · full (8B, GPU)
runpod/          pod orchestration for the GPU runs
scripts/         local setup, CLM checkout, credential guard
tests/
```

## Running E1

```bash
bash scripts/setup_local.sh                                      # venv, pinned deps, CLM @ pinned commit
.venv/bin/python -m pytest -q
.venv/bin/python -m cvlm run --config configs/e1/tiny.yaml       # ~1 min on CPU once data is cached
.venv/bin/python -m cvlm run --config configs/e1/local.yaml      # real small models on MPS
```

The stages are `prepare → embed-vl → embed-text → fit → eval → report`. Each writes `runs/<name>/<stage>/` with a
manifest and is skipped when its inputs haven't changed. So a run resumes after an interruption, and `fit`/`eval`
can be re-run on a laptop against embeddings produced on a GPU.

## Running on a RunPod GPU

Only the 8B forward passes need a GPU. Everything else (tests, dataset preparation, image extraction, and re-fitting
or re-evaluating afterwards) runs on the laptop.

```bash
cp .env.example .env                     # add RUNPOD_API_KEY (console → Settings → API Keys)
.venv/bin/python runpod/launch.py run --config configs/e1/full.yaml --dry-run    # prints the request, creates nothing
.venv/bin/python runpod/launch.py run --config configs/e1/full.yaml              # asks before creating a paid pod
.venv/bin/python runpod/launch.py list                                           # what is running
.venv/bin/python runpod/launch.py terminate --all                                # panic button (cvlm-* pods only)
```

What `run` does:

1. **Preflight (free):** tests pass, data is prepared locally, the image tag exists, and no `cvlm-<run>` pod already exists.
2. **Create one pod:** RTX 4090 first, then 48 GB fallbacks; community cloud; `volumeInGb: 0`, so nothing is billed
   after termination; at least 500 Mbps download, since 34 GB of weights over a slow link wastes GPU time.
3. **Upload and run:** upload the non-ignored files plus `runs/<run>/` over SSH, run a CUDA smoke test on the tiny
   models, then the config.
4. **Stream and pull back:** stream the log, and pull each stage back as soon as its manifest appears. A preempted or
   failed run resumes from those stages when re-run.
5. **Terminate** on success, error, Ctrl-C, SIGTERM/SIGHUP, a price above `--max-price` (default $1/hr), or the
   `--max-minutes` cap (default 90). A watchdog on the pod also deletes it at the cap if the laptop disappears. The
   command runs under `caffeinate` so the Mac doesn't sleep mid-run.

Expected cost of `configs/e1/full.yaml`: about 45–60 minutes on one RTX 4090, roughly $0.30–0.70. `--spot` halves
that at the risk of preemption.

## Credentials

- `RUNPOD_API_KEY` (and optionally `HF_TOKEN`) live only in `.env`, which is gitignored. See `.env.example`.
- `.githooks/` runs `scripts/check_secrets.sh` on every commit (staged changes) and push (all tracked files). It blocks
  token-shaped strings, private keys, `.env` files and files over 5 MB, and reports file and line only, never the value.
- Pods get a dedicated SSH key (`~/.ssh/runpod_ed25519`) through the pod's `PUBLIC_KEY` variable. Nothing is added to
  your RunPod account settings.
