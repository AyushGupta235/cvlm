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

The 8B run (`configs/e1/full.yaml`) runs on a RunPod GPU. See [runpod/](runpod/) (in progress).

## Credentials

- `RUNPOD_API_KEY` (and optionally `HF_TOKEN`) live only in `.env`, which is gitignored. See `.env.example`.
- `.githooks/` runs `scripts/check_secrets.sh` on every commit (staged changes) and push (all tracked files). It blocks
  token-shaped strings, private keys, `.env` files and files over 5 MB, and reports file and line only, never the value.
- Pods get a dedicated SSH key (`~/.ssh/runpod_ed25519`) through the pod's `PUBLIC_KEY` variable. Nothing is added to
  your RunPod account settings.
