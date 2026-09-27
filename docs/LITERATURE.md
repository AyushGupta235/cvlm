# Literature for Probe-1

Readings for the linear-stitch probe (Qwen3-VL-8B hidden states → CLM-8B's Qwen3-8B space), ordered by how directly
they bear on it. Every arXiv link was checked against its abstract page.

If time is short, read 1–3: between them they predict most of what the probe will show.

## The probe's hypothesis

1. **[Linearly Mapping from Image to Text Space](https://arxiv.org/abs/2209.15162)** (Merullo et al., ICLR 2023).
   A single linear projection from a frozen image encoder into a frozen LM works surprisingly well. This is the same
   "is a linear map enough?" bet the probe makes.
2. **[E5-V: Universal Embeddings with Multimodal Large Language Models](https://arxiv.org/abs/2407.12580)** (Jiang et al., 2024).
   An MLLM trained on text-only pairs yields embeddings that transfer to image inputs. This is the closest prior result
   to "fit the map on text, apply it to image states".
3. **[Mind the Gap: the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053)** (Liang et al., NeurIPS 2022).
   Explains the modality gap, the expected failure mode when `vl_ridge` is compared with `caption_then_clm`.

## Why a stitch should (or shouldn't) work

4. **[Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682)** (Bansal, Nakkiran, Barak, 2021).
   How to interpret stitching results rigorously.
5. **[Relative representations enable zero-shot latent space communication](https://arxiv.org/abs/2209.15430)** (Moschella et al., ICLR 2023).
   Latent spaces often differ by near-isometries, which motivates the Procrustes variant.
6. **[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)** (Huh et al., ICML 2024).
   The broader argument for why vision and language representations converge.

## The geometry of CLM's embeddings

7. **[Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)** (Wang et al., 2024; E5-Mistral).
   The last-token-pooling recipe CLM builds on.
8. **[Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)** (Sun et al., 2024).
   A few huge hidden dimensions dominate last-token vectors. Keep this in mind when reading raw cosine numbers in the report.

## Contrastive foundations

9. **[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)** (van den Oord et al., 2018), the origin of InfoNCE,
   then **[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)** (Radford et al., 2021; CLIP).
10. **[LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)** (Zhai et al., CVPR 2022).
    Keep one tower locked and train the other. Route 2 is a version of this, with the CLM text side frozen.

## Toward routes 2 and 3

11. **[Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)** (Liu et al., NeurIPS 2023; LLaVA). The projector recipe.
12. **[Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07865)** (Karamcheti et al., ICML 2024).
    Ablations on freezing vs. unfreezing the LLM and on the choice of vision encoder.
13. **[VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160)** (Jiang et al., 2024).
    Turns a VLM into a contrastive embedder, i.e. route 3.
14. **[Qwen3-VL-Embedding and Qwen3-VL-Reranker](https://arxiv.org/abs/2601.04720)** (Qwen, 2026).
    The strongest off-the-shelf backbone candidate for route 3.

## Evaluation set

15. **[A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge](https://arxiv.org/abs/2206.01718)** (Schwenk et al., ECCV 2022).
    Probe-1's image eval (the validation split, multiple choice). Know its quirks before reading the numbers.
