# CVLM experimental design

**Goal.** A contrastive visual language model: a System One model that scores candidate actions against states
that include images (screens, scenes, documents), while keeping what makes CLM fast:
- one encoder pass per state
- cached action embeddings
- a dot product per candidate
- no decoding

**Starting point.** [CLM-8B](https://github.com/Contrastive-LM/CLM) is a frozen Qwen3-8B plus two ~20M-parameter
projection heads. Its embedding is the last token's hidden state after the final norm, 4096-d and L2-normalised. The
state head and action head each map it to 512-d, and a candidate's score is `exp(logit_scale) · cos(state, action)`.
The heads were trained with bidirectional InfoNCE in three stages:
- pre-training on 60M Nemotron Q&A pairs
- mid-training on 30M synthetic hard negatives
- post-training on 1M agent steps

Only the heads were trained. The backbone was never fine-tuned, and a head is only valid on the encoder it was
trained against.

**Consequences for the design.**
- Only the state side needs vision. Actions stay text, so the action head and the action cache carry over.
- The tokenizer does not matter here. What matters is whether image-conditioned states land where CLM's heads can read them.
- The work runs in three experiments, cheapest first. Each result decides whether the next one is needed and how it starts.

---

## E1: linear alignment of a VLM into CLM's encoder space (implemented)

**Question.** Qwen3-VL-8B is built on Qwen3-8B, with the same 36 layers and 4096-d hidden size. Is its last-token space
still close enough that a map fitted on text alone lets CLM's existing heads score image+question states?

**Method.**
1. Embed about 40k texts with both models: typed-decisions states and options, plus A-OKVQA train text.
2. Fit three maps from the VL space into Qwen3-8B's space on 90% of them:
   - identity
   - ridge: affine, with λ chosen on an inner validation split
   - orthogonal Procrustes
3. Measure fidelity on the held-out 10%: cosine, cosine after the state head, and retrieval@1.

**Evaluations.**
- **Typed decisions (text only).** Does the aligned VL path make the same decisions as CLM's own path?
  Measured as accuracy, agreement with CLM, and total-variation distance to CLM's distribution.
- **A-OKVQA validation (images, four-way multiple choice).** Image+question states go through the VL model, a map and
  CLM's state head. Candidates come from CLM's own action path. Controls:

  | condition | what it isolates |
  |---|---|
  | `question_only` | the prior; how guessable the answer is without the image |
  | `caption_then_clm` | the zero-training pipeline alternative: VL captions the image, CLM scores the text |
  | `vl_raw` | whether the VL space alone carries the answer, with no heads |
  | `vl_generative` | what the VL model itself knows (ceiling) |

**Decision rule.**
| outcome | reading | next |
|---|---|---|
| `vl_ridge` ≫ `question_only` and ≈ `caption_then_clm` | the text-fitted alignment transfers to images | warm-start E2/E3 from the aligned heads; E1 is already a usable baseline CVLM |
| typed-decisions agreement high, image accuracy ≈ `question_only` | modality gap: text alignment doesn't reach image states | image-conditioned contrastive training is required (E2 or E3) |
| typed-decisions agreement low even for text | the VL model's geometry has diverged from Qwen3-8B | skip E2 and go to E3 (retrain heads on the VLM) |

**Cost.** Everything is developed on Apple Silicon with stand-in models (tiny random models, then Qwen3-1.7B with
Qwen3-VL-2B). The 8B run takes about an hour on one 24 GB GPU (about $0.30–0.70 on RunPod).

---

## E2: vision projector on frozen Qwen3-8B, text path unchanged (planned)

**Question.** Can a vision encoder feed CLM-8B directly while every text-only input stays bit-for-bit identical?
If so, CLM's heads, its fine-tuned heads (DeepSWE and others) and its action caches all remain valid.

**Method.**
- A SigLIP2 or Qwen3-VL vision tower feeds an MLP projector that emits soft tokens, placed before the text. The
  question comes last, so the pooled last token sees the image.
- **Stage A:** captioning with a next-token loss, which is possible because Qwen3-8B still has its LM head.
- **Stage B:** InfoNCE against the frozen CLM heads (text answers as positives), plus distillation that pulls
  (image + question) toward CLM's embedding of (detailed caption + question).
- **Optional:** LoRA active only at image-token positions ("partial LoRA"), which leaves text-only inputs untouched.

**Risk.** A frozen text LLM with a projector is weak at fine visual detail (OCR, small UI elements), which is exactly
what computer-use needs.

---

## E3: VLM backbone with retrained contrastive heads (planned)

**Question.** What does CLM's full recipe produce when the frozen backbone is a VLM, and the heads are trained on
image-conditioned states from the start?

**Method.**
- Serve Qwen3-VL-8B (or Qwen3-VL-Embedding-8B) as the frozen encoder.
- Re-embed CLM's text recipe and retrain the heads, initialised from E1's aligned heads.
- Add multimodal stages that mirror CLM's recipe:

  | stage | data |
  |---|---|
  | pre-training | visual question answering and caption pairs |
  | mid-training | LLM-generated hard-negative answers, plus near-duplicate images with different answers |
  | post-training | screenshot agent trajectories (Mind2Web, AndroidControl, AgentNet), with 40% text replay |

**Cost.** Dominated by the one-off embedding pass: roughly 27B tokens for the text recipe, a few hundred H100-hours.

---

## Shared evaluation (grows with the experiments)
- **A-OKVQA (four-way multiple choice):** scene understanding, and the E1 gate.
- **Typed decisions (text):** regression check against CLM-8B on its own ground.
- **Planned:** GUI decisions (a screenshot as the state, candidate UI actions as the actions) for the computer-use target
  on CLM's roadmap.
- **Baseline to beat, always:** `caption_then_clm` at matched latency.
