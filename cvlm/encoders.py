"""Last-token embeddings from Qwen3 (text) and Qwen3-VL (text and image+text), plus the VL
model's captions and multiple-choice answers.

The embedding is the final hidden state (after the model's final norm) at the last real
token, L2-normalised: the same vector vLLM's pooling runner serves to CLM. Batches are
right-padded and the vector is gathered at ``attention_mask.sum() - 1``, so every real token
keeps the position id it would have unpadded. Texts are tokenized with CLM's recipe
(train/embed_utils.Recipe): no special tokens, states keep their tail, actions their head,
at most ``max_len - 1`` tokens.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, Sequence

import numpy as np
import torch


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(pref: str, device: str) -> torch.dtype:
    if pref != "auto":
        return getattr(torch, pref)
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)


def free(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_model(cls, repo: str, dtype: torch.dtype, device: str):
    """Load a checkpoint in ``dtype`` onto ``device``.

    CUDA: ``device_map`` puts the weights straight on the GPU, so no full copy is staged in CPU
    RAM (pods can be short on it). MPS/CPU: cast on the CPU, then move. Qwen checkpoints are
    bfloat16, and casting them on an M1's GPU (no native bfloat16) hangs in PyTorch's Metal
    cast kernel; moving already-cast float16 tensors does not.
    """
    if device == "cuda":
        return cls.from_pretrained(repo, dtype=dtype, device_map="cuda").eval()
    return cls.from_pretrained(repo, dtype=dtype, device_map="cpu").to(device).eval()


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def length_batches(lengths: Sequence[int], token_budget: int, max_batch: int) -> list[list[int]]:
    """Group indices by length (longest first) so that padded tokens per batch stay under the budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        width = lengths[cur[0]] if cur else lengths[i]      # longest first: the first item sets the width
        if cur and (len(cur) >= max_batch or width * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def pool_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state at the last real token of each right-padded row."""
    if not bool(attention_mask[:, 0].all()):
        raise ValueError("pool_last expects right padding")
    idx = attention_mask.long().sum(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def run_split_on_oom(compute: Callable[[list[int]], tuple[np.ndarray, int]], batch: list[int],
                     device: str) -> list[tuple[list[int], np.ndarray, int]]:
    """Run ``compute`` on a batch; if the accelerator runs out of memory, halve the batch and retry.

    Returns (indices, outputs, tokens) pieces. A single item that still does not fit re-raises.
    """
    try:
        out, tokens = compute(batch)
        return [(batch, out, tokens)]
    except torch.OutOfMemoryError:
        if len(batch) == 1:
            raise
        free(device)
        print(f"  [oom] batch of {len(batch)} did not fit; splitting", flush=True)
        mid = len(batch) // 2
        return run_split_on_oom(compute, batch[:mid], device) + run_split_on_oom(compute, batch[mid:], device)


def pad_right(id_lists: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for r, x in enumerate(id_lists):
        ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[r, :len(x)] = 1
    return ids, mask


class _Progress:
    def __init__(self, label: str | None, total: int):
        self.label, self.total, self.done, self.tokens = label, total, 0, 0
        self.t0 = self.last = time.time()

    def step(self, n: int, tokens: int) -> None:
        self.done += n
        self.tokens += tokens
        now = time.time()
        if self.label is None:
            return
        if now - self.last > 15 or self.done == self.total:
            dt = max(now - self.t0, 1e-6)
            eta = (self.total - self.done) * dt / max(self.done, 1)
            print(f"  [{self.label}] {self.done}/{self.total}  {self.tokens / dt:,.0f} tok/s  eta {eta:,.0f}s", flush=True)
            self.last = now


class _Base:
    """Shared tokenization and batched last-token pooling."""

    tok = None
    device = "cpu"

    def __init__(self, max_len: int, token_budget: int, max_batch: int):
        self.cap = max_len - 1
        self.token_budget, self.max_batch = token_budget, max_batch

    def _truncate(self, ids: list[int], keep: str) -> list[int]:
        if not ids:
            ids = self.tok(" ", add_special_tokens=False)["input_ids"]
        return ids[:self.cap] if keep == "head" else ids[-self.cap:]

    def ids(self, text: str, keep: str) -> list[int]:
        return self._truncate(self.tok(text, add_special_tokens=False)["input_ids"], keep)

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def embed_ids(self, id_lists: list[list[int]], label: str | None = "embed") -> np.ndarray:
        out = np.zeros((len(id_lists), self.hidden_size), dtype=np.float32)
        prog = _Progress(label, len(id_lists))
        pad = self.tok.pad_token_id

        def compute(batch):
            ids, mask = pad_right([id_lists[i] for i in batch], pad)
            ids, mask = ids.to(self.device), mask.to(self.device)
            return pool_last(self._forward(ids, mask), mask).float().cpu().numpy(), int(mask.sum())

        for batch in length_batches([len(x) for x in id_lists], self.token_budget, self.max_batch):
            for idx, h, tokens in run_split_on_oom(compute, batch, self.device):
                out[idx] = h
                prog.step(len(idx), tokens)
        if not np.isfinite(out).all():
            raise FloatingPointError(f"{label}: non-finite embeddings (try --set dtype=float32)")
        return l2(out)

    def embed_texts(self, texts: list[str], keep: str | list[str] = "tail",
                    label: str = "embed") -> tuple[np.ndarray, int]:
        """-> (embeddings, number of texts that were truncated). ``keep`` is per text or shared."""
        keeps = [keep] * len(texts) if isinstance(keep, str) else keep
        full = self.tok(list(texts), add_special_tokens=False)["input_ids"] if texts else []
        ids = [self._truncate(f, k) for f, k in zip(full, keeps)]
        truncated = sum(len(f) > self.cap for f in full)
        return self.embed_ids(ids, label), truncated

    def batch_parity(self, texts: list[str]) -> float:
        """Min cosine between batched and one-at-a-time embeddings (padding must not leak in)."""
        ids = [self.ids(t, "tail") for t in texts]
        batched = self.embed_ids(ids, None)
        single = np.concatenate([self.embed_ids([x], None) for x in ids])
        return float((batched * single).sum(1).min())


class TextEncoder(_Base):
    """Qwen3 causal LM loaded as its base model (no lm_head)."""

    def __init__(self, repo: str, device: str, dtype: torch.dtype, max_len: int, token_budget: int, max_batch: int):
        from transformers import AutoModel, AutoTokenizer
        super().__init__(max_len, token_budget, max_batch)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = load_model(AutoModel, repo, dtype, device)
        self.hidden_size = self.model.config.hidden_size

    def _forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state


class VLEncoder(_Base):
    """Qwen3-VL with its lm_head (needed for captions and multiple choice); embeddings come from ``.model``."""

    def __init__(self, repo: str, device: str, dtype: torch.dtype, max_len: int, token_budget: int, max_batch: int,
                 min_pixels: int, max_pixels: int):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        super().__init__(max_len, token_budget, max_batch)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(repo)
        self.tok = self.processor.tokenizer
        self.model = load_model(Qwen3VLForConditionalGeneration, repo, dtype, device)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.size = {"longest_edge": max_pixels, "shortest_edge": min_pixels}

    def _forward(self, input_ids, attention_mask, **mm):
        self.model.model.rope_deltas = None           # cached M-RoPE state from a previous call must not leak
        return self.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                                **mm).last_hidden_state

    # ------------------------------------------------------------------ image + text
    def _mm_inputs(self, texts: list[str], images: list, padding_side: str):
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True,
                              padding_side=padding_side, images_kwargs={"size": self.size}).to(self.device)

    @torch.no_grad()
    def _mm_pooled(self, texts: list[str], image_paths: list[str], label: str,
                   head: Callable[[torch.Tensor], torch.Tensor] | None = None) -> np.ndarray:
        """Pooled last-token vectors for image+text prompts (optionally mapped through ``head``)."""
        from PIL import Image
        # Image token counts are unknown before preprocessing; the text length is a good enough sort key.
        lengths = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        per_image = max(1, self.size["longest_edge"] // (32 * 32))
        batches = length_batches([n + per_image for n in lengths], self.token_budget, self.max_batch)
        prog = _Progress(label, len(texts))
        out = None

        def compute(batch):
            imgs = [Image.open(image_paths[i]).convert("RGB") for i in batch]
            inputs = self._mm_inputs([texts[i] for i in batch], imgs, "right")
            mask = inputs.pop("attention_mask")
            h = pool_last(self._forward(inputs.pop("input_ids"), mask, **inputs), mask)
            return (head(h) if head is not None else h).float().cpu().numpy(), int(mask.sum())

        for batch in batches:
            for idx, v, tokens in run_split_on_oom(compute, batch, self.device):
                if out is None:
                    out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
                out[idx] = v
                prog.step(len(idx), tokens)
        if not np.isfinite(out).all():
            raise FloatingPointError(f"{label}: non-finite outputs (try --set dtype=float32)")
        return out

    def embed_image_states(self, texts: list[str], image_paths: list[str]) -> np.ndarray:
        return l2(self._mm_pooled(texts, image_paths, "image states"))

    def image_token_count(self, image_path: str) -> int:
        from PIL import Image
        inputs = self._mm_inputs(["<|vision_start|><|image_pad|><|vision_end|>"],
                                 [Image.open(image_path).convert("RGB")], "right")
        pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        return int((inputs["input_ids"] == pad_id).sum())

    def _chat(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def mc_letter_logits(self, questions: list[str], choices: list[list[str]], image_paths: list[str]) -> np.ndarray:
        """Generative ceiling: next-token logits over the option letters after a standard MC prompt."""
        letters = "ABCDEFGH"
        prompts = []
        for q, cs in zip(questions, choices):
            opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(cs))
            prompts.append(self._chat(f"{q}\n{opts}\nAnswer with the option's letter from the given choices directly."))
        n_opt = max(len(cs) for cs in choices)
        letter_ids = [self.tok.convert_tokens_to_ids(letters[j]) for j in range(n_opt)]
        lm_head = self.model.lm_head
        return self._mm_pooled(prompts, image_paths, "multiple choice",
                               head=lambda h: lm_head(h)[:, letter_ids])

    @torch.no_grad()
    def caption(self, image_paths: list[str], prompt: str, max_new_tokens: int, batch_size: int) -> list[str]:
        from PIL import Image
        text = self._chat(prompt)
        prog = _Progress("captions", len(image_paths))
        captions: dict[int, str] = {}

        def compute(batch):
            imgs = [Image.open(image_paths[i]).convert("RGB") for i in batch]
            inputs = self._mm_inputs([text] * len(batch), imgs, "left")
            self.model.model.rope_deltas = None
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1]:]
            return self.tok.batch_decode(new, skip_special_tokens=True), int(new.numel())

        for s in range(0, len(image_paths), batch_size):
            for idx, caps, tokens in run_split_on_oom(compute, list(range(s, min(s + batch_size, len(image_paths)))),
                                                      self.device):
                captions.update({i: c.strip() for i, c in zip(idx, caps)})
                prog.step(len(idx), tokens)
        return [captions[i] for i in range(len(image_paths))]
