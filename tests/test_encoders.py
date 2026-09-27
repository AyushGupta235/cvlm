import numpy as np
import pytest
import torch

from cvlm.encoders import length_batches, pad_right, pool_last

TEXTS = ["short", "a somewhat longer piece of text for padding",
         "The agent rotated the TLS certificate.\n\nHow did this agent run turn out?", "ok"]


def test_length_batches_cover_once_and_respect_budget():
    lengths = [5, 50, 7, 30, 30, 1, 64, 12]
    batches = length_batches(lengths, token_budget=64, max_batch=3)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 3
        assert max(lengths[i] for i in b) * len(b) <= 64 or len(b) == 1


def test_pool_last_right_padding_only():
    ids, mask = pad_right([[1, 2, 3], [4]], pad_id=0)
    h = torch.arange(2 * 3, dtype=torch.float32).view(2, 3, 1)
    assert pool_last(h, mask).squeeze(-1).tolist() == [2.0, 3.0]
    with pytest.raises(ValueError):
        pool_last(h, mask.flip(1))


def test_text_batched_equals_single(text_encoder):
    assert text_encoder.batch_parity(TEXTS) > 0.9999


def test_truncation_keeps_head_or_tail(text_encoder):
    text = " ".join(f"w{i}" for i in range(200))
    full = text_encoder.tok(text, add_special_tokens=False)["input_ids"]
    assert text_encoder.ids(text, "head") == full[:63]
    assert text_encoder.ids(text, "tail") == full[-63:]
    _, truncated = text_encoder.embed_texts([text, "short"], "tail", label=None)
    assert truncated == 1


def test_vl_text_batched_equals_single(vl_encoder):
    assert vl_encoder.batch_parity(TEXTS) > 0.9999


def test_tokenizers_agree(text_encoder, vl_encoder):
    for t in TEXTS:
        assert text_encoder.ids(t, "tail") == vl_encoder.ids(t, "tail")


def test_image_states_batched_equals_single(vl_encoder, images):
    from cvlm.config import IMAGE_TOKENS
    texts = [IMAGE_TOKENS + "\n\nWhat is shown?", IMAGE_TOKENS + "\n\nA longer question about this picture?"]
    assert vl_encoder.image_token_count(images[0]) != vl_encoder.image_token_count(images[1])
    batched = vl_encoder.embed_image_states(texts, images)
    single = np.concatenate([vl_encoder.embed_image_states([t], [p]) for t, p in zip(texts, images)])
    assert batched.shape == (2, vl_encoder.hidden_size)
    assert (batched * single).sum(1).min() > 0.9999
    np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1.0, atol=1e-5)


def test_mc_logits_and_captions(vl_encoder, images):
    logits = vl_encoder.mc_letter_logits(["Which?", "What?"], [["a", "b", "c", "d"], ["e", "f", "g", "h"]], images)
    assert logits.shape == (2, 4) and np.isfinite(logits).all()
    caps = vl_encoder.caption(images, "Describe.", max_new_tokens=4, batch_size=2)
    assert len(caps) == 2 and all(isinstance(c, str) for c in caps)


def test_oom_splits_batch_and_keeps_order():
    from cvlm.encoders import run_split_on_oom

    def compute(batch):
        if len(batch) > 2:
            raise torch.OutOfMemoryError("fake")
        return np.array([[i] for i in batch], dtype=np.float32), len(batch)

    pieces = run_split_on_oom(compute, [0, 1, 2, 3, 4], "cpu")
    assert [i for idx, _, _ in pieces for i in idx] == [0, 1, 2, 3, 4]
    assert all(len(idx) <= 2 and out[:, 0].tolist() == idx for idx, out, _ in pieces)
    with pytest.raises(torch.OutOfMemoryError):
        run_split_on_oom(lambda b: (_ for _ in ()).throw(torch.OutOfMemoryError("x")), [0], "cpu")
