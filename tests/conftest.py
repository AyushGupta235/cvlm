import numpy as np
import pytest
from PIL import Image

TINY_TEXT = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_VL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"


@pytest.fixture(scope="session")
def images(tmp_path_factory):
    """Two images of different sizes, so image-token counts differ and batches need padding."""
    d = tmp_path_factory.mktemp("img")
    rng = np.random.default_rng(0)
    paths = []
    for i, (h, w) in enumerate([(480, 640), (256, 320)]):
        p = d / f"{i}.jpg"
        Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8")).save(p)
        paths.append(str(p))
    return paths


@pytest.fixture(scope="session")
def text_encoder():
    import torch
    from cvlm.encoders import TextEncoder
    return TextEncoder(TINY_TEXT, "cpu", torch.float32, max_len=64, token_budget=512, max_batch=4)


@pytest.fixture(scope="session")
def vl_encoder():
    import torch
    from cvlm.encoders import VLEncoder
    return VLEncoder(TINY_VL, "cpu", torch.float32, max_len=64, token_budget=2048, max_batch=4,
                     min_pixels=64 * 64, max_pixels=224 * 224)
