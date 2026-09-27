import numpy as np
import pytest

from cvlm import config, evaluate
from cvlm.align import Map
from cvlm.encoders import l2
from cvlm.scoring import Scorer


def _perfect(n_items=30, n_opt=4, d=16, seed=0):
    """Candidates whose correct option embedding equals the state embedding."""
    rng = np.random.default_rng(seed)
    cands = [f"c{i}" for i in range(n_items * n_opt)]
    A = l2(rng.normal(size=(len(cands), d)).astype(np.float32))
    items, S = [], []
    for i in range(n_items):
        opts = cands[i * n_opt:(i + 1) * n_opt]
        label = int(rng.integers(n_opt))
        items.append((opts, label))
        S.append(A[i * n_opt + label])
    return items, np.stack(S), A, {c: j for j, c in enumerate(cands)}


def test_bootstrap_ci_brackets_mean():
    x = np.r_[np.ones(70), np.zeros(30)]
    lo, hi = evaluate.bootstrap_ci(x, 500, 0)
    assert lo < 0.7 < hi


def test_typed_decisions_perfect_embeddings_score_100():
    items, S, A, index = _perfect()
    td = [{"candidates": o, "label": lab, "target": np.eye(4)[lab].tolist(), "type": "choice"} for o, lab in items]
    E = {"text/td_states": S, "text/td_cands": A, "vl/td_states": S, "vl/td_cands": A}
    out = evaluate.typed_decisions(td, E, index, {"identity": Map("identity")}, Scorer("raw"), 100, 0)
    assert out["ref"]["acc"] == 1.0 and out["vl_identity"]["agree_ref"] == 1.0


def test_aokvqa_conditions_and_deltas():
    items, S, A, index = _perfect()
    ok = [{"choices": o, "label": lab} for o, lab in items]
    rng = np.random.default_rng(1)
    noise = l2(rng.normal(size=S.shape).astype(np.float32))
    E = {"text/ok_q": noise, "text/ok_cap": S, "text/ok_choices": A, "vl/ok_img": S, "vl/ok_choices": A}
    mc = np.eye(4)[[lab for _, lab in items]] * 5
    out = evaluate.aokvqa(ok, E, index, {"identity": Map("identity")}, Scorer("raw"), mc, 100, 0)
    assert out["caption_then_clm"]["acc"] == out["vl_identity"]["acc"] == out["vl_generative"]["acc"] == 1.0
    assert out["chance"]["acc"] == 0.25
    assert out["vl_identity"]["delta"]["question_only"]["mean"] > 0


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ntext_model: a\nvl_model: b\n")
    c = config.load(str(p), ["fit_limit=10", "td_limit=none", "ridge_lambdas=0.1,1", "device=cpu"])
    assert (c.fit_limit, c.td_limit, c.ridge_lambdas, c.device) == (10, None, [0.1, 1.0], "cpu")
    with pytest.raises(ValueError):
        config.load(str(p), ["nope=1"])


def test_clm_head_loads_and_scores():
    """The real CLM-v0.1-8B reference head (75 MB, downloaded once into ~/.cache/clm)."""
    try:
        s = Scorer("clm")
    except Exception as e:  # offline
        pytest.skip(f"CLM head unavailable: {e}")
    rng = np.random.default_rng(0)
    E = l2(rng.normal(size=(5, 4096)).astype(np.float32))
    zs, za = s.states(E), s.actions(E)
    assert zs.shape == za.shape == (5, s.pair.proj_dim)
    np.testing.assert_allclose(np.linalg.norm(zs, axis=1), 1.0, atol=1e-5)
    assert 1.0 < s.scale <= 100.0
    with pytest.raises(ValueError):
        s.states(E[:, :2048])
