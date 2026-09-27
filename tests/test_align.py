import numpy as np

from cvlm import align
from cvlm.encoders import l2


def _data(n=3000, dx=24, dy=24, seed=0):
    rng = np.random.default_rng(seed)
    X = l2(rng.normal(size=(n, dx)).astype(np.float32))
    return X, rng


def test_ridge_recovers_affine_map():
    X, rng = _data(dy=16)
    W, b = rng.normal(size=(24, 16)), rng.normal(size=16)
    Y = l2((X @ W + b + 0.01 * rng.normal(size=(len(X), 16))).astype(np.float32))
    m, scores = align.fit_ridge(X[:2500], Y[:2500], [1e-6, 1e-4, 1e-2, 1.0])
    assert align.mean_cos(m(X[2500:]), Y[2500:]) > 0.98
    assert m.lam == max(scores, key=scores.get)


def test_procrustes_recovers_rotation():
    X, rng = _data()
    Q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    Y = l2((X @ Q).astype(np.float32))
    m = align.fit_procrustes(X[:2500], Y[:2500])
    assert align.mean_cos(m(X[2500:]), Y[2500:]) > 0.999


def test_identity_only_when_widths_match():
    X, rng = _data()
    maps, _ = align.fit_all(X, X.copy(), [1e-3])
    assert set(maps) == {"identity", "ridge", "procrustes"}
    maps, _ = align.fit_all(X, l2(rng.normal(size=(len(X), 8)).astype(np.float32)), [1e-3])
    assert "identity" not in maps and maps["procrustes"].W.shape == (24, 8)


def test_save_load_roundtrip(tmp_path):
    X, rng = _data(n=500)
    Y = l2(X @ rng.normal(size=(24, 24)).astype(np.float32))
    maps, _ = align.fit_all(X, Y, [1e-3, 1e-1])
    align.save(tmp_path / "maps.npz", maps)
    back = align.load(tmp_path / "maps.npz")
    for k in maps:
        np.testing.assert_allclose(maps[k](X), back[k](X), atol=1e-5)
    assert back["ridge"].lam == maps["ridge"].lam
