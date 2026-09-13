import numpy as np

from wrongpaste.artifacts import Artifact
from wrongpaste.similarity import cosine, stratified_pick


def test_cosine_of_identical_vectors_is_one():
    v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.isclose(cosine(v, v)[0][0], 1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.isclose(cosine(a, b)[0][0], 0.0)


def _fake_ranked(n):
    arts = [Artifact(f"a{i}", "k", "t", ("e",)) for i in range(n)]
    return [(a, i / (n - 1)) for i, a in enumerate(arts)]


def test_stratified_pick_covers_the_range():
    ranked = _fake_ranked(100)
    rng = np.random.default_rng(0)
    picked = stratified_pick(ranked, k=12, rng=rng)
    sims = sorted(s for _, s in picked)
    assert len(picked) == 12
    assert sims[0] < 0.2, "el estrato bajo no está representado"
    assert sims[-1] > 0.8, "el estrato alto no está representado"


def test_stratified_pick_is_deterministic_for_a_seed():
    ranked = _fake_ranked(100)
    one = stratified_pick(ranked, k=12, rng=np.random.default_rng(7))
    two = stratified_pick(ranked, k=12, rng=np.random.default_rng(7))
    assert [a.id for a, _ in one] == [a.id for a, _ in two]
