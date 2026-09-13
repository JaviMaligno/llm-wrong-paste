"""Tests de similaridad. Ninguno llama al endpoint de embeddings.

Donde hace falta embeber se monkeypatchea `wrongpaste.similarity.embed` por una
función determinista: lo que se está probando es la selección de texto (D2), la
guarda de longitud y el orden del ranking, no el embedder.
"""

import numpy as np
import pytest

from wrongpaste.artifacts import Artifact
from wrongpaste.similarity import (
    MAX_EMBED_CHARS,
    cosine,
    rank_artifacts,
    rank_stats,
    stratified_pick,
    truncate_for_embedding,
    user_text,
)


def test_cosine_of_identical_vectors_is_one():
    v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.isclose(cosine(v, v)[0][0], 1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.isclose(cosine(a, b)[0][0], 0.0)


# --- user_text (D2) -------------------------------------------------------


def test_user_text_ignores_assistant_turns():
    transcript = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "PROSA DEL MODELO"},
        {"role": "user", "content": "y el horno?"},
        {"role": "assistant", "content": "MÁS PROSA"},
    ]
    assert user_text(transcript) == "hola\ny el horno?"


def test_user_text_keeps_the_order_of_the_user_turns():
    transcript = [
        {"role": "user", "content": "uno"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "dos"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "tres"},
    ]
    assert user_text(transcript) == "uno\ndos\ntres"


def test_user_text_selects_opening_and_user_sim_by_tag():
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "prosa", "tag": "assistant"},
        {"role": "user", "content": "seguimiento", "tag": "user_sim"},
    ]
    assert user_text(transcript) == "apertura\nseguimiento"


def test_user_text_excludes_the_paste_and_the_repair():
    # `paste` y `repair` son role=user pero posteriores al pegote: no definen
    # de qué va la conversación, así que no entran en la similaridad.
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "prosa", "tag": "assistant"},
        {"role": "user", "content": "STACKTRACE", "tag": "paste"},
        {"role": "assistant", "content": "reacción", "tag": "assistant"},
        {"role": "user", "content": "perdón, pegote", "tag": "repair"},
    ]
    assert user_text(transcript) == "apertura"


def test_user_text_of_a_transcript_without_user_turns_is_empty():
    assert user_text([{"role": "assistant", "content": "solo prosa"}]) == ""


# --- truncate_for_embedding (D2) ------------------------------------------


def test_truncate_for_embedding_leaves_short_text_alone():
    text, truncated = truncate_for_embedding("corto", max_chars=100)
    assert text == "corto"
    assert truncated is False


def test_truncate_for_embedding_keeps_the_end():
    text = "INICIO" + "x" * 100 + "FINAL"
    clipped, truncated = truncate_for_embedding(text, max_chars=10)
    assert truncated is True
    assert len(clipped) == 10
    assert clipped.endswith("FINAL")
    assert "INICIO" not in clipped
    assert text.endswith(clipped)


def test_truncate_for_embedding_at_the_exact_limit_does_not_truncate():
    text = "a" * 24
    clipped, truncated = truncate_for_embedding(text, max_chars=24)
    assert clipped == text
    assert truncated is False


def test_truncate_for_embedding_default_is_the_documented_guard():
    text = "b" * (MAX_EMBED_CHARS + 500)
    clipped, truncated = truncate_for_embedding(text)
    assert truncated is True
    assert len(clipped) == MAX_EMBED_CHARS == 24_000


def test_truncate_for_embedding_rejects_a_non_positive_limit():
    # Sin la guarda, `text[-0:]` devolvería el texto entero sin avisar.
    with pytest.raises(ValueError):
        truncate_for_embedding("hola", max_chars=0)


# --- rank_artifacts -------------------------------------------------------


def _arts(n: int) -> list[Artifact]:
    return [Artifact(f"a{i}", "k", f"texto {i}", ("e",)) for i in range(n)]


def _fake_embed(calls: list[list[str]]):
    """Embedder falso: el artefacto i apunta cada vez más lejos del texto.

    Registra en `calls` los textos recibidos para poder comprobar que se
    embebió el texto ya recortado.
    """

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        vecs = [[1.0, 0.0]]
        for i in range(len(texts) - 1):
            angle = (i + 1) * (np.pi / 4) / len(texts)
            vecs.append([float(np.cos(angle)), float(np.sin(angle))])
        return np.array(vecs, dtype=np.float32)

    return embed


def test_rank_artifacts_returns_ranking_and_truncation_flag(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("wrongpaste.similarity.embed", _fake_embed(calls))

    ranking, truncated = rank_artifacts("de qué va esto", _arts(4))

    assert truncated is False
    assert [a.id for a, _ in ranking] == ["a3", "a2", "a1", "a0"]
    assert all(isinstance(s, float) for _, s in ranking)
    sims = [s for _, s in ranking]
    assert sims == sorted(sims), "el ranking va de menor a mayor coseno"
    assert calls == [["de qué va esto", "texto 0", "texto 1", "texto 2", "texto 3"]]


def test_rank_artifacts_truncates_before_embedding(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("wrongpaste.similarity.embed", _fake_embed(calls))

    text = "PRINCIPIO" + "z" * 50 + "COLA"
    _, truncated = rank_artifacts(text, _arts(2), max_chars=20)

    assert truncated is True
    embedded = calls[0][0]
    assert len(embedded) == 20
    assert embedded.endswith("COLA")
    assert "PRINCIPIO" not in embedded


# --- rank_stats (D12) -----------------------------------------------------


def _ranked_from(sims: list[float]) -> list[tuple[Artifact, float]]:
    arts = _arts(len(sims))
    return list(zip(arts, sims))


def test_rank_stats_computes_min_median_and_max():
    stats = rank_stats(_ranked_from([0.1, 0.2, 0.3, 0.4, 0.9]))
    assert stats == {"min": 0.1, "median": 0.3, "max": 0.9}


def test_rank_stats_median_of_an_even_number_averages_the_middle_pair():
    stats = rank_stats(_ranked_from([0.0, 0.2, 0.4, 1.0]))
    assert stats["median"] == pytest.approx(0.3)
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0


def test_rank_stats_does_not_depend_on_the_input_order():
    ordered = rank_stats(_ranked_from([0.1, 0.5, 0.7]))
    shuffled = rank_stats(_ranked_from([0.7, 0.1, 0.5]))
    assert ordered == shuffled


def test_rank_stats_of_a_single_artifact_collapses_to_one_value():
    assert rank_stats(_ranked_from([0.42])) == {
        "min": 0.42,
        "median": 0.42,
        "max": 0.42,
    }


def test_rank_stats_rejects_an_empty_ranking():
    with pytest.raises(ValueError):
        rank_stats([])


def test_rank_stats_measures_the_width_of_the_axis(monkeypatch):
    # D12: el ancho del rango es criterio de GO/NO-GO, así que tiene que salir
    # del mismo ranking que usa el runner.
    monkeypatch.setattr("wrongpaste.similarity.embed", _fake_embed([]))
    ranking, _ = rank_artifacts("tema", _arts(8))
    stats = rank_stats(ranking)
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["max"] - stats["min"] > 0


# --- stratified_pick (intacta, la usa la Fase 1) --------------------------


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
