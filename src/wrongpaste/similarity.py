import numpy as np

from wrongpaste.artifacts import Artifact
from wrongpaste.clients import embed


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def rank_artifacts(
    conversation_text: str, arts: list[Artifact]
) -> list[tuple[Artifact, float]]:
    vecs = embed([conversation_text] + [a.text for a in arts])
    sims = cosine(vecs[:1], vecs[1:])[0]
    return sorted(zip(arts, (float(s) for s in sims)), key=lambda p: p[1])


def stratified_pick(
    ranked: list[tuple[Artifact, float]], k: int, rng: np.random.Generator
) -> list[tuple[Artifact, float]]:
    """Un artefacto por estrato de igual anchura sobre el rango observado.

    Estratifica por posición en el ranking, no por valor de similaridad: la
    distribución real está muy concentrada y estratificar por valor dejaría
    estratos vacíos.
    """
    if len(ranked) < k:
        raise ValueError(f"el banco tiene {len(ranked)} artefactos, se piden {k}")
    edges = np.linspace(0, len(ranked), k + 1).astype(int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = int(rng.integers(lo, max(hi, lo + 1)))
        picked.append(ranked[idx])
    return picked
