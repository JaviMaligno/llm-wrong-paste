"""Similaridad coseno entre la conversación y el banco de artefactos.

D2: la similaridad **primaria** se mide contra el lado del usuario (apertura +
turnos del usuario simulado), no contra la conversación entera. Dos razones:

1. El texto del asistente lo genera el modelo bajo prueba, así que embeber la
   conversación completa hace que el mismo artefacto reciba cosenos distintos
   según quién conteste — y el eje x dejaría de ser comparable entre modelos.
2. La conversación entera se pasa del límite de 8.191 tokens del embedder en el
   brazo de 10 turnos, con truncado silencioso *del final*, que es justo la
   parte que define de qué va la charla ahora mismo.

De ahí la guarda de longitud: se recorta por el principio y se registra que se
recortó (`similarity_text_truncated` en la fila del JSONL).
"""

import numpy as np

from wrongpaste.artifacts import Artifact
from wrongpaste.clients import embed

# Tope de caracteres antes de embeber (D2): ≈6.000 tokens estimados, holgado
# frente a los 8.191 del embedder incluso con texto denso en símbolos.
MAX_EMBED_CHARS = 24_000

# Etiquetas de mensaje que cuentan como "lado del usuario" a efectos de
# similaridad. `paste` y `repair` también son role=user, pero son posteriores
# al pegote: entran en la conversación, no en la definición del tema.
USER_TAGS: frozenset[str] = frozenset({"opening", "user_sim"})


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def user_text(transcript: list[dict]) -> str:
    """Concatena, en orden, solo los turnos del lado del usuario (D2).

    Un mensaje cuenta si lleva `tag` en `USER_TAGS` (apertura o usuario
    simulado); los mensajes sin `tag` —transcripciones crudas, anteriores a
    D5— se seleccionan por `role == "user"`. Así el pegote y la reparación,
    que van etiquetados, quedan fuera aunque su role sea `user`.
    """
    parts = []
    for message in transcript:
        tag = message.get("tag")
        if tag is None:
            if message.get("role") == "user":
                parts.append(message["content"])
        elif tag in USER_TAGS:
            parts.append(message["content"])
    return "\n".join(parts)


def truncate_for_embedding(
    text: str, max_chars: int = MAX_EMBED_CHARS
) -> tuple[str, bool]:
    """Recorta **por el principio** conservando el final (guarda de D2).

    Devuelve `(texto, truncado)`. Se conserva el final porque es lo que fija de
    qué se está hablando ahora: recortar la cola dejaría la similaridad anclada
    a una apertura que la conversación ya ha abandonado.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars tiene que ser positivo, recibí {max_chars}")
    if len(text) <= max_chars:
        return text, False
    return text[-max_chars:], True


def rank_artifacts(
    text: str, arts: list[Artifact], max_chars: int = MAX_EMBED_CHARS
) -> tuple[list[tuple[Artifact, float]], bool]:
    """Ordena el banco por coseno ascendente contra `text`.

    Aplica la guarda de longitud antes de embeber. Devuelve el ranking (lista
    de `(Artifact, coseno)` de menor a mayor) y un booleano de si hubo que
    recortar el texto, que el runner copia a la fila del JSONL.
    """
    clipped, truncated = truncate_for_embedding(text, max_chars=max_chars)
    vecs = embed([clipped] + [a.text for a in arts])
    sims = cosine(vecs[:1], vecs[1:])[0]
    ranking = sorted(zip(arts, (float(s) for s in sims)), key=lambda p: p[1])
    return ranking, truncated


def rank_stats(ranking: list[tuple[Artifact, float]]) -> dict:
    """Mínimo, mediana y máximo del coseno observado en un ranking (D12).

    Es la medida del **ancho del eje**: si un tema tiene un rango estrecho, la
    estratificación por similaridad no separa nada y el GO/NO-GO tiene que
    verlo antes de gastar el presupuesto.
    """
    if not ranking:
        raise ValueError("el ranking está vacío: no hay estadísticos que dar")
    sims = np.array([s for _, s in ranking], dtype=float)
    return {
        "min": float(sims.min()),
        "median": float(np.median(sims)),
        "max": float(sims.max()),
    }


def stratified_pick(
    ranked: list[tuple[Artifact, float]], k: int, rng: np.random.Generator
) -> list[tuple[Artifact, float]]:
    """Un artefacto por estrato de igual anchura sobre el rango observado.

    Estratifica por posición en el ranking, no por valor de similaridad: la
    distribución real está muy concentrada y estratificar por valor dejaría
    estratos vacíos.

    La Fase 0 **no** la usa (`run_phase0.main()` inlinea su propia lógica para
    garantizar la cobertura de `kind` de D4); queda para la Fase 1.
    """
    if len(ranked) < k:
        raise ValueError(f"el banco tiene {len(ranked)} artefactos, se piden {k}")
    edges = np.linspace(0, len(ranked), k + 1).astype(int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = int(rng.integers(lo, max(hi, lo + 1)))
        picked.append(ranked[idx])
    return picked
