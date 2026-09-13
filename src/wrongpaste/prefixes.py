"""Prefijos compartidos: uno por (tema, longitud), reproducido a todos (D1).

El eje x del experimento es la similaridad entre el pegote y la conversación
previa. Si cada modelo evaluado se fabrica su propio prefijo, ese eje depende de
quién conteste — Opus diluye el embedding con mil tokens de prosa y `luna` no —
y las curvas dejan de ser comparables entre modelos, que es justo lo que el
spec §9 quiere comparar.

Por eso el prefijo se genera **una vez** por (tema, longitud), con el usuario
simulado y un **modelo de asistente fijo** (`PREFIX_MODEL`, el mismo que hace de
usuario), se persiste en `runs/prefixes/<prefix_id>.json` y se le sirve idéntico
a todos los modelos evaluados.

Coste declarado: el modelo evaluado no reacciona a sus propias palabras, sino a
las de otro. Es menos natural, y va en el artículo como limitación (D1).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from wrongpaste.clients import Reply
from wrongpaste.conversation import build_prefix, reply_traces
from wrongpaste.simulated_user import USER_MODEL
from wrongpaste.topics import Topic

# Modelo que hace de asistente al fabricar el prefijo. Es el mismo que hace de
# usuario simulado a propósito: un solo cuerpo barato fabrica la conversación
# entera, y ninguno de los modelos evaluados influye en el eje x.
PREFIX_MODEL = "gpt-5.6-terra-tst"

# Dónde viven los prefijos ya fabricados. Se lee como global en cada función
# para que los tests lo puedan redirigir a un `tmp_path`.
PREFIX_DIR = Path(__file__).resolve().parents[2] / "runs" / "prefixes"

# Longitud en hex del identificador. 16 hex = 64 bits: de sobra para ocho
# prefijos y corto como para caber en un nombre de fichero legible.
PREFIX_ID_LEN = 16


def prefix_id(
    topic_id: str, n_turns: int, prefix_model: str, transcript: list[dict]
) -> str:
    """Huella del prefijo: sha256 de su serialización determinista, 16 hex.

    Entra todo lo que puede cambiar el contexto que ve el modelo evaluado: el
    tema, la longitud, quién hizo de asistente y la transcripción completa
    (incluidas las etiquetas `tag`). Dos prefijos con el mismo `prefix_id` son
    el mismo contexto, palabra por palabra.
    """
    payload = {
        "topic_id": topic_id,
        "n_turns": int(n_turns),
        "prefix_model": prefix_model,
        "transcript": [
            {
                "role": m["role"],
                "content": m["content"],
                "tag": m.get("tag", ""),
            }
            for m in transcript
        ],
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:PREFIX_ID_LEN]


def generate_prefix(topic: Topic, n_turns: int) -> dict[str, Any]:
    """Fabrica un prefijo nuevo con `PREFIX_MODEL` de asistente.

    No lo guarda: eso es cosa de `save_prefix`, para que los tests y el paso de
    medición de ejes (D12) puedan generar sin ensuciar el repo.

    `conversation.build_prefix` devuelve **tres** valores desde D5:
    `(transcript, usages, replies)`. Los `Reply` NO se guardan tal cual —
    arrastran `raw`, la respuesta entera del proveedor, que no es serializable
    a JSON— sino que pasan por `conversation.reply_traces()`, que deja el
    `stop_reason`, el `response_model`, los `attempts` y la latencia de cada
    llamada. El prefijo también es una tirada contra un modelo y también hay
    que poder auditar si alguno de sus turnos se cortó por `max_tokens` o
    necesitó reintentos (D5/D6).

    **Las dos mitades del prefijo se auditan por separado.** En un prefijo
    hablan dos papeles: el asistente (`PREFIX_MODEL`) y el usuario simulado. Sus
    trazas van en `reply_traces` y `user_reply_traces` respectivamente. Sin la
    segunda —que es lo que pasaba antes, porque `build_prefix` se llamaba sin
    `user_replies_out`— el `stop_reason` de los turnos del usuario simulado no
    llegaba a ninguna parte: un turno suyo cortado por el tope se quedaba dentro
    del contexto compartido, se le servía idéntico a los tres modelos evaluados
    (D1) y ninguna fila del JSONL podía decirlo. `run_phase0` lee
    `user_reply_traces` al abrir un prefijo y manda la celda a `harness_error`
    si trae un turno roto.
    """
    user_replies: list[Reply] = []
    transcript, usages, replies = build_prefix(
        PREFIX_MODEL, topic, n_turns, user_replies_out=user_replies
    )
    return {
        "prefix_id": prefix_id(topic.id, n_turns, PREFIX_MODEL, transcript),
        "topic_id": topic.id,
        "n_turns": int(n_turns),
        "prefix_model": PREFIX_MODEL,
        "user_model": USER_MODEL,
        "transcript": transcript,
        "usages": usages,
        "reply_traces": reply_traces(replies),
        "user_reply_traces": reply_traces(user_replies),
    }


def prefix_path(pid: str) -> Path:
    """Ruta del fichero de un prefijo, exista o no."""
    return PREFIX_DIR / f"{pid}.json"


def save_prefix(prefix: dict[str, Any]) -> Path:
    """Persiste el prefijo en `runs/prefixes/<prefix_id>.json`."""
    path = prefix_path(prefix["prefix_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prefix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_prefix(pid: str) -> dict[str, Any]:
    """Lee un prefijo ya guardado. Revienta si no está: no lo regenera solo."""
    return json.loads(prefix_path(pid).read_text(encoding="utf-8"))


def find_prefix(
    topic_id: str, n_turns: int, prefix_model: str = PREFIX_MODEL
) -> dict[str, Any] | None:
    """Busca en disco el prefijo de esa celda del diseño; None si no hay.

    Se busca por (tema, longitud, modelo) y no por `prefix_id` porque el id
    solo se conoce *después* de generar la transcripción. Se recorre el
    directorio en orden de nombre para que dos ejecuciones elijan el mismo si
    por lo que fuera hubiera dos candidatos.
    """
    if not PREFIX_DIR.is_dir():
        return None
    for path in sorted(PREFIX_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("topic_id") == topic_id
            and data.get("n_turns") == int(n_turns)
            and data.get("prefix_model") == prefix_model
        ):
            return data
    return None


def ensure_prefix(topic: Topic, n_turns: int) -> dict[str, Any]:
    """Devuelve el prefijo de (tema, longitud), generándolo solo si falta.

    Es la puerta que usa el runner: idempotente y barata al reanudar una
    tirada, y garantía de que los tres modelos de la Fase 0 ven exactamente el
    mismo contexto antes del pegote (D1).
    """
    existing = find_prefix(topic.id, n_turns, PREFIX_MODEL)
    if existing is not None:
        return existing
    prefix = generate_prefix(topic, n_turns)
    save_prefix(prefix)
    return prefix
