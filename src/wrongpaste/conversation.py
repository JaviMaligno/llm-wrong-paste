"""Construcción de la conversación: prefijo, pegote y los dos turnos de después.

Cada mensaje del transcript lleva una etiqueta `tag` ∈ MESSAGE_TAGS (D5), que es
lo que permite al análisis saber qué escribió el usuario simulado, qué es el
pegote y qué vino después sin contar posiciones a mano.

Ojo: `tag` es metadato nuestro, no del protocolo. Todas las llamadas pasan por
`api_messages()`, que deja solo `role` y `content`, porque los proveedores
rechazan campos desconocidos en `messages`.
"""

from wrongpaste.artifacts import Artifact
from wrongpaste.clients import chat
from wrongpaste.simulated_user import next_user_turn
from wrongpaste.topics import Topic

MAX_TOKENS = 1024

# Cuántos turnos de usuario van después de la reacción al pegote (D7).
N_POST_TURNS = 2


def api_messages(transcript: list[dict]) -> list[dict]:
    """La transcripción tal y como se le manda al proveedor: sin `tag`."""
    return [{"role": m["role"], "content": m["content"]} for m in transcript]


def build_prefix(
    model_id: str, topic: Topic, n_turns: int
) -> tuple[list[dict], list[dict]]:
    """Conduce n_turns de conversación normal sobre el tema.

    Devuelve la transcripción (siempre terminada en assistant, que es lo que
    Claude exige para poder continuar) y los `usage` de cada llamada.

    Etiquetas (D5): el mensaje de apertura es `opening`, los demás mensajes de
    usuario son `user_sim` (los escribe el usuario simulado) y las respuestas
    son `assistant`.

    En la Fase 0 este prefijo lo fabrica siempre `PREFIX_MODEL` y se comparte
    entre modelos evaluados (D1); ver `wrongpaste.prefixes`.
    """
    transcript: list[dict] = [
        {"role": "user", "content": topic.opening, "tag": "opening"}
    ]
    usages: list[dict] = []

    reply = chat(model_id, api_messages(transcript), max_tokens=MAX_TOKENS)
    transcript.append({"role": "assistant", "content": reply.text, "tag": "assistant"})
    usages.append(reply.usage)

    for _ in range(n_turns - 1):
        transcript.append(
            {
                "role": "user",
                "content": next_user_turn(topic, transcript),
                "tag": "user_sim",
            }
        )
        reply = chat(model_id, api_messages(transcript), max_tokens=MAX_TOKENS)
        transcript.append(
            {"role": "assistant", "content": reply.text, "tag": "assistant"}
        )
        usages.append(reply.usage)

    return transcript, usages


def conversation_text(transcript: list[dict]) -> str:
    """La conversación entera como texto (similaridad secundaria, D2)."""
    return "\n".join(m["content"] for m in transcript)


def inject_paste(
    model_id: str, transcript: list[dict], artifact: Artifact
) -> tuple[str, dict, int]:
    """Añade el pegote TAL CUAL, sin preámbulo ni envoltorio, y pide respuesta.

    Muta `transcript` in place: el registro guarda la conversación completa.

    Devuelve (reacción, usage, índice del mensaje del pegote). El índice va al
    campo `paste_index` de la fila (D5): sin él, el análisis tendría que volver
    a localizar el pegote comparando textos.
    """
    paste_index = len(transcript)
    transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
    reply = chat(model_id, api_messages(transcript), max_tokens=MAX_TOKENS)
    transcript.append({"role": "assistant", "content": reply.text, "tag": "assistant"})
    return reply.text, reply.usage, paste_index


def continue_after_paste(
    model_id: str,
    transcript: list[dict],
    topic: Topic,
    n_post: int = N_POST_TURNS,
) -> tuple[list[int], list[dict]]:
    """Dos turnos más con el usuario simulado, SIN reparación (D7).

    Es la variante (a) del spec §7: el usuario sigue a lo suyo como si el
    pegote no existiera. Sirve para separar "pivota en silencio" de "puente
    confabulado" — que a menudo solo se distinguen en el turno +1 — y para
    pilotar ya el recuento de entidades de la Fase 2.

    Muta `transcript` in place. El mensaje de usuario se etiqueta `post` y la
    respuesta `assistant`.

    Devuelve (post_indices, usages). `post_indices` son los índices de **todos**
    los mensajes añadidos aquí, usuario y asistente, en orden: el recuento de
    fuga de entidades se hace sobre las respuestas, así que dejarlas fuera
    obligaría al análisis a recalcular posiciones.
    """
    post_indices: list[int] = []
    usages: list[dict] = []

    for _ in range(n_post):
        post_indices.append(len(transcript))
        transcript.append(
            {
                "role": "user",
                "content": next_user_turn(topic, transcript),
                "tag": "post",
            }
        )
        reply = chat(model_id, api_messages(transcript), max_tokens=MAX_TOKENS)
        post_indices.append(len(transcript))
        transcript.append(
            {"role": "assistant", "content": reply.text, "tag": "assistant"}
        )
        usages.append(reply.usage)

    return post_indices, usages
