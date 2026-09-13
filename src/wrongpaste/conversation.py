from wrongpaste.artifacts import Artifact
from wrongpaste.clients import chat
from wrongpaste.simulated_user import next_user_turn
from wrongpaste.topics import Topic

MAX_TOKENS = 1024


def build_prefix(
    model_id: str, topic: Topic, n_turns: int
) -> tuple[list[dict], list[dict]]:
    """Conduce n_turns de conversación normal sobre el tema.

    Devuelve la transcripción (siempre terminada en assistant, que es lo que
    Claude exige para poder continuar) y los `usage` de cada llamada.
    """
    transcript: list[dict] = [{"role": "user", "content": topic.opening}]
    usages: list[dict] = []

    reply = chat(model_id, transcript, max_tokens=MAX_TOKENS)
    transcript.append({"role": "assistant", "content": reply.text})
    usages.append(reply.usage)

    for _ in range(n_turns - 1):
        transcript.append({"role": "user", "content": next_user_turn(topic, transcript)})
        reply = chat(model_id, transcript, max_tokens=MAX_TOKENS)
        transcript.append({"role": "assistant", "content": reply.text})
        usages.append(reply.usage)

    return transcript, usages


def conversation_text(transcript: list[dict]) -> str:
    """El texto contra el que se mide la similaridad del banco."""
    return "\n".join(m["content"] for m in transcript)


def inject_paste(
    model_id: str, transcript: list[dict], artifact: Artifact
) -> tuple[str, dict]:
    """Añade el pegote TAL CUAL, sin preámbulo ni envoltorio, y pide respuesta.

    Muta `transcript` in place: el registro guarda la conversación completa.
    """
    transcript.append({"role": "user", "content": artifact.text})
    reply = chat(model_id, transcript, max_tokens=MAX_TOKENS)
    transcript.append({"role": "assistant", "content": reply.text})
    return reply.text, reply.usage
