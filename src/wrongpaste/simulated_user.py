from wrongpaste.clients import chat
from wrongpaste.topics import Topic

USER_MODEL = "gpt-5.6-terra-tst"

_SYSTEM = """Eres una persona normal conversando con un asistente.
Escribe UN único mensaje breve (una o dos frases), en primera persona, en español.
No eres un asistente: no ofrezcas ayuda, no resumas, no hagas listas.
Avanza hacia el siguiente objetivo pendiente de forma natural, reaccionando a
lo que te acaban de decir. No menciones nunca que tienes objetivos."""


def next_user_turn(topic: Topic, history: list[dict]) -> str:
    pending = "\n".join(f"- {g}" for g in topic.goals)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (
        f"Tema de la conversación: {topic.opening}\n\n"
        f"Cosas que quieres acabar sabiendo:\n{pending}\n\n"
        f"Conversación hasta ahora:\n{transcript}\n\n"
        f"Escribe tu siguiente mensaje."
    )
    reply = chat(
        USER_MODEL,
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return reply.text.strip()
