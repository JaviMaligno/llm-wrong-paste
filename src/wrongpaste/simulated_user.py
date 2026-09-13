"""El usuario simulado: quien escribe los turnos de usuario que no son el pegote.

Lo paga un modelo distinto del evaluado (`USER_MODEL`), porque sus turnos no son
conducta del modelo bajo prueba sino parte del decorado (D1).

**Lo que este módulo NO puede ver.** La variante (a) del spec §7, fijada en D7,
dice literalmente que tras el pegote «el usuario sigue con el tema como si el
pegote no existiera». Si el histórico que se serializa aquí incluyera el mensaje
etiquetado `paste` (o, en la Fase 2, el `repair`) y la reacción inmediata a él,
el usuario simulado lo leería y podría comentarlo, disculparse o pivotar: el
brazo «sin reparación» se convertiría en silencio en un brazo de reparación y la
comparación de D7 dejaría de medir lo que dice medir.

Por eso `next_user_turn` **rechaza** un histórico que traiga esas etiquetas en
vez de tragárselo: quien llame tiene que filtrarlo antes con
`conversation.user_visible_history`. El modelo evaluado sí sigue viendo la
conversación entera —la contaminación que mide el experimento es la suya, no la
del usuario simulado—, y por eso el filtro se aplica a la vista del usuario, no
al transcript.

**Los fallos del usuario simulado no pueden ser invisibles.** Este módulo no
habla con el modelo bajo prueba: si su turno sale vacío o cortado por el tope de
tokens, la conversación que luego se mide está rota, y una fila que dijera `ok`
estaría certificando un dato que no existe. Peor aún, un mensaje de usuario
vacío es un 400 seguro en `vertex_anthropic`, así que devolverlo tal cual
convierte un fallo del arnés en un error HTTP atribuido al modelo evaluado tres
llamadas más tarde. Por eso `next_user_turn` levanta `SimulatedUserError` en vez
de devolver un turno inservible: el runner la traduce a `harness_error` (D6), que
es el único status que no se puede confundir con conducta del modelo evaluado.
"""

from typing import Any

from wrongpaste.clients import Reply, chat
from wrongpaste.topics import Topic

USER_MODEL = "gpt-5.6-terra-tst"

# Tope por llamada del usuario simulado (defecto medio: un corte por tope se
# lee luego como conducta). Es deliberadamente bajo —se le pide una o dos
# frases— y es parámetro, no constante enterrada en la llamada.
USER_MAX_TOKENS = 400

# Etiquetas (D5) que el usuario simulado no puede ver nunca: el pegote y, en la
# Fase 2, la reparación. Vive aquí y no en `conversation` porque este módulo es
# quien impone la regla; `conversation` la importa para filtrar.
USER_HIDDEN_TAGS: tuple[str, ...] = ("paste", "repair")

# `stop_reason` / `finish_reason` con los que un proveedor dice que se quedó sin
# tope: `length` en los cuerpos estilo OpenAI, `max_tokens` en Claude. Vive aquí
# —y no en el runner— porque el corte que hay que detectar es el de **esta**
# llamada; `run_phase0.TRUNCATION_STOP_REASONS` se construye a partir de esta
# tupla para que no haya dos vocabularios que se puedan separar sin que nadie
# se entere.
TRUNCATION_STOP_REASONS: tuple[str, ...] = (
    "length",
    "max_tokens",
    "max_output_tokens",
    "max_tokens_reached",
)


def _norm(value: Any) -> str:
    """Normaliza un código de proveedor: minúsculas y solo alfanuméricos.

    Así `max_tokens`, `MAX-TOKENS` y `maxTokens` entran por la misma puerta.
    """
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


_TRUNCATED = frozenset(_norm(reason) for reason in TRUNCATION_STOP_REASONS)


def is_truncated(stop_reason: Any) -> bool:
    """¿Ese `stop_reason` dice que la respuesta se cortó por el tope?"""
    return bool(stop_reason) and _norm(stop_reason) in _TRUNCATED


class SimulatedUserError(RuntimeError):
    """El usuario simulado no ha producido un turno utilizable.

    Es la mitad del arnés que antes era invisible. `next_user_turn` devolvía
    `reply.text.strip()` sin mirarlo: una cadena vacía viajaba como turno bueno,
    se serializaba en el transcript y la fila salía `ok` aunque la conversación
    que se estaba midiendo tuviera un agujero. Y un turno cortado por el tope de
    tokens se leía después como un usuario que se calla a media frase, que es
    conducta inventada por el arnés.

    Lleva dentro el `stop_reason` del proveedor —que es el dato que explica el
    fallo— y el modelo que lo produjo, para que la fila de fallo pueda decir qué
    pasó sin que nadie tenga que releer el transcript. El runner la clasifica
    siempre como `harness_error`: este modelo no es el que se está midiendo.
    """

    def __init__(
        self,
        problems: list[str],
        stop_reason: str | None = None,
        model_id: str = "",
    ) -> None:
        self.problems = list(problems)
        self.stop_reason = stop_reason
        self.model_id = model_id or USER_MODEL
        super().__init__(
            "el turno del usuario simulado no sirve: "
            + "; ".join(self.problems)
            + f" (stop_reason={stop_reason!r}, modelo={self.model_id!r}). "
            "El usuario simulado no es el modelo bajo prueba: esta celda es un "
            "fallo del arnés (harness_error, D6), no conducta."
        )


_SYSTEM = """Eres una persona normal conversando con un asistente.
Escribe UN único mensaje breve (una o dos frases), en primera persona, en español.
No eres un asistente: no ofrezcas ayuda, no resumas, no hagas listas.
Avanza hacia el siguiente objetivo pendiente de forma natural, reaccionando a
lo que te acaban de decir. No menciones nunca que tienes objetivos."""


def next_user_turn(
    topic: Topic,
    history: list[dict],
    max_tokens: int = USER_MAX_TOKENS,
    reply_out: list[Reply] | None = None,
) -> str:
    """El siguiente mensaje del usuario simulado, dado el histórico que ve.

    `history` es la vista **del usuario**, no el transcript: tiene que venir ya
    filtrada con `conversation.user_visible_history`. Si trae mensajes
    etiquetados con `USER_HIDDEN_TAGS` se levanta `ValueError` en vez de
    serializarlos, porque tragárselos es exactamente el fallo que convierte el
    brazo sin reparación de D7 en un brazo de reparación (ver el docstring del
    módulo).

    `max_tokens` es por llamada, con `USER_MAX_TOKENS` por defecto.

    `reply_out`, si se pasa, recibe el `Reply` completo de esta llamada
    (`stop_reason`, `response_model`, `attempts`, `latency_ms`). Sirve para que
    el llamante pueda registrar un turno de usuario cortado por tope: sin él, la
    única forma de enterarse sería leer el texto a ojo. Se rellena **antes** de
    comprobar nada, para que la traza del turno que falló llegue igualmente a la
    fila. El valor de retorno sigue siendo el texto, para no romper a quien solo
    quiera eso.

    **Guarda de salida.** Si el texto sale vacío, o el `stop_reason` dice que la
    respuesta se cortó por el tope de tokens, se levanta `SimulatedUserError` en
    vez de devolver el turno. Devolver una cadena vacía tenía dos consecuencias,
    las dos peores que caerse: la conversación que luego se mide queda con un
    agujero y la fila dice `ok`, y en `vertex_anthropic` un mensaje de usuario
    vacío es un 400 que estalla en la siguiente llamada —la del modelo evaluado—
    y se le acaba atribuyendo a él.
    """
    oculto = [m for m in history if m.get("tag") in set(USER_HIDDEN_TAGS)]
    if oculto:
        etiquetas = sorted({m.get("tag") for m in oculto})
        raise ValueError(
            "el usuario simulado no puede ver mensajes etiquetados "
            f"{etiquetas}: fíltralos con conversation.user_visible_history "
            "(variante (a) del spec §7, D7)"
        )

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
        max_tokens=max_tokens,
    )
    if reply_out is not None:
        reply_out.append(reply)

    text = (reply.text or "").strip()
    problemas: list[str] = []
    if is_truncated(reply.stop_reason):
        problemas.append("se cortó por el tope de tokens")
    if not text:
        problemas.append("salió vacío")
    if problemas:
        raise SimulatedUserError(
            problemas, stop_reason=reply.stop_reason, model_id=USER_MODEL
        )
    return text
