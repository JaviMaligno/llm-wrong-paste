"""Construcción de la conversación: prefijo, pegote y los dos turnos de después.

Cada mensaje del transcript lleva una etiqueta `tag` ∈ MESSAGE_TAGS (D5), que es
lo que permite al análisis saber qué escribió el usuario simulado, qué es el
pegote y qué vino después sin contar posiciones a mano.

Ojo: `tag` es metadato nuestro, no del protocolo, y los proveedores rechazan
campos desconocidos dentro de `messages`. Pero el saneado **no se hace aquí**:
`chat()` recibe la transcripción etiquetada tal cual y quita el `tag` en el
último momento, ya dentro de `clients`, **después** de haber colocado el
breakpoint de caché (D10). Quitarlo antes —como hacía este módulo a través de
una `api_messages()` que ya no existe— dejaba a `clients` sin las etiquetas y le
obligaba a caer siempre en su rama degradada «el prefijo acaba en el penúltimo
mensaje», que solo acierta en la llamada del pegote: en los dos turnos `post` de
D7 y en las celdas de control de D11 colgaba el `cache_control` de la reacción al
pegote o de una respuesta posterior, contenido que cambia en cada celda y que por
tanto no se puede servir de caché entre celdas que comparten `prefix_id`.

Lo que se le pasa a `chat()` es `list(transcript)`, una copia superficial: el
transcript se sigue mutando in place después de cada llamada (la respuesta se
añade al final), y una llamada tiene que quedarse con los mensajes que se le
enviaron, no con una vista viva que al mirarla luego ya lleva el pegote dentro.
Los dicts no se copian porque nadie los muta: solo se añaden nuevos.

**Forma elegida para la trazabilidad (D5/D6/D9).** `clients.Reply` ya trae
`stop_reason`, `response_model`, `attempts` y `latency_ms`; este módulo NO los
descarta. La regla, uniforme en las tres funciones, es:

    lo que ya se devolvía + los `Reply` completos al final.

- `build_prefix`   -> `(transcript, usages, replies)`
- `inject_paste`   -> `(reaction, usage, paste_index, reply)`
- `continue_after_paste` -> `(post_indices, usages, replies)`

`replies` es una lista de objetos `clients.Reply` **completos** (no de dicts), en
orden de llamada, y solo de las llamadas al **modelo evaluado**: los turnos que
escribe el usuario simulado los paga `simulated_user.next_user_turn` con otro
modelo y no son conducta del modelo bajo prueba. `reply.text` y `reply.usage` son
exactamente los mismos objetos que viajan por las posiciones anteriores del
tuple; la redundancia se mantiene a propósito para que el llamante solo tenga que
añadir un nombre al final de su desempaquetado.

Para escribir eso en el JSONL sin volver a saber de `Reply`, está
`reply_traces()`, que devuelve la lista de dicts que espera cada fila.

**Dos vistas de la misma conversación.** El transcript es lo que ve el modelo
evaluado y lo que se guarda en el JSONL: lleva el pegote y todo lo demás. El
usuario simulado ve otra cosa, más corta, que fabrica `user_visible_history()`;
el porqué está en su docstring.
"""

from wrongpaste.artifacts import Artifact
from wrongpaste.clients import Reply, chat
from wrongpaste.simulated_user import (
    USER_HIDDEN_TAGS,
    USER_MAX_TOKENS,
    next_user_turn,
)
from wrongpaste.topics import Topic

# Tope por llamada del modelo evaluado. Es el **valor por defecto**, no una
# constante enterrada en cada llamada: un corte por tope se lee luego como
# conducta ("el modelo se calló a media frase"), así que quien tire tiene que
# poder subirlo por celda sin tocar el módulo. Viaja a la fila vía
# `request_params` (D9) y el corte, cuando ocurre, viaja como `stop_reason`
# dentro de cada `Reply` (D6).
MAX_TOKENS = 1024

# Cuántos turnos de usuario van después de la reacción al pegote (D7).
N_POST_TURNS = 2


def user_visible_history(
    transcript: list[dict],
    hide_tags: tuple[str, ...] = USER_HIDDEN_TAGS,
) -> list[dict]:
    """El histórico tal y como lo ve el **usuario simulado**: sin el pegote.

    **Por qué se oculta.** El brazo sin reparación de D7 es la variante (a) del
    spec §7, que dice literalmente que «el usuario sigue con el tema como si el
    pegote no existiera». `next_user_turn` serializa el histórico entero en su
    prompt, así que pasarle el transcript ya mutado —con el mensaje etiquetado
    `paste` y la reacción del modelo a él— le enseña el pegote: el usuario
    simulado puede entonces comentarlo, disculparse o pivotar, y el brazo «sin
    reparación» se convierte en silencio en un brazo **de** reparación. Los dos
    brazos de D7 medirían lo mismo y la comparación perdería su sentido.

    **Qué se oculta.** Los mensajes con `tag` en `hide_tags` (`paste` y, cuando
    exista en la Fase 2, `repair`) y la **reacción inmediata** a ellos, es decir
    la tanda contigua de mensajes `assistant` que los sigue. Lo demás —apertura,
    turnos `user_sim`, respuestas del asistente del prefijo y los turnos `post`
    con sus respuestas— se conserva en orden: el usuario sí lee lo que el
    asistente le contesta después, y ahí es donde se verá si el modelo arrastra
    el pegote.

    **Qué NO se oculta.** El modelo evaluado sigue recibiendo el transcript
    completo, y el JSONL lo guarda completo. La contaminación que mide el
    experimento es la del modelo evaluado; la del usuario simulado sería un
    artefacto del arnés.

    Devuelve una lista nueva con **los mismos dicts** (no copia los mensajes):
    no muta el transcript del llamante ni lo desincroniza.
    """
    hidden = set(hide_tags)
    visible: list[dict] = []
    skipping_reaction = False

    for message in transcript:
        tag = message.get("tag")
        if tag in hidden:
            skipping_reaction = True
            continue
        if skipping_reaction:
            if tag == "assistant":
                continue
            skipping_reaction = False
        visible.append(message)

    return visible


def reply_traces(replies: list[Reply]) -> list[dict]:
    """Los campos de trazabilidad de cada `Reply`, listos para el JSONL (D5).

    Un dict por llamada al modelo evaluado, en orden de llamada, con
    `stop_reason`, `response_model`, `attempts` y `latency_ms`. Existe para que
    el runner no tenga que conocer el dataclass ni repetir el mismo bucle en
    tres sitios: `Reply` no es serializable a JSON tal cual (lleva `raw`, que es
    la respuesta entera del proveedor).
    """
    return [
        {
            "stop_reason": r.stop_reason,
            "response_model": r.response_model,
            "attempts": r.attempts,
            "latency_ms": r.latency_ms,
        }
        for r in replies
    ]


def build_prefix(
    model_id: str,
    topic: Topic,
    n_turns: int,
    request_params_out: dict | None = None,
    max_tokens: int = MAX_TOKENS,
    user_max_tokens: int = USER_MAX_TOKENS,
    user_replies_out: list[Reply] | None = None,
) -> tuple[list[dict], list[dict], list[Reply]]:
    """Conduce n_turns de conversación normal sobre el tema.

    Devuelve la transcripción (siempre terminada en assistant, que es lo que
    Claude exige para poder continuar), los `usage` de cada llamada y los
    `Reply` completos de esas mismas llamadas, en el mismo orden: sin ellos, las
    filas del JSONL saldrían con `stop_reasons=[]`, `response_model=""` y
    `attempts=1` aunque hubiera habido reintentos (D5/D6).

    `request_params_out`, si se pasa, se le cede a `chat()`, que lo rellena con
    el cuerpo enviado sin `messages` (D9). Queda con el de la **última** llamada:
    todas las llamadas de una celda van al mismo modelo con el mismo
    `max_tokens`, así que el cuerpo es el mismo turno a turno y el último es
    representativo. `chat()` lo vacía antes de rellenarlo, así que no se
    acumulan restos de la llamada anterior.

    Etiquetas (D5): el mensaje de apertura es `opening`, los demás mensajes de
    usuario son `user_sim` (los escribe el usuario simulado) y las respuestas
    son `assistant`.

    `max_tokens` es el tope del modelo evaluado y `user_max_tokens` el del
    usuario simulado; son distintos porque los papeles son distintos (al usuario
    se le piden una o dos frases). `user_replies_out`, si se pasa, recibe los
    `Reply` de los turnos del usuario simulado, que no son conducta del modelo
    evaluado pero sí pueden salir cortados por tope.

    Aquí el histórico que se le pasa al usuario simulado ya pasa por
    `user_visible_history()` aunque en el prefijo no haya nada que ocultar: hay
    un solo camino hacia `next_user_turn` en todo el módulo, y así no depende de
    que quien añada una fase futura se acuerde de filtrar.

    En la Fase 0 este prefijo lo fabrica siempre `PREFIX_MODEL` y se comparte
    entre modelos evaluados (D1); ver `wrongpaste.prefixes`.
    """
    transcript: list[dict] = [
        {"role": "user", "content": topic.opening, "tag": "opening"}
    ]
    usages: list[dict] = []
    replies: list[Reply] = []

    reply = chat(
        model_id,
        list(transcript),
        max_tokens=max_tokens,
        request_params_out=request_params_out,
    )
    transcript.append({"role": "assistant", "content": reply.text, "tag": "assistant"})
    usages.append(reply.usage)
    replies.append(reply)

    for _ in range(n_turns - 1):
        transcript.append(
            {
                "role": "user",
                "content": next_user_turn(
                    topic,
                    user_visible_history(transcript),
                    max_tokens=user_max_tokens,
                    reply_out=user_replies_out,
                ),
                "tag": "user_sim",
            }
        )
        reply = chat(
            model_id,
            list(transcript),
            max_tokens=max_tokens,
            request_params_out=request_params_out,
        )
        transcript.append(
            {"role": "assistant", "content": reply.text, "tag": "assistant"}
        )
        usages.append(reply.usage)
        replies.append(reply)

    return transcript, usages, replies


def conversation_text(transcript: list[dict]) -> str:
    """La conversación entera como texto (similaridad secundaria, D2)."""
    return "\n".join(m["content"] for m in transcript)


def inject_paste(
    model_id: str,
    transcript: list[dict],
    artifact: Artifact,
    request_params_out: dict | None = None,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, dict, int, Reply]:
    """Añade el pegote TAL CUAL, sin preámbulo ni envoltorio, y pide respuesta.

    Muta `transcript` in place: el registro guarda la conversación completa.

    Devuelve (reacción, usage, índice del mensaje del pegote, `Reply` completo).
    El índice va al campo `paste_index` de la fila (D5): sin él, el análisis
    tendría que volver a localizar el pegote comparando textos. El `Reply` lleva
    `stop_reason`, `response_model`, `attempts` y `latency_ms` de **la llamada
    del pegote**, que es la que decide la conducta que mide el experimento: si un
    modelo se corta por `max_tokens` en vez de terminar, la fila tiene que
    decirlo (D6).

    `request_params_out`, si se pasa, se le cede a `chat()` y queda con el cuerpo
    de esta llamada (D9). `max_tokens` es el tope de **esta** llamada, que es la
    que decide la conducta medida: subirlo aquí sin tocar el resto es legítimo
    cuando el corte por tope amenaza con leerse como conducta.
    """
    paste_index = len(transcript)
    transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
    reply = chat(
        model_id,
        list(transcript),
        max_tokens=max_tokens,
        request_params_out=request_params_out,
    )
    transcript.append({"role": "assistant", "content": reply.text, "tag": "assistant"})
    return reply.text, reply.usage, paste_index, reply


def continue_after_paste(
    model_id: str,
    transcript: list[dict],
    topic: Topic,
    n_post: int = N_POST_TURNS,
    request_params_out: dict | None = None,
    max_tokens: int = MAX_TOKENS,
    user_max_tokens: int = USER_MAX_TOKENS,
    user_replies_out: list[Reply] | None = None,
) -> tuple[list[int], list[dict], list[Reply]]:
    """Dos turnos más con el usuario simulado, SIN reparación (D7).

    Es la variante (a) del spec §7: el usuario sigue a lo suyo **como si el
    pegote no existiera**. Sirve para separar "pivota en silencio" de "puente
    confabulado" — que a menudo solo se distinguen en el turno +1 — y para
    pilotar ya el recuento de entidades de la Fase 2.

    **"Como si no existiera" es una propiedad del arnés, no una esperanza.** Lo
    que se le pasa a `next_user_turn` es `user_visible_history(transcript)`, sin
    el mensaje `paste` (ni el `repair` de la Fase 2) ni la reacción inmediata a
    él. Pasarle el transcript ya mutado —que es lo que hacía antes esta
    función— le enseñaba el pegote entero, porque `next_user_turn` serializa el
    histórico en su prompt: el usuario simulado podía comentarlo o disculparse y
    este brazo se convertía en silencio en un brazo de reparación. El modelo
    evaluado sí sigue recibiendo el transcript completo en cada llamada: la
    contaminación que se mide es la suya.

    Muta `transcript` in place. El mensaje de usuario se etiqueta `post` y la
    respuesta `assistant`.

    Devuelve (post_indices, usages, replies). `post_indices` son los índices de
    **todos** los mensajes añadidos aquí, usuario y asistente, en orden: el
    recuento de fuga de entidades se hace sobre las respuestas, así que dejarlas
    fuera obligaría al análisis a recalcular posiciones. `replies` son los
    `Reply` completos de los `n_post` turnos del modelo evaluado, en orden, con
    la misma trazabilidad que los demás (D5/D6).

    `request_params_out`, si se pasa, se le cede a `chat()` y queda con el cuerpo
    de la última de esas llamadas (D9). `max_tokens` y `user_max_tokens` son los
    topes por llamada del modelo evaluado y del usuario simulado;
    `user_replies_out`, si se pasa, recibe los `Reply` de los turnos del usuario
    simulado, por si alguno sale cortado por tope.
    """
    post_indices: list[int] = []
    usages: list[dict] = []
    replies: list[Reply] = []

    for _ in range(n_post):
        post_indices.append(len(transcript))
        transcript.append(
            {
                "role": "user",
                "content": next_user_turn(
                    topic,
                    user_visible_history(transcript),
                    max_tokens=user_max_tokens,
                    reply_out=user_replies_out,
                ),
                "tag": "post",
            }
        )
        reply = chat(
            model_id,
            list(transcript),
            max_tokens=max_tokens,
            request_params_out=request_params_out,
        )
        post_indices.append(len(transcript))
        transcript.append(
            {"role": "assistant", "content": reply.text, "tag": "assistant"}
        )
        usages.append(reply.usage)
        replies.append(reply)

    return post_indices, usages, replies
