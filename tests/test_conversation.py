import pytest

import wrongpaste.conversation as conv
import wrongpaste.simulated_user as su
from wrongpaste.artifacts import Artifact
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic(id="t", opening="quiero mudarme", goals=("a", "b", "c", "d"))
ART = Artifact("art1", "recipe", "300 g de lentejas", ("lentejas",))


def _stub(monkeypatch, replies=None, seen=None, params_seen=None):
    """Sustituye las dos únicas puertas de red del módulo.

    Si se pasa `seen`, se van apuntando los `messages` con los que se llamó a
    `chat`, para poder comprobar que salen sin el `tag` (que es metadato
    nuestro y los proveedores rechazan campos desconocidos).

    Si se pasa `params_seen`, se apunta el `request_params_out` que recibió cada
    llamada. El doble imita el contrato real de `clients.chat`: vacía el dict y
    lo rellena con el cuerpo enviado sin `messages` (D9).
    """
    seq = iter(replies or [Reply(f"r{i}", {}, {}) for i in range(20)])

    def fake_chat(model_id, messages, **kwargs):
        if seen is not None:
            seen.append(messages)
        out = kwargs.get("request_params_out")
        if params_seen is not None:
            params_seen.append(out)
        if out is not None:
            out.clear()
            out.update(
                {"model": model_id, "max_tokens": kwargs.get("max_tokens")}
            )
        return next(seq)

    monkeypatch.setattr(conv, "chat", fake_chat)
    monkeypatch.setattr(conv, "next_user_turn", lambda *a, **k: "turno de usuario")


def _traced(n: int) -> list[Reply]:
    """`n` respuestas con trazabilidad distinta en cada una.

    Los valores son distintos a propósito: así un test que los compruebe falla
    si alguien devuelve el `Reply` equivocado, no solo si los descarta.
    """
    return [
        Reply(
            text=f"r{i}",
            usage={"total_tokens": i},
            raw={},
            stop_reason=f"stop{i}",
            response_model=f"modelo-real-{i}",
            attempts=i + 1,
            latency_ms=100.0 * (i + 1),
        )
        for i in range(n)
    ]


def test_prefix_alternates_roles_and_ends_on_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, usages, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=3)

    assert transcript[0]["role"] == "user"
    assert transcript[-1]["role"] == "assistant"
    assert [m["role"] for m in transcript] == ["user", "assistant"] * 3
    assert len(usages) == 3


def test_prefix_never_ends_on_user_so_claude_accepts_it(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("claude-opus-5", TOPIC, n_turns=2)
    assert transcript[-1]["role"] == "assistant"


def test_prefix_tags_opening_user_sim_and_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=3)

    assert [m["tag"] for m in transcript] == [
        "opening",
        "assistant",
        "user_sim",
        "assistant",
        "user_sim",
        "assistant",
    ]


def test_only_the_first_message_is_tagged_opening(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=4)

    assert [m["tag"] for m in transcript].count("opening") == 1
    assert transcript[0]["content"] == TOPIC.opening


def test_chat_recibe_la_transcripcion_etiquetada(monkeypatch):
    """`chat()` tiene que ver el `tag`: es lo que coloca el breakpoint de caché.

    Antes este módulo saneaba los mensajes antes de llamar, y `clients` se
    quedaba sin etiquetas: su `_prefix_breakpoint_index` caía SIEMPRE en la rama
    degradada «el prefijo acaba en el penúltimo mensaje», que solo acierta en la
    llamada del pegote. El saneado es ahora cosa de `clients`, en el último
    momento; que ningún proveedor reciba el `tag` se comprueba sobre el cuerpo
    HTTP realmente enviado, no aquí (ver `test_clients.py`, ruta real).
    """
    seen: list[list[dict]] = []
    _stub(monkeypatch, seen=seen)

    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert seen, "el stub tiene que haber visto llamadas"
    for messages in seen:
        for m in messages:
            assert set(m) == {"role", "content", "tag"}

    # La llamada del pegote lo lleva etiquetado como tal, que es exactamente lo
    # que `clients` necesita para saber dónde acaba el prefijo compartido.
    llamada_del_pegote = seen[2]
    assert llamada_del_pegote[-1]["tag"] == "paste"
    assert llamada_del_pegote[-1]["content"] == ART.text


def test_cada_llamada_se_queda_con_los_mensajes_de_esa_llamada(monkeypatch):
    """Lo que se le pasa a `chat()` es una foto, no una vista viva.

    El transcript se muta in place después de cada llamada. Si se pasara el
    mismo objeto lista, los mensajes apuntados por cualquiera que inspeccione,
    registre o reintente la llamada del prefijo acabarían incluyendo el pegote
    y los turnos posteriores, que aún no se habían enviado.
    """
    seen: list[list[dict]] = []
    _stub(monkeypatch, seen=seen)

    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    assert len(seen[0]) == 1, "la primera llamada solo llevaba la apertura"
    assert all(ART.text not in m["content"] for m in seen[0])
    assert len(seen[-1]) == len(transcript) - 1


def test_inject_paste_appends_artifact_verbatim(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    before = len(transcript)

    reaction, usage, paste_index, _ = conv.inject_paste(
        "gpt-5.6-luna-tst", transcript, ART
    )

    assert transcript[before]["role"] == "user"
    assert transcript[before]["content"] == ART.text
    assert transcript[-1]["role"] == "assistant"
    assert reaction == transcript[-1]["content"]
    assert paste_index == before


def test_inject_paste_tags_the_paste_and_the_reaction(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    _, _, paste_index, _ = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    assert transcript[paste_index]["tag"] == "paste"
    assert transcript[paste_index]["content"] == ART.text
    assert transcript[-1]["tag"] == "assistant"
    assert [m["tag"] for m in transcript].count("paste") == 1


def test_inject_paste_adds_no_preamble(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    pasted = [m for m in transcript if m["content"] == ART.text]
    assert len(pasted) == 1, "el pegote debe ir tal cual, sin envoltorio"


def test_continue_after_paste_adds_four_messages(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    before = len(transcript)

    post_indices, usages, _ = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC
    )

    assert len(transcript) - before == 4
    assert len(usages) == 2
    assert post_indices == [before, before + 1, before + 2, before + 3]


def test_continue_after_paste_tags_post_and_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    before = len(transcript)

    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert [m["tag"] for m in transcript[before:]] == [
        "post",
        "assistant",
        "post",
        "assistant",
    ]
    assert [m["role"] for m in transcript[before:]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_continue_after_paste_never_repairs(monkeypatch):
    """La variante (a) del spec §7: nadie menciona el pegote por el usuario."""
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert all(m["tag"] != "repair" for m in transcript)


def test_post_indices_point_at_the_messages_added_after_the_reaction(monkeypatch):
    _stub(monkeypatch)
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    _, _, paste_index, _ = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    post_indices, _, _ = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, n_post=3
    )

    assert len(post_indices) == 6
    assert all(i > paste_index for i in post_indices)
    assert [transcript[i]["tag"] for i in post_indices[::2]] == ["post"] * 3


# --- D5/D6: la trazabilidad del `Reply` sobrevive el viaje ------------------


def test_build_prefix_devuelve_los_reply_completos(monkeypatch):
    """Sin esto, las filas saldrían con stop_reasons=[] y attempts=1."""
    _stub(monkeypatch, replies=_traced(3))

    _, usages, replies = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=3)

    assert [r.stop_reason for r in replies] == ["stop0", "stop1", "stop2"]
    assert [r.response_model for r in replies] == [
        "modelo-real-0",
        "modelo-real-1",
        "modelo-real-2",
    ]
    assert [r.attempts for r in replies] == [1, 2, 3]
    assert [r.latency_ms for r in replies] == [100.0, 200.0, 300.0]
    # `usages` sigue siendo exactamente lo que ya devolvía, sin duplicarlo mal.
    assert usages == [r.usage for r in replies]


def test_build_prefix_solo_cuenta_las_llamadas_al_modelo_evaluado(monkeypatch):
    """Los turnos del usuario simulado los paga otro modelo: no son conducta.

    Con n_turns=3 hay tres respuestas del modelo evaluado y dos mensajes del
    usuario simulado; `replies` tiene que traer tres, no cinco.
    """
    _stub(monkeypatch, replies=_traced(3))

    transcript, _, replies = conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=3
    )

    assert len(replies) == 3
    assert len(transcript) == 6
    assert [r.text for r in replies] == [
        m["content"] for m in transcript if m["role"] == "assistant"
    ]


def test_inject_paste_devuelve_el_reply_de_la_llamada_del_pegote(monkeypatch):
    # Cuatro respuestas: dos del prefijo y la del pegote es la tercera.
    _stub(monkeypatch, replies=_traced(4))
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    reaction, usage, _, reply = conv.inject_paste(
        "gpt-5.6-luna-tst", transcript, ART
    )

    assert reply.stop_reason == "stop2"
    assert reply.response_model == "modelo-real-2"
    assert reply.attempts == 3
    assert reply.latency_ms == 300.0
    # Las posiciones viejas siguen apuntando al mismo sitio.
    assert reaction == reply.text
    assert usage == reply.usage


def test_continue_after_paste_devuelve_un_reply_por_turno(monkeypatch):
    _stub(monkeypatch, replies=_traced(5))
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    _, usages, replies = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC
    )

    assert len(replies) == 2
    assert [r.stop_reason for r in replies] == ["stop3", "stop4"]
    assert [r.response_model for r in replies] == ["modelo-real-3", "modelo-real-4"]
    assert [r.attempts for r in replies] == [4, 5]
    assert usages == [r.usage for r in replies]


def test_la_trazabilidad_llega_entera_de_extremo_a_extremo(monkeypatch):
    """El viaje completo: prefijo, pegote y dos turnos después.

    Es el defecto que dejaba las filas con `stop_reasons=[]`,
    `response_model=""` y `attempts=1`: lo que el runner escribe en la fila
    tiene que poder reconstruirse juntando los tres tramos.
    """
    _stub(monkeypatch, replies=_traced(5))

    transcript, _, prefix_replies = conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=2
    )
    _, _, _, paste_reply = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    _, _, post_replies = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC
    )

    todos = [*prefix_replies, paste_reply, *post_replies]

    assert [r.stop_reason for r in todos] == [f"stop{i}" for i in range(5)]
    assert [r.response_model for r in todos] == [
        f"modelo-real-{i}" for i in range(5)
    ]
    assert [r.attempts for r in todos] == [1, 2, 3, 4, 5]
    # Un reintento de verdad no puede acabar registrado como `attempts=1`.
    assert max(r.attempts for r in todos) == 5


def test_reply_traces_da_las_columnas_que_espera_la_fila():
    traces = conv.reply_traces(_traced(2))

    assert traces == [
        {
            "stop_reason": "stop0",
            "response_model": "modelo-real-0",
            "attempts": 1,
            "latency_ms": 100.0,
        },
        {
            "stop_reason": "stop1",
            "response_model": "modelo-real-1",
            "attempts": 2,
            "latency_ms": 200.0,
        },
    ]


def test_reply_traces_aguanta_un_reply_sin_trazas():
    """Un doble de test o un proveedor callado no puede tumbar la escritura."""
    assert conv.reply_traces([Reply("x", {}, {})]) == [
        {
            "stop_reason": None,
            "response_model": None,
            "attempts": 1,
            "latency_ms": None,
        }
    ]


# --- D9: el cuerpo enviado se puede capturar --------------------------------


def test_las_tres_funciones_ceden_request_params_out_a_chat(monkeypatch):
    params_seen: list = []
    _stub(monkeypatch, params_seen=params_seen)
    out: dict = {}

    transcript, _, _ = conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=2, request_params_out=out
    )
    conv.inject_paste(
        "gpt-5.6-luna-tst", transcript, ART, request_params_out=out
    )
    conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, request_params_out=out
    )

    assert len(params_seen) == 5
    assert all(p is out for p in params_seen), (
        "el dict tiene que llegar a chat() en todas las llamadas al modelo"
    )


def test_request_params_out_queda_relleno_con_el_cuerpo_enviado(monkeypatch):
    _stub(monkeypatch)
    out: dict = {}

    conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=2, request_params_out=out
    )

    assert out == {"model": "gpt-5.6-luna-tst", "max_tokens": conv.MAX_TOKENS}


def test_inject_paste_deja_en_request_params_out_el_cuerpo_del_pegote(monkeypatch):
    _stub(monkeypatch)
    out: dict = {}
    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    conv.inject_paste("claude-opus-5", transcript, ART, request_params_out=out)

    assert out == {"model": "claude-opus-5", "max_tokens": conv.MAX_TOKENS}


def test_sin_request_params_out_no_se_le_pasa_dict_a_chat(monkeypatch):
    """El parámetro es opcional: quien no lo pida no paga nada por él."""
    params_seen: list = []
    _stub(monkeypatch, params_seen=params_seen)

    conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    assert params_seen == [None, None]


# --- El brazo sin reparación de D7 lo es de verdad ---------------------------
#
# Bloqueante de validez: `continue_after_paste` le pasaba al usuario simulado el
# transcript YA MUTADO, con el mensaje `paste` y la reacción del modelo dentro.
# `next_user_turn` serializa el histórico entero en su prompt, así que el usuario
# veía el pegote y podía comentarlo: el brazo "sin reparación" se convertía en
# silencio en un brazo DE reparación. La variante (a) del spec §7 (D7) dice que
# el usuario sigue con el tema como si el pegote no existiera.


def test_user_visible_history_oculta_el_pegote_y_su_reaccion():
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "respuesta al tema", "tag": "assistant"},
        {"role": "user", "content": "EL PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "reacción al pegote", "tag": "assistant"},
        {"role": "user", "content": "sigo a lo mío", "tag": "post"},
        {"role": "assistant", "content": "respuesta de después", "tag": "assistant"},
    ]

    visible = conv.user_visible_history(transcript)

    assert [m["content"] for m in visible] == [
        "apertura",
        "respuesta al tema",
        "sigo a lo mío",
        "respuesta de después",
    ]


def test_user_visible_history_oculta_tambien_la_reparacion_de_la_fase_2():
    """`repair` no existe en la Fase 0, pero el filtro ya tiene que cubrirlo."""
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "a", "tag": "assistant"},
        {"role": "user", "content": "EL PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "reacción", "tag": "assistant"},
        {"role": "user", "content": "perdón, pegué otra cosa", "tag": "repair"},
        {"role": "assistant", "content": "tranquilo", "tag": "assistant"},
        {"role": "user", "content": "sigo", "tag": "post"},
    ]

    visible = conv.user_visible_history(transcript)

    assert [m["tag"] for m in visible] == ["opening", "assistant", "post"]


def test_user_visible_history_solo_oculta_la_reaccion_inmediata():
    """Lo que el modelo conteste DESPUÉS sí lo lee el usuario.

    Ahí es justamente donde se ve si arrastra el pegote: ocultarlo dejaría al
    usuario simulado respondiendo a un vacío.
    """
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "user", "content": "EL PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "reacción", "tag": "assistant"},
        {"role": "user", "content": "sigo", "tag": "post"},
        {"role": "assistant", "content": "respuesta posterior", "tag": "assistant"},
    ]

    visible = conv.user_visible_history(transcript)

    assert "reacción" not in [m["content"] for m in visible]
    assert "respuesta posterior" in [m["content"] for m in visible]


def test_user_visible_history_no_muta_el_transcript():
    transcript = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "user", "content": "EL PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "reacción", "tag": "assistant"},
    ]
    copia = [dict(m) for m in transcript]

    visible = conv.user_visible_history(transcript)

    assert transcript == copia, "el transcript del llamante se queda entero"
    assert visible is not transcript
    assert visible[0] is transcript[0], "misma identidad, no copias desincronizadas"


def test_user_visible_history_con_hide_tags_vacio_no_oculta_nada():
    transcript = [
        {"role": "user", "content": "EL PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "reacción", "tag": "assistant"},
    ]

    assert conv.user_visible_history(transcript, hide_tags=()) == transcript


def test_user_visible_history_deja_pasar_mensajes_sin_etiqueta():
    """Transcripciones crudas (tests, herramientas sueltas) no se pierden."""
    transcript = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "qué tal"},
    ]

    assert conv.user_visible_history(transcript) == transcript


def _stub_modelo_evaluado(monkeypatch, seen=None, n=10):
    """Corta la red del modelo evaluado dejando el usuario simulado REAL.

    `_stub` de arriba sustituye también `next_user_turn`, así que no puede ver
    este fallo: el bloqueante está precisamente en lo que se le pasa a esa
    función. Aquí solo se corta `chat`.
    """
    seq = iter(
        [
            Reply(f"respuesta-{i}", {}, {}, stop_reason="end_turn")
            for i in range(n)
        ]
    )

    def fake_chat(model_id, messages, **kwargs):
        if seen is not None:
            seen.append(messages)
        return next(seq)

    monkeypatch.setattr(conv, "chat", fake_chat)


def _stub_red_del_usuario(monkeypatch, prompts, stop_reason="stop"):
    """Corta la red del usuario simulado y apunta el prompt que recibe."""

    def fake_user_chat(model_id, messages, max_tokens=None, **kwargs):
        prompts.append(messages[-1]["content"])
        return Reply(
            "sigo con lo mío",
            {},
            {},
            stop_reason=stop_reason,
            response_model="modelo-usuario",
        )

    monkeypatch.setattr(su, "chat", fake_user_chat)


def _conversacion_completa(monkeypatch, prompts, seen=None):
    """Prefijo + pegote + dos turnos de después, con el usuario simulado real."""
    _stub_modelo_evaluado(monkeypatch, seen=seen)
    _stub_red_del_usuario(monkeypatch, prompts)

    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)
    return transcript


def test_el_prompt_del_usuario_simulado_no_contiene_el_pegote(monkeypatch):
    """El bloqueante, en una línea: el usuario no puede leer lo que pegó mal."""
    prompts: list[str] = []
    _conversacion_completa(monkeypatch, prompts)

    assert prompts, "el usuario simulado tiene que haber escrito algún turno"
    for prompt in prompts:
        assert ART.text not in prompt


def test_el_prompt_del_usuario_simulado_no_contiene_la_reaccion_al_pegote(
    monkeypatch,
):
    """Ocultar el pegote y dejar la reacción sería el mismo fallo con otra cara.

    Con n_turns=2 las respuestas 0 y 1 son del prefijo y la 2 es la reacción al
    pegote.
    """
    prompts: list[str] = []
    _conversacion_completa(monkeypatch, prompts)

    posteriores = prompts[-2:]
    for prompt in posteriores:
        assert "respuesta-2" not in prompt, "la reacción al pegote se oculta"
        assert "respuesta-1" in prompt, "el prefijo sí se ve: hay conversación"


def test_el_usuario_simulado_si_ve_lo_que_el_modelo_contesta_despues(monkeypatch):
    """Sin esto, el turno +2 respondería al vacío y no a un modelo contaminado."""
    prompts: list[str] = []
    _conversacion_completa(monkeypatch, prompts)

    assert "respuesta-3" in prompts[-1]


def test_el_modelo_evaluado_si_ve_el_pegote_entero(monkeypatch):
    """La contaminación que se mide es la suya: a él no se le oculta nada."""
    prompts: list[str] = []
    seen: list[list[dict]] = []
    transcript = _conversacion_completa(monkeypatch, prompts, seen=seen)

    # Llamadas 0 y 1: prefijo, aún no hay pegote. De la 2 en adelante, sí.
    assert len(seen) == 5
    assert all(ART.text not in str(m) for m in seen[0])
    for messages in seen[2:]:
        assert any(m["content"] == ART.text for m in messages)

    # Y el transcript que se guarda en el JSONL lo conserva.
    assert [m["content"] for m in transcript].count(ART.text) == 1
    assert [m["tag"] for m in transcript].count("paste") == 1


def test_ocultar_el_pegote_no_altera_las_etiquetas_posteriores(monkeypatch):
    prompts: list[str] = []
    transcript = _conversacion_completa(monkeypatch, prompts)

    assert [m["tag"] for m in transcript] == [
        "opening",
        "assistant",
        "user_sim",
        "assistant",
        "paste",
        "assistant",
        "post",
        "assistant",
        "post",
        "assistant",
    ]
    assert [m["role"] for m in transcript] == ["user", "assistant"] * 5


def test_el_usuario_simulado_se_planta_si_le_cuelan_el_pegote(monkeypatch):
    """Defensa en profundidad: el fallo silencioso pasa a ser ruidoso."""
    prompts: list[str] = []
    _stub_red_del_usuario(monkeypatch, prompts)

    with pytest.raises(ValueError, match="paste"):
        su.next_user_turn(
            TOPIC,
            [
                {"role": "user", "content": "apertura", "tag": "opening"},
                {"role": "user", "content": ART.text, "tag": "paste"},
            ],
        )

    assert prompts == [], "no se llega ni a llamar al modelo"


# --- Defecto medio: el tope de tokens es parámetro, no constante -------------


def test_max_tokens_por_defecto_sigue_siendo_el_de_siempre():
    assert conv.MAX_TOKENS == 1024


def test_las_tres_funciones_aceptan_su_propio_max_tokens(monkeypatch):
    """Un corte por tope se lee luego como conducta: hay que poder subirlo."""
    vistos: list[int] = []

    def fake_chat(model_id, messages, **kwargs):
        vistos.append(kwargs.get("max_tokens"))
        return Reply("r", {}, {})

    monkeypatch.setattr(conv, "chat", fake_chat)
    monkeypatch.setattr(conv, "next_user_turn", lambda *a, **k: "turno")

    transcript, _, _ = conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=2, max_tokens=111
    )
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART, max_tokens=222)
    conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, n_post=1, max_tokens=333
    )

    assert vistos == [111, 111, 222, 333]


def test_el_tope_del_usuario_simulado_es_suyo_y_configurable(monkeypatch):
    topes: list[int] = []

    def fake_user_chat(model_id, messages, max_tokens=None, **kwargs):
        topes.append(max_tokens)
        return Reply("sigo", {}, {})

    monkeypatch.setattr(su, "chat", fake_user_chat)
    _stub_modelo_evaluado(monkeypatch)

    transcript, _, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, n_post=1, user_max_tokens=64
    )

    assert topes == [su.USER_MAX_TOKENS, 64]


def test_el_stop_reason_de_los_turnos_del_usuario_tambien_llega(monkeypatch):
    """El runner no puede marcar la fila por lo que no le llega (D6).

    El `stop_reason` es `content_filter` y no `max_tokens` porque un turno de
    usuario cortado por el tope ya no llega hasta aquí: `simulated_user` lo
    convierte en `SimulatedUserError` (fallo del arnés, no conducta). Lo que
    este test comprueba sigue siendo lo mismo —que el `Reply` del usuario
    simulado viaja entero hasta el llamante— y el valor sigue sin ser el de por
    defecto, así que falla igual si alguien devuelve el `Reply` equivocado.
    """
    prompts: list[str] = []
    _stub_modelo_evaluado(monkeypatch)
    _stub_red_del_usuario(monkeypatch, prompts, stop_reason="content_filter")

    user_replies: list[Reply] = []
    transcript, _, _ = conv.build_prefix(
        "gpt-5.6-luna-tst", TOPIC, n_turns=2, user_replies_out=user_replies
    )
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, user_replies_out=user_replies
    )

    assert len(user_replies) == 3, "un turno de prefijo y los dos de después"
    assert [r.stop_reason for r in user_replies] == ["content_filter"] * 3
    assert all(
        t["stop_reason"] == "content_filter" for t in conv.reply_traces(user_replies)
    )


def test_sin_user_replies_out_el_usuario_simulado_no_paga_nada(monkeypatch):
    prompts: list[str] = []
    _stub_modelo_evaluado(monkeypatch)
    _stub_red_del_usuario(monkeypatch, prompts)

    transcript, _, replies = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    assert len(replies) == 2, "`replies` sigue siendo solo del modelo evaluado"
