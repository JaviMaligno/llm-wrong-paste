import httpx
import pytest

import wrongpaste.conversation as conv
from wrongpaste import clients
from wrongpaste.artifacts import Artifact
from wrongpaste.topics import Topic
from wrongpaste.clients import (
    TEMPERATURE,
    _anthropic_body,
    _backoff_seconds,
    _gateway_body,
    _vertex_openai_body,
    chat,
)


class _FakeResponse:
    """Respuesta mínima con la superficie que usa el cliente."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("POST", "https://example.test"),
                response=httpx.Response(self.status_code),
            )


def _install_fake_post(monkeypatch, responses: list[_FakeResponse]) -> list[dict]:
    """Sustituye la red por una cola de respuestas y registra las llamadas."""
    calls: list[dict] = []
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return queue.pop(0)

    monkeypatch.setattr(clients.httpx, "post", fake_post)
    monkeypatch.setattr(clients.time, "sleep", lambda _s: None)
    monkeypatch.setattr(clients.config, "gateway_key", lambda: "clave-de-prueba")
    monkeypatch.setattr(clients, "_gcp_token", lambda: "token-de-prueba")
    return calls


def _openai_payload(content="hola", finish="stop", model="gpt-5.6-sol-tst") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10},
    }


def _anthropic_payload(text="hola", stop="end_turn", model="claude-opus-5") -> dict:
    return {
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 0},
    }


# --- forma de los cuerpos ---------------------------------------------------


def test_anthropic_body_moves_system_out_of_messages():
    msgs = [
        {"role": "system", "content": "eres útil"},
        {"role": "user", "content": "hola"},
    ]
    body = _anthropic_body(msgs, max_tokens=16)
    assert body["system"] == "eres útil"
    assert body["messages"] == [{"role": "user", "content": "hola"}]
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert body["max_tokens"] == 16


def test_anthropic_body_never_ends_on_assistant_turn():
    # El prefill está eliminado en la familia 5: un último turno de
    # assistant daría 400. El cliente debe negarse antes de llamar.
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    try:
        _anthropic_body(msgs, max_tokens=16)
    except ValueError as exc:
        assert "prefill" in str(exc).lower()
    else:
        raise AssertionError("debería haber lanzado ValueError")


def test_gateway_body_uses_max_completion_tokens():
    body = _gateway_body("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}], 32)
    assert body["max_completion_tokens"] == 32
    assert "max_tokens" not in body


def test_vertex_openai_body_prefixes_model_with_google():
    body = _vertex_openai_body("gemini-2.5-pro", [{"role": "user", "content": "x"}], 32)
    assert body["model"] == "google/gemini-2.5-pro"


# --- D9: muestreo explícito y razonamiento ---------------------------------


def test_temperature_explicita_donde_el_proveedor_la_admite():
    # D9 corregido: temperatura explícita en gateway y Vertex-OpenAI, para no
    # heredar defaults que ni coinciden entre proveedores ni están congelados.
    # En Claude 5 NO: el muestreo explícito devuelve 400 (ver el test de abajo).
    msgs = [{"role": "user", "content": "x"}]
    assert _gateway_body("gpt-5.6-sol-tst", msgs, 32)["temperature"] == TEMPERATURE
    assert _vertex_openai_body("gemini-2.5-pro", msgs, 32)["temperature"] == TEMPERATURE
    assert TEMPERATURE == 1.0


def test_anthropic_body_manda_thinking_adaptive_siempre():
    # Sin esto, Sonnet 5 correría sin razonamiento mientras Opus 5 lo lleva por
    # defecto, y los dos Claude dejarían de ser comparables.
    body = _anthropic_body([{"role": "user", "content": "x"}], max_tokens=32)
    assert body["thinking"] == {"type": "adaptive"}


def test_anthropic_body_nunca_manda_budget_tokens():
    # `budget_tokens` devuelve 400 en Opus 5 y Sonnet 5: tumbaría la tirada.
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "pegote"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert "budget_tokens" not in body
    assert "budget_tokens" not in body["thinking"]


# --- D10: cache_control -----------------------------------------------------


def _cache_control_of(message: dict):
    content = message["content"]
    if isinstance(content, str):
        return None
    return content[-1].get("cache_control")


def test_cache_control_marca_el_final_del_prefijo_no_el_pegote():
    # El breakpoint va en el último mensaje que NO es el pegote, para que el
    # prefijo compartido se sirva de caché en la llamada del pegote (D10).
    msgs = [
        {"role": "user", "content": "apertura"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "PEGOTE"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    enviados = body["messages"]
    assert _cache_control_of(enviados[-2]) == {"type": "ephemeral"}
    assert _cache_control_of(enviados[-1]) is None
    assert enviados[-1]["content"] == "PEGOTE"


def test_cache_control_convierte_la_cadena_en_bloques():
    msgs = [
        {"role": "user", "content": "apertura"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "PEGOTE"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    prefijo = body["messages"][-2]["content"]
    assert prefijo == [
        {
            "type": "text",
            "text": "respuesta",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_cache_control_respeta_el_contenido_que_ya_venia_en_bloques():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": "PEGOTE"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    bloques = body["messages"][-2]["content"]
    assert len(bloques) == 2
    assert "cache_control" not in bloques[0]
    assert bloques[1]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_no_muta_la_transcripcion_del_llamante():
    # La transcripción se guarda tal cual en el JSONL: el cuerpo de la llamada
    # no puede ensuciarla con bloques ni con cache_control.
    msgs = [
        {"role": "user", "content": "apertura"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "PEGOTE"},
    ]
    _anthropic_body(msgs, max_tokens=32)
    assert msgs[1]["content"] == "respuesta"


def test_sin_prefijo_no_hay_cache_control():
    # Un único mensaje no tiene delante nada estable que cachear.
    body = _anthropic_body([{"role": "user", "content": "x"}], max_tokens=32)
    assert body["messages"] == [{"role": "user", "content": "x"}]


# --- D10 × D7 × D11: el breakpoint se decide por etiquetas, no por posición --


def _prefijo_etiquetado() -> list[dict]:
    """Prefijo de dos turnos tal y como lo fabrica `conversation.build_prefix`."""
    return [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "resp-1", "tag": "assistant"},
        {"role": "user", "content": "seguimiento", "tag": "user_sim"},
        {"role": "assistant", "content": "FIN-DEL-PREFIJO", "tag": "assistant"},
    ]


def _texto_marcado(messages: list[dict]) -> str:
    """Texto del único bloque que lleva `cache_control`."""
    marcados = [
        block["text"]
        for message in messages
        if not isinstance(message["content"], str)
        for block in message["content"]
        if "cache_control" in block
    ]
    assert len(marcados) == 1, f"se esperaba un único breakpoint, hay {len(marcados)}"
    return marcados[0]


def test_breakpoint_en_el_pegote_sigue_al_final_del_prefijo():
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert _texto_marcado(body["messages"]) == "FIN-DEL-PREFIJO"
    assert body["messages"][-1]["content"] == "PEGOTE"


def test_breakpoint_con_turnos_post_no_cae_sobre_la_reaccion():
    # D7: tras la reacción vienen dos turnos más. El último mensaje ya no es el
    # pegote, así que marcar el penúltimo pondría el breakpoint sobre texto que
    # cambia en cada celda y la caché no se leería nunca entre celdas.
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
        {"role": "assistant", "content": "REACCION", "tag": "assistant"},
        {"role": "user", "content": "post-1", "tag": "post"},
        {"role": "assistant", "content": "POSTERIOR", "tag": "assistant"},
        {"role": "user", "content": "post-2", "tag": "post"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    marcado = _texto_marcado(body["messages"])
    assert marcado == "FIN-DEL-PREFIJO"
    assert marcado not in ("REACCION", "POSTERIOR")


def test_breakpoint_en_celda_de_control_sin_pegote():
    # D11: el brazo de control no tiene mensaje `paste`. El prefijo acaba igual,
    # justo antes del primer turno `post`.
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "post-1", "tag": "post"},
        {"role": "assistant", "content": "POSTERIOR", "tag": "assistant"},
        {"role": "user", "content": "post-2", "tag": "post"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert _texto_marcado(body["messages"]) == "FIN-DEL-PREFIJO"


def test_breakpoint_durante_la_construccion_del_prefijo():
    # Todavía no hay pegote ni turnos post: el prefijo es toda la lista, y el
    # breakpoint va en la última respuesta del asistente.
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "seguimiento-2", "tag": "user_sim"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert _texto_marcado(body["messages"]) == "FIN-DEL-PREFIJO"


def test_breakpoint_sin_tags_degrada_al_penultimo_mensaje():
    # Comportamiento degradado y documentado: sin etiquetas no hay forma de
    # saber dónde acaba el prefijo, así que se supone que el último mensaje es
    # el pegote. Solo vale para llamadas sueltas.
    msgs = [
        {"role": "user", "content": "apertura"},
        {"role": "assistant", "content": "PENULTIMO"},
        {"role": "user", "content": "ultimo"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert _texto_marcado(body["messages"]) == "PENULTIMO"


def test_breakpoint_sin_respuesta_de_asistente_en_el_prefijo():
    # Pegote inmediatamente después de la apertura: no hay nada estable que
    # cachear delante, así que no se marca nada en vez de marcar la apertura.
    msgs = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    assert all(isinstance(m["content"], str) for m in body["messages"])


def test_los_tags_no_se_envian_a_ningun_proveedor():
    # `tag` es metadato nuestro (D5): mandarlo dentro de `messages` devuelve 400.
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
    ]
    cuerpos = [
        _anthropic_body(msgs, max_tokens=32),
        _gateway_body("gpt-5.6-sol-tst", msgs, 32),
        _vertex_openai_body("gemini-2.5-pro", msgs, 32),
    ]
    for body in cuerpos:
        for message in body["messages"]:
            assert set(message) == {"role", "content"}


def test_marcar_el_prefijo_no_muta_la_transcripcion_etiquetada():
    msgs = _prefijo_etiquetado() + [
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
    ]
    _anthropic_body(msgs, max_tokens=32)
    assert msgs[3] == {
        "role": "assistant",
        "content": "FIN-DEL-PREFIJO",
        "tag": "assistant",
    }


# --- D6: reintentos, stop_reason y trazas -----------------------------------


def test_backoff_es_exponencial_de_un_segundo():
    assert [_backoff_seconds(i) for i in (1, 2, 3)] == [1.0, 2.0, 4.0]


def test_chat_devuelve_stop_reason_modelo_y_trazas_en_gateway(monkeypatch):
    _install_fake_post(monkeypatch, [_FakeResponse(200, _openai_payload())])
    reply = chat("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}])
    assert reply.text == "hola"
    assert reply.stop_reason == "stop"
    assert reply.response_model == "gpt-5.6-sol-tst"
    assert reply.attempts == 1
    assert reply.latency_ms is not None and reply.latency_ms >= 0


def test_chat_devuelve_stop_reason_de_anthropic(monkeypatch):
    _install_fake_post(monkeypatch, [_FakeResponse(200, _anthropic_payload(stop="refusal"))])
    reply = chat("claude-opus-5", [{"role": "user", "content": "x"}])
    assert reply.stop_reason == "refusal"
    assert reply.response_model == "claude-opus-5"


def test_chat_reintenta_el_429_y_cuenta_los_intentos(monkeypatch):
    calls = _install_fake_post(
        monkeypatch,
        [_FakeResponse(429), _FakeResponse(200, _openai_payload())],
    )
    reply = chat("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}])
    assert reply.attempts == 2
    assert len(calls) == 2


def test_chat_reintenta_los_5xx_y_se_rinde_al_tercero(monkeypatch):
    calls = _install_fake_post(
        monkeypatch,
        [_FakeResponse(503), _FakeResponse(500), _FakeResponse(500)],
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        chat("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}])
    assert len(calls) == 3
    assert exc.value.attempts == 3


def test_chat_no_reintenta_un_400(monkeypatch):
    # Un 400 es un error nuestro: esperar no lo arregla y gasta la tirada.
    calls = _install_fake_post(monkeypatch, [_FakeResponse(400)])
    with pytest.raises(httpx.HTTPStatusError):
        chat("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_chat_rellena_request_params_out_sin_los_mensajes(monkeypatch):
    _install_fake_post(monkeypatch, [_FakeResponse(200, _anthropic_payload())])
    params: dict = {}
    chat(
        "claude-opus-5",
        [{"role": "user", "content": "x"}],
        max_tokens=64,
        request_params_out=params,
    )
    assert "messages" not in params
    # Claude 5 no admite muestreo explícito, así que aquí no debe aparecer.
    assert "temperature" not in params
    assert params["thinking"] == {"type": "adaptive"}
    assert params["max_tokens"] == 64


def test_chat_rellena_request_params_out_aunque_la_llamada_falle(monkeypatch):
    # La fila se escribe igual (D6), así que los parámetros tienen que estar.
    _install_fake_post(monkeypatch, [_FakeResponse(400)])
    params: dict = {}
    with pytest.raises(httpx.HTTPStatusError):
        chat(
            "gpt-5.6-sol-tst",
            [{"role": "user", "content": "x"}],
            request_params_out=params,
        )
    assert params["temperature"] == TEMPERATURE
    assert "messages" not in params


def test_chat_devuelve_texto_vacio_sin_reventar(monkeypatch):
    # Un `content: null` es un resultado (status `empty`), no una excepción.
    _install_fake_post(monkeypatch, [_FakeResponse(200, _openai_payload(content=None))])
    reply = chat("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}])
    assert reply.text == ""


def test_anthropic_body_nunca_manda_parametros_de_muestreo():
    """En la familia Claude 5, temperature/top_p/top_k devuelven 400.

    La version original de D9 pedia temperatura explicita en los tres cuerpos.
    Era un error: habria tumbado todas las celdas de Claude de la tirada.
    """
    body = _anthropic_body([{"role": "user", "content": "x"}], max_tokens=16)
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert "budget_tokens" not in body.get("thinking", {})


# --- D10 por la RUTA REAL: transcripción de `conversation`, cuerpo enviado ---
#
# Los tests de arriba construyen las listas de mensajes a mano. Eso dejó pasar
# el fallo que cerró este bloqueante: `conversation.api_messages()` quitaba el
# `tag` ANTES de llamar a `chat()`, así que en la tirada de verdad
# `_prefix_breakpoint_index` entraba siempre por la rama degradada
# `len(convo) - 2` y 17 de los 26 cuerpos anthropic colgaban el `cache_control`
# de la reacción al pegote o de una respuesta a un turno `post` —contenido que
# cambia en cada celda—. La suite seguía verde porque ningún test recorría el
# camino que recorre `main()`.
#
# Estos tests no construyen mensajes: los fabrican `conversation.build_prefix`,
# `inject_paste` y `continue_after_paste`, y comprueban el cuerpo **realmente
# enviado** por `httpx.post`, que es lo único que ve el proveedor.


class _RedFalsa:
    """Dobla solo `clients.httpx.post`: de ahí para arriba, todo es el código real.

    Responde como Anthropic a las URLs de `rawPredict` y como OpenAI al resto
    (gateway y Vertex-OpenAI), con un texto distinto por llamada para poder
    distinguir el final del prefijo, la reacción al pegote y las respuestas a
    los turnos posteriores.
    """

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        monkeypatch.setattr(clients.httpx, "post", self._post)
        monkeypatch.setattr(clients.time, "sleep", lambda _s: None)
        monkeypatch.setattr(clients.config, "gateway_key", lambda: "clave-de-prueba")
        monkeypatch.setattr(clients, "_gcp_token", lambda: "token-de-prueba")

    def _post(self, url, headers=None, json=None, timeout=None):
        numero = len(self.calls) + 1
        es_anthropic = ":rawPredict" in url
        self.calls.append({"url": url, "body": json, "anthropic": es_anthropic})
        if es_anthropic:
            return _FakeResponse(200, _anthropic_payload(text=f"CLAUDE-{numero}"))
        return _FakeResponse(200, _openai_payload(content=f"OPENAI-{numero}"))

    def cuerpos_anthropic(self) -> list[dict]:
        return [c["body"] for c in self.calls if c["anthropic"]]

    def cuerpos(self) -> list[dict]:
        return [c["body"] for c in self.calls]


_TOPIC_REAL = Topic(
    id="mudanza",
    opening="estoy pensando en mudarme de ciudad y no sé por dónde empezar",
    goals=("presupuesto", "barrios", "colegios", "fechas"),
)
_ARTIFACT_REAL = Artifact(
    id="stacktrace-1",
    kind="stacktrace",
    text="Traceback (most recent call last):\n  File 'app.py', line 12\nKeyError: 'saldo'",
    entities=("KeyError", "app.py"),
)


def test_ruta_real_el_breakpoint_cae_al_final_del_prefijo_en_el_pegote(monkeypatch):
    red = _RedFalsa(monkeypatch)

    transcript, _, _ = conv.build_prefix("claude-opus-5", _TOPIC_REAL, n_turns=2)
    fin_del_prefijo = transcript[-1]["content"]
    ya_enviados = len(red.cuerpos_anthropic())

    reaccion, _, _, _ = conv.inject_paste("claude-opus-5", transcript, _ARTIFACT_REAL)

    cuerpos = red.cuerpos_anthropic()
    assert len(cuerpos) == ya_enviados + 1, "el pegote es una llamada más"
    enviado = cuerpos[ya_enviados]

    marcado = _texto_marcado(enviado["messages"])
    assert marcado == fin_del_prefijo
    assert marcado != reaccion
    assert marcado != _ARTIFACT_REAL.text
    # El pegote va tal cual y sin breakpoint: cambia en cada celda.
    assert enviado["messages"][-1]["content"] == _ARTIFACT_REAL.text
    assert _cache_control_of(enviado["messages"][-1]) is None


def test_ruta_real_el_breakpoint_no_cae_sobre_la_reaccion_en_los_turnos_post(monkeypatch):
    # Este es el caso que la rama degradada `len(convo) - 2` fallaba: el último
    # mensaje ya no es el pegote, así que el penúltimo es la reacción o una
    # respuesta a un turno `post`.
    red = _RedFalsa(monkeypatch)

    transcript, _, _ = conv.build_prefix("claude-opus-5", _TOPIC_REAL, n_turns=2)
    fin_del_prefijo = transcript[-1]["content"]
    reaccion, _, _, _ = conv.inject_paste("claude-opus-5", transcript, _ARTIFACT_REAL)
    ya_enviados = len(red.cuerpos_anthropic())

    conv.continue_after_paste("claude-opus-5", transcript, _TOPIC_REAL)

    cuerpos_post = red.cuerpos_anthropic()[ya_enviados:]
    assert len(cuerpos_post) == conv.N_POST_TURNS

    for enviado in cuerpos_post:
        marcado = _texto_marcado(enviado["messages"])
        assert marcado == fin_del_prefijo
        assert marcado != reaccion
        assert marcado != _ARTIFACT_REAL.text
        # La reacción viajó en el cuerpo, pero sin breakpoint encima.
        assert any(m["content"] == reaccion for m in enviado["messages"])
        assert _cache_control_of(enviado["messages"][-2]) is None


def test_ruta_real_celda_de_control_sin_pegote(monkeypatch):
    # D11: el brazo de control no inyecta pegote; va del prefijo a los turnos
    # `post` directamente. El prefijo acaba en el mismo sitio.
    red = _RedFalsa(monkeypatch)

    transcript, _, _ = conv.build_prefix("claude-opus-5", _TOPIC_REAL, n_turns=2)
    fin_del_prefijo = transcript[-1]["content"]
    ya_enviados = len(red.cuerpos_anthropic())

    conv.continue_after_paste("claude-opus-5", transcript, _TOPIC_REAL)

    assert all(m.get("tag") != "paste" for m in transcript)
    cuerpos_post = red.cuerpos_anthropic()[ya_enviados:]
    assert len(cuerpos_post) == conv.N_POST_TURNS
    for enviado in cuerpos_post:
        assert _texto_marcado(enviado["messages"]) == fin_del_prefijo


def test_ruta_real_el_prefijo_enviado_es_identico_en_las_tres_llamadas(monkeypatch):
    # Lo que hace útil el breakpoint: los mensajes hasta el punto marcado son
    # byte a byte los mismos en la llamada del pegote y en los dos turnos post.
    # Si el breakpoint se moviera, esta igualdad dejaría de servir de nada.
    red = _RedFalsa(monkeypatch)

    transcript, _, _ = conv.build_prefix("claude-opus-5", _TOPIC_REAL, n_turns=2)
    largo_prefijo = len(transcript)
    conv.inject_paste("claude-opus-5", transcript, _ARTIFACT_REAL)
    conv.continue_after_paste("claude-opus-5", transcript, _TOPIC_REAL)

    cuerpos = red.cuerpos_anthropic()[largo_prefijo // 2 :]
    assert len(cuerpos) == 1 + conv.N_POST_TURNS
    prefijos = [tuple(str(m) for m in c["messages"][:largo_prefijo]) for c in cuerpos]
    assert len(set(prefijos)) == 1, "el prefijo enviado tiene que ser idéntico"


@pytest.mark.parametrize(
    "model_id", ["claude-opus-5", "gpt-5.6-sol-tst", "gemini-2.5-pro"]
)
def test_ruta_real_ningun_proveedor_recibe_la_clave_tag(monkeypatch, model_id):
    # El saneado se hace ahora dentro de `clients`, en el último momento. Que
    # siga haciéndose se comprueba donde importa: en el cuerpo HTTP enviado,
    # por los tres proveedores y en las tres fases de la conversación.
    red = _RedFalsa(monkeypatch)

    transcript, _, _ = conv.build_prefix(model_id, _TOPIC_REAL, n_turns=2)
    conv.inject_paste(model_id, transcript, _ARTIFACT_REAL)
    conv.continue_after_paste(model_id, transcript, _TOPIC_REAL)

    # Prefijo (2) + pegote (1) + post (2) del modelo evaluado, más los turnos
    # del usuario simulado, que también pasan por aquí.
    assert len(red.cuerpos()) >= 5
    for enviado in red.cuerpos():
        for message in enviado["messages"]:
            assert set(message) == {"role", "content"}, message

    # Y el transcript que se guarda en el JSONL sigue etiquetado.
    assert all(m.get("tag") for m in transcript)


def test_no_se_marca_como_cacheable_un_bloque_de_texto_vacio():
    """La API devuelve 400 si el bloque del breakpoint esta vacio.

    Ocurre de verdad: si un turno anterior se corto por `max_tokens`, su
    respuesta entra vacia en la transcripcion y el breakpoint cae encima.
    Paso en 3 de las 27 celdas de la primera tirada de Fase 0.
    """
    msgs = [
        {"role": "user", "content": "apertura", "tag": "opening"},
        {"role": "assistant", "content": "", "tag": "assistant"},
        {"role": "user", "content": "PEGOTE", "tag": "paste"},
    ]
    body = _anthropic_body(msgs, max_tokens=32)
    for mensaje in body["messages"]:
        contenido = mensaje["content"]
        if isinstance(contenido, list):
            for bloque in contenido:
                assert "cache_control" not in bloque


def test_embed_reintenta_el_408_del_gateway(monkeypatch):
    """Un 408 transitorio en embeddings tumbo la segunda tirada de Fase 0.

    Los reintentos vivian solo en chat(); embed() llamaba a httpx.post a pelo y
    measure_axis corre antes de abrir el fichero de salida y fuera de todo try,
    asi que un timeout del gateway mataba la tirada sin escribir ni la cabecera.
    """
    payload = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
    llamadas = _install_fake_post(
        monkeypatch,
        [_FakeResponse(408), _FakeResponse(200, payload)],
    )
    monkeypatch.setattr(clients.config, "gateway_url", lambda: "https://gw.test")

    vecs = clients.embed(["hola"])

    assert vecs.shape == (1, 3)
    assert len(llamadas) == 2, "no reintento el 408"


def test_el_408_es_reintentable():
    assert clients._is_retriable(408)
    assert clients._is_retriable(429)
    assert clients._is_retriable(503)
    assert not clients._is_retriable(400)
