import httpx
import pytest

from wrongpaste import clients
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
