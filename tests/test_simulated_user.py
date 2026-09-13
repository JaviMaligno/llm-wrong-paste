"""El usuario simulado: qué ve, con qué tope habla y qué deja registrado.

La única puerta de red del módulo es `chat`, y aquí se sustituye siempre. Lo que
se comprueba es el contrato que el resto del arnés da por hecho: que el prompt se
arma con el histórico **que se le pasa** (y solo con ese), que el tope de tokens
es un parámetro, y que el `stop_reason` de su llamada puede salir del módulo.
"""

import pytest

import wrongpaste.simulated_user as su
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic(id="t", opening="quiero mudarme", goals=("precio", "fechas"))

HISTORIA = [
    {"role": "user", "content": "quiero mudarme", "tag": "opening"},
    {"role": "assistant", "content": "¿a qué ciudad?", "tag": "assistant"},
]


def _stub(monkeypatch, reply=None, calls=None):
    def fake_chat(model_id, messages, max_tokens=None, **kwargs):
        if calls is not None:
            calls.append(
                {"model_id": model_id, "messages": messages, "max_tokens": max_tokens}
            )
        return reply or Reply("  me mudo a Lisboa  ", {}, {})

    monkeypatch.setattr(su, "chat", fake_chat)


def test_el_prompt_lleva_tema_objetivos_e_historico(monkeypatch):
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)

    su.next_user_turn(TOPIC, HISTORIA)

    prompt = calls[0]["messages"][-1]["content"]
    assert TOPIC.opening in prompt
    assert "- precio" in prompt and "- fechas" in prompt
    assert "assistant: ¿a qué ciudad?" in prompt


def test_habla_con_el_modelo_del_usuario_y_con_su_system(monkeypatch):
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)

    su.next_user_turn(TOPIC, HISTORIA)

    assert calls[0]["model_id"] == su.USER_MODEL
    assert calls[0]["messages"][0]["role"] == "system"


def test_devuelve_el_texto_sin_espacios_de_alrededor(monkeypatch):
    _stub(monkeypatch)
    assert su.next_user_turn(TOPIC, HISTORIA) == "me mudo a Lisboa"


# --- Defecto medio: el tope de tokens es parámetro --------------------------


def test_el_tope_por_defecto_es_el_del_usuario_simulado(monkeypatch):
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)

    su.next_user_turn(TOPIC, HISTORIA)

    assert su.USER_MAX_TOKENS == 400
    assert calls[0]["max_tokens"] == su.USER_MAX_TOKENS


def test_el_tope_se_puede_subir_por_llamada(monkeypatch):
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)

    su.next_user_turn(TOPIC, HISTORIA, max_tokens=900)

    assert calls[0]["max_tokens"] == 900


def test_el_stop_reason_del_turno_de_usuario_puede_salir_del_modulo(monkeypatch):
    """La traza completa de la llamada sale del módulo, turno bueno incluido."""
    _stub(
        monkeypatch,
        reply=Reply(
            "y entonces yo",
            {},
            {},
            stop_reason="stop",
            response_model="modelo-real",
            attempts=2,
            latency_ms=12.5,
        ),
    )
    trazas: list[Reply] = []

    texto = su.next_user_turn(TOPIC, HISTORIA, reply_out=trazas)

    assert texto == "y entonces yo"
    assert len(trazas) == 1
    assert trazas[0].stop_reason == "stop"
    assert trazas[0].response_model == "modelo-real"
    assert trazas[0].attempts == 2
    assert trazas[0].latency_ms == 12.5


def test_sin_reply_out_no_pasa_nada(monkeypatch):
    _stub(monkeypatch)
    assert su.next_user_turn(TOPIC, HISTORIA)


def test_reply_out_acumula_un_reply_por_turno(monkeypatch):
    _stub(monkeypatch)
    trazas: list[Reply] = []

    su.next_user_turn(TOPIC, HISTORIA, reply_out=trazas)
    su.next_user_turn(TOPIC, HISTORIA, reply_out=trazas)

    assert len(trazas) == 2


# --- BLOQUEANTE: un turno inservible no puede salir por la puerta -----------
#
# `next_user_turn` devolvía `reply.text.strip()` sin mirarlo. Una cadena vacía
# viajaba como turno bueno: la conversación que después se mide quedaba con un
# agujero, la fila del JSONL decía `ok` y, en `vertex_anthropic`, el mensaje de
# usuario vacío reventaba con un 400 en la llamada siguiente —la del modelo
# evaluado—, que es a quien se le acababa atribuyendo.


@pytest.mark.parametrize("texto", ["", "   ", "\n\t "])
def test_un_turno_vacio_levanta_en_vez_de_devolverse(monkeypatch, texto):
    _stub(monkeypatch, reply=Reply(texto, {}, {}, stop_reason="stop"))

    with pytest.raises(su.SimulatedUserError) as exc:
        su.next_user_turn(TOPIC, HISTORIA)

    assert "vacío" in str(exc.value)
    # El stop_reason va dentro: es el dato que explica el fallo.
    assert exc.value.stop_reason == "stop"
    assert "stop" in str(exc.value)
    assert exc.value.model_id == su.USER_MODEL


@pytest.mark.parametrize(
    "stop_reason", ["max_tokens", "length", "MAX_TOKENS", "max-tokens"]
)
def test_un_turno_cortado_por_tope_levanta_aunque_traiga_texto(
    monkeypatch, stop_reason
):
    """Con texto y todo: un turno cortado a media frase no es un turno."""
    _stub(monkeypatch, reply=Reply("y entonces yo", {}, {}, stop_reason=stop_reason))

    with pytest.raises(su.SimulatedUserError) as exc:
        su.next_user_turn(TOPIC, HISTORIA)

    assert "tope de tokens" in str(exc.value)
    assert exc.value.stop_reason == stop_reason
    assert stop_reason in str(exc.value)


def test_la_traza_del_turno_roto_sale_igual_por_reply_out(monkeypatch):
    """El `Reply` se apila antes de comprobar nada: la fila lo necesita."""
    _stub(
        monkeypatch,
        reply=Reply("", {}, {}, stop_reason="max_tokens", attempts=3),
    )
    trazas: list[Reply] = []

    with pytest.raises(su.SimulatedUserError):
        su.next_user_turn(TOPIC, HISTORIA, reply_out=trazas)

    assert len(trazas) == 1
    assert trazas[0].stop_reason == "max_tokens"
    assert trazas[0].attempts == 3


def test_un_turno_normal_no_lo_toca_la_guarda(monkeypatch):
    """La guarda no puede estorbar al caso bueno: `stop` y texto pasan."""
    _stub(monkeypatch, reply=Reply(" me mudo a Lisboa ", {}, {}, stop_reason="stop"))
    assert su.next_user_turn(TOPIC, HISTORIA) == "me mudo a Lisboa"


def test_el_vocabulario_de_corte_es_el_mismo_que_el_del_runner():
    """Dos listas separadas dejan pasar por bueno lo que la otra ya rechaza."""
    from wrongpaste import run_phase0 as rp

    assert {
        "".join(c for c in r.lower() if c.isalnum())
        for r in su.TRUNCATION_STOP_REASONS
    } == set(rp.TRUNCATION_STOP_REASONS)


# --- D7 variante (a): el usuario no puede ver el pegote ---------------------


@pytest.mark.parametrize("tag", ["paste", "repair"])
def test_se_niega_a_serializar_un_historico_contaminado(monkeypatch, tag):
    """Si alguien le pasa el transcript crudo, se planta en vez de tragárselo.

    Tragárselo es lo que convertía el brazo "sin reparación" de D7 en un brazo
    de reparación: el usuario simulado leía el pegote y podía comentarlo.
    """
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)
    historia = [*HISTORIA, {"role": "user", "content": "PEGOTE", "tag": tag}]

    with pytest.raises(ValueError) as exc:
        su.next_user_turn(TOPIC, historia)

    assert tag in str(exc.value)
    assert "user_visible_history" in str(exc.value)
    assert calls == [], "ni siquiera se gasta la llamada"


def test_un_historico_limpio_pasa_sin_ruido(monkeypatch):
    """La guarda no puede estorbar al caso normal: `post` y `user_sim` valen."""
    calls: list[dict] = []
    _stub(monkeypatch, calls=calls)
    historia = [
        *HISTORIA,
        {"role": "user", "content": "sigo", "tag": "user_sim"},
        {"role": "assistant", "content": "claro", "tag": "assistant"},
        {"role": "user", "content": "y ahora", "tag": "post"},
        {"role": "assistant", "content": "vale", "tag": "assistant"},
    ]

    su.next_user_turn(TOPIC, historia)

    assert len(calls) == 1


def test_un_historico_sin_etiquetas_tambien_pasa(monkeypatch):
    _stub(monkeypatch)
    assert su.next_user_turn(TOPIC, [{"role": "user", "content": "hola"}])
