import inspect
import json

import pytest

import wrongpaste.conversation as conv
import wrongpaste.prefixes as prefixes
import wrongpaste.simulated_user as su
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic(id="mudanza", opening="quiero mudarme", goals=("a", "b"))

TRANSCRIPT = [
    {"role": "user", "content": "quiero mudarme", "tag": "opening"},
    {"role": "assistant", "content": "cuéntame más", "tag": "assistant"},
]

# Un `Reply` como el que devuelve de verdad `clients.chat`: con `raw`, que es
# la respuesta entera del proveedor y lo que hace que un `Reply` NO sea
# serializable a JSON. El doble tiene que arrastrarlo, porque si devolviera
# dicts limpios el test dejaría pasar un `generate_prefix` que mete objetos
# vivos en el fichero del prefijo.
REPLY = Reply(
    text="cuéntame más",
    usage={"total_tokens": 1},
    raw={"choices": [{"message": {"content": "cuéntame más"}}]},
    stop_reason="stop",
    response_model="gpt-5.6-terra-tst-20260101",
    attempts=2,
    latency_ms=12.5,
)

TRACES = [
    {
        "stop_reason": "stop",
        "response_model": "gpt-5.6-terra-tst-20260101",
        "attempts": 2,
        "latency_ms": 12.5,
    }
]

# El `Reply` de un turno del **usuario simulado**. El prefijo lo fabrican dos
# papeles —el asistente (`PREFIX_MODEL`) y el usuario simulado— y sus trazas van
# por separado: si un turno del usuario sale cortado por el tope, el contexto
# compartido que ven los tres modelos evaluados (D1) queda roto, y sin
# `user_reply_traces` ninguna fila del JSONL podría decirlo.
USER_REPLY = Reply(
    text="y luego qué",
    usage={"total_tokens": 2},
    raw={"choices": [{"message": {"content": "y luego qué"}}]},
    stop_reason="stop",
    response_model="gpt-5.6-terra-tst-20260101",
    attempts=1,
    latency_ms=3.5,
)

USER_TRACES = [
    {
        "stop_reason": "stop",
        "response_model": "gpt-5.6-terra-tst-20260101",
        "attempts": 1,
        "latency_ms": 3.5,
    }
]


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Ningún test escribe en `runs/prefixes` de verdad."""
    monkeypatch.setattr(prefixes, "PREFIX_DIR", tmp_path / "prefixes")


def _stub_build(monkeypatch, transcript=None, calls=None):
    """Corta la única puerta de red: `build_prefix`.

    El doble copia la firma REAL de `conversation.build_prefix`: devuelve
    **tres** valores —`(transcript, usages, replies)`— y acepta
    `request_params_out` como kwarg. Un doble que devolviera dos se queda
    verde mientras el código de verdad revienta con `ValueError: too many
    values to unpack`, que es exactamente lo que pasó. Contra eso hay dos
    guardas abajo: `test_the_build_prefix_double_matches_the_real_signature`
    y, sobre todo, el test de costura, que no dobla `build_prefix` en
    absoluto.
    """

    def fake_build(
        model_id, topic, n_turns, request_params_out=None, user_replies_out=None
    ):
        if calls is not None:
            calls.append((model_id, topic.id, n_turns))
        if request_params_out is not None:
            request_params_out.clear()
            request_params_out.update({"max_tokens": conv.MAX_TOKENS})
        # `build_prefix` apila aquí un `Reply` por turno del usuario simulado
        # (uno menos que turnos: la apertura la escribimos nosotros). El doble
        # lo imita porque `generate_prefix` guarda esas trazas en el prefijo.
        if user_replies_out is not None:
            user_replies_out.extend([USER_REPLY] * max(n_turns - 1, 0))
        return list(transcript or TRANSCRIPT), [{"total_tokens": 1}], [REPLY]

    monkeypatch.setattr(prefixes, "build_prefix", fake_build)
    return fake_build


def test_prefix_id_is_deterministic():
    first = prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT)
    second = prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT)

    assert first == second
    assert len(first) == 16
    assert all(c in "0123456789abcdef" for c in first)


def test_prefix_id_changes_with_the_transcript():
    base = prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT)
    other = [dict(m) for m in TRANSCRIPT]
    other[-1]["content"] = "otra respuesta"

    assert prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, other) != base


def test_prefix_id_changes_with_the_tags():
    """Las etiquetas son parte del contexto guardado, no adorno."""
    base = prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT)
    other = [dict(m) for m in TRANSCRIPT]
    other[0]["tag"] = "user_sim"

    assert prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, other) != base


def test_prefix_id_changes_with_topic_length_and_model():
    base = prefixes.prefix_id("mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT)

    assert prefixes.prefix_id("hacer-pan", 2, prefixes.PREFIX_MODEL, TRANSCRIPT) != base
    assert prefixes.prefix_id("mudanza", 10, prefixes.PREFIX_MODEL, TRANSCRIPT) != base
    assert prefixes.prefix_id("mudanza", 2, "claude-opus-5", TRANSCRIPT) != base


def test_generate_prefix_uses_the_fixed_prefix_model(monkeypatch):
    calls: list[tuple] = []
    _stub_build(monkeypatch, calls=calls)

    prefix = prefixes.generate_prefix(TOPIC, 2)

    assert calls == [(prefixes.PREFIX_MODEL, "mudanza", 2)]
    assert prefix["prefix_model"] == prefixes.PREFIX_MODEL
    assert prefix["topic_id"] == "mudanza"
    assert prefix["n_turns"] == 2
    assert prefix["transcript"] == TRANSCRIPT
    assert prefix["usages"] == [{"total_tokens": 1}]
    assert prefix["reply_traces"] == TRACES
    assert prefix["prefix_id"] == prefixes.prefix_id(
        "mudanza", 2, prefixes.PREFIX_MODEL, TRANSCRIPT
    )


# --- BLOQUEANTE: el usuario simulado también deja traza en el prefijo -------


def test_generate_prefix_records_the_simulated_users_traces(monkeypatch):
    """Sin esto, el `stop_reason` de sus turnos no llegaba a ninguna parte.

    El prefijo lo fabrican dos papeles y se le sirve idéntico a los tres
    modelos evaluados (D1). Si un turno del usuario simulado sale cortado por el
    tope, el agujero está en el contexto **compartido**, así que afecta a tres
    filas a la vez; antes, ninguna de las tres podía decirlo porque
    `build_prefix` se llamaba sin `user_replies_out` y esas trazas se perdían.
    """
    _stub_build(monkeypatch)

    prefix = prefixes.generate_prefix(TOPIC, 3)

    # Dos turnos de usuario en un prefijo de tres (la apertura no la escribe él).
    assert prefix["user_reply_traces"] == USER_TRACES * 2
    assert prefix["user_model"] == su.USER_MODEL
    # Y no se mezclan con las del asistente del prefijo.
    assert prefix["reply_traces"] == TRACES


def test_the_two_turn_prefix_has_exactly_one_user_turn(monkeypatch):
    _stub_build(monkeypatch)
    assert prefixes.generate_prefix(TOPIC, 2)["user_reply_traces"] == USER_TRACES


def test_the_user_traces_survive_the_round_trip_to_disk(monkeypatch):
    """De poco sirve la traza si se queda en memoria: el runner la lee de disco."""
    _stub_build(monkeypatch)
    prefix = prefixes.generate_prefix(TOPIC, 3)

    prefixes.save_prefix(prefix)
    leido = prefixes.load_prefix(prefix["prefix_id"])

    assert leido["user_reply_traces"] == prefix["user_reply_traces"]


def test_generate_prefix_does_not_write_anything(monkeypatch):
    _stub_build(monkeypatch)
    prefixes.generate_prefix(TOPIC, 2)

    assert not prefixes.PREFIX_DIR.exists()


def test_save_and_load_round_trip(monkeypatch):
    _stub_build(monkeypatch)
    prefix = prefixes.generate_prefix(TOPIC, 2)

    path = prefixes.save_prefix(prefix)

    assert path.name == f"{prefix['prefix_id']}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == prefix
    assert prefixes.load_prefix(prefix["prefix_id"]) == prefix


def test_ensure_prefix_generates_when_missing(monkeypatch):
    calls: list[tuple] = []
    _stub_build(monkeypatch, calls=calls)

    prefix = prefixes.ensure_prefix(TOPIC, 2)

    assert len(calls) == 1
    assert prefixes.prefix_path(prefix["prefix_id"]).exists()


def test_ensure_prefix_does_not_regenerate_when_the_file_exists(monkeypatch):
    calls: list[tuple] = []
    _stub_build(monkeypatch, calls=calls)
    first = prefixes.ensure_prefix(TOPIC, 2)

    second = prefixes.ensure_prefix(TOPIC, 2)

    assert len(calls) == 1, "el segundo paso no debe volver a llamar al modelo"
    assert second == first
    assert len(list(prefixes.PREFIX_DIR.glob("*.json"))) == 1


def test_ensure_prefix_is_per_topic_and_length(monkeypatch):
    _stub_build(monkeypatch)
    short = prefixes.ensure_prefix(TOPIC, 2)
    long = prefixes.ensure_prefix(TOPIC, 10)
    other = prefixes.ensure_prefix(
        Topic(id="hacer-pan", opening="quiero hacer pan", goals=("a",)), 2
    )

    ids = {short["prefix_id"], long["prefix_id"], other["prefix_id"]}
    assert len(ids) == 3
    assert len(list(prefixes.PREFIX_DIR.glob("*.json"))) == 3


def test_every_model_gets_the_very_same_prefix(monkeypatch):
    """D1: el eje x no puede depender de quién conteste."""
    _stub_build(monkeypatch)
    prefixes.ensure_prefix(TOPIC, 2)

    served = [prefixes.ensure_prefix(TOPIC, 2) for _ in range(3)]

    assert all(p["transcript"] == served[0]["transcript"] for p in served)
    assert len({p["prefix_id"] for p in served}) == 1


def test_find_prefix_returns_none_when_the_dir_is_empty(monkeypatch):
    _stub_build(monkeypatch)
    assert prefixes.find_prefix("mudanza", 2) is None


# --- Costuras: que el doble no se separe de la firma real -------------------


def _fake_user_turn(text: str):
    """Doble de `simulated_user.next_user_turn` con su firma real.

    Copia sus cuatro parámetros —`reply_out` incluido, que `build_prefix` ya le
    pasa— y lo alimenta como el original: así el doble no esconde el contrato
    de salida del usuario simulado. Un doble escrito como `lambda *a, **k` se
    lo tragaría todo y dejaría de avisar de cualquier cambio de firma.
    """

    def fake(topic, history, max_tokens=su.USER_MAX_TOKENS, reply_out=None):
        if reply_out is not None:
            reply_out.append(Reply(text=text, usage={}, raw={}, stop_reason="stop"))
        return text

    return fake


def test_the_build_prefix_double_matches_the_real_signature(monkeypatch):
    """El doble no puede declarar parámetros que el original no tenga.

    La comprobación es de inclusión y no de igualdad a propósito: que
    `build_prefix` gane un parámetro **opcional** nuevo no invalida al doble
    mientras `generate_prefix` no se lo pase. Lo que sí lo invalida —y es lo
    que este test caza— es que un parámetro se renombre o desaparezca y el
    doble siga aceptando el nombre viejo, que es cómo un doble empieza a
    mentir sobre la firma real.

    Del número de valores devueltos no se ocupa este test, sino el de costura:
    una firma no dice nada del tuple que sale por la otra punta, y era ahí
    donde estaba el fallo.
    """
    fake = _stub_build(monkeypatch)
    reales = inspect.signature(conv.build_prefix).parameters
    dobles = inspect.signature(fake).parameters

    assert set(dobles) <= set(reales), (
        f"el doble declara parámetros que `build_prefix` no tiene: "
        f"{sorted(set(dobles) - set(reales))}"
    )
    # Y la llamada que hace `generate_prefix` tiene que seguir casando.
    inspect.signature(conv.build_prefix).bind(prefixes.PREFIX_MODEL, TOPIC, 2)


def test_generate_prefix_against_the_real_build_prefix(monkeypatch):
    """Costura: `generate_prefix` contra el `build_prefix` DE VERDAD.

    Aquí NO se dobla `build_prefix`. Se doblan solo las dos puertas de red que
    hay debajo —`clients.chat` y `simulated_user.next_user_turn`, importadas
    en el espacio de nombres de `conversation`— y se deja que corra el código
    real de las dos funciones. Es el test que faltaba: el que comprueba que el
    número de valores que devuelve `build_prefix` es el que `generate_prefix`
    desempaqueta, sin que ningún doble pueda mentir sobre ello.
    """
    turnos = 3
    llamadas: list[int] = []

    def fake_chat(model_id, messages, max_tokens=1024, request_params_out=None):
        llamadas.append(len(messages))
        if request_params_out is not None:
            request_params_out.clear()
            request_params_out.update({"max_tokens": max_tokens})
        return Reply(
            text=f"respuesta {len(llamadas)}",
            usage={"total_tokens": 10 * len(llamadas)},
            raw={"content": [{"type": "text", "text": "respuesta"}]},
            stop_reason="end_turn",
            response_model=f"{model_id}-20260101",
            attempts=1,
            latency_ms=7.5,
        )

    monkeypatch.setattr(conv, "chat", fake_chat)
    monkeypatch.setattr(conv, "next_user_turn", _fake_user_turn("y luego qué"))

    prefix = prefixes.generate_prefix(TOPIC, turnos)

    # Una llamada al modelo por turno, y la transcripción alterna usuario y
    # asistente empezando por la apertura del tema.
    assert len(llamadas) == turnos
    assert len(prefix["transcript"]) == 2 * turnos
    assert prefix["transcript"][0] == {
        "role": "user",
        "content": TOPIC.opening,
        "tag": "opening",
    }
    assert [m["tag"] for m in prefix["transcript"]] == [
        "opening",
        "assistant",
        "user_sim",
        "assistant",
        "user_sim",
        "assistant",
    ]

    # La trazabilidad del prefijo: un dict por llamada, con los campos de D5.
    assert prefix["usages"] == [
        {"total_tokens": 10},
        {"total_tokens": 20},
        {"total_tokens": 30},
    ]
    assert prefix["reply_traces"] == [
        {
            "stop_reason": "end_turn",
            "response_model": f"{prefixes.PREFIX_MODEL}-20260101",
            "attempts": 1,
            "latency_ms": 7.5,
        }
    ] * turnos

    # Y la del otro papel, la que antes se perdía: `generate_prefix` le pasa
    # `user_replies_out` al `build_prefix` de verdad, y este se lo pasa a
    # `next_user_turn`. Dos turnos de usuario en un prefijo de tres.
    assert prefix["user_reply_traces"] == [
        {
            "stop_reason": "stop",
            "response_model": None,
            "attempts": 1,
            "latency_ms": None,
        }
    ] * (turnos - 1)


def test_the_real_prefix_survives_being_written_to_disk(monkeypatch):
    """Ningún `Reply` vivo puede colarse en el fichero del prefijo.

    `Reply` arrastra `raw`, la respuesta entera del proveedor, y no es
    serializable a JSON: guardarlo tal cual reventaría `save_prefix` en la
    primera celda de la tirada, con el prefijo ya pagado.
    """
    monkeypatch.setattr(
        conv,
        "chat",
        lambda model_id, messages, max_tokens=1024, request_params_out=None: Reply(
            text="vale",
            usage={"total_tokens": 3},
            raw={"objeto": object()},  # no serializable a propósito
            stop_reason="end_turn",
            response_model="gpt-5.6-terra-tst-20260101",
        ),
    )
    monkeypatch.setattr(conv, "next_user_turn", _fake_user_turn("sigue"))

    prefix = prefixes.generate_prefix(TOPIC, 2)
    path = prefixes.save_prefix(prefix)

    assert json.loads(path.read_text(encoding="utf-8")) == prefix
