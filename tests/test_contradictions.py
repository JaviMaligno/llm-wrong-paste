"""Tests del generador N2: pegotes fabricados CONTRA la conversación.

`test_el_generador_ve_la_conversacion_a_proposito` es el test raro de esta
suite, y lo es a propósito: comprueba que aquí el pegote se escribe sabiendo de
qué va la conversación, que es exactamente lo que N0 y N1 tienen prohibido. Ese
test es la documentación ejecutable de por qué N2 no se compara nunca con N0 ni
con N1 (spec §5): no mide un accidente, mide un distractor.
"""

import inspect

import pytest

import wrongpaste.clients as clients
import wrongpaste.contradictions as cd
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic("mudanza", "me mudo el mes que viene", ("a", "b", "c", "d"))
TRANSCRIPT = [
    {"role": "user", "content": "Me mudo a Gijón el 12 de mayo", "tag": "opening"},
    {"role": "assistant", "content": "Vale, ¿piso o casa?", "tag": "assistant"},
]


def test_el_artefacto_generado_se_marca_como_n2(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '{"text": "Te confirmo Santander para el 3 de junio", '
        '"entities": ["Santander", "3 de junio"]}', {}, {}))
    art = cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)
    assert art.level == "N2"
    assert art.signal == "contradiccion"
    assert art.entities


def test_el_generador_ve_la_conversacion_a_proposito(monkeypatch):
    # Es la diferencia con N0/N1: aquí SÍ se condiciona al tema, y por eso
    # N2 queda fuera de toda comparación (spec §5).
    visto = {}
    def fake(model_id, messages, **kw):
        visto["prompt"] = messages[-1]["content"]
        return Reply('{"text": "x", "entities": ["x"]}', {}, {})
    monkeypatch.setattr(cd, "chat", fake)
    cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)
    assert "Gijón" in visto["prompt"]


def test_es_determinista_para_una_semilla(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '{"text": "y", "entities": ["y"]}', {}, {}))
    a = cd.make_contradiction(TOPIC, TRANSCRIPT, seed=7)
    b = cd.make_contradiction(TOPIC, TRANSCRIPT, seed=7)
    assert a.id == b.id


def test_una_respuesta_no_json_revienta_con_mensaje_claro(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply("lo siento", {}, {}))
    try:
        cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("debería haber lanzado ValueError")


def test_el_doble_de_chat_casa_con_la_firma_real(monkeypatch):
    """Comprueba la llamada REAL contra la firma REAL de `chat()`.

    Es el patrón que ya ha roto este repositorio varias veces: un doble que
    acepta cualquier cosa (`*a, **k`), una suite en verde y una llamada que en
    producción reventaría porque el argumento no existe. Aquí se capturan los
    argumentos con los que `make_contradiction` llama de verdad y se atan
    contra `inspect.signature(clients.chat)`, que es quien manda.
    """
    llamada: dict = {}

    def fake(*args, **kwargs):
        llamada["args"] = args
        llamada["kwargs"] = kwargs
        return Reply('{"text": "z", "entities": ["z"]}', {}, {})

    monkeypatch.setattr(cd, "chat", fake)
    cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)

    inspect.signature(clients.chat).bind(*llamada["args"], **llamada["kwargs"])
    # Y el `Reply` de tres posicionales que devuelven los dobles de arriba
    # tiene que ser construible con la firma real del dataclass.
    inspect.signature(Reply).bind("texto", {}, {})


def test_el_generador_es_invocable_y_es_el_de_los_prefijos():
    from wrongpaste.config import MODELS
    from wrongpaste.prefixes import PREFIX_MODEL

    assert cd.CONTRADICTION_MODEL in MODELS, "chat() tiene que poder llamarlo"
    # El comentario del plan dice «el mismo que escribe los prefijos»; si
    # alguien cambia uno de los dos, que se entere aquí y no leyendo prosa.
    assert cd.CONTRADICTION_MODEL == PREFIX_MODEL


def test_un_json_sin_entities_revienta_en_vez_de_callarse(monkeypatch):
    # Un artefacto sin entities no rompe nada visible: produce un Artifact
    # válido con `entities=()` y deja el recuento de fuga (D8) midiendo cero
    # para siempre en esa celda. Eso es peor que un fallo, así que falla.
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '{"text": "Te confirmo Santander"}', {}, {}))
    with pytest.raises(ValueError, match="entities"):
        cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)


def test_una_lista_de_entities_vacia_tambien_revienta(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '{"text": "Te confirmo Santander", "entities": []}', {}, {}))
    with pytest.raises(ValueError, match="entities"):
        cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)


def test_un_json_sin_texto_revienta(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '{"entities": ["Santander"]}', {}, {}))
    with pytest.raises(ValueError, match="text"):
        cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)


def test_acepta_el_json_envuelto_en_una_valla_de_codigo(monkeypatch):
    monkeypatch.setattr(cd, "chat", lambda *a, **k: Reply(
        '```json\n{"text": "Santander, 3 de junio", '
        '"entities": ["Santander"]}\n```', {}, {}))
    art = cd.make_contradiction(TOPIC, TRANSCRIPT, seed=1)
    assert art.text == "Santander, 3 de junio"


# --- Persistencia: una contradicción por (prefijo, semilla) ------------------
# D1 exige que las dos réplicas de una celda compartan prefijo Y artefacto. En
# N0 y N1 eso sale gratis porque el artefacto se elige de un banco en disco; en
# N2 el artefacto lo escribe un modelo, así que dos llamadas darían dos textos
# distintos y las réplicas dejarían de medir lo que deben medir. Por eso el
# runner no llama a `make_contradiction`, llama a `ensure_contradiction`.


@pytest.fixture
def dir_temporal(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "CONTRADICTION_DIR", tmp_path)
    return tmp_path


def _fake_chat(texto="Santander el 3 de junio", entities=("Santander",)):
    import json as _json

    cuerpo = _json.dumps({"text": texto, "entities": list(entities)})
    return lambda *a, **k: Reply(cuerpo, {}, {})


def test_la_ruta_cuelga_del_directorio_de_contradicciones(dir_temporal):
    ruta = cd.contradiction_path("abc123", 4)
    assert ruta.parent == dir_temporal
    assert "abc123" in ruta.name and "4" in ruta.name


def test_guardar_y_releer_devuelve_el_mismo_artefacto(dir_temporal, monkeypatch):
    monkeypatch.setattr(cd, "chat", _fake_chat())
    art = cd.make_contradiction(TOPIC, TRANSCRIPT, seed=3)
    cd.save_contradiction("pref1", 3, art)
    assert cd.load_contradiction("pref1", 3) == art


def test_sin_fichero_load_devuelve_none(dir_temporal):
    assert cd.load_contradiction("no-existe", 1) is None


def test_ensure_no_vuelve_a_pagar_el_modelo(dir_temporal, monkeypatch):
    llamadas = []

    def fake(*a, **k):
        llamadas.append(1)
        return Reply('{"text": "Santander", "entities": ["Santander"]}', {}, {})

    monkeypatch.setattr(cd, "chat", fake)
    primero = cd.ensure_contradiction("pref1", TOPIC, TRANSCRIPT, seed=3)
    segundo = cd.ensure_contradiction("pref1", TOPIC, TRANSCRIPT, seed=3)
    assert primero == segundo, "las dos réplicas comparten artefacto (D1)"
    assert len(llamadas) == 1, "la segunda réplica no vuelve a llamar al modelo"


def test_ensure_distingue_prefijo_y_semilla(dir_temporal, monkeypatch):
    monkeypatch.setattr(cd, "chat", _fake_chat())
    cd.ensure_contradiction("pref1", TOPIC, TRANSCRIPT, seed=3)
    cd.ensure_contradiction("pref2", TOPIC, TRANSCRIPT, seed=3)
    cd.ensure_contradiction("pref1", TOPIC, TRANSCRIPT, seed=4)
    assert len(list(dir_temporal.glob("*.json"))) == 3
