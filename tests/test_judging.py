import inspect

import pytest

import wrongpaste.clients as clients
import wrongpaste.judging as jd
from wrongpaste.clients import Reply


def _fake(texto):
    return lambda *a, **k: Reply(texto, {}, {})


def test_devuelve_categoria_y_cita(monkeypatch):
    monkeypatch.setattr(jd, "chat", _fake(
        '{"category": "C", "quote": "Veo que has cambiado de tema", '
        '"confidence": 0.9}'))
    v = jd.judge_one("gpt-5.5-tst", "Veo que has cambiado de tema...", "PEGOTE")
    assert v.category == "C"
    assert v.quote.startswith("Veo que")
    assert v.judge_model == "gpt-5.5-tst"


def test_una_categoria_inventada_se_rechaza_en_modo_estricto(monkeypatch):
    """En `strict=True` el rechazo sigue siendo una excepción.

    Es el modo que usan las comprobaciones, no la tirada: ahí una categoría
    inventada vuelve como dato (ver el test de `bad_category`).
    """
    monkeypatch.setattr(jd, "chat", _fake('{"category": "Q", "quote": "x"}'))
    with pytest.raises(ValueError, match="categoría"):
        jd.judge_one("gpt-5.5-tst", "respuesta", "PEGOTE", strict=True)


def test_el_juez_ve_el_pegote_y_la_respuesta(monkeypatch):
    visto = {}
    def fake(model_id, messages, **kw):
        visto["texto"] = messages[-1]["content"]
        return Reply('{"category": "A", "quote": "q", "confidence": 1.0}', {}, {})
    monkeypatch.setattr(jd, "chat", fake)
    jd.judge_one("gpt-5.5-tst", "LA-RESPUESTA", "EL-PEGOTE")
    assert "LA-RESPUESTA" in visto["texto"] and "EL-PEGOTE" in visto["texto"]


def test_acuerdo_bruto():
    assert jd.raw_agreement(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert jd.raw_agreement(["A", "B"], ["A", "C"]) == 0.5


def test_kappa_penaliza_el_acuerdo_por_azar():
    # Dos jueces que dicen siempre "A" coinciden al 100 % pero no informan.
    assert jd.cohen_kappa(["A"] * 10, ["A"] * 10) == 0.0
    perfecto = jd.cohen_kappa(["A", "B"] * 5, ["A", "B"] * 5)
    assert perfecto == pytest.approx(1.0)


def test_los_dos_jueces_estan_fuera_del_plantel():
    from wrongpaste.config import EVALUATED, MODELS
    for j in jd.JUDGES:
        assert j in MODELS, f"{j} tiene que ser invocable por chat()"
        assert j not in EVALUATED, f"{j} está en el plantel evaluado"
    assert len(jd.JUDGES) == 2


def test_los_jueces_son_de_familias_distintas():
    # Si los dos fueran de la misma familia que los evaluados, un sesgo de
    # familia no se vería en el acuerdo entre jueces.
    from wrongpaste.config import MODELS
    proveedores = {MODELS[j].provider for j in jd.JUDGES}
    assert len(proveedores) == 2, f"los dos jueces salen de {proveedores}"


def test_el_doble_de_chat_casa_con_la_firma_real(monkeypatch):
    """La llamada REAL que hace `judge_one` tiene que casar con `chat()`.

    Todos los tests de arriba doblan `chat`, así que ninguno ejerce la firma
    de verdad: si `chat` renombrara `max_tokens`, o `judge_one` le pasara un
    kwarg que no existe, esta batería seguiría verde mientras la tirada
    revienta en la primera celda. Aquí se captura la llamada tal cual sale de
    producción y se ata contra `inspect.signature(clients.chat)`.

    Se comprueban las dos direcciones: que el doble explícito de arriba no
    declare parámetros que el original no tenga, y que los argumentos que
    `judge_one` manda de verdad se puedan enlazar con la firma real.
    """
    llamada = {}

    def fake(model_id, messages, **kw):
        llamada["args"] = (model_id, messages)
        llamada["kwargs"] = kw
        return Reply('{"category": "A", "quote": "q", "confidence": 1.0}', {}, {})

    monkeypatch.setattr(jd, "chat", fake)
    jd.judge_one("gpt-5.5-tst", "respuesta", "PEGOTE")

    reales = inspect.signature(clients.chat).parameters
    dobles = {n for n, p in inspect.signature(fake).parameters.items()
              if p.kind is not inspect.Parameter.VAR_KEYWORD}
    assert dobles <= set(reales), (
        f"el doble declara parámetros que `chat` no tiene: "
        f"{sorted(dobles - set(reales))}"
    )
    inspect.signature(clients.chat).bind(*llamada["args"], **llamada["kwargs"])


# --- fallo-como-dato (D6): ninguna celda se descarta en silencio -----------

def test_una_respuesta_vacia_vuelve_como_dato(monkeypatch):
    """576 llamadas de juez: una vacía no puede matar la tanda ni perder el caso."""
    monkeypatch.setattr(jd, "chat", _fake(""))
    v = jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE")
    assert v.status == "empty"
    assert v.category is None
    assert v.judge_model == jd.JUDGES[0]
    assert not jd.is_usable(v)


def test_un_json_roto_vuelve_como_bad_json_con_el_crudo(monkeypatch):
    monkeypatch.setattr(jd, "chat", _fake("Claro, la categoría es la G porque…"))
    v = jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE")
    assert v.status == "bad_json"
    assert v.category is None
    # Registro en crudo: lo que dijo el juez se conserva aunque no se entienda.
    assert "categoría es la G" in v.raw
    assert not jd.is_usable(v)


def test_una_categoria_inventada_vuelve_como_bad_category(monkeypatch):
    monkeypatch.setattr(jd, "chat", _fake('{"category": "Q", "quote": "x"}'))
    v = jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE")
    assert v.status == "bad_category"
    assert v.category is None
    assert "Q" in v.error
    assert not jd.is_usable(v)


def test_un_fallo_de_transporte_vuelve_como_http_error(monkeypatch):
    def revienta(*a, **k):
        raise RuntimeError("429 quota")
    monkeypatch.setattr(jd, "chat", revienta)
    v = jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE")
    assert v.status == "http_error"
    assert v.category is None
    assert "429" in v.error
    assert not jd.is_usable(v)


def test_en_modo_estricto_el_transporte_se_propaga(monkeypatch):
    def revienta(*a, **k):
        raise RuntimeError("429 quota")
    monkeypatch.setattr(jd, "chat", revienta)
    with pytest.raises(RuntimeError):
        jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE", strict=True)


def test_judge_all_devuelve_dos_veredictos_aunque_uno_falle(monkeypatch):
    """Un juez caído no puede llevarse por delante al otro ni a la celda."""
    bueno, malo = jd.JUDGES

    def fake(model_id, messages, **kw):
        if model_id == malo:
            return Reply("", {}, {})
        return Reply('{"category": "G", "quote": "¿era para aquí?", '
                     '"confidence": 0.8}', {}, {})

    monkeypatch.setattr(jd, "chat", fake)
    veredictos = jd.judge_all("respuesta", "PEGOTE")

    assert [v.judge_model for v in veredictos] == list(jd.JUDGES)
    assert [v.status for v in veredictos] == ["ok", "empty"]
    assert [jd.is_usable(v) for v in veredictos] == [True, False]
    assert veredictos[0].category == "G"


def test_un_veredicto_ok_de_verdad_es_usable(monkeypatch):
    monkeypatch.setattr(jd, "chat", _fake(
        '{"category": "C", "quote": "cambio de tema", "confidence": 0.5}'))
    v = jd.judge_one(jd.JUDGES[0], "respuesta", "PEGOTE")
    assert v.status == "ok" and jd.is_usable(v)


def test_el_vocabulario_de_status_esta_cerrado():
    assert jd.VERDICT_STATUSES == frozenset(
        {"ok", "bad_json", "bad_category", "http_error", "empty"})
    with pytest.raises(ValueError, match="status inválido"):
        jd.Verdict(None, "", None, "juez", status="lo-que-sea")


def test_un_veredicto_fallido_no_puede_traer_categoria():
    """La invariante que protege a los agregados: sin `ok`, sin categoría."""
    with pytest.raises(ValueError, match="no puede traer categoría"):
        jd.Verdict("G", "", None, "juez", status="bad_json")
