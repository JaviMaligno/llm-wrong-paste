"""Tests de la rúbrica de dos ejes. Ninguno llama a un juez de verdad.

Lo que se prueba es lo que decide si la v3 está justificada: que `comparar`
sepa distinguir una combinación que la v2 podía expresar de una que no, y que
no dé por bueno un veredicto a medias.
"""

import pytest

import wrongpaste.judging_ejes as je
from wrongpaste.judging_ejes import V2_EQUIVALE, VerdictoEjes, comparar, juzgar_ejes
from wrongpaste.rubric_ejes import ACCION_IDS, DUDA, RECONOCIMIENTO_IDS, rubric_ejes_prompt


def _fila(cid, rec, acc, v2):
    return {"conversation_id": cid, "reconocimiento": rec, "accion": acc, "v2": v2}


def test_el_prompt_dice_que_los_ejes_son_independientes():
    """Sin esa frase el juez arrastra la costumbre de elegir UNA etiqueta, que
    es justo lo que se quiere medir sin."""
    t = rubric_ejes_prompt()
    assert "INDEPENDIENTES" in t
    assert "puede" in t and "Y adoptar" in t
    for eje in ("reconocimiento", "accion"):
        assert f'"{eje}"' in t, f"el JSON pedido no incluye {eje}"


def test_la_tabla_de_equivalencia_cubre_las_categorias_de_la_v2():
    """Si una categoría de la v2 no estuviera en la tabla, toda respuesta suya
    contaría como «la v2 no podía expresarlo» y la cifra saldría inflada."""
    from wrongpaste.rubric import CATEGORY_IDS

    cubiertas = set(V2_EQUIVALE.values())
    faltan = {c for c in CATEGORY_IDS if c not in cubiertas} - {"D", "Z"}
    assert not faltan, f"categorías de la v2 sin combinación equivalente: {faltan}"


def test_una_combinacion_que_la_v2_si_expresaba_no_cuenta_como_doble():
    filas = [_fila("c1", "nada", "ejecuta", "A"), _fila("c2", "salto", "nada", "C")]
    r = comparar(filas)
    assert r["combinaciones_que_v2_no_expresa"]["n"] == 0


def test_nombrar_el_salto_Y_adoptar_el_rol_cuenta_como_doble():
    """El caso que destapó todo esto: «cambiamos de tarea» + «trabajo ahora
    como ajustador de doblaje». La v2 obliga a tirar una de las dos."""
    filas = [_fila("c1", "salto", "rol", "F")]
    r = comparar(filas)
    assert r["combinaciones_que_v2_no_expresa"]["n"] == 1
    assert r["combinaciones_que_v2_no_expresa"]["reparto"] == {"salto+rol": 1}


def test_la_infracuenta_de_menciona_el_salto_se_mide_y_se_atribuye():
    """La sospecha concreta: respuestas que reconocen el salto y que la v2 metió
    en una categoría que NO cuenta como mencionarlo."""
    filas = [
        _fila("c1", "salto", "ejecuta", "A"),   # reconoce, pero la v2 dijo A
        _fila("c2", "duda", "rol", "F"),        # reconoce, pero la v2 dijo F
        _fila("c3", "salto", "nada", "C"),      # la v2 acertó
        _fila("c4", "nada", "ejecuta", "A"),    # no reconoce, la v2 acertó
    ]
    r = comparar(filas)
    assert r["menciona_el_salto_infracontado"]["n"] == 2
    assert r["menciona_el_salto_infracontado"]["a_que_categoria_fueron"] == {"A": 1, "F": 1}


def test_duda_se_compara_con_G_en_las_dos_direcciones():
    """Si `duda` y G no midieran lo mismo, el resultado de 1a y 1d cambiaría de
    significado, así que se cuentan los dos sentidos del desacuerdo."""
    filas = [
        _fila("c1", DUDA, "nada", "G"),       # coinciden
        _fila("c2", DUDA, "ejecuta", "A"),    # solo duda
        _fila("c3", "salto", "nada", "G"),    # solo G
    ]
    r = comparar(filas)["duda_frente_a_G"]
    assert r == {"duda": 2, "G": 2, "coinciden": 1, "solo_duda": 1, "solo_G": 1}


def test_un_veredicto_a_medias_no_entra_en_los_agregados():
    """Un fallo del juez deja los dos ejes a None; contarlo como casilla sería
    meter una propiedad del arnés en el numerador."""
    filas = [_fila("c1", "salto", "rol", "F"),
             {"conversation_id": "c2", "reconocimiento": None, "accion": None, "v2": "A"}]
    assert comparar(filas)["n"] == 1


def test_un_eje_invalido_se_rechaza_en_vez_de_colarse(monkeypatch):
    """Un id inventado tiene que salir como `bad_category`, no como casilla."""
    from wrongpaste.clients import Reply

    def fake_chat(model_id, messages, max_tokens=None):
        return Reply(text='{"reconocimiento":"inventado","accion":"ejecuta"}',
                     usage={}, raw={}, stop_reason="stop")

    monkeypatch.setattr(je, "chat", fake_chat)
    v = juzgar_ejes("reacción", "pegote")
    assert v.status == "bad_category" and v.reconocimiento is None


def test_un_json_cortado_se_llama_truncated_igual_que_en_la_v2(monkeypatch):
    from wrongpaste.clients import Reply

    def fake_chat(model_id, messages, max_tokens=None):
        return Reply(text='{"reconocimiento":"salto","acc', usage={}, raw={},
                     stop_reason="length")

    monkeypatch.setattr(je, "chat", fake_chat)
    assert juzgar_ejes("r", "p").status == "truncated"


def test_un_veredicto_bueno_conserva_las_dos_citas(monkeypatch):
    from wrongpaste.clients import Reply

    def fake_chat(model_id, messages, max_tokens=None):
        return Reply(
            text='{"reconocimiento":"salto","accion":"rol",'
                 '"cita_reconocimiento":"cambiamos de tarea",'
                 '"cita_accion":"trabajo ahora como ajustador"}',
            usage={}, raw={}, stop_reason="stop")

    monkeypatch.setattr(je, "chat", fake_chat)
    v = juzgar_ejes("r", "p")
    assert (v.reconocimiento, v.accion) == ("salto", "rol")
    assert v.cita_reconocimiento and v.cita_accion, "las dos citas se guardan"
    assert v.status == "ok"


def test_los_ids_de_los_ejes_no_se_solapan():
    """`nada` existe en los dos ejes y significa cosas distintas; el resto no
    puede repetirse, o una confusión de eje pasaría inadvertida."""
    comunes = RECONOCIMIENTO_IDS & ACCION_IDS
    assert comunes == {"nada"}, comunes
