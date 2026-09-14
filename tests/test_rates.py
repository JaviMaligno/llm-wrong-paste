"""Las dos tasas del §4 y la puerta del §7.

Las filas se construyen con `ConversationRecord` y la categoría se toma de un
`Verdict` de verdad: si el nombre de un campo cambia en el esquema de la fila o
en el veredicto del juez, estos tests tienen que enterarse. Una fila inventada
a mano en el test pasaría verde contra una producción rota.
"""

import pytest

from wrongpaste.agreement import _es_del_juez
from wrongpaste.judging import Verdict
from wrongpaste.rates import (
    BASELINE,
    CATEGORY_FIELD,
    COMPARABLE_LEVELS,
    LEVELS,
    compare_levels,
    gate_decision,
    rates_by_level,
    usable_category,
)
from wrongpaste.records import ConversationRecord
from wrongpaste.rubric import CATEGORY_IDS, ENTERTAINS_ERROR, MENTIONS_JUMP

MODELOS = ("gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5")


def _veredicto(category: str) -> Verdict:
    """Un veredicto bueno del juez, construido con la clase de verdad."""
    return Verdict(category, "cita literal", 0.9, "gpt-5.5-tst")


def _fallo_del_juez(status: str) -> Verdict:
    """Un veredicto fallido: sin categoría y con el motivo en `status` (D6)."""
    return Verdict(None, "", None, "gpt-5.5-tst", status=status, error="lo que fuera")


def _fila(
    level: str | None,
    category: str | Verdict | None,
    *,
    status: str = "ok",
    idx: int = 0,
    model_id: str = MODELOS[0],
) -> dict:
    """Una fila de tirada tal y como se escribe, con su veredicto pegado.

    El veredicto viaja por donde viaja de verdad: el campo que el análisis une
    al JSONL y que `agreement.blind_sample` esconde al etiquetador. `category`
    puede ser la letra ya resuelta o un `Verdict` entero, que son dos de las
    tres formas reales del campo (la tercera, su dict, se prueba aparte).
    """
    rec = ConversationRecord(
        model_id=model_id,
        topic_id=f"t{idx}",
        n_turns=2,
        replicate_idx=idx,
        paste_level=level,
        condition="paste" if level else "no_paste",
        status=status,
    )
    fila = rec.to_json()
    if isinstance(category, Verdict):
        fila[CATEGORY_FIELD] = category
    elif category in CATEGORY_IDS:
        # La letra pasa por la clase real: si `Verdict` dejara de aceptarla,
        # el test se entera aquí y no fingiendo un dato que no existe.
        fila[CATEGORY_FIELD] = _veredicto(category).category
    elif category is not None:
        # Una letra que la rúbrica no conoce: un join corrupto, que `Verdict`
        # no puede ni construir. Se escribe tal cual, que es como llegaría.
        fila[CATEGORY_FIELD] = category
    return fila


def _filas(level: str, categorias: str, **kw) -> list[dict]:
    return [_fila(level, c, idx=i, **kw) for i, c in enumerate(categorias)]


def test_el_campo_que_leemos_es_el_que_se_esconde_al_etiquetar():
    # Si los dos módulos dejaran de llamarlo igual, uno estaría leyendo o
    # escondiendo el campo equivocado, y el acuerdo del §6 no mediría nada.
    assert _es_del_juez(CATEGORY_FIELD)


def test_las_tasas_anidadas_se_respetan():
    # A y B no mencionan el salto; C, D y E lo mencionan; G además contempla
    # el error. 8 filas usables: 4 en MENTIONS_JUMP, 1 de ellas en ENTERTAINS.
    tasas = rates_by_level(_filas("N0", "AABBCDEG"))
    n0 = tasas["N0"]
    assert n0.n == 8
    assert n0.mentions_jump == pytest.approx(4 / 8)
    assert n0.entertains_error == pytest.approx(1 / 8)
    assert n0.entertains_error <= n0.mentions_jump
    assert n0.counts["A"] == 2 and n0.counts["G"] == 1
    assert sum(n0.counts.values()) == n0.n
    assert set(n0.counts) == set(CATEGORY_IDS), "el recuento cubre la rúbrica entera"


def test_las_tasas_salen_de_la_rubrica_y_no_de_letras_a_mano():
    # Una fila por cada categoría de la rúbrica: cada tasa tiene que valer
    # exactamente el tamaño de su conjunto sobre el total.
    categorias = sorted(CATEGORY_IDS)
    tasas = rates_by_level(_filas("N1", "".join(categorias)))["N1"]
    assert tasas.mentions_jump == pytest.approx(len(MENTIONS_JUMP) / len(categorias))
    assert tasas.entertains_error == pytest.approx(
        len(ENTERTAINS_ERROR) / len(categorias)
    )


def test_cada_nivel_se_cuenta_por_separado():
    filas = _filas("N0", "AAAA") + _filas("N1", "GGGG") + _filas("N2", "CCCC")
    tasas = rates_by_level(filas)
    assert list(tasas) == list(LEVELS)
    assert tasas["N0"].mentions_jump == 0.0
    assert tasas["N1"].entertains_error == 1.0
    assert tasas["N2"].mentions_jump == 1.0
    assert tasas["N2"].entertains_error == 0.0


def test_las_filas_con_veredicto_no_usable_no_cuentan_en_el_denominador():
    filas = [
        _fila("N0", "G", idx=0),
        _fila("N0", "A", idx=1),
        # Se cayó la llamada: no hay reacción que clasificar (D6).
        _fila("N0", None, status="http_error", idx=2),
        # El arnés falló: ni siquiera es conducta del modelo evaluado.
        _fila("N0", "G", status="harness_error", idx=3),
        # Se cortó por nuestro max_tokens: la respuesta no está entera.
        _fila("N0", "G", status="truncated", idx=4),
        # El juez devolvió una letra que no es de la rúbrica: viene como
        # veredicto fallido, que es la forma real de ese caso.
        _fila("N0", _fallo_del_juez("bad_category"), idx=5),
        # El juez no devolvió JSON.
        _fila("N0", _fallo_del_juez("bad_json"), idx=7),
        # Una letra suelta que no es de la rúbrica (join corrupto).
        _fila("N0", "Q", idx=8),
        # Fila sin veredicto: nadie la clasificó todavía.
        _fila("N0", None, idx=6),
    ]
    n0 = rates_by_level(filas)["N0"]
    assert n0.n == 2, "solo las dos filas ok con categoría válida"
    assert n0.n_excluded == 7
    assert n0.entertains_error == pytest.approx(1 / 2)
    assert n0.mentions_jump == pytest.approx(1 / 2)
    assert usable_category(_fila("N0", "Q")) is None
    assert usable_category(_fila("N0", _fallo_del_juez("http_error"))) is None
    assert usable_category(_fila("N0", "G", status="truncated")) is None
    assert usable_category(_fila("N0", "G")) == "G"
    assert usable_category(_fila("N0", _veredicto("G"))) == "G"


def test_el_veredicto_tambien_se_lee_en_su_forma_de_dict():
    """Es como viaja en `verdicts-<run_id>.jsonl`: `asdict` de un `Verdict`."""
    from dataclasses import asdict

    bueno = _fila("N0", None)
    bueno[CATEGORY_FIELD] = asdict(_veredicto("G"))
    malo = _fila("N0", None, idx=1)
    malo[CATEGORY_FIELD] = asdict(_fallo_del_juez("empty"))

    n0 = rates_by_level([bueno, malo])["N0"]
    assert n0.n == 1
    assert n0.n_excluded == 1
    assert n0.entertains_error == 1.0


def test_el_brazo_de_control_no_es_un_nivel():
    filas = _filas("N0", "AG") + [_fila(None, None, idx=9)]
    tasas = rates_by_level(filas)
    assert set(tasas) == {"N0"}
    assert tasas["N0"].n == 2


def test_un_nivel_escrito_de_otra_forma_revienta():
    # Un "n1" en minúsculas partiría el brazo en dos sin que nadie lo notara.
    fila = _fila("N1", "G")
    fila["paste_level"] = "n1"
    with pytest.raises(ValueError, match="paste_level"):
        rates_by_level([fila])


def test_cero_de_cero_no_es_cero():
    tasas = rates_by_level([_fila("N0", None, status="timeout")])["N0"]
    assert tasas.n == 0
    assert tasas.mentions_jump is None, "una celda vacía no es el resultado buscado"
    assert tasas.entertains_error is None
    assert tasas.z_share is None


def test_comparar_n2_revienta_con_un_mensaje_que_explica_por_que():
    filas = _filas("N0", "AAAA") + _filas("N2", "GGGG")
    with pytest.raises(ValueError) as exc:
        compare_levels(filas, ("N0", "N2"))
    mensaje = str(exc.value)
    assert "N2" in mensaje
    assert "distractor" in mensaje
    assert "§5" in mensaje
    assert "rates_by_level" in mensaje, "el mensaje dice dónde SÍ se reporta N2"


def test_n2_no_esta_entre_los_niveles_comparables():
    assert COMPARABLE_LEVELS == ("N0", "N1")
    assert "N2" not in COMPARABLE_LEVELS
    assert BASELINE == "N0"


def test_la_comparacion_por_defecto_es_n1_contra_n0():
    filas = _filas("N0", "AAAA") + _filas("N1", "CCGG") + _filas("N2", "GGGG")
    comp = compare_levels(filas)
    assert set(comp) == {"N1"}
    n1 = comp["N1"]
    assert n1.baseline == "N0"
    assert n1.mentions_jump_delta == pytest.approx(1.0)
    assert n1.entertains_error_delta == pytest.approx(0.5)
    assert n1.moved() == ("mentions_jump", "entertains_error")


def test_comparar_sin_linea_base_revienta():
    with pytest.raises(ValueError, match="N0"):
        compare_levels(_filas("N1", "GG"), ("N1",))


def test_la_puerta_para_si_los_tres_niveles_dan_lo_mismo():
    filas = (
        _filas("N0", "AABC")
        + _filas("N1", "AABC")
        + _filas("N2", "AABC")
    )
    decision = gate_decision(filas)
    assert decision.should_continue is False
    assert decision.moved == ()
    assert "PARA" in decision.reason
    assert "no hay pegote que les haga preguntar" in decision.reason
    # El porqué trae los números, no solo el veredicto.
    assert "+0.000" in decision.reason
    assert decision.rates["N2"].n == 4


def test_la_puerta_sigue_si_n1_sube():
    filas = _filas("N0", "AAAAAAAA") + _filas("N1", "GGGGAAAA") + _filas("N2", "AAAAAAAA")
    decision = gate_decision(filas)
    assert decision.should_continue is True
    assert decision.moved == ("N1",)
    assert "SIGUE" in decision.reason
    assert decision.movements["N1"]["entertains_error_delta"] == pytest.approx(0.5)
    assert decision.movements["N1"]["role"] == "comparable"


def test_si_solo_se_mueve_n2_la_puerta_lo_dice_por_su_nombre():
    # N2 cuenta en la puerta (§7) pero es el techo con trampa, y el motivo
    # tiene que decirlo: ese movimiento no se reporta como comparación.
    filas = _filas("N0", "AAAA") + _filas("N1", "AAAA") + _filas("N2", "GGGG")
    decision = gate_decision(filas)
    assert decision.should_continue is True
    assert decision.moved == ("N2",)
    assert "techo" in decision.reason and "trampa" in decision.reason
    assert decision.movements["N2"]["role"] == "techo (distractor de diseño)"


def test_un_movimiento_por_debajo_del_umbral_no_abre_la_puerta():
    # 1 de 40 frente a 0 de 40 son 2,5 puntos: por debajo del umbral declarado.
    filas = _filas("N0", "A" * 40) + _filas("N1", "G" + "A" * 39)
    assert gate_decision(filas).should_continue is False
    assert gate_decision(filas, threshold=0.01).should_continue is True


def test_la_puerta_sin_linea_base_usable_revienta():
    with pytest.raises(ValueError, match="línea base"):
        gate_decision(_filas("N1", "GGGG"))
