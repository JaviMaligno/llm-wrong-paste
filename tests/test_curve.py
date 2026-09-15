"""El análisis de la Fase 1b: la curva por posición y la prueba de tendencia.

Los números de Cochran-Armitage salen de la forma cerrada y se comprueban a
mano: no hay `scipy` contra el que contrastar, así que el test tiene que ser el
que fija la respuesta. Ningún test de aquí llama a un modelo ni lee disco.
"""

import numpy as np
import pytest

from wrongpaste.curve import (
    PRIMARY_HYPOTHESES,
    REPLICATE_AGREEMENT_N0,
    by_kind,
    cochran_armitage,
    holm,
    primary_family_report,
    rate_by_position,
    trend_report,
)
from wrongpaste.rates import CATEGORY_FIELD


def test_una_tabla_plana_no_tiene_tendencia():
    counts = [(p, 24, 6) for p in range(12)]  # 25 % en todas las posiciones
    t = cochran_armitage(counts)
    assert abs(t.z) < 1e-9
    assert t.p == pytest.approx(1.0)
    assert t.slope_sign == 0


def test_una_tabla_monotona_da_z_positivo_y_p_pequeno():
    # de 2/24 en la posición 0 a 20/24 en la 11, subiendo
    ks = [2, 4, 6, 8, 10, 11, 13, 15, 16, 18, 19, 20]
    t = cochran_armitage([(p, 24, k) for p, k in enumerate(ks)])
    assert t.slope_sign == 1
    assert t.z > 5
    assert t.p < 1e-6


def test_la_cola_normal_esta_fijada_a_valores_calculados_a_mano():
    """`p` no es una desigualdad: es la cifra que se reporta, y va clavada.

    La conversión de z a p es `erfc(|z|/raíz(2))`, la cola normal a dos colas.
    Escalarla mal —`erfc(|z|)`, que es la tentación, porque `erfc` ya parece una
    cola— sigue dando un número pequeño para un z grande y sigue dando 1 en la
    tabla plana, así que ninguna desigualdad lo pilla. Los dos anclas de abajo
    están calculados a mano y son los que fijan la escala:

    - z = 8,0918 → p = 5,8776e-16 (el mal escalado da 2,53e-30, catorce órdenes
      de magnitud por debajo).
    - z = 1,6572 → p = 0,09748, el valor de tabla de toda la vida para un z de
      1,66 a dos colas, y el que decide si algo cruza 0,05 o no (el mal escalado
      da 0,0191, que cruzaría).

    El segundo importa más que el primero: es el tramo donde vive de verdad la
    puerta de decisión de la fase.
    """
    ks = [2, 4, 6, 8, 10, 11, 13, 15, 16, 18, 19, 20]
    fuerte = cochran_armitage([(p, 24, k) for p, k in enumerate(ks)])
    assert fuerte.z == pytest.approx(8.09182709, abs=1e-8)
    assert fuerte.p == pytest.approx(5.877630812e-16, rel=1e-6)

    ks_suave = [6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    suave = cochran_armitage([(p, 24, k) for p, k in enumerate(ks_suave)])
    assert suave.z == pytest.approx(1.65720987, abs=1e-8)
    assert suave.p == pytest.approx(0.0974770511, rel=1e-6)
    assert suave.p > 0.05, "un z de 1,66 no es significativo a dos colas"


def test_la_varianza_pondera_cada_posicion_por_su_denominador():
    """Con n igual en las 12 posiciones la ponderación es invisible, y hay que verla.

    Sobre la tabla monótona de arriba, `Σn·x² − (Σn·x)²/N` y la varianza de los
    scores SIN ponderar por n dan exactamente el mismo z (8,09182709 las dos), así
    que el ancla anterior no distingue una de otra por mucho que fije la cifra.
    Solo una tabla con denominadores desiguales las separa.

    Y desiguales es el caso normal, no el raro: `rate_by_position` devuelve una
    entrada por posición incluidas las vacías (D-curva), y en una tanda real cada
    posición pierde filas distintas —fallos, respuestas sin veredicto—, así que la
    tabla que llega aquí casi nunca es uniforme.

    `n = [10, 40, 40, 10]`, `k = [1, 12, 24, 7]`, calculada a mano en forma
    cerrada: z = 3,7481 / p = 1,7816e-4. Sin ponderar por n saldría 2,7028 /
    6,876e-3, y quitando el término de varianza entero, 30,218.
    """
    t = cochran_armitage([(0, 10, 1), (1, 40, 12), (2, 40, 24), (3, 10, 7)])
    assert t.z == pytest.approx(3.74812640, abs=1e-8)
    assert t.p == pytest.approx(1.781604650e-4, rel=1e-6)
    assert t.n == 100 and t.k == 44


def test_la_direccion_se_invierte_al_invertir_la_tabla():
    ks = [2, 4, 6, 8, 10, 11, 13, 15, 16, 18, 19, 20]
    sube = cochran_armitage([(p, 24, k) for p, k in enumerate(ks)])
    baja = cochran_armitage([(p, 24, k) for p, k in enumerate(reversed(ks))])
    assert baja.slope_sign == -1
    assert baja.z == pytest.approx(-sube.z)
    assert baja.p == pytest.approx(sube.p)


def test_todo_ceros_o_todo_unos_no_revienta_y_no_tiene_tendencia():
    """Varianza cero: no hay tendencia que medir, y no se divide por cero."""
    for k in (0, 24):
        t = cochran_armitage([(p, 24, k) for p in range(12)])
        assert t.z == 0.0 and t.p == pytest.approx(1.0) and t.slope_sign == 0
    # Y una tabla sin una sola observación tampoco: no hay denominador que
    # dividir, y devolver NaN convertiría el hueco en un número que alguien
    # acabaría leyendo.
    vacia = cochran_armitage([(p, 0, 0) for p in range(12)])
    assert vacia.z == 0.0 and vacia.p == pytest.approx(1.0) and vacia.n == 0
    assert cochran_armitage([]).n == 0


def _fila(pos, cat, kind="email", model="gpt-5.6-sol-tst"):
    return {
        "sweep_position": pos,
        CATEGORY_FIELD: cat,
        "artifact_kind": kind,
        "model_id": model,
        "similarity_user": 0.1 + 0.03 * pos,
        "status": "ok",
    }


def test_la_tasa_por_posicion_cuenta_pertenencia_al_conjunto():
    filas = [_fila(0, "A"), _fila(0, "C"), _fila(1, "C"), _fila(1, "E")]
    # Dos posiciones, porque la curva devuelve una entrada por posición SIEMPRE:
    # con el barrido entero saldrían también las diez vacías, que es justo lo que
    # comprueba el test siguiente.
    tasas = rate_by_position(filas, {"C", "E"}, positions=2)
    assert [(t.position, t.n, t.k) for t in tasas] == [(0, 2, 1), (1, 2, 2)]
    assert tasas[0].rate == 0.5 and tasas[1].rate == 1.0


def test_una_posicion_sin_filas_usables_no_desaparece_de_la_curva():
    """Un hueco en la curva es un dato; borrarlo la convierte en otra curva."""
    filas = [_fila(0, "A"), _fila(2, "A")]
    tasas = rate_by_position(filas, {"A"}, positions=3)
    assert [t.position for t in tasas] == [0, 1, 2]
    assert tasas[1].n == 0 and tasas[1].rate is None


def test_las_filas_sin_categoria_no_entran_en_el_denominador():
    """Un veredicto ausente no es una respuesta que no mencionó el salto."""
    filas = [_fila(0, "A"), {**_fila(0, "A"), CATEGORY_FIELD: None}]
    tasas = rate_by_position(filas, {"A"})
    assert tasas[0].n == 1 and tasas[0].k == 1


def test_el_informe_separa_por_modelo_y_tambien_agrega():
    """Cada grupo tiene que ver SOLO sus filas, y la agregada todas.

    Los dos modelos reciben curvas de **signo opuesto** a propósito: con la
    misma curva para los dos, el test pasaría igual aunque el reparto le diera a
    cada grupo las filas de todos, y entonces no estaría comprobando que el
    reparto reparta. Aquí las dos pendientes se cancelan al mezclarse, así que
    un grupo mal repartido sale plano y el test lo ve.

    Que ese sea el caso interesante es el punto 3 del plan: la prueba primaria
    va por modelo porque `gpt-5.6-luna` y `claude-opus-5` se comportan de forma
    distinta, y una agregada sobre dos tendencias opuestas describe a ninguno.
    """
    filas = [_fila(p, "C" if p > 5 else "A", model="gpt-5.6-sol-tst")
             for p in range(12) for _ in range(8)]
    filas += [_fila(p, "C" if p < 6 else "A", model="claude-opus-5")
              for p in range(12) for _ in range(8)]
    rep = trend_report(filas, {"C"}, by="model_id")
    assert set(rep["by"]) == {"gpt-5.6-sol-tst", "claude-opus-5"}

    sol = rep["by"]["gpt-5.6-sol-tst"]
    opus = rep["by"]["claude-opus-5"]
    assert sol["trend"]["slope_sign"] == 1
    assert opus["trend"]["slope_sign"] == -1
    assert sol["trend"]["z"] == pytest.approx(-opus["trend"]["z"])
    # 96 y no 192: el denominador es la prueba de que a cada grupo le llegaron
    # sus filas y solo las suyas.
    assert sol["trend"]["n"] == opus["trend"]["n"] == 96
    assert len(sol["curve"]) == len(opus["curve"]) == 12

    # La agregada sí ve las 192, y por eso las dos pendientes se cancelan: es
    # exactamente por lo que `pooled` se reporta pero no manda.
    assert rep["pooled"]["trend"]["n"] == 192
    assert rep["pooled"]["trend"]["slope_sign"] == 0


def test_el_informe_lleva_el_suelo_de_ruido_de_replica():
    """Un p pequeño sobre una diferencia menor que el ruido no es un hallazgo."""
    filas = [_fila(p, "A") for p in range(12) for _ in range(8)]
    rep = trend_report(filas, {"A"})
    assert rep["replicate_agreement_n0"] == REPLICATE_AGREEMENT_N0
    assert 0 < REPLICATE_AGREEMENT_N0 < 1


def test_solo_se_informa_la_pendiente_de_los_generos_con_rango_ancho():
    """Un género que no recorre el eje no puede decir nada sobre el eje.

    Es la mitad medible de D15: dentro de `job_ad`, que va de 0,234 a 0,546, la
    pendiente distingue parcialmente «se parece más» de «otro registro». Dentro
    de `prompt`, que recorre 0,108, no distingue nada, y reportarla invitaría a
    leer ruido como resultado.
    """
    anchas = [{**_fila(p, "A" if p > 5 else "C", kind="job_ad"),
               "similarity_user": 0.2 + 0.03 * p} for p in range(12)]
    estrechas = [{**_fila(p, "A", kind="prompt"),
                  "similarity_user": 0.20 + 0.002 * p} for p in range(12)]
    rep = by_kind(anchas + estrechas, {"A"}, min_span=0.20)
    assert "job_ad" in rep["wide"]
    assert "prompt" in rep["narrow"]
    assert "trend" in rep["wide"]["job_ad"]
    assert rep["narrow"]["prompt"]["span"] < 0.20
    assert "trend" not in rep["narrow"]["prompt"]


# --- La familia de nueve pruebas primarias -----------------------------------
#
# El plan corre tres hipótesis (H1, H2, H3) por tres modelos y reporta las nueve
# por separado, cada una contra 0,05. Sin corregir por multiplicidad, una tanda
# en la que las tres curvas son planas produce al menos un «significativo» una
# de cada tres veces. Los tests de aquí abajo fijan la corrección.


def test_holm_no_toca_una_familia_de_una_sola_prueba():
    """Corregir cuando no hay nada que corregir sería pagar potencia por nada."""
    assert holm({"unica": 0.04}) == {"unica": 0.04}


def test_holm_escalona_el_multiplicador_de_mayor_a_menor():
    """Holm y no Bonferroni: solo la menor paga el tamaño entero de la familia.

    Con p = 0,01 / 0,02 / 0,04 y familia de tres, los multiplicadores son 3, 2 y
    1: 0,03 / 0,04 / 0,04. Bonferroni multiplicaría las tres por 3 y tiraría la
    tercera a 0,12, que es potencia regalada sin necesidad.
    """
    ajustados = holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert ajustados["a"] == pytest.approx(0.03)
    assert ajustados["b"] == pytest.approx(0.04)
    assert ajustados["c"] == pytest.approx(0.04)


def test_holm_es_monotono_y_no_devuelve_p_mayores_que_uno():
    """Un p ajustado nunca puede quedar por debajo de otro más pequeño que él.

    Sin el arrastre del máximo, `b` saldría 0,40 y `a` 0,60: la prueba con el p
    crudo más grande parecería la más fuerte de las dos.
    """
    ajustados = holm({"a": 0.30, "b": 0.40})
    assert ajustados["a"] == pytest.approx(0.60)
    assert ajustados["b"] == pytest.approx(0.60)
    assert all(p <= 1.0 for p in holm({"a": 0.6, "b": 0.7}).values())


def _tanda_plana(modelos, k_por_posicion=4):
    """Filas sin ninguna tendencia: la mitad A y la mitad B en cada posición."""
    return [
        {**_fila(pos, "A" if i < k_por_posicion else "B", model=m)}
        for m in modelos
        for pos in range(12)
        for i in range(8)
    ]


def test_la_familia_primaria_de_la_fase_1b_son_nueve_pruebas():
    """Tres hipótesis por tres modelos. El número viaja en el informe: sin él,
    el lector no puede corregir por su cuenta lo que el informe no corrigió."""
    assert len(PRIMARY_HYPOTHESES) == 3
    rep = primary_family_report(
        _tanda_plana(("claude-opus-5", "gpt-5.6-luna-tst", "gpt-5.6-sol-tst"))
    )
    assert rep["family_size"] == 9
    assert set(rep["hypotheses"]) == set(PRIMARY_HYPOTHESES)


def test_un_p_de_0_04_en_un_solo_modelo_no_sobrevive_a_la_familia():
    """El caso concreto: `luna` saca p=0,038 en H3 y los otros ocho son planos.

    Sin corregir, eso se reporta como «luna mueve la mezcla con el coseno y opus
    no», que es justo el contraste por familia que el artículo busca. Con nueve
    pruebas en la familia, la probabilidad de que algo así salga por azar con las
    tres curvas planas es del 32 %, así que un 0,038 suelto no es un hallazgo:
    ajustado por Holm son 0,345.
    """
    # z = 2,0714 / p = 0,03832, calculado en forma cerrada sobre esta rampa.
    rampa = [3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6]
    filas = _tanda_plana(("claude-opus-5", "gpt-5.6-sol-tst"))
    filas += [
        {**_fila(pos, "A" if i < rampa[pos] else "B", model="gpt-5.6-luna-tst")}
        for pos in range(12)
        for i in range(8)
    ]
    rep = primary_family_report(filas)
    luna = rep["hypotheses"]["H3 ejecuta en silencio"]["by"]["gpt-5.6-luna-tst"]["trend"]
    assert luna["p"] == pytest.approx(0.03832, rel=1e-3), "el caso de partida"
    assert luna["p_holm"] == pytest.approx(0.3449, rel=1e-3)
    assert luna["significant_holm"] is False

    # Y ninguna de las nueve pasa: la tanda entera es ruido.
    for informe in rep["hypotheses"].values():
        for sub in informe["by"].values():
            assert sub["trend"]["significant_holm"] is False


def test_el_informe_por_modelo_declara_cuantas_pruebas_lleva():
    """`trend_report` corre una prueba por grupo y no corrige nada: es un tercio
    de la familia y no puede corregir lo que no ve. Lo que sí puede es decir
    cuántas pruebas acaba de correr, para que nadie lea las tres como una."""
    rep = trend_report(_tanda_plana(("a", "b", "c")), {"A"}, by="model_id")
    assert rep["n_tests"] == 3
    assert "p_holm" not in rep["by"]["a"]["trend"]


def test_sin_corregir_una_de_cada_tres_tandas_planas_da_un_falso_positivo():
    """La simulación que justifica el número, con semilla fija.

    288 filas generadas con las tasas base de N0 de la Fase 1a (A 55 %, E 7,7 %,
    menciona el salto 25,3 %) y la categoría SIN ninguna dependencia de la
    posición: por construcción las nueve hipótesis nulas son ciertas. Aun así,
    una de cada tres tandas produce al menos un p<0,05 entre las nueve. Con Holm
    sobre la familia, la tasa vuelve al 5 % declarado.
    """
    rng = np.random.default_rng(20260915)
    p_a, p_e, p_jump = 50 / 91, 7 / 91, 0.253
    modelos = ("claude-opus-5", "gpt-5.6-luna-tst", "gpt-5.6-sol-tst")
    crudos = corregidos = 0
    tandas = 300
    for _ in range(tandas):
        filas = []
        for m in modelos:
            for pos in range(12):
                for _ in range(8):
                    u = rng.random()
                    cat = ("A" if u < p_a else "E" if u < p_a + p_e
                           else "C" if u < p_a + p_e + p_jump else "B")
                    filas.append({**_fila(pos, cat, model=m)})
        pruebas = [sub["trend"] for inf in primary_family_report(filas)["hypotheses"].values()
                   for sub in inf["by"].values()]
        assert len(pruebas) == 9
        crudos += any(t["p"] < 0.05 for t in pruebas)
        corregidos += any(t["significant_holm"] for t in pruebas)
    assert crudos / tandas > 0.20, crudos / tandas
    assert corregidos / tandas <= 0.08, corregidos / tandas


def test_cada_hipotesis_lleva_el_suelo_de_ruido_de_SU_conjunto():
    """Confundir los dos suelos infla el umbral casi al triple.

    El acuerdo entre réplicas sobre la etiqueta A-G es 0,73, pero cada hipótesis
    es una tasa binaria y ahí la mayoría de los desacuerdos de categoría no
    cruzan la frontera del conjunto: de los 12 pares que cambian de letra, solo
    4 cambian de lado en MENTIONS_JUMP. Medido sobre los mismos 45 pares de la
    Fase 1a, el acuerdo binario va de 0,867 a 0,933, no 0,73.
    """
    from wrongpaste.curve import REPLICATE_AGREEMENT_N0, replicate_agreement
    from wrongpaste.rubric import MENTIONS_JUMP

    for member in (MENTIONS_JUMP, {"E"}, {"A"}):
        suelo = replicate_agreement(member)
        assert suelo is not None, f"{sorted(member)} sin suelo medido"
        assert suelo > REPLICATE_AGREEMENT_N0, (
            f"{sorted(member)}: el suelo binario ({suelo}) tiene que ser MÁS alto "
            f"que el de la etiqueta ({REPLICATE_AGREEMENT_N0}); si no, alguien ha "
            "vuelto a copiar el número de las siete categorías"
        )


def test_un_conjunto_sin_suelo_medido_devuelve_none_y_no_un_numero_plausible():
    """Inventar un suelo es lo que convierte un umbral en un adorno."""
    from wrongpaste.curve import replicate_agreement

    assert replicate_agreement({"B"}) is None
    assert replicate_agreement({"A", "B"}) is None


def test_el_informe_lleva_el_suelo_del_conjunto_que_mide():
    from wrongpaste.curve import replicate_agreement, trend_report
    from wrongpaste.rubric import MENTIONS_JUMP

    filas = [_fila(p, "C" if p > 5 else "A") for p in range(12) for _ in range(2)]
    assert trend_report(filas, {"A"})["replicate_agreement_member"] == replicate_agreement({"A"})
    assert trend_report(filas, MENTIONS_JUMP)["replicate_agreement_member"] == replicate_agreement(
        MENTIONS_JUMP
    )
