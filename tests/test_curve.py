"""El análisis de la Fase 1b: la curva por posición y la prueba de tendencia.

Los números de Cochran-Armitage salen de la forma cerrada y se comprueban a
mano: no hay `scipy` contra el que contrastar, así que el test tiene que ser el
que fija la respuesta. Ningún test de aquí llama a un modelo ni lee disco.
"""

import numpy as np
import pytest

from wrongpaste.curve import (
    ALPHA,
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


def _fila(pos, cat, kind="email", model="gpt-5.6-sol-tst", signal=None):
    return {
        "sweep_position": pos,
        CATEGORY_FIELD: cat,
        "artifact_kind": kind,
        # `None` es lo que trae el brazo neutro: los 67 artefactos de N0 no
        # llevan señal, y por eso el control de composición no le aplica.
        "artifact_signal": signal,
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
    """Filas sin ninguna tendencia: la mitad A y la mitad B en cada posición.

    Llevan `n_turns` alternando 2 y 10 porque desde la Fase 1d hay hipótesis que
    se contrastan sobre ese eje (H5): sin la columna, esas pruebas saldrían con
    n = 0 y el fixture certificaría una familia que nadie corrió.
    """
    return [
        {**_fila(pos, "A" if i < k_por_posicion else "B", model=m), "n_turns": 2 + 8 * (i % 2)}
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


# --- Cada tanda declara SU familia -------------------------------------------
#
# Hasta la Fase 1d la familia era una constante única, porque solo había una
# tanda que corregir. Con dos tandas de hipótesis distintas sobre el mismo
# código, una familia global corrige por pruebas que esta tanda no ha corrido y
# deja fuera las que sí: las dos cosas mueven el `p` ajustado, y en direcciones
# opuestas.


def test_cada_tanda_declara_su_propia_familia():
    """Una familia global para todas las tandas corrige por pruebas que esta
    tanda no ha corrido, y deja fuera las que sí."""
    from wrongpaste.curve import HYPOTHESIS_FAMILIES

    assert "1b" in HYPOTHESIS_FAMILIES and "1d" in HYPOTHESIS_FAMILIES
    assert set(HYPOTHESIS_FAMILIES["1b"]) == {
        "H1 menciona el salto", "H2 puente confabulado", "H3 ejecuta en silencio"}
    assert set(HYPOTHESIS_FAMILIES["1d"]) == {
        "H4 contempla el error", "H5 contempla el error en las largas"}


def test_el_informe_de_familia_corrige_por_el_tamano_declarado():
    """Seis pruebas y no nueve: Holm tiene que ver la familia de ESTA tanda.

    Seis y no tres: el plan declara «2 hipótesis x 3 modelos» antes de mirar, y
    H5 es una prueba con su `p` como cualquier otra.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    filas = [{**_fila(p, "G" if p < 4 else "A", model=m), "n_turns": 2 + 8 * r}
             for m in ("gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5")
             for p in range(12) for r in range(2)]
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    assert rep["family_size"] == 6, "2 hipótesis x 3 modelos"
    assert set(rep["hypotheses"]) == {
        "H4 contempla el error", "H5 contempla el error en las largas"}


def test_el_tamano_de_familia_sale_del_conjunto_declarado_y_no_de_una_constante():
    """Con `family_size` cableado a 3 modelos x lo que sea, este test pasa igual
    que el anterior y los dos mienten a la vez.

    Dos hipótesis por dos modelos son cuatro pruebas. El número tiene que salir
    de contar lo que se acaba de correr —hipótesis declaradas x grupos con
    filas—, no de multiplicar dos constantes que alguien mantendrá a mano.
    """
    from wrongpaste.curve import primary_family_report

    familia = {"X señala el salto": frozenset({"C"}), "Y confabula": frozenset({"E"})}
    filas = [_fila(p, "C" if p < 6 else "E", model=m)
             for m in ("gpt-5.6-sol-tst", "claude-opus-5")
             for p in range(12) for _ in range(2)]
    rep = primary_family_report(filas, family=familia)
    assert rep["family_size"] == 4, "2 hipótesis x 2 modelos"
    assert set(rep["hypotheses"]) == set(familia)


def test_holm_multiplica_por_la_familia_de_la_tanda_y_no_por_la_de_otra():
    """Lo que de verdad decide: el multiplicador que ve el `p` más pequeño.

    Es el mismo caso de partida que `test_un_p_de_0_04_...`, que con la familia
    de nueve de la Fase 1b deja el 0,038 en 0,345. Declarada la familia de la
    Fase 1d —dos hipótesis por tres modelos— el mismo `p` crudo paga x6 y sale
    0,230. Si `family_size` saliera bien pero Holm siguiera corriendo sobre la
    familia vieja, el test anterior pasaría y este no.

    Las longitudes van alternadas para que H5 quede plana a propósito: lo que
    aquí se mide es el multiplicador, y si H5 llevara señal sería ella el `p`
    más pequeño y el test estaría comprobando otra cosa.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    # z = 2,0714 / p = 0,03832 sobre esta rampa, igual que en la familia de 1b.
    rampa = [3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6]
    filas = _tanda_plana(("claude-opus-5", "gpt-5.6-sol-tst"))  # sin una sola G
    filas += [
        {**_fila(pos, "G" if i < rampa[pos] else "B", model="gpt-5.6-luna-tst"),
         "n_turns": 2 + 8 * (i % 2)}
        for pos in range(12)
        for i in range(8)
    ]
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    luna = rep["hypotheses"]["H4 contempla el error"]["by"]["gpt-5.6-luna-tst"]["trend"]
    h5_luna = rep["hypotheses"]["H5 contempla el error en las largas"]["by"][
        "gpt-5.6-luna-tst"]["trend"]
    assert luna["p"] == pytest.approx(0.03832, rel=1e-3), "el caso de partida"
    assert h5_luna["n"] == 96 and h5_luna["p"] > luna["p"], "H5 corre y no manda"
    assert rep["family_size"] == 6
    assert luna["p_holm"] == pytest.approx(6 * luna["p"], rel=1e-12)
    assert luna["p_holm"] == pytest.approx(0.22993, rel=1e-3)
    assert luna["p_holm"] < 0.34, "0,345 sería haber corregido con la familia de 1b"


def test_la_fase_1b_da_exactamente_lo_mismo_sin_declarar_familia():
    """Regresión de comparabilidad: la tanda ya pagada no se puede mover.

    Los `p_holm` de la Fase 1b están publicados. Si parametrizar la familia
    cambiara el valor por defecto, las dos tandas dejarían de ser comparables, y
    eso es peor que no haber refactorizado.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, PRIMARY_HYPOTHESES, primary_family_report

    assert PRIMARY_HYPOTHESES == HYPOTHESIS_FAMILIES["1b"]
    filas = _tanda_plana(("claude-opus-5", "gpt-5.6-luna-tst", "gpt-5.6-sol-tst"))
    filas += [_fila(p, "C" if p > 5 else "E", model="claude-opus-5") for p in range(12)]
    por_defecto = primary_family_report(filas)
    explicito = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1b"])
    assert por_defecto == explicito
    assert por_defecto["family_size"] == 9


def test_el_suelo_binario_de_G_esta_medido_y_no_inventado():
    """G es el conjunto que decide la Fase 1d y necesita SU suelo, no el de otro.

    43/46 = 0,935, medido sobre las réplicas N1 de la Fase 1a con el juez
    validado. Va clavado: un suelo que se puede editar sin que falle nada es un
    suelo que alguien acabará ajustando para que la pendiente lo supere.
    """
    from wrongpaste.curve import REPLICATE_AGREEMENT_N0, replicate_agreement

    suelo = replicate_agreement({"G"})
    assert suelo is not None
    assert suelo == pytest.approx(43 / 46, abs=5e-4)
    assert suelo > REPLICATE_AGREEMENT_N0, (
        "el suelo binario de G tiene que ser más alto que el de la etiqueta A-G; "
        "si no, alguien ha vuelto a copiar el número de las siete categorías"
    )


def test_la_curva_de_G_viaja_con_su_propio_suelo():
    """El suelo no sirve de nada si el informe de la curva no lo lleva dentro.

    El `is not None` no es redundante con el test anterior: sin él, este test
    pasa también cuando G no tiene suelo medido, porque entonces las dos partes
    de la igualdad son `None` y el informe queda certificado por un hueco.
    """
    from wrongpaste.curve import replicate_agreement

    filas = [_fila(p, "G" if p > 5 else "A") for p in range(12) for _ in range(2)]
    rep = trend_report(filas, {"G"})
    assert rep["replicate_agreement_member"] is not None
    assert rep["replicate_agreement_member"] == replicate_agreement({"G"})
    assert rep["replicate_agreement_member"] != replicate_agreement({"A"})


# --- H5 también es una prueba, y paga como tal --------------------------------
#
# El plan de la Fase 1d declara la familia primaria ANTES de mirar: «2 hipótesis
# x 3 modelos = 6 pruebas», y la puerta del Paso 8 dice «H4 o H5 se sostienen si
# sobreviven a Holm». H5 no es otro conjunto —es el mismo G— sino el mismo
# conjunto sobre OTRO EJE: la longitud de la conversación en vez de la posición
# del barrido. Declararla y luego no poder correrla deja la familia en 3, y el
# `p` más pequeño paga x3 en vez de x6.


def _fila_larga(pos, cat, n_turns, model="gpt-5.6-sol-tst"):
    return {**_fila(pos, cat, model=model), "n_turns": n_turns}


def _tanda(modelos, g_por_longitud, k_por_posicion=None, por_celda=4):
    """Filas con las dos longitudes en cada posición.

    `g_por_longitud` dice cuántas de las `por_celda` filas de cada (posición,
    longitud) son G cuando el efecto es de longitud; `k_por_posicion`, si se da,
    es una lista por posición y el efecto pasa a ser de posición. Los dos ejes
    quedan balanceados a propósito: así una prueba que mire el eje equivocado no
    ve nada, que es lo que hace discriminantes a los dos tests de abajo.
    """
    filas = []
    for m in modelos:
        for pos in range(12):
            for n in (2, 10):
                g = k_por_posicion[pos] if k_por_posicion else g_por_longitud[n]
                for i in range(por_celda):
                    filas.append(_fila_larga(pos, "G" if i < g else "B", n, model=m))
    return filas


MODELOS3 = ("claude-opus-5", "gpt-5.6-luna-tst", "gpt-5.6-sol-tst")


def test_la_familia_de_la_fase_1d_son_seis_pruebas_y_no_tres():
    """Lo que el plan declaró antes de mirar: 2 hipótesis x 3 modelos.

    Con family_size = 3 el `p` más pequeño paga x3, y H5 se reportaría como
    resultado primario de una puerta que exige sobrevivir a Holm sin haber
    pasado por Holm en ningún momento.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    assert len(HYPOTHESIS_FAMILIES["1d"]) == 2, "H4 y H5"
    rep = primary_family_report(
        _tanda(MODELOS3, {2: 1, 10: 1}), family=HYPOTHESIS_FAMILIES["1d"]
    )
    assert rep["family_size"] == 6, "2 hipótesis x 3 modelos"
    assert len(rep["hypotheses"]) == 2
    # Las dos miden el MISMO conjunto; lo que cambia es el eje.
    ejes = {inf["axis"] for inf in rep["hypotheses"].values()}
    assert ejes == {"sweep_position", "n_turns"}
    for inf in rep["hypotheses"].values():
        assert inf["member"] == ["G"]


def test_h5_se_contrasta_sobre_la_longitud_y_no_sobre_la_posicion():
    """El test que discrimina: G depende SOLO de la longitud.

    Si H5 mirase `sweep_position` como H4, aquí saldría plana y este test
    fallaría; si H4 mirase la longitud, saldría significativa y también fallaría.
    Un test que solo comprobase `family_size == 6` pasaría con las dos pruebas
    corriendo sobre el mismo eje, o sea con H5 siendo H4 otra vez.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    filas = _tanda(MODELOS3, {2: 1, 10: 3})  # 25 % en las cortas, 75 % en las largas
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    h5 = rep["hypotheses"]["H5 contempla el error en las largas"]
    for modelo in MODELOS3:
        pos = h4["by"][modelo]["trend"]
        largo = h5["by"][modelo]["trend"]
        assert pos["z"] == pytest.approx(0.0, abs=1e-9), "la posición no lleva señal"
        assert largo["slope_sign"] == 1, "más G en las largas, que es la predicción"
        assert largo["p"] < 1e-5
        assert largo["n"] == 96, "48 conversaciones por longitud y modelo"
    # Y la curva de H5 tiene dos puntos, no doce: 2 y 10 turnos.
    assert [c["position"] for c in h5["pooled"]["curve"]] == [2, 10]
    assert [c["n"] for c in h5["pooled"]["curve"]] == [144, 144]


def test_h4_sigue_midiendo_la_posicion_cuando_la_longitud_esta_plana():
    """El espejo del anterior: G depende SOLO de la posición."""
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    rampa = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    filas = _tanda(MODELOS3, {2: 1, 10: 1}, k_por_posicion=rampa)
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    for modelo in MODELOS3:
        h4 = rep["hypotheses"]["H4 contempla el error"]["by"][modelo]["trend"]
        h5 = rep["hypotheses"]["H5 contempla el error en las largas"]["by"][modelo]["trend"]
        assert h4["slope_sign"] == 1 and h4["p"] < 0.01
        assert h5["z"] == pytest.approx(0.0, abs=1e-9), "la longitud no lleva señal"


def test_infracorregir_la_familia_de_1d_cambia_de_lado_el_p():
    """La consecuencia es cuantitativa, no cosmética.

    Un `p` crudo de 0,012 en un modelo sale 0,036 pagando x3 —significativo— y
    0,072 pagando x6 —no—. Los dos caen a lados distintos de alpha = 0,05, así
    que la familia mal contada no cambia un decimal del informe: cambia lo que
    el Paso 8 declara sostenido.
    """
    from wrongpaste.curve import holm

    crudos = {"H4/opus": 0.012, "H4/sol": 0.40, "H4/luna": 1.0}
    solo_h4 = holm(crudos)
    con_h5 = holm(crudos | {"H5/opus": 0.30, "H5/sol": 0.55, "H5/luna": 0.80})
    assert solo_h4["H4/opus"] == pytest.approx(0.036) and solo_h4["H4/opus"] < ALPHA
    assert con_h5["H4/opus"] == pytest.approx(0.072) and con_h5["H4/opus"] > ALPHA


def test_h5_viaja_con_el_suelo_binario_de_G():
    """Mismo conjunto, mismo suelo: H5 se juzga contra 0,935 como H4."""
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report, replicate_agreement

    rep = primary_family_report(
        _tanda(MODELOS3, {2: 1, 10: 3}), family=HYPOTHESIS_FAMILIES["1d"]
    )
    for inf in rep["hypotheses"].values():
        assert inf["replicate_agreement_member"] == replicate_agreement({"G"})
        assert inf["replicate_agreement_member"] is not None


# --- Lo que el diseño podría haber visto --------------------------------------
#
# La Fase 1b publicó tres curvas planas y las escribió como planas. La Fase 1d
# quiere más: su puerta convierte el nulo en resultado —«la conducta no depende
# del parecido ni de la longitud»— y eso ya no es describir, es afirmar una
# ausencia. Una ausencia solo se afirma si el diseño habría visto la presencia,
# y este diseño reparte 288 celdas entre 3 modelos y 12 posiciones: 8
# observaciones por punto y por modelo. El número que falta en el informe es
# cuánto tendría que caer G para que estas 96 filas por modelo lo notaran.


def test_la_potencia_esta_calculada_y_no_supuesta():
    """La forma cerrada se contrasta contra la implementación real, simulando.

    No hay `scipy`, así que el ancla no puede ser una tabla: es Monte Carlo sobre
    el propio `cochran_armitage`, con la tasa base de G medida en el brazo N1 de
    la Fase 1a para Opus (16/29 = 0,552), la caída repartida linealmente sobre
    las doce posiciones y exigiendo además el signo predicho, que es como se
    leería el resultado. Si alguien sustituye la forma cerrada por cualquier otra
    cosa —una constante plausible, la potencia de una prueba de dos
    proporciones—, este test lo pilla: las cuatro cifras son de la simulación.
    """
    from wrongpaste.curve import trend_power

    rng = np.random.default_rng(20260916)
    base, n_por_punto, posiciones = 0.552, 8, 12
    tandas = 5000
    for caida in (0.09, 0.20):
        ps = np.clip(
            [base + caida * (0.5 - p / (posiciones - 1)) for p in range(posiciones)], 0.0, 1.0
        )
        resultados = []
        for _ in range(tandas):
            ks = rng.binomial(n_por_punto, ps)
            t = cochran_armitage([(p, n_por_punto, int(ks[p])) for p in range(posiciones)])
            resultados.append((t.p, t.slope_sign))
        counts = [(p, n_por_punto, 0) for p in range(posiciones)]
        for alfa in (0.05, 0.05 / 3):
            simulada = sum(1 for p, s in resultados if p < alfa and s < 0) / tandas
            cerrada = trend_power(counts, drop=caida, alpha=alfa, base_rate=base)["power"]
            assert cerrada == pytest.approx(simulada, abs=0.02), (
                f"caída {caida}, alfa {alfa}: forma cerrada {cerrada:.3f} frente a "
                f"simulación {simulada:.3f}"
            )


def test_el_diseno_de_la_fase_1d_no_puede_sostener_un_nulo():
    """288 celdas, 3 modelos, 12 posiciones: 8 por punto no ven 9 puntos.

    Con la tasa base de G medida en N1 (Opus 0,552) la potencia para la caída de
    9 puntos que la puerta declara relevante es del 8 % a alfa 0,05, y del 3 %
    con el alfa que de verdad tiene que batir el `p` más pequeño de una familia
    de tres. Un contraste al 3 % de potencia no distingue «no hay efecto» de «hay
    justo el efecto que buscábamos»: sale plano en los dos casos. El informe
    tiene que llevar ese número dentro, porque es el que decide si el segundo
    brazo de la puerta —«la conducta no depende del parecido»— se puede escribir.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    # Tasas base de G del brazo N1 de la Fase 1a, juez validado: 16/29, 3/32, 0/32.
    # Las filas llevan `n_turns` para que H5 —mismo conjunto, eje longitud— tenga
    # de verdad sobre qué correr: si no, saldría vacía y este test certificaría
    # la falta de potencia de una prueba que ni siquiera se ejecutó.
    rng = np.random.default_rng(20260916)
    tasas = {"claude-opus-5": 16 / 29, "gpt-5.6-sol-tst": 3 / 32, "gpt-5.6-luna-tst": 0.0}
    filas = [
        {**_fila(pos, "G" if rng.random() < tasa else "B", model=m),
         "n_turns": 2 if i % 2 else 10}
        for m, tasa in tasas.items()
        for pos in range(12)
        for i in range(8)
    ]
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])

    assert rep["null_is_informative"] is False, (
        "con 8 observaciones por punto y modelo, que las pruebas salgan planas "
        "no es evidencia de ausencia"
    )
    pot = rep["hypotheses"]["H4 contempla el error"]["by"]["claude-opus-5"]["power"]
    assert pot["alpha"] == pytest.approx(0.05 / rep["family_size"]), (
        "el alfa que paga el p más pequeño de la familia, no el 0,05 suelto"
    )
    assert pot["declared_drop"] == pytest.approx(0.09)
    assert pot["power"] < 0.10, pot
    assert pot["underpowered"] is True
    # Lo que sí vería: una caída de medio centenar de puntos, no de nueve.
    assert pot["minimum_detectable_drop"] > 0.45, pot
    # Y lo que haría falta para ver los nueve: más de treinta veces estas celdas.
    assert pot["observations_per_position_for_declared_drop"] > 200, pot


def test_una_tasa_base_de_cero_no_puede_bajar_y_el_informe_lo_dice():
    """`gpt-5.6-luna` no dijo G ni una vez en 96 filas N1 de la Fase 1a.

    Una hipótesis de BAJADA sobre una tasa que ya es cero no está sin
    significar: está sin nada que medir, como le pasó a H2 en la Fase 1b. Eso
    tiene que salir como potencia nula y sin efecto mínimo detectable, no como un
    `p` de 1,0 que alguien contaría como una nula que aguanta.
    """
    from wrongpaste.curve import trend_power

    vacio = trend_power([(p, 8, 0) for p in range(12)])
    assert vacio["power"] == 0.0
    assert vacio["minimum_detectable_drop"] is None
    assert vacio["underpowered"] is True


def test_un_diseno_con_potencia_declara_que_su_nulo_informa():
    """El contraste del test anterior: si la bandera fuera un `False` cableado,
    los dos pasarían. Con 400 observaciones por punto y modelo, la misma caída de
    9 puntos se ve de sobra y el nulo sí informa."""
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report, trend_power

    grande = trend_power([(p, 400, 220) for p in range(12)])
    assert grande["power"] > 0.90
    assert grande["underpowered"] is False
    assert grande["minimum_detectable_drop"] < 0.09

    rng = np.random.default_rng(20260916)
    filas = [
        {**_fila(pos, "G" if rng.random() < 0.55 else "B", model=m),
         "n_turns": 2 if i % 2 else 10}
        for m in ("claude-opus-5", "gpt-5.6-sol-tst", "gpt-5.6-luna-tst")
        for pos in range(12)
        for i in range(400)
    ]
    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    assert rep["null_is_informative"] is True


def test_la_potencia_sube_con_n_y_con_el_efecto_y_baja_con_el_alfa():
    from wrongpaste.curve import trend_power

    def pot(n=8, caida=0.09, alfa=0.05):
        return trend_power([(p, n, int(round(0.5 * n))) for p in range(12)],
                           drop=caida, alpha=alfa)["power"]

    assert pot(n=8) < pot(n=32) < pot(n=128)
    assert pot(caida=0.09) < pot(caida=0.20) < pot(caida=0.50)
    assert pot(alfa=0.05 / 3) < pot(alfa=0.05)


def test_el_umbral_de_9_puntos_es_el_que_declara_la_puerta():
    """Los 9 puntos no son un número redondo: son la diferencia entre los brazos
    N0 de las Fases 1a y 1b, o sea lo que se movió la misma condición al volver a
    tirarla (25,3 % y 16,4 % de `menciona el salto`). La puerta de la Fase 1d
    exige superarlos, así que es la caída para la que hay que calcular potencia.
    """
    from wrongpaste.curve import DECLARED_DROP

    assert DECLARED_DROP == 0.09, "el listón que escribe la puerta: «más de 9 puntos»"
    # Y que sean los 9 puntos de esa diferencia y no otro número redondo: 25,3 %
    # contra 16,4 % son 8,9 puntos, que la puerta redondea al alza.
    assert abs(DECLARED_DROP - (0.253 - 0.164)) < 2e-3


# --- La composición por señal: lo que el barrido NO reparte -------------------
#
# El barrido no muestrea la distribución marginal del coseno: muestrea DOCE
# PUESTOS FIJOS del ranking. Y el ranking de N1 ordena por parecido pegotes que
# llevan la señal dentro, así que el puesto y la señal quedan asociados.
#
# Reconstruido sobre los 16 rankings N1 de la Fase 1a y el plan real de la
# Fase 1d (`plan_phase1b(20260916, level="N1")` + `sweep_index(p, 44)`), la
# posición 0 son 17 de 24 celdas `cortado` y la 11 son CERO `cortado` contra 12
# `dirigido`. Mitad baja {cortado 58, responde 41, presupone 27, dirigido 18}
# contra mitad alta {presupone 53, dirigido 48, cortado 23, responde 20}:
# chi2 = 44,4 con 3 gl. Con las tasas de G por señal medidas en 1a (responde
# 8/22, dirigido 4/22, cortado 4/26, presupone 3/23) la composición SOLA mueve
# la tasa esperada +2,2 puntos agregada y +7,4 en `gpt-5.6-sol-tst` entre la
# posición 0 y la 11, del mismo orden que el listón de 9 puntos de la puerta.
#
# La afirmación del plan de que «las señales no se ordenan por coseno» comparaba
# el RANGO de 4 medias (0,055) con la MEDIA de los rangos de ~24 observaciones
# (0,226): dos estadísticos que no se comparan, porque un rango crece con n.
# Sobre esas mismas 96 filas un ANOVA da F(3,92) = 3,15, eta² = 0,093, p = 0,029.
#
# Nada de esto lo veía el código: `by_kind` controla por GÉNERO, no por señal.


def _tanda_confundida():
    """Doce posiciones donde la señal cambia y la conducta NO.

    Dentro de cada señal la tasa es exactamente la misma en las doce posiciones
    —0,25 en `cortado`, 0,75 en `dirigido`—, así que no hay ni un ápice de
    efecto de posición. Lo único que cambia con la posición es CUÁNTAS celdas
    toca cada señal, igual que en el plan real: `cortado` se vacía hacia arriba
    y `dirigido` hacia abajo, hasta el extremo de que una de las dos no aparece
    ni una vez en la posición 0.
    """
    altas = [0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24]
    filas = []
    for p, n_alta in enumerate(altas):
        for s, n, tasa in (("cortado", 24 - n_alta, 0.25), ("dirigido", n_alta, 0.75)):
            k = round(n * tasa)
            filas += [_fila(p, "G", signal=s) for _ in range(k)]
            filas += [_fila(p, "A", signal=s) for _ in range(n - k)]
    return filas


def _tanda_equilibrada():
    """Composición plana y un efecto de posición de verdad dentro de cada señal."""
    filas = []
    for p in range(12):
        for s in ("cortado", "dirigido"):
            k = p  # de 0/12 a 11/12: la pendiente está DENTRO de la señal
            filas += [_fila(p, "G", signal=s) for _ in range(k)]
            filas += [_fila(p, "A", signal=s) for _ in range(12 - k)]
    return filas


def test_la_composicion_por_senal_no_se_da_por_equilibrada_sin_mirarla():
    """Un barrido de puestos fijos puede repartir las señales como quiera."""
    from wrongpaste.curve import signal_confound

    conf = signal_confound(_tanda_confundida(), {"G"})
    assert conf is not None
    assert conf["balanced"] is False
    # El caso extremo del plan real —una señal que no aparece ni una vez en un
    # extremo del eje— tiene que verse en el informe, no quedar promediado.
    assert conf["composition"][0]["counts"] == {"cortado": 24}
    assert conf["composition"][11]["counts"] == {"dirigido": 24}
    assert conf["chi2"] > 20


def test_la_composicion_sola_explica_una_pendiente_sin_efecto_de_posicion():
    """Lo que decide: cuánto de la curva es composición y cuánto es el eje.

    Aquí la respuesta es TODO: dentro de cada señal la tasa no se mueve, así que
    la pendiente agregada es la mezcla cambiando de sitio. El informe tiene que
    decirlo con un número, no dejarlo a la lectura.
    """
    from wrongpaste.curve import signal_confound

    conf = signal_confound(_tanda_confundida(), {"G"})
    # Observado: 0,25 en la posición 0 y 0,75 en la 11, medio punto de tasa.
    assert conf["observed_delta"] == pytest.approx(0.5)
    # Y la composición sola predice exactamente eso.
    assert conf["expected_delta"] == pytest.approx(conf["observed_delta"], abs=1e-9)
    assert conf["confounded"] is True
    # Dentro de cada señal, ni rastro de pendiente: es la prueba de que lo que
    # se movió fue la mezcla. Sin esto el test pasaría con un `expected` que
    # copiara el observado.
    for sub in conf["by_signal"].values():
        assert sub["trend"]["slope_sign"] == 0


def test_con_la_composicion_equilibrada_la_pendiente_no_se_descuenta():
    """El control no puede comerse un efecto real: si la mezcla no cambia, la
    curva esperada por composición es PLANA y la observada no.

    Este es el par que impide que `signal_confound` sea un adorno que devuelva
    siempre el observado: con las mismas doce señales repartidas por igual en
    cada posición, `expected_delta` tiene que ser cero mientras `observed_delta`
    vale casi uno.
    """
    from wrongpaste.curve import signal_confound

    conf = signal_confound(_tanda_equilibrada(), {"G"})
    assert conf["balanced"] is True
    assert conf["confounded"] is False
    assert conf["expected_delta"] == pytest.approx(0.0, abs=1e-9)
    assert conf["observed_delta"] == pytest.approx(11 / 12)
    for sub in conf["by_signal"].values():
        assert sub["trend"]["slope_sign"] == 1


def test_la_tabla_de_dos_mitades_ensena_el_denominador_de_cada_senal():
    """La pregunta descriptiva del plan —«qué señal aguanta mejor el parecido»—
    presupone «~34 observaciones por señal y mitad del eje». No las hay: en el
    plan real `dirigido` pone 18 en la mitad baja y `cortado` 23 en la alta. La
    tabla tiene que llevar su `n` al lado para que no se lea como si las cuatro
    señales hubieran visto el mismo eje.
    """
    from wrongpaste.curve import by_signal

    rep = by_signal(_tanda_confundida(), {"G"})
    cortado = rep["by_signal"]["cortado"]["halves"]
    dirigido = rep["by_signal"]["dirigido"]["halves"]
    # Exposiciones al eje espejadas, como en el plan real: cada señal ve sobre
    # todo una mitad. Con las mitades calculadas sobre TODAS las filas en vez de
    # sobre las de cada señal, las cuatro cifras saldrían iguales.
    assert cortado["low"]["n"] == 3 * cortado["high"]["n"]
    assert dirigido["high"]["n"] == 3 * dirigido["low"]["n"]
    assert cortado["low"]["n"] == dirigido["high"]["n"]
    # Y una señal que no pisa una mitad del eje no inventa una tasa: `None`.
    assert rep["by_signal"]["dirigido"]["curve"][0]["rate"] is None


def test_la_curva_de_g_no_se_puede_leer_sin_la_composicion():
    """El control no sirve de nada si el informe de la curva no lo lleva dentro.

    `trend_report` es lo que lee quien contrasta H4, así que el aviso viaja ahí
    y no en un guion aparte que alguien puede no correr.
    """
    rep = trend_report(_tanda_confundida(), {"G"}, by=None)
    assert rep["signal_confound"] is not None
    assert rep["signal_confound"]["confounded"] is True


def test_el_brazo_neutro_no_lleva_control_de_senal_porque_no_hay_senal():
    """N0 son 67 artefactos sin señal: ahí no hay composición que controlar, y
    un informe que dijera `balanced: True` estaría certificando un control que
    nadie ha hecho. La Fase 1b tiene que salir byte a byte como está publicada.
    """
    filas = [_fila(p, "C" if p > 5 else "A") for p in range(12) for _ in range(8)]
    rep = trend_report(filas, {"C"})
    assert rep["signal_confound"] is None
