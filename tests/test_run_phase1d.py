"""Tests de la Fase 1d rediseñada: bandas, un modelo por estímulo. Sin red.

La Fase 1b barrió el eje con **doce puestos exactos** del ranking: para cada
prefijo, la posición `p` daba el artefacto del puesto `sweep_index(p, N)`. Con 16
prefijos (8 temas x 2 longitudes) eso son 16 estímulos distintos por posición, y
los tres modelos se los repartían viendo cada uno los 16. Mucha conversación y
poca información: la potencia para la caída de 9 puntos que el proyecto declara
relevante salía del 16 %.

Lo que se prueba aquí es lo que el rediseño cambia, y cada test mira una de las
tres patas:

1. **Bandas en vez de puestos**: el ranking se parte en 4 tramos por rango y de
   cada tramo se sortean 18 artefactos distintos por prefijo, sin reemplazo.
2. **Un modelo por estímulo**: cada (prefijo, artefacto) lo ve UN modelo, no los
   tres, y el reparto es equilibrado y determinista.
3. **Que la Fase 1b no se haya movido**: está pagada y publicada.

El test que justifica el rediseño entero es
`test_las_filas_no_repiten_ningun_estimulo`: 1.152 celdas con 1.152 pares
(prefijo, artefacto) DISTINTOS. Bajo el diseño viejo ese número habría sido 192.
"""

import collections
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys

import pytest

import wrongpaste.conversation as conv
import wrongpaste.run_phase0 as rp0
import wrongpaste.run_phase1b as rp1b
import wrongpaste.run_judging as run_judging
import wrongpaste.run_phase1d as rp
import wrongpaste.similarity as sim
import wrongpaste.simulated_user as su
from wrongpaste.artifacts import Artifact
from wrongpaste.rates import CATEGORY_FIELD
from wrongpaste.run_phase0 import stratum_window
from wrongpaste.run_phase1d import (
    ARTIFACTS_PER_BAND,
    BANDS,
    LENGTHS,
    PHASE1D_MODELS,
    band_order,
    band_window,
    choose_band_artifact,
    plan_phase1d,
)

# El banco N1 de verdad. Es lo que hace que una banda traiga exactamente los
# `ARTIFACTS_PER_BAND` slots del diseño; con otro tamaño, los números del plan
# dejarían de ser los que se van a correr.
BANK_N1 = 84

# Los recuentos del diseño se DERIVAN de las constantes, no se clavan a mano.
# Clavarlos obligaba a repasar cuarenta y cinco números cada vez que el diseño
# cambiaba de tamaño —y el tamaño ha cambiado dos veces ya, de 18 a 19 y de 19 a
# 21— con el riesgo de dejar la mitad viejos y la suite verde. Lo que sí se clava
# es cada constante por separado, en `test_las_constantes_del_diseno_son_las_que
# _se_va_a_pagar`: si alguien mueve una, ese test lo dice, y estos recuentos se
# mueven con ella en vez de mentir.
PREFIXES = 16  # 8 temas x 2 longitudes
CELLS = BANDS * ARTIFACTS_PER_BAND * PREFIXES
CELLS_PER_BAND = CELLS // BANDS
CELLS_PER_MODEL = CELLS // len(PHASE1D_MODELS)
CELLS_PER_BAND_MODEL = CELLS_PER_BAND // len(PHASE1D_MODELS)
CELLS_PER_BAND_MODEL_LENGTH = CELLS_PER_BAND_MODEL // len(LENGTHS)
# El contraste primario pliega las BANDS bandas a dos, así que cada punto de la
# curva primaria son la mitad de las celdas.
CELLS_PER_PRIMARY_BAND = CELLS // 2

# Potencia de una prueba primaria con el plantel de TRES modelos: cada modelo ve
# CELLS_PER_MODEL celdas, plegadas a dos puntos. Se clava porque es el número
# que la puerta compara contra MIN_POWER; si el diseño cambia de tamaño, cambia.
POTENCIA_H4_TRES_MODELOS = 0.2348

# La misma prueba, mismo tamaño y mismo alfa, para `sol`. Sale distinta porque
# lo que más pesa en un diseño binario es la tasa base, y la de `sol` en N1 es
# 3/32 frente a los 16/29 de Opus. Va aparte y clavada por lo mismo que la otra:
# es un resultado del estimador, no una constante del diseño.
POTENCIA_H4_SOL_TRES_MODELOS = 0.7355

# Y con el plantel de UN modelo —la tanda que de verdad se va a pagar—. Cambian
# las dos cosas a la vez: la familia encoge a dos pruebas (alfa 0,05/2 en vez de
# 0,05/6) y Opus se queda las CELLS celdas enteras, CELLS_PER_PRIMARY_BAND por
# punto del contraste plegado. Con el tamaño nuevo del diseño (21 artefactos por
# banda, 1.344 celdas) este número cruza el MIN_POWER de 0,80; con el viejo, de
# 18 y 1.152, se quedaba en 0,797. Es el número por el que el diseño creció.
POTENCIA_H4_UN_MODELO = 0.859


def test_las_constantes_del_diseno_son_las_que_se_va_a_pagar():
    """Los recuentos de esta suite se derivan de aquí, así que aquí se clavan.

    `ARTIFACTS_PER_BAND` tiene que dividir entre 3: la guarda de `plan_phase1d`
    exige que el plantel reparta por igual los slots de cada (prefijo, banda), y
    con un primo ningún plantel de tres modelos lo hace.
    """
    assert BANDS == 4
    assert ARTIFACTS_PER_BAND == 21
    assert ARTIFACTS_PER_BAND % len(PHASE1D_MODELS) == 0
    assert BANK_N1 == BANDS * ARTIFACTS_PER_BAND
    assert CELLS == 1344

SIGNALS = ("cortado", "dirigido", "responde", "presupone")
KINDS = ("email", "nota", "mensaje", "albaran", "aviso", "sms", "carta", "acta")

def _fake_bank(n: int) -> list[Artifact]:
    """Un banco N1 de `n` artefactos. El tamaño es parámetro porque el tamaño
    del banco es justo lo que un test tiene que poder mover: el plan declara
    4 x 18 artefactos y nada en el código los ataba al banco de disco."""
    return [
        Artifact(
            f"n1-{i:02d}",
            KINDS[(i * 3) % len(KINDS)],
            f"texto n1 {i}",
            (f"e{i}",),
            level="N1",
            signal=SIGNALS[i % len(SIGNALS)],
        )
        for i in range(n)
    ]


FAKE_BANK_N1 = _fake_bank(BANK_N1)


def fake_load_artifacts(level: str | None = None) -> list[Artifact]:
    """Doble de `load_artifacts` con el TAMAÑO real del banco N1 (72).

    Exige el nivel por lo mismo que el de la Fase 1a: pedir el banco entero
    mezclaría N0 con N1 y el ranking —que aquí ES el eje— saldría contaminado
    con artefactos que esta tanda no puede pegar.
    """
    assert level == "N1", f"la Fase 1d corre sobre N1, no sobre {level!r}"
    return list(FAKE_BANK_N1)


def _ranking(n=BANK_N1):
    """Ranking ascendente por coseno, como lo devuelve `rank_artifacts` (D2)."""
    return [(art, i / (n - 1)) for i, art in enumerate(FAKE_BANK_N1[:n])]


# --- las bandas ------------------------------------------------------------


def test_band_window_es_stratum_window_y_no_una_copia():
    """`stratum_window` ya parte un ranking por rango, que es justo lo que una
    banda necesita. Reimplementarlo daría dos definiciones de «tramo del
    ranking» que se pueden separar en los bordes sin que nadie lo note."""
    for n in (72, 70, 3):
        r = _ranking(min(n, BANK_N1)) if n <= BANK_N1 else _ranking()
        for b in range(BANDS):
            assert band_window(r, b, BANDS) == stratum_window(r, b, BANDS), (n, b)


def test_las_cuatro_bandas_parten_el_banco_sin_huecos_ni_solapes():
    """Si las bandas se solaparan, dos bandas podrían dar el mismo artefacto y
    el contraste entre bandas mediría menos eje del que dice medir."""
    r = _ranking()
    visto = []
    for b in range(BANDS):
        ventana = band_window(r, b, BANDS)
        assert len(ventana) == ARTIFACTS_PER_BAND, b
        visto += [art.id for art, _ in ventana]
    assert visto == [art.id for art, _ in r]
    assert len(set(visto)) == BANK_N1


def test_la_banda_baja_es_la_menos_parecida_y_la_alta_la_mas():
    r = _ranking()
    baja = [s for _, s in band_window(r, 0, BANDS)]
    alta = [s for _, s in band_window(r, BANDS - 1, BANDS)]
    assert max(baja) < min(alta)


def test_una_banda_fuera_de_rango_es_un_error():
    with pytest.raises(ValueError):
        band_window(_ranking(), BANDS, BANDS)
    with pytest.raises(ValueError):
        band_window(_ranking(), -1, BANDS)


def test_las_bandas_que_pliega_el_analisis_son_las_que_muestrea_este_runner():
    """El test que el comentario de `curve.BANDS` dice que existe, y no existía.

    `curve` no importa la constante del runner a propósito —el análisis no sabe
    de tiradas—, así que el 4 está escrito dos veces. Lo que hace que duplicarlo
    sea aceptable es exactamente esto: que moverlo en un lado ponga la suite
    roja. Sin este test, subir las bandas del runner a 6 no revienta nada —
    `fold_band` devuelve `None` para las bandas que el análisis no declara y los
    dos sitios que pliegan las descartan—, y el contraste primario sale de dos
    puntos bien formados sobre 768 de las 1.152 conversaciones pagadas.

    La guarda de `curve.rate_by_position` sobre el `n_strata` de la fila cubre
    las filas ya escritas; ésta cubre el caso en que las dos mitades del proyecto
    se separan ANTES de pagar la tanda.
    """
    from wrongpaste import curve

    assert curve.BANDS == rp.BANDS, (
        "el análisis pliega desde el muestreo del runner: si no son el mismo "
        "número, las bandas que sobran se caen del contraste en silencio"
    )
    # Y el campo en el que la fila declara su muestreo es el que el runner
    # escribe: la guarda del análisis lee `n_strata` y `run_cell` lo pone ahí.
    assert curve.BAND_COUNT_FIELD == "n_strata"
    assert curve.BAND_AXIS == "stratum"


# --- el muestreo sin reemplazo dentro de la banda --------------------------


def test_los_18_slots_de_una_banda_agotan_la_banda_sin_repetir():
    """El muestreo es SIN REEMPLAZO: los 18 slots de un (prefijo, banda) son los
    18 artefactos de esa banda. Con reemplazo, dos celdas del mismo prefijo
    podrían llevar el mismo pegote y el recuento de estímulos independientes
    volvería a estar inflado, que es el fallo de la Fase 1b."""
    r = _ranking()
    for b in range(BANDS):
        elegidos = [
            choose_band_artifact(r, b, s, "pfx-tema-0-2")[0].id
            for s in range(ARTIFACTS_PER_BAND)
        ]
        assert len(set(elegidos)) == ARTIFACTS_PER_BAND, b
        assert set(elegidos) == {art.id for art, _ in band_window(r, b, BANDS)}, b


def test_el_artefacto_elegido_lleva_su_coseno_del_ranking():
    """La fila guarda el coseno que venía con el artefacto. Si el muestreo
    devolviera el par de otro puesto, el eje x diría una cosa y el pegote otra."""
    r = _ranking()
    por_id = {art.id: s for art, s in r}
    for b in range(BANDS):
        for s in range(ARTIFACTS_PER_BAND):
            art, coseno = choose_band_artifact(r, b, s, "pfx-tema-3-10")
            assert coseno == por_id[art.id]


def test_el_mismo_slot_da_siempre_el_mismo_artefacto():
    """Sin esto la tirada no se puede reanudar: una celda reintentada dentro de
    un mes tiene que recibir el pegote que le tocaba hoy."""
    r = _ranking()
    assert choose_band_artifact(r, 2, 7, "pfx-tema-1-10") == choose_band_artifact(
        r, 2, 7, "pfx-tema-1-10"
    )


def test_el_orden_de_la_banda_cambia_con_el_prefijo_y_con_la_banda():
    """El barajado se siembra con (prefijo, banda). Si ignorara el prefijo, el
    slot —y con él el modelo, que se asigna por slot— quedaría pegado siempre a
    los mismos artefactos; si ignorara la banda, las cuatro bandas recorrerían
    sus ventanas en el mismo orden."""
    a = band_order("pfx-tema-0-2", 1, ARTIFACTS_PER_BAND)
    b = band_order("pfx-tema-1-2", 1, ARTIFACTS_PER_BAND)
    c = band_order("pfx-tema-0-2", 2, ARTIFACTS_PER_BAND)
    assert sorted(a) == list(range(ARTIFACTS_PER_BAND))
    assert a != b, "el barajado no mira el prefijo"
    assert a != c, "el barajado no mira la banda"


def test_un_slot_fuera_de_la_banda_revienta_en_vez_de_repetir():
    """Con un banco más pequeño que 4 x 18 no hay 18 artefactos distintos por
    banda, así que la garantía del rediseño —ningún estímulo repetido— no se
    puede cumplir. Se para en vez de dar la vuelta en silencio: un plan que
    repite pegotes sin decirlo es exactamente lo que esta fase viene a arreglar.
    """
    with pytest.raises(ValueError):
        choose_band_artifact(_ranking(), 0, ARTIFACTS_PER_BAND, "pfx-tema-0-2")
    with pytest.raises(ValueError):
        choose_band_artifact(_ranking(40), 0, 12, "pfx-tema-0-2")


def test_el_barajado_no_depende_de_la_semilla_de_hash_del_proceso():
    """`hash()` de Python está aleatorizado por proceso (PYTHONHASHSEED), así que
    sembrar el barajado con él daría un plan distinto en cada arranque: la misma
    celda reintentada mañana recibiría otro pegote y la tirada dejaría de ser
    reproducible. Este test corre el mismo barajado en dos procesos con semillas
    de hash distintas y exige el mismo orden."""
    codigo = (
        "from wrongpaste.run_phase1d import band_order;"
        "print(band_order('pfx-tema-0-2', 1, 18))"
    )
    salidas = []
    for semilla in ("0", "1", "12345"):
        entorno = {**os.environ, "PYTHONHASHSEED": semilla}
        out = subprocess.run(
            [sys.executable, "-c", codigo],
            check=True,
            capture_output=True,
            text=True,
            env=entorno,
        )
        salidas.append(out.stdout.strip())
    assert len(set(salidas)) == 1, salidas
    assert salidas[0] == str(band_order("pfx-tema-0-2", 1, 18))


# --- el plan ---------------------------------------------------------------


def test_el_plan_tiene_1152_celdas():
    # 4 bandas x 8 temas x 2 longitudes x 18 artefactos
    assert BANDS * 8 * len(LENGTHS) * ARTIFACTS_PER_BAND == CELLS
    assert len(plan_phase1d(1)) == CELLS


def test_cada_celda_declara_banda_slot_modelo_y_nivel():
    celda = plan_phase1d(1)[0]
    for campo in (
        "band",
        "artifact_slot",
        "model_id",
        "topic_id",
        "n_turns",
        "seed",
        "conversation_id",
        "paste_level",
        "condition",
    ):
        assert campo in celda, campo
    assert celda["conversation_id"].startswith("p1d-")
    assert {c["paste_level"] for c in plan_phase1d(1)} == {"N1"}


def test_cada_prefijo_y_banda_agota_los_18_slots():
    """El (prefijo, banda) es la unidad del muestreo sin reemplazo: si a uno le
    faltara un slot o le sobrara, la banda no quedaría agotada y los 1.152
    estímulos distintos dejarían de serlo."""
    por_celda = collections.defaultdict(list)
    for c in plan_phase1d(1):
        por_celda[(c["topic_id"], c["n_turns"], c["band"])].append(c["artifact_slot"])
    assert len(por_celda) == 8 * len(LENGTHS) * BANDS
    for clave, slots in por_celda.items():
        assert sorted(slots) == list(range(ARTIFACTS_PER_BAND)), clave


def test_el_reparto_de_modelos_es_exacto_en_las_tres_particiones():
    """CELLS_PER_MODEL por modelo, y su reparto exacto por banda y por longitud.

    El equilibrio por banda no es cosmético: si un modelo cayera más en las
    bandas altas, el contraste entre modelos mediría la banda, y el contraste
    entre bandas mediría el modelo. Los dos son las preguntas de la tanda.
    """
    plan = plan_phase1d(1)
    assert collections.Counter(c["model_id"] for c in plan) == {
        m: CELLS_PER_MODEL for m in PHASE1D_MODELS
    }
    por_banda = collections.Counter((c["model_id"], c["band"]) for c in plan)
    assert set(por_banda.values()) == {CELLS_PER_BAND_MODEL}
    assert len(por_banda) == len(PHASE1D_MODELS) * BANDS
    por_largo = collections.Counter(
        (c["model_id"], c["band"], c["n_turns"]) for c in plan
    )
    assert set(por_largo.values()) == {CELLS_PER_BAND_MODEL_LENGTH}
    assert len(por_largo) == len(PHASE1D_MODELS) * BANDS * len(LENGTHS)


def test_las_longitudes_estan_equilibradas_dentro_de_cada_banda():
    """Si una banda cayera entera en conversaciones cortas, el contraste entre
    bandas mediría la longitud disfrazada de parecido (la confusión que D3
    deshacía en 1a y que 1b volvía a comprobar en cada posición)."""
    plan = plan_phase1d(1)
    for b in range(BANDS):
        largos = collections.Counter(c["n_turns"] for c in plan if c["band"] == b)
        assert set(largos) == set(LENGTHS), b
        assert len(set(largos.values())) == 1, f"banda {b}: {dict(largos)}"


def test_las_1152_celdas_son_1152_pares_prefijo_artefacto_distintos():
    """**El test que justifica el rediseño.** Sobre el plan, resolviendo el
    artefacto de cada celda con el mismo muestreo que usa el runner.

    Bajo el diseño de la Fase 1b este número era 192 (16 prefijos x 12
    posiciones) repartido en 576 celdas: cada estímulo lo veían los tres
    modelos. Aquí cada par aparece UNA vez, así que las 1.152 conversaciones son
    1.152 observaciones independientes en vez de 192 repetidas.
    """
    r = _ranking()
    pares = [
        (
            (c["topic_id"], c["n_turns"]),
            choose_band_artifact(
                r, c["band"], c["artifact_slot"], f"pfx-{c['topic_id']}-{c['n_turns']}"
            )[0].id,
        )
        for c in plan_phase1d(1)
    ]
    assert len(pares) == CELLS
    assert len(set(pares)) == CELLS, collections.Counter(pares).most_common(3)


def test_los_estimulos_distintos_de_cada_banda_son_288():
    """16 prefijos x 18 artefactos. Es el denominador de la comparación entre
    bandas, y es lo que el diseño viejo dejaba en 16 por posición."""
    r = _ranking()
    por_banda = collections.defaultdict(set)
    for c in plan_phase1d(1):
        art, _ = choose_band_artifact(
            r, c["band"], c["artifact_slot"], f"pfx-{c['topic_id']}-{c['n_turns']}"
        )
        por_banda[c["band"]].add(((c["topic_id"], c["n_turns"]), art.id))
    assert {b: len(v) for b, v in sorted(por_banda.items())} == {
        b: 16 * ARTIFACTS_PER_BAND for b in range(BANDS)
    }


def test_cada_modelo_ve_los_18_artefactos_de_cada_banda():
    """Dentro de una banda, cada modelo tiene que ver los 18 artefactos, no 6.

    Si un modelo viera solo su tercio de la banda, la comparación entre modelos
    mediría en parte qué pegotes le tocaron a cada uno.

    La propiedad la sostienen DOS mecanismos a la vez, y conviene saberlo antes
    de fiarse de este test: el desplazamiento por tema y longitud de
    `plan_phase1d` —que hace que el mismo slot caiga en modelos distintos según
    el prefijo— y el barajado por (prefijo, banda). Comprobado rompiendo cada uno
    por separado: con el otro en pie, este test sigue verde. O sea que es un test
    de la PROPIEDAD, no del mecanismo; quien vigila el barajado es
    `test_el_slot_no_es_el_puesto_dentro_de_la_banda`.
    """
    r = _ranking()
    vistos = collections.defaultdict(set)
    for c in plan_phase1d(1):
        art, _ = choose_band_artifact(
            r, c["band"], c["artifact_slot"], f"pfx-{c['topic_id']}-{c['n_turns']}"
        )
        vistos[(c["model_id"], c["band"])].add(art.id)
    assert {len(v) for v in vistos.values()} == {ARTIFACTS_PER_BAND}


def test_el_slot_no_es_el_puesto_dentro_de_la_banda():
    """Lo que compra el barajado, y es lo único que compra.

    Sin él, `artifact_slot` sería el rango fino dentro del tramo —el puesto 3 de
    la banda 1, el mismo artefacto en los 16 prefijos—, y como el modelo se
    asigna por slot, quedaría asignado por la similaridad de grano fino de forma
    determinista y repetida en todas las bandas. El sesgo es pequeño (2 puestos
    de 18 en la media de rango de cada modelo) pero sistemático.

    El test mira el puesto que cada slot recibe a lo largo de los 16 prefijos:
    con barajado son muchos puestos distintos, sin barajado sería siempre uno.
    Es el test que caza `return ventana[slot]`, que deja verde a todos los demás.

    Va por `choose_band_artifact` y no por `band_order` a propósito: lo que hay
    que cazar es que el muestreo **use** el barajado, y un test contra
    `band_order` a pelo se queda verde con la función bien y el runner
    saltándosela. Comprobado: esa versión del test no notaba la mutación.
    """
    r = _ranking()
    ventana = [art.id for art, _ in band_window(r, 1, BANDS)]
    puestos = collections.defaultdict(set)
    for t_i in range(8):
        for n in LENGTHS:
            for slot in range(ARTIFACTS_PER_BAND):
                art, _ = choose_band_artifact(r, 1, slot, f"pfx-tema-{t_i}-{n}")
                puestos[slot].add(ventana.index(art.id))
    assert len(puestos) == ARTIFACTS_PER_BAND
    # Con 16 prefijos barajando 18 puestos hay colisiones, así que no se exige
    # 16: se exige que el slot NO determine el puesto, ni de lejos.
    assert min(len(v) for v in puestos.values()) > 1, {
        s: sorted(v) for s, v in puestos.items() if len(v) == 1
    }


def test_los_modelos_rotan_celda_a_celda_en_el_orden_de_la_tirada():
    """El plan se corre en su orden, y una tirada dura horas contra un gateway
    compartido: dos celdas seguidas del mismo modelo empiezan a correlacionar el
    modelo con la hora de reloj. Es la misma razón por la que 1a y 1b ponían el
    modelo en round-robin.

    Es lo que sostiene el término de la **banda** en el desplazamiento: sin él,
    las cuatro celdas seguidas que comparten slot —las cuatro bandas— irían al
    mismo modelo. El equilibrio de CELLS_PER_MODEL/96/48 no lo notaría, porque sigue saliendo
    exacto.
    """
    plan = plan_phase1d(1)
    seguidas = [
        (a["conversation_id"], b["conversation_id"])
        for a, b in zip(plan, plan[1:])
        if a["model_id"] == b["model_id"]
    ]
    assert not seguidas, seguidas[:3]


def test_conversation_id_unico_y_el_plan_es_determinista():
    ids = [c["conversation_id"] for c in plan_phase1d(1)]
    assert len(set(ids)) == CELLS
    assert plan_phase1d(9) == plan_phase1d(9)


def test_los_identificadores_no_chocan_con_los_del_barrido_de_1b():
    """Las tandas viven en el mismo repositorio y se reanudan por identificador:
    dos filas con el mismo `conversation_id` y distinto diseño harían que una
    reanudación diera por hecha una celda que corrió el otro muestreo."""
    nuevos = {c["conversation_id"] for c in plan_phase1d(1)}
    for nivel in ("N0", "N1"):
        viejos = {c["conversation_id"] for c in rp1b.plan_phase1b(1, level=nivel)}
        assert not (nuevos & viejos), sorted(nuevos & viejos)[:3]


def test_el_plan_no_lleva_posicion_de_barrido():
    """La posición era el instrumento de 1b. Dejar las dos invitaría a analizar
    por la que no toca."""
    assert "sweep_position" not in plan_phase1d(1)[0]


# --- que la Fase 1b no se haya movido --------------------------------------


def _huella(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_el_plan_de_la_fase_1b_sigue_dando_exactamente_lo_mismo():
    """La Fase 1b está pagada y publicada: su plan no se puede mover ni un bit.

    Las huellas se tomaron antes de escribir una línea de la Fase 1d. Un test que
    solo contara CELLS_PER_BAND celdas no vería un cambio de semillas, de longitudes o de
    identificadores, que es justo lo que rompería la reanudación de una tirada ya
    hecha.
    """
    assert rp1b.MASTER_SEED == 20260915
    assert rp1b.SWEEP_POSITIONS == 12
    assert (
        _huella(rp1b.plan_phase1b(rp1b.MASTER_SEED))
        == "207ecd152e949fc79c985a3024fdc7b9a72092a216817817799791b5881e0623"
    )
    assert (
        _huella(rp1b.plan_phase1b(rp1b.MASTER_SEED, level="N1"))
        == "9ecce98cc646ffa0f59aa2c7a5854200965dc40ace0e18afead76f26aacc4f6f"
    )


# --- el runner -------------------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Las mismas puertas de red que doblan 1a y 1b, reapuntadas a 1d.

    El banco es el de esta suite —72 artefactos, el tamaño real de N1— y no el
    de 32 de la Fase 1a: con 32 no hay 18 artefactos por banda y el diseño que
    se está probando no cabe.
    """
    from tests.test_run_phase1a import (
        FAKE_TOPICS,
        _reset_calls,
        fake_continue_after_paste,
        fake_ensure_prefix,
        fake_inject_paste,
        fake_rank_artifacts,
    )

    _reset_calls()
    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(rp, "ensure_prefix", fake_ensure_prefix)
    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue_after_paste)
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase1d")
    monkeypatch.setattr(rp0, "rank_artifacts", fake_rank_artifacts)
    for modulo in (conv, su, sim):
        monkeypatch.setattr(
            modulo,
            "chat" if modulo is not sim else "embed",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("llamada real a un modelo por un camino sin doblar")
            ),
        )
    return tmp_path


def _rows(path):
    lineas = [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return lineas[0], lineas[1:]


def test_los_dobles_declaran_la_firma_de_la_funcion_real():
    from wrongpaste import artifacts as arts_mod

    assert list(inspect.signature(fake_load_artifacts).parameters) == list(
        inspect.signature(arts_mod.load_artifacts).parameters
    )


def test_el_banco_n1_de_disco_da_para_el_plan_declarado():
    """El tamaño del plan contra el banco **de verdad**, no contra el doble.

    Todo lo demás en esta suite corre sobre `FAKE_BANK_N1`, 72 artefactos
    escritos a mano, así que el día que alguien retire un artefacto de
    `data/artifacts-n1/` la suite se queda verde y el plan deja de caber en el
    banco sin que nadie se entere. Este es el único test que mira el disco, y
    mira las dos cosas que tienen que seguir cuadrando: que el banco dé para las
    4 x 18 del plan, y que el doble siga teniendo el tamaño del banco real —un
    doble más grande que el banco convierte esta suite en una comprobación sobre
    un banco que no existe.
    """
    from wrongpaste.artifacts import load_artifacts as load_real

    n1 = load_real(level="N1")
    assert len(n1) >= BANDS * ARTIFACTS_PER_BAND, (
        f"el banco N1 tiene {len(n1)} artefactos y el plan pide "
        f"{BANDS} x {ARTIFACTS_PER_BAND}"
    )
    assert len(n1) == BANK_N1, (
        "el doble de esta suite ya no tiene el tamaño del banco real: los "
        "recuentos de estímulos que se prueban aquí dejan de ser los de la tanda"
    )


def test_un_banco_que_no_da_para_el_plan_para_la_tirada_antes_de_escribir(
    harness, tmp_path, monkeypatch
):
    """Un banco de 71 no da 18 por banda, y la tirada tiene que NO empezar.

    Es el caso que el docstring de `ARTIFACTS_PER_BAND` prometía y no cumplía.
    Con 71 artefactos las ventanas salen 17/18/18/18, el slot 17 de la banda 0
    no existe y `choose_band_artifact` revienta — pero revienta *dentro* de la
    celda, y D6 convierte eso en una fila fallida, así que la tanda entera se
    pagaba igual y terminaba con 1.136 estímulos de los 1.152 y las 16 pérdidas
    **todas en la banda 0**: un hueco sistemático en un extremo de la variable
    independiente, que es exactamente el daño que `by_band` existe para poder
    contar después. Se puede saber antes de llamar a nadie, porque el banco está
    cargado antes de escribir la cabecera.

    Por eso el test no se conforma con que salte una excepción: comprueba que no
    quedó fichero. Una versión que dejara la cabecera escrita —o que solo
    fallara celda a celda— ya sería la tanda a medio pagar.
    """
    monkeypatch.setattr(rp, "load_artifacts", lambda level=None: _fake_bank(71))
    with pytest.raises(ValueError, match="71"):
        rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    assert not (tmp_path / "t.jsonl").exists(), (
        "la tirada no empieza, así que no deja cabecera: un fichero con "
        "cabecera declara un diseño que no se corrió"
    )


def test_la_tirada_escribe_una_fila_por_celda_con_su_banda(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    header, rows = _rows(path)
    assert header["phase"] == "1d"
    # La procedencia de D5 no es decorativa: es el campo por el que se sabe qué
    # plan se siguió sin reconstruirlo.
    assert header["plan_path"] == PLAN_BANDAS, header["plan_path"]
    assert header["plan_path"] != rp1b.PLAN_PATH_BY_LEVEL["N1"]
    assert header["planned_cells"] == CELLS
    assert header["bands"] == BANDS
    assert header["artifacts_per_band"] == ARTIFACTS_PER_BAND
    assert "sweep_positions" not in header, "1d no barre posiciones: barre bandas"
    assert len(rows) == CELLS
    assert all(r["paste_level"] == "N1" for r in rows)
    # La banda viaja en la fila como el estrato por rango que es, con su
    # denominador al lado: sin `n_strata`, un 3 no distingue 4 bandas de 8.
    assert {r["stratum"] for r in rows} == set(range(BANDS))
    assert {r["n_strata"] for r in rows} == {BANDS}
    assert all(r["sweep_position"] is None for r in rows), (
        "esta tanda no barrió posiciones: escribir una diría que sí"
    )


def test_las_filas_no_repiten_ningun_estimulo(harness, tmp_path):
    """**El test que justifica el rediseño**, ahora de punta a punta.

    1.152 filas y 1.152 pares (prefijo, artefacto) distintos: cada estímulo lo
    ve un solo modelo y ningún prefijo repite pegote. El test anterior lo
    comprueba sobre el plan; este lo comprueba sobre lo que el runner escribe de
    verdad, que es donde se vería que `run_cell` eligió el pegote por su cuenta.
    """
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    pares = [(r["prefix_id"], r["artifact_id"]) for r in rows]
    assert len(pares) == CELLS
    assert len(set(pares)) == CELLS, collections.Counter(pares).most_common(3)
    # Y el corolario: nadie ve el mismo estímulo dos veces con dos modelos.
    modelos = collections.defaultdict(set)
    for r in rows:
        modelos[(r["prefix_id"], r["artifact_id"])].add(r["model_id"])
    assert {len(v) for v in modelos.values()} == {1}


def test_los_estimulos_por_banda_de_la_tirada_son_288(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    por_banda = collections.defaultdict(set)
    for r in rows:
        por_banda[r["stratum"]].add((r["prefix_id"], r["artifact_id"]))
    assert {b: len(v) for b, v in sorted(por_banda.items())} == {
        b: 16 * ARTIFACTS_PER_BAND for b in range(BANDS)
    }


def test_la_similaridad_media_crece_con_la_banda(harness, tmp_path):
    """Es la comprobación que dice que el eje existe. Si la similaridad media no
    creciera con la banda, las bandas estarían midiendo otra cosa."""
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    medias = []
    for b in range(BANDS):
        sims = [r["similarity_user"] for r in rows if r["stratum"] == b]
        medias.append(sum(sims) / len(sims))
    assert medias == sorted(medias), medias
    assert medias[-1] > medias[0]


def test_el_artefacto_de_la_fila_cae_dentro_de_su_banda(harness, tmp_path):
    """La fila guarda el ranking entero, así que el puesto del artefacto se puede
    recomprobar contra la ventana de su banda sin creerse nada."""
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    for r in rows:
        n = len(r["ranking"])
        lo, hi = n * r["stratum"] // BANDS, n * (r["stratum"] + 1) // BANDS
        assert lo <= r["similarity_rank"] < hi, r["conversation_id"]


def test_una_celda_rota_no_tumba_la_tirada(harness, tmp_path, monkeypatch):
    from tests.test_run_phase1a import fake_inject_paste

    def revienta(
        model_id,
        transcript,
        artifact,
        request_params_out=None,
        max_tokens=conv.MAX_TOKENS,
    ):
        if artifact.id.endswith("3"):
            raise RuntimeError("se cayó el gateway")
        return fake_inject_paste(
            model_id,
            transcript,
            artifact,
            request_params_out=request_params_out,
            max_tokens=max_tokens,
        )

    monkeypatch.setattr(rp, "inject_paste", revienta)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    assert len(rows) == CELLS, "una celda rota no se descarta en silencio"
    rotas = [r for r in rows if r["status"] != "ok"]
    assert rotas
    for r in rotas:
        assert r["status"] == "http_error"
        # La banda y el nivel viajan aunque la celda falle: son datos del
        # diseño, no resultados, y sin ellos no se puede saber si las pérdidas
        # se concentran en un extremo del eje.
        assert r["stratum"] in range(BANDS)
        assert r["n_strata"] == BANDS
        assert r["paste_level"] == "N1"


def test_la_tirada_se_reanuda_sin_duplicar_filas(harness, tmp_path):
    path = tmp_path / "t.jsonl"
    rp.main(seed=1, out=path, measure=False)
    header, quedan = _rows(path)
    path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *quedan[:400]])
        + "\n",
        encoding="utf-8",
    )
    rp.main(seed=1, out=path, measure=False)
    _, rows = _rows(path)
    assert len(rows) == CELLS
    assert len({r["conversation_id"] for r in rows}) == CELLS
    # Y reanudar no cambia el pegote de ninguna celda: el muestreo depende de
    # (prefijo, banda, slot) y de nada más.
    pares = {(r["prefix_id"], r["artifact_id"]) for r in rows}
    assert len(pares) == CELLS


def test_la_puerta_del_eje_para_la_tirada_si_el_eje_es_estrecho(
    harness, tmp_path, monkeypatch
):
    """La misma puerta de D12 que la Fase 1b, y no una copia: con un eje
    estrecho, partirlo en cuatro bandas es todavía más decorativo que barrerlo
    en doce puestos."""
    from tests.test_run_phase1b import fake_measure_axis_estrecho

    monkeypatch.setattr(rp1b, "measure_axis", fake_measure_axis_estrecho)
    # `acknowledge_underpowered` va puesto para que la puerta de POTENCIA no se
    # dispare antes y tape la que este test mide. Son dos puertas distintas y el
    # orden importa: la de potencia mira el diseño y la del eje mira los datos.
    with pytest.raises(rp0.NarrowAxisError):
        rp.main(
            seed=1,
            out=tmp_path / "t.jsonl",
            measure=True,
            acknowledge_underpowered=True,
        )
    assert not (tmp_path / "t.jsonl").exists(), "no se escribe nada si no se corre"
    assert rp0.axis_path("t", out=tmp_path / "t.jsonl").exists()


def test_no_se_reanuda_un_fichero_del_barrido_viejo(harness, tmp_path):
    """Reanudar una tirada de la Fase 1d **anterior** —la de doce posiciones—
    tiene que parar, y no es paranoia: las dos declaran `phase: "1d"` y
    `levels: ["N1"]`, así que el control de nivel de 1b las da por compatibles.

    Lo que pasaría sin esto es el fallo silencioso de siempre: ninguno de los
    CELLS_PER_BAND identificadores `...-s05` del fichero viejo está entre los 1.152
    `...-b1-a07` del plan nuevo, así que no se saltaría ni una celda, se
    volvería a pagar la tanda entera y quedarían 1.440 filas de dos diseños
    distintos bajo una cabecera que dice `sweep_positions: 12`. El resumen solo
    diría «saltadas: 0».
    """
    path = tmp_path / "vieja.jsonl"
    header = rp1b.build_header(
        "vieja", 1, rp1b.plan_phase1b(1, level="N1"), [], [], 0.0, level="N1"
    )
    path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="posiciones"):
        rp.main(seed=1, out=path, measure=False)



# --- la procedencia que declara la cabecera (D5) ---------------------------
#
# El campo `plan_path` existe para que quien abra el JSONL dentro de seis meses
# no tenga que reconstruir el diseño: abre el documento y lo lee. Eso solo vale
# si el documento describe ESTA tanda, y aquí hay una trampa concreta: el plan
# «Fase 1d» de 1b (`PLAN_PATH_BY_LEVEL["N1"]`) declara el mismo brazo y la misma
# fase, pero describe el barrido de doce posiciones —CELLS_PER_BAND celdas, banco de 44,
# «no hay runner nuevo», los tres modelos viendo cada pegote—, que es justo el
# diseño que este módulo NO corre.

PLAN_BANDAS = "docs/superpowers/plans/2026-09-16-pegado-accidental-fase-1d-bandas.md"

# Con qué se reconoce el repo del blog: el plan de la Fase 1b, que está
# pagada y publicada, así que de ahí no se va a mover.
ANCLA_DEL_REPO_DE_PLANES = (
    "docs/superpowers/plans/2026-09-15-pegado-accidental-fase-1b.md"
)


def _texto_del_plan(rel_path: str) -> str:
    """El documento que hay detrás de un `plan_path`, si está a mano.

    Los planes viven en el repo del blog, no en este, así que la comprobación de
    contenido solo puede correr donde ese repo esté al lado: se busca por
    `WRONGPASTE_PLANS_ROOT` y, si no, entre los hermanos del repo. Si no aparece
    se salta, porque un test que se inventa el veredicto cuando no puede mirar es
    peor que no tenerlo — y el test hermano, el que compara las rutas, no
    necesita el documento para correr.
    """
    raiz = pathlib.Path(__file__).resolve().parents[1]
    candidatos = []
    env = os.environ.get("WRONGPASTE_PLANS_ROOT")
    if env:
        candidatos.append(pathlib.Path(env))
    candidatos.append(raiz.parent / "personal-website")
    # El repo de planes se reconoce por un documento que existe seguro —el plan
    # de la Fase 1b, que está pagada y publicada— y NO por el documento que se
    # va a leer. Si el ancla fuera el propio `rel_path`, declarar una ruta que no
    # lleva a ningún documento SALTARÍA el test en vez de suspenderlo, y una
    # procedencia colgando del vacío pasaría tan verde como una correcta.
    for base in candidatos:
        if not (base / ANCLA_DEL_REPO_DE_PLANES).exists():
            continue
        doc = base / rel_path
        assert doc.exists(), (
            f"{rel_path} no existe en {base}: la cabecera estaría declarando "
            "una procedencia que no lleva a ningún documento"
        )
        return doc.read_text(encoding="utf-8")
    pytest.skip(
        "no encuentro el repo de los planes: viven en el repo del blog. Apunta "
        "WRONGPASTE_PLANS_ROOT a su raíz para correr esta comprobación."
    )


def _describe_el_muestreo_por_bandas(texto: str) -> bool:
    """¿Este documento describe la tanda que corre `run_phase1d`?

    Cuatro marcas del diseño, y las cuatro faltan en el plan del barrido por
    posiciones: la palabra banda, el recuento de celdas, la constante que fija
    cuántos artefactos salen de cada banda y el nombre del runner nuevo —aquel
    plan dice literalmente que no lo hay—.

    **El recuento se DERIVA de `CELLS`, no se clava.** Clavado, este predicado
    estuvo un tiempo exigiendo «1.152» contra un plan que decía «1.152» mientras
    el diseño ya iba por 1.344: los dos lados viejos a la vez, y el test pasando
    sin comprobar nada. Derivado, el día que el diseño cambie de tamaño el test
    exige que el documento cambie con él, que es justo para lo que existe.
    """
    bajo = texto.lower()
    return (
        "banda" in bajo
        and f"{CELLS:,}".replace(",", ".") in texto
        and "artifacts_per_band" in bajo
        and "run_phase1d" in bajo
    )


def test_el_plan_declarado_no_es_el_del_barrido_por_posiciones():
    """La ruta de D5 tiene que ser propia, no heredada de 1b.

    `endswith("-fase-1d.md")` no vale como control: los dos documentos acaban
    igual, y era el único test que había sobre el campo.
    """
    assert rp.PLAN_PATH != rp1b.PLAN_PATH_BY_LEVEL["N1"], (
        "la cabecera declararía el plan del barrido de doce posiciones, que "
        "describe CELLS_PER_BAND celdas sobre un banco de 44 y sin runner nuevo"
    )
    assert rp.PLAN_PATH == PLAN_BANDAS


def test_el_plan_declarado_describe_el_muestreo_por_bandas():
    """Y el documento de esa ruta describe de verdad esta tanda.

    El test se sostiene sobre que el predicado SEPARA los dos documentos: se
    comprueba también que el plan del barrido no lo cumple, porque si lo
    cumpliera —si midiera algo que ambos tienen— pasaría igual con la
    procedencia falsa, que es el fallo que este test viene a cerrar.
    """
    assert _describe_el_muestreo_por_bandas(_texto_del_plan(rp.PLAN_PATH))
    viejo = _texto_del_plan(rp1b.PLAN_PATH_BY_LEVEL["N1"])
    assert not _describe_el_muestreo_por_bandas(viejo), (
        "el predicado no distingue los dos diseños, así que no prueba nada"
    )


# --- la banda tiene que llegar al análisis -----------------------------------
#
# El runner escribe la banda en `stratum` y deja `sweep_position` a `None`. Si la
# familia de hipótesis de la tanda sigue declarando H4 sobre `sweep_position`,
# las 1.152 filas se caen del contraste sin una sola queja: `rate_by_position`
# salta las filas cuyo eje viene a `None`, y H4 —la pregunta por la que se paga
# la tanda— sale z = 0, p = 1 sobre n = 0. Los tests de `test_curve.py` lo miran
# sobre filas sintéticas; éste lo mira sobre las filas que el runner escribe de
# verdad, que es donde se vería que las dos mitades del proyecto dejaron de
# hablar el mismo idioma.


def test_la_familia_de_1d_declara_el_eje_que_el_runner_escribe():
    """El eje del análisis y el campo del runner no pueden separarse.

    Sin esto, mover uno de los dos deja el otro apuntando a una columna que ya no
    existe, y el modo de fallo no es una excepción: es un nulo impecable.
    """
    from wrongpaste.curve import (
        HYPOTHESIS_FAMILIES,
        PRIMARY_BAND_LEVELS,
        _como_hipotesis,
    )

    h4 = _como_hipotesis(HYPOTHESIS_FAMILIES["1d"]["H4 contempla el error"])
    assert h4.axis == "stratum", "es el campo en el que `run_cell` escribe la banda"
    # Y los niveles son los del CONTRASTE —dos bandas—, que no tienen por qué
    # ser los del muestreo. Lo que ata la familia a ESTE runner es `fold_from`:
    # si alguien sube `BANDS` a 6, el pliegue lo sigue y ninguna banda se queda
    # fuera. Clavar aquí `range(BANDS)` sería exigir que el contraste tuviera la
    # resolución del muestreo, que es justo lo que `PRIMARY_BANDS` decide que no.
    assert h4.levels == PRIMARY_BAND_LEVELS
    assert h4.fold_from == BANDS, (
        "sin esto, un `levels` de (0, 1) no agrupa las bandas 2 y 3: las tira, "
        "y el contraste primario se queda con media tanda"
    )


def test_el_analisis_de_1d_cuenta_las_1152_filas_que_escribe_el_runner(
    harness, tmp_path
):
    """De punta a punta: tirada real, veredictos encima, familia declarada.

    Las categorías se ponen aquí porque el juez es otro paso y no se llama sin
    red; lo que importa es que las FILAS son las del runner, con su `stratum` y
    su `sweep_position` a `None`. La caída va metida en la banda y es enorme, así
    que si H4 mirase el eje viejo no saldría discreta: saldría n = 0.
    """
    from wrongpaste.curve import (
        HYPOTHESIS_FAMILIES,
        PRIMARY_BANDS,
        primary_family_report,
    )

    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    assert len(rows) == CELLS
    assert all(r["sweep_position"] is None for r in rows), "el runner no barre"

    # G cae con la banda, de forma determinista: 75 % en la 0 y 25 % en la 3.
    g_por_banda = {0: 3, 1: 2, 2: 1, 3: 0}
    for i, r in enumerate(rows):
        r["judge_category"] = "G" if i % 4 <= g_por_banda[r["stratum"]] else "B"

    rep = primary_family_report(rows, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    assert h4["axis"] == "stratum"
    assert sum(c["n"] for c in h4["pooled"]["curve"]) == CELLS, (
        "las 1.152 conversaciones que se pagan tienen que entrar en el contraste"
    )
    # Dos puntos, no cuatro: H4 contrasta las bandas plegadas de dos en dos
    # (0+1 contra 2+3), así que las CELLS_PER_BAND de cada banda muestreada se suman de par
    # en par. Que la curva tuviera cuatro puntos aquí querría decir que el
    # análisis no está plegando y que las 1.152 se reparten a 288.
    assert [c["n"] for c in h4["pooled"]["curve"]] == [CELLS_PER_PRIMARY_BAND] * PRIMARY_BANDS
    for modelo in PHASE1D_MODELS:
        t = h4["by"][modelo]["trend"]
        assert t["n"] == CELLS_PER_MODEL, "1.152 celdas entre 3 modelos"
        assert t["slope_sign"] == -1 and t["p"] < 1e-9
    # Las seis pruebas de la familia, todas sobre filas de verdad.
    assert rep["family_size"] == 6
    for nombre, inf in rep["hypotheses"].items():
        for modelo, sub in inf["by"].items():
            assert sub["trend"]["n"] > 0, (nombre, modelo)


# --- el espejo: que H4 y H5 no acaben leyendo el mismo eje -------------------
#
# El test de arriba planta la caída en la BANDA y comprueba que el informe la
# ve. Con ese solo, un análisis que hubiera acabado con las DOS hipótesis sobre
# `stratum` —el copia y pega más probable el día que se corrigió el eje de H4—
# pasaría igual: H5 tendría filas, `n > 0`, y ninguna afirmación que la
# contradijera. Éste pone la señal en la LONGITUD sobre las mismas 1.152 filas
# del runner y exige lo contrario de cada hipótesis: H4 exactamente plana y H5
# significativa. Los dos juntos son lo que ata cada hipótesis a SU columna, y no
# solo a una columna que exista.


def _con_veredicto_por_longitud(rows, g_por_longitud):
    """Le pega a cada fila la categoría que habría puesto el juez.

    La cuota se reparte por (modelo, banda, longitud), que en esta tanda son
    grupos de `CELLS_PER_BAND_MODEL_LENGTH` filas exactas: con la misma cuota en
    las cuatro bandas, la tasa
    de cada banda sale idéntica y H4 tiene que dar z = 0 **exactamente**, no
    «pequeña». Una tolerancia sobre una señal que no se plantó dejaría pasar
    justo el caso que este test busca.

    Las categorías se ponen aquí y no salen del juez porque clasificar es otro
    paso y no se llama a ningún modelo en los tests; lo que importa es que las
    FILAS sean las que el runner escribe, con su `stratum` y su
    `sweep_position` a `None`.
    """
    grupos = collections.defaultdict(list)
    for r in rows:
        grupos[(r["model_id"], r["stratum"], r["n_turns"])].append(r)
    for (_, _, n_turns), filas in grupos.items():
        # El orden de la tirada no reparte las longitudes dentro del grupo, así
        # que se fija por identificador: el veredicto plantado tiene que ser el
        # mismo en cada ejecución o el test mediría la suerte.
        filas.sort(key=lambda r: r["conversation_id"])
        for i, r in enumerate(filas):
            r[CATEGORY_FIELD] = "G" if i < g_por_longitud[n_turns] else "B"
    return rows


def test_el_analisis_ve_la_senal_de_longitud_cuando_esta_en_la_longitud(
    harness, tmp_path
):
    """H5 sobre las filas del runner, y H4 plana sobre esas mismas filas."""
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    # Un cuarto de G en las cortas y tres cuartos en las largas: 25 % contra
    # 75 %. La cuota se planta por (modelo, banda, longitud), así que lo que hay
    # que partir en cuartos es el tamaño de ESE grupo. Derivado y no clavado
    # porque el grupo encoge y crece con `ARTIFACTS_PER_BAND`: con un 12 escrito
    # a mano la señal plantada dejaría de ser el 25 % que el test dice medir en
    # cuanto el diseño cambiara de tamaño, y las k de abajo saldrían torcidas.
    cuarto = CELLS_PER_BAND_MODEL_LENGTH // 4
    filas = _con_veredicto_por_longitud(rows, {2: cuarto, 10: 3 * cuarto})

    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    h5 = rep["hypotheses"]["H5 contempla el error en las largas"]
    assert h5["axis"] == "n_turns" and h5["levels"] == [2, 10]
    for modelo in PHASE1D_MODELS:
        # H4 sigue midiendo sus CELLS_PER_MODEL filas —no es que se haya quedado sin
        # denominador— y sale plana porque en la banda no se plantó nada.
        assert h4["by"][modelo]["trend"]["n"] == CELLS_PER_MODEL
        assert h4["by"][modelo]["trend"]["z"] == pytest.approx(0.0, abs=1e-9)
        largo = h5["by"][modelo]["trend"]
        assert largo["n"] == CELLS_PER_MODEL and largo["slope_sign"] == 1
        assert largo["significant_holm"] is True
    # Un cuarto en las cortas y tres cuartos en las largas: los dos puntos de la
    # curva de H5 salen de las CELLS filas de la tanda y de ninguna otra parte.
    assert [c["n"] for c in h5["pooled"]["curve"]] == [CELLS_PER_PRIMARY_BAND] * 2
    assert [c["k"] for c in h5["pooled"]["curve"]] == [
        CELLS_PER_PRIMARY_BAND // 4,
        3 * CELLS_PER_PRIMARY_BAND // 4,
    ]


# --- la potencia del diseño -------------------------------------------------
#
# El rediseño entero existe por un número —la potencia del 16 % de la Fase 1b— y
# hasta aquí ese número solo vivía en la prosa del módulo. Estos tests lo ponen
# en código: cuánta potencia tiene ESTE plan, medida con el estimador del propio
# repositorio y con las tasas de G medidas en la Fase 1a, y qué hace el runner
# cuando la respuesta es «no la suficiente».


def test_la_potencia_cuenta_TODAS_las_celdas_del_plan_de_cada_hipotesis():
    """Ninguna celda pagada puede caerse de la cuenta que autoriza el gasto.

    Es el invariante, no un número: sea cual sea el eje y sean cuantos sean los
    niveles que el contraste declare, la potencia de una hipótesis tiene que
    estar calculada sobre las celdas que la tanda va a comprar — todas.

    Existe porque el modo de fallo no se ve. `H4` se contrasta sobre DOS bandas
    (`PRIMARY_BAND_LEVELS`) mientras el plan escribe las CUATRO del muestreo, y
    contar `celda["band"] == nivel` contra un `levels` de (0, 1) se queda con las
    bandas 0 y 1 y tira las otras dos: el informe sale bien formado, con un
    número plausible, y es la potencia de media tanda. `curve.fold_band` pliega
    las FILAS en el análisis; quien cuente celdas del PLAN tiene que plegar
    igual, y eso es lo que esto vigila.

    Se comprueba por modelo y no solo en total para que un pliegue que agrupara
    mal —juntando las cuatro bandas en un nivel, por ejemplo— tampoco pasara: la
    suma seguiría cuadrando, pero el reparto por nivel no.
    """
    plan = plan_phase1d(1)
    rep = rp.design_power(plan)
    celdas_por_modelo = collections.Counter(c["model_id"] for c in plan)

    for nombre, por_modelo in rep["tests"].items():
        for modelo, inf in por_modelo.items():
            assert sum(n for _, n in inf["counts"]) == celdas_por_modelo[modelo], (
                f"{nombre} / {modelo}: la potencia se calculó sobre "
                f"{sum(n for _, n in inf['counts'])} celdas de las "
                f"{celdas_por_modelo[modelo]} que este modelo paga"
            )
            # Y repartidas a partes iguales: el diseño es equilibrado por
            # construcción, así que un reparto desigual delata un pliegue que
            # mandó bandas contiguas al nivel que no era.
            assert len({n for _, n in inf["counts"]}) == 1, (nombre, modelo)


def test_la_potencia_de_h4_es_la_de_las_dos_bandas_que_contrasta():
    """El número que la puerta usa es el del contraste declarado, no otro.

    H4 muestrea cuatro bandas y contrasta dos. Hay tres números en danza y solo
    uno es el bueno (tirada de un solo modelo, alfa de una familia de dos):

    - sobre las cuatro bandas del muestreo (CELLS_PER_BAND por punto) la
      potencia es 0,59;
    - plegando a las dos que el contraste declara (CELLS_PER_PRIMARY_BAND por
      punto) es **0,859**, que es el motivo por el que el pliegue existe;
    - y contando `band == nivel` sin plegar salen 0,54 — la potencia de media
      tanda, que no es ninguno de los dos y es el que salía antes.

    Los tres subieron con el diseño: con 18 artefactos por banda el plegado daba
    0,797 y se quedaba corto, y es justo por eso por lo que `ARTIFACTS_PER_BAND`
    se subió a 21. Ahora **cruza** el MIN_POWER de 0,80, así que lo que este test
    ancla ya no es «sigue sin llegar» sino «llega»: `underpowered` a False y un
    nulo de esta tanda que sí se puede leer como ausencia.

    Se ancla contra H5 a propósito: H5 va sobre `n_turns`, cuyos niveles (2, 10)
    son los valores que el plan ya escribe, así que no necesita pliegue ninguno.
    Con la misma tasa base, el mismo alfa y el mismo tamaño por punto las dos
    potencias tienen que salir IGUALES. Es la comprobación que no depende de que
    yo haya copiado bien un decimal.
    """
    plan = plan_phase1d(1, models=["claude-opus-5"])
    rep = rp.design_power(plan)
    h4 = rep["tests"]["H4 contempla el error"]["claude-opus-5"]
    h5 = rep["tests"]["H5 contempla el error en las largas"]["claude-opus-5"]

    assert h4["counts"] == [
        [0, CELLS_PER_PRIMARY_BAND],
        [1, CELLS_PER_PRIMARY_BAND],
    ], "las celdas del plan, plegadas a dos"
    assert h5["counts"] == [
        [2, CELLS_PER_PRIMARY_BAND],
        [10, CELLS_PER_PRIMARY_BAND],
    ]
    assert h4["power"] == pytest.approx(POTENCIA_H4_UN_MODELO, abs=5e-3)
    assert h4["power"] == pytest.approx(h5["power"], abs=1e-9), (
        "mismo tamaño, misma tasa y mismo alfa: si no coinciden, una de las dos "
        "no está contando las celdas que dice"
    )
    # Y ahora SÍ pasa del 0,80 declarado, que es para lo que el diseño creció de
    # 18 artefactos por banda a 21. Se ancla igual de fuerte que cuando no
    # llegaba: si alguien encoge el diseño, la puerta vuelve a saltar y este test
    # lo dice antes de que la tanda se pague creyendo que decide algo.
    assert h4["underpowered"] is False
    assert rep["null_would_be_informative"] is True


def test_la_potencia_del_diseno_se_calcula_desde_el_plan_y_no_se_supone():
    """Las seis pruebas de la familia declarada, con su potencia, antes de gastar.

    Los números son el resultado, no la promesa: repartiendo las CELLS celdas
    entre los TRES modelos quedan CELLS_PER_MODEL // 2 por (modelo, banda
    plegada) y, con las tasas base de N1 de la Fase 1a, la caída de 9 puntos que
    la puerta declara relevante se vería el 23 % de las veces en Opus y el 74 %
    en `sol`. Van clavados —no como desigualdad— porque son lo que decide si esta
    tanda se puede leer: una versión que devolviera «0,8 y pico» para todo
    pasaría cualquier `>` y volvería a certificar lo que este módulo vino a
    arreglar.

    Que ninguno llegue al 0,80 con el plantel de tres no contradice al test de
    arriba, donde Opus solo sí llega: ahí la familia es de dos pruebas y Opus se
    queda la tanda entera; aquí es de seis —el alfa de Holm se divide entre tres
    modelos más— y cada modelo se lleva un tercio de las celdas. Es la razón por
    la que la tanda que se paga corre un solo modelo.

    H4 y H5 dan el mismo número sobre este plan y no es casualidad ni copia:
    plegadas las cuatro bandas a dos, los dos ejes parten las mismas celdas de
    cada modelo por la mitad, y a igual denominador y misma tasa base la potencia
    es la misma. Lo que las separa es el eje, no el tamaño.
    """
    rep = rp.design_power(plan_phase1d())

    assert rep["family_size"] == 6, "2 hipótesis declaradas x 3 modelos"
    assert rep["alpha_effective"] == pytest.approx(0.05 / 6)
    assert rep["min_power"] == 0.80 and rep["declared_drop"] == pytest.approx(0.09)

    h4 = rep["tests"]["H4 contempla el error"]
    h5 = rep["tests"]["H5 contempla el error en las largas"]
    assert h4["claude-opus-5"]["power"] == pytest.approx(POTENCIA_H4_TRES_MODELOS, abs=5e-3)
    assert h4["gpt-5.6-sol-tst"]["power"] == pytest.approx(
        POTENCIA_H4_SOL_TRES_MODELOS, abs=5e-3
    )
    assert h4["gpt-5.6-luna-tst"]["power"] == 0.0
    assert h5["claude-opus-5"]["power"] == pytest.approx(POTENCIA_H4_TRES_MODELOS, abs=5e-3)
    assert h5["gpt-5.6-sol-tst"]["power"] == pytest.approx(
        POTENCIA_H4_SOL_TRES_MODELOS, abs=5e-3
    )
    assert h5["gpt-5.6-luna-tst"]["power"] == 0.0

    # El denominador de cada prueba, que es lo que hace comparables los números.
    # H4 va sobre las DOS bandas del contraste, no sobre las cuatro del
    # muestreo: las CELLS_PER_MODEL celdas de un modelo plegadas de dos en dos
    # son CELLS_PER_MODEL // 2 por punto. Que aquí pusiera cuatro niveles de
    # CELLS_PER_BAND_MODEL era la cuenta sin plegar, y con ella la puerta juzgaba
    # media tanda.
    assert h4["claude-opus-5"]["counts"] == [
        [0, CELLS_PER_MODEL // 2],
        [1, CELLS_PER_MODEL // 2],
    ]
    assert h5["claude-opus-5"]["counts"] == [
        [2, CELLS_PER_MODEL // 2],
        [10, CELLS_PER_MODEL // 2],
    ]
    assert sum(n for _, n in h4["claude-opus-5"]["counts"]) == sum(
        n for _, n in h5["claude-opus-5"]["counts"]
    ), "los dos contrastes se pagan con las MISMAS conversaciones"

    assert rep["null_would_be_informative"] is False, (
        "con estas potencias un nulo de esta tanda vuelve a ser «no lo hemos "
        "podido ver», que es justo lo que el rediseño venía a evitar"
    )
    assert rep["underpowered_tests"], "y el informe tiene que decir cuáles"


def test_la_potencia_sale_del_plan_y_no_de_una_tabla_escrita_a_mano():
    """El mismo diseño con más celdas da más potencia. Con una tabla no.

    Es la comprobación que separa «calcular» de «escribir el resultado»: una
    `design_power` que devolviera los seis números de arriba cableados pasaría
    el test anterior entero. Aquí el plan se multiplica por ocho —misma forma,
    mismos modelos, mismas bandas— y la potencia de Opus tiene que cruzar el
    0,80 que el proyecto declara; y se divide, y tiene que bajar.
    """
    plan = plan_phase1d()
    grande = [dict(celda) for _ in range(8) for celda in plan]
    pequeno = [celda for celda in plan if celda["artifact_slot"] < 3]

    base = rp.design_power(plan)["tests"]["H4 contempla el error"]["claude-opus-5"]
    mas = rp.design_power(grande)["tests"]["H4 contempla el error"]["claude-opus-5"]
    menos = rp.design_power(pequeno)["tests"]["H4 contempla el error"]["claude-opus-5"]

    assert menos["power"] < base["power"] < mas["power"]
    assert mas["power"] > 0.80, "ocho veces la tanda sí vería la caída declarada"
    # Ocho veces las CELLS_PER_MODEL celdas de Opus, plegadas a los dos puntos
    # del contraste: 8 * CELLS_PER_MODEL / 2 por punto.
    assert mas["counts"] == [
        [0, 4 * CELLS_PER_MODEL],
        [1, 4 * CELLS_PER_MODEL],
    ]


def test_la_potencia_de_h4_cuenta_las_celdas_plegadas_igual_que_el_analisis():
    """La puerta y el análisis tienen que contar la MISMA tanda.

    H4 se declara con `levels=(0, 1)` y `fold_from=BANDS`: la celda trae una de
    las cuatro bandas del muestreo y el contraste va sobre dos, agrupando. El
    análisis pliega (`curve.rate_by_position`), pero `design_power` contaba
    `celda["band"] == nivel` a pelo, así que se quedaba con las bandas 0 y 1 y
    tiraba las 576 celdas de las bandas 2 y 3 —la potencia de la puerta salía
    la de media tanda, 0,47 en vez de 0,80—.

    El fallo no se ve mirando el número: 0,47 y 0,80 son los dos plausibles.
    Por eso el test no compara contra una constante escrita a mano sino contra
    lo que el análisis cuenta sobre filas con esas mismas bandas: si los dos
    lados vuelven a separarse, el que se mueva se lleva el test por delante.
    """
    from wrongpaste.curve import (
        HYPOTHESIS_FAMILIES,
        _como_hipotesis,
        rate_by_position,
    )

    plan = plan_phase1d(models=["claude-opus-5"])
    assert len(plan) == CELLS, "la tanda entera, un solo modelo"

    h4 = _como_hipotesis(HYPOTHESIS_FAMILIES["1d"]["H4 contempla el error"])
    counts = rp.design_power(plan)["tests"]["H4 contempla el error"]["claude-opus-5"][
        "counts"
    ]

    # El denominador que cuenta el ANÁLISIS sobre filas con esas mismas bandas.
    # `stratum` es el campo de la fila; `band`, el de la celda del plan.
    filas = [{"stratum": c["band"], "judge_category": "B"} for c in plan]
    esperado = [
        [tasa.position, tasa.n]
        for tasa in rate_by_position(
            filas,
            set(h4.member),
            axis=h4.axis,
            levels=h4.levels,
            fold_from=h4.fold_from,
        )
    ]

    assert counts == esperado, (
        "la puerta cuenta las celdas del plan de otra forma que el análisis "
        "cuenta las filas: una de las dos está mirando media tanda"
    )
    assert counts == [
        [0, CELLS_PER_PRIMARY_BAND],
        [1, CELLS_PER_PRIMARY_BAND],
    ]
    assert sum(n for _, n in counts) == len(plan), (
        "las 1.152 conversaciones que se pagan tienen que entrar en el "
        "contraste que las justifica, no la mitad"
    )


def test_una_tasa_base_de_cero_no_es_falta_de_tamano_y_el_informe_lo_dice():
    """`luna` no se arregla comprando conversaciones: no hay nada que medir.

    0/32 en la Fase 1a. Sobre una tasa de cero, una hipótesis de bajada no está
    refutada —está sin denominador—, así que su potencia es 0 por muchas celdas
    que se compren. Importa que el informe lo distinga: si el 0,00 de `luna` se
    leyera como «falta tanda», alguien multiplicaría el presupuesto esperando
    que subiera.
    """
    grande = [dict(c) for _ in range(8) for c in plan_phase1d()]
    luna = rp.design_power(grande)["tests"]["H4 contempla el error"]["gpt-5.6-luna-tst"]

    assert luna["power"] == 0.0
    assert luna["base_rate"] == 0.0
    assert luna["minimum_detectable_drop"] is None, (
        "sin varianza no hay efecto mínimo detectable, y devolver uno aquí "
        "sería inventarlo"
    )
    assert luna["nothing_to_measure"] is True


def test_las_tasas_base_estan_medidas_en_la_fase_1a_y_no_supuestas():
    """El único test de esta sección que mira el disco, y el que la sostiene.

    La potencia de un diseño binario es casi toda la tasa base: con 0,55 y con
    0,20 salen números distintos para el mismo plan. Si las tasas fueran tres
    constantes plausibles escritas a mano, todo lo demás de esta sección sería
    aritmética correcta sobre una entrada inventada — que es la forma que tiene
    este repositorio de quedarse verde certificando algo falso.

    Se recuentan de los veredictos de la Fase 1a: brazo N1, juez validado
    (`JUDGES[0]`), veredictos `ok`, categoría G.
    """
    from wrongpaste.judging import JUDGES

    verdicts = sorted(
        pathlib.Path(rp.OUT_DIR).resolve().parents[0].glob("phase1a/verdicts-*.jsonl")
    )
    assert verdicts, "sin los veredictos de la Fase 1a no hay tasas que medir"

    medidas: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for linea in verdicts[0].read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        v = json.loads(linea)
        if (
            v["paste_level"] != "N1"
            or v["judge_model"] != JUDGES[0]
            or v["status"] != "ok"
        ):
            continue
        medidas[v["model_id"]][1] += 1
        if v["category"] == "G":
            medidas[v["model_id"]][0] += 1

    assert dict(medidas) == {m: list(kn) for m, kn in rp.N1_G_RATE_BY_MODEL.items()}, (
        "las tasas base declaradas ya no son las que se midieron en la Fase 1a"
    )


def test_la_tirada_no_empieza_sin_reconocer_que_el_diseno_no_veria_la_caida(
    harness, tmp_path, monkeypatch
):
    """1.152 conversaciones no se compran sin haber mirado si deciden algo.

    La puerta va delante de la del eje (D12) y delante de la cabecera: lo que se
    castiga es gastar, y el fallo tiene que ocurrir con el fichero todavía sin
    existir. Un aviso por pantalla no vale — un control que hay que acordarse de
    leer es un control que no se lee.
    """
    monkeypatch.setattr(rp, "check_axis", lambda *a, **k: None)
    with pytest.raises(ValueError, match="potencia"):
        rp.main(seed=1, out=tmp_path / "t.jsonl")
    assert not (tmp_path / "t.jsonl").exists(), (
        "la tirada no empieza, así que no deja cabecera"
    )


def test_con_el_reconocimiento_explicito_la_tirada_corre_y_queda_escrito(
    harness, tmp_path, monkeypatch
):
    """Reconocerlo es una decisión del dueño de la tanda, y se firma en el fichero.

    La puerta no es un veto: el diseño puede correrse sabiendo lo que no va a
    poder concluir. Lo que no puede es correrse sin que el fichero lo diga, que
    es lo que convierte un nulo en «no lo hemos podido ver» seis meses después.
    """
    monkeypatch.setattr(rp, "check_axis", lambda *a, **k: None)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", acknowledge_underpowered=True)
    header, rows = _rows(path)
    assert len(rows) == CELLS
    assert header["underpowered_acknowledged"] is True
    assert header["design_power"]["null_would_be_informative"] is False


def test_la_cabecera_lleva_la_potencia_de_la_tanda_que_declara(harness, tmp_path):
    """La potencia viaja con la tanda, como el presupuesto y el banco.

    Quien abra el JSONL dentro de seis meses tiene que poder leer, sin
    reconstruir el plan, con qué probabilidad esa tanda habría visto la caída
    que su propia puerta declara relevante.
    """
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    header, _ = _rows(path)
    potencia = header["design_power"]
    assert potencia["family_size"] == 6
    assert potencia["alpha_effective"] == pytest.approx(0.05 / 6)
    assert potencia["tests"]["H4 contempla el error"]["claude-opus-5"][
        "power"
    ] == pytest.approx(POTENCIA_H4_TRES_MODELOS, abs=5e-3)
    # Y el registro D5 tiene que llevar el denominador PLEGADO, que es el que el
    # análisis usará dentro de seis meses. Si la cabecera guardara las cuatro
    # bandas sin agrupar, quien audite la tanda leería la potencia de media.
    assert potencia["tests"]["H4 contempla el error"]["claude-opus-5"]["counts"] == [
        [0, CELLS_PER_MODEL // 2],
        [1, CELLS_PER_MODEL // 2],
    ]
    assert header["underpowered_acknowledged"] is False


# --- la tanda partida en dos: la parte 1 no paga los turnos posteriores ------
#
# La decisión, tomada antes de escribir esto: la tirada se parte en dos partes
# que se pagan por separado. La **parte 1** es prefijo + pegote + reacción, que
# es lo único que decide la categoría G — la variable de H4 y H5— y por tanto lo
# único que hay que comprar para responder la pregunta de la fase. La **parte
# 2** son los dos turnos posteriores (D7), que la Fase 2 necesitará para el
# recuento de fuga de entidades y que se generan más adelante retomando la
# transcripción guardada.
#
# Lo que estos tests vigilan es que la parte 1 no sea una tanda mutilada: que la
# fila siga siendo válida y clasificable, que la transcripción conserve sus
# etiquetas —de las que depende que el usuario simulado no vea el pegote cuando
# se retome—, que la cabecera diga cuál de las dos cosas es, y que la parte 2
# sea de verdad posible sin volver a pagar la reacción.


def _part1(tmp_path, **kwargs):
    """Una tirada de parte 1 en `tmp_path`, con los dobles del `harness`."""
    kwargs.setdefault("seed", 1)
    kwargs.setdefault("measure", False)
    kwargs.setdefault("post_turns", 0)
    return rp.main(out=tmp_path / "parte1.jsonl", **kwargs)


def test_con_post_turns_cero_no_se_llama_a_continue_after_paste(harness, tmp_path):
    """El test que justifica la partición: lo que no se corre, no se paga.

    Se cuenta con el doble, no con el presupuesto declarado ni con la forma de
    la fila: un runner que llamara a `continue_after_paste` y luego tirara los
    turnos dejaría filas idénticas a las de la parte 1 y habría gastado las dos
    terceras partes del dinero de la tanda igualmente.
    """
    from tests.test_run_phase1a import CALLS

    path = _part1(tmp_path)
    _, rows = _rows(path)
    assert len(rows) == CELLS
    assert CALLS["post"] == [], (
        f"{len(CALLS['post'])} llamadas a los turnos posteriores en una tanda "
        "que no los compra"
    )
    # Y la reacción sí se paga, una por celda: si no, la parte 1 no mide nada.
    assert len(CALLS["paste"]) == CELLS


def test_la_fila_de_parte_1_termina_en_la_reaccion_y_conserva_las_etiquetas(
    harness, tmp_path
):
    """La transcripción acaba en la reacción, y con sus `tag` (D5).

    Las etiquetas son lo que hace posible la parte 2: `user_visible_history`
    esconde el mensaje `paste` y la reacción a él, que es lo que sostiene el
    brazo «sin reparación» de D7. Una transcripción guardada sin `tag` se
    retomaría igual de bien —el modelo evaluado no las ve— pero el usuario
    simulado leería el pegote entero, y el brazo se convertiría en silencio en
    uno de reparación meses después de haberse pagado.
    """
    path = _part1(tmp_path)
    _, rows = _rows(path)
    for r in rows:
        assert r["post_indices"] == [], r["conversation_id"]
        etiquetas = [m["tag"] for m in r["transcript"]]
        assert etiquetas == ["opening", "assistant", "paste", "assistant"], etiquetas
        assert r["transcript"][r["paste_index"]]["content"] == r["artifact_text"]
        assert r["transcript"][-1]["content"] == r["reaction"]
        visible = conv.user_visible_history(r["transcript"])
        assert [m["tag"] for m in visible] == ["opening", "assistant"]
        assert all(r["artifact_text"] not in m["content"] for m in visible), (
            "el usuario simulado vería el pegote al retomar esta transcripción"
        )


def test_la_fila_de_parte_1_sigue_siendo_valida_y_clasificable(harness, tmp_path):
    """Una fila de parte 1 entra en el paso de clasificación como cualquier otra.

    Es la condición de que la partición no rompa nada: el juez mira la reacción
    al pegote y el texto pegado, y las dos cosas están. Si la fila saliera con
    `status` distinto de `ok` —porque el arnés leyera la ausencia de turnos
    posteriores como un agujero— las 1.152 conversaciones se caerían del
    denominador enteras y en silencio, que es el modo de fallo de esta casa.
    """
    from wrongpaste.run_judging import judgeable

    path = _part1(tmp_path)
    _, rows = _rows(path)
    assert {r["status"] for r in rows} == {"ok"}
    assert len(judgeable(rows)) == CELLS
    for r in rows:
        assert r["reaction"] and r["artifact_text"]
        assert r["similarity_user"] is not None and r["similarity_rank"] is not None
        assert r["stratum"] in range(BANDS) and r["n_strata"] == BANDS
        assert r["paste_level"] == "N1" and r["sweep_position"] is None
        assert r["stop_reasons"] == ["end_turn"], (
            "la parte 1 tiene una sola llamada al modelo evaluado, la del pegote"
        )


def test_la_parte_1_sola_decide_la_categoria_g_de_h4_y_h5(harness, tmp_path):
    """Y por eso se puede pagar sola: la familia declarada sale entera de ella.

    Mismo contraste que el test de punta a punta del protocolo completo, sobre
    filas que no tienen turnos posteriores. Si G dependiera de lo que viene
    después del pegote, la curva se quedaría sin denominador aquí.
    """
    from wrongpaste.curve import (
        HYPOTHESIS_FAMILIES,
        PRIMARY_BANDS,
        primary_family_report,
    )

    path = _part1(tmp_path)
    _, rows = _rows(path)
    g_por_banda = {0: 3, 1: 2, 2: 1, 3: 0}
    for i, r in enumerate(rows):
        r[CATEGORY_FIELD] = "G" if i % 4 <= g_por_banda[r["stratum"]] else "B"

    rep = primary_family_report(rows, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    # Las mismas 1.152 de la tanda completa, plegadas a los dos puntos del
    # contraste: la parte 1 no encoge el denominador de H4.
    assert [c["n"] for c in h4["pooled"]["curve"]] == [CELLS_PER_PRIMARY_BAND] * PRIMARY_BANDS
    for modelo in PHASE1D_MODELS:
        t = h4["by"][modelo]["trend"]
        assert t["n"] == CELLS_PER_MODEL and t["slope_sign"] == -1 and t["p"] < 1e-9


def test_la_cabecera_declara_post_turns_y_el_plantel_realmente_corrido(
    harness, tmp_path
):
    """Sin esto, una tanda de parte 1 y una de protocolo completo son iguales.

    Y no son comparables: la métrica de fuga de la Fase 2 se cuenta sobre los
    turnos posteriores, así que mezclar las dos tandas mete un denominador de
    conversaciones que nunca tuvieron esos turnos. El plantel va por lo mismo:
    esta tirada corre SOLO Opus —es el único modelo que produce G en N1, 55 %
    frente a 9 % y 0 % en la Fase 1a—, y una cabecera que declarase los tres
    diría que los otros dos salieron a cero cuando lo que pasa es que no
    corrieron.
    """
    parte1 = rp.main(
        seed=1,
        out=tmp_path / "parte1.jsonl",
        measure=False,
        post_turns=0,
        models=["claude-opus-5"],
    )
    completa = rp.main(seed=1, out=tmp_path / "completa.jsonl", measure=False)

    h1, filas1 = _rows(parte1)
    h2, filas2 = _rows(completa)

    assert h1["post_turns"] == 0
    assert h1["roster"] == ["claude-opus-5"]
    assert h2["post_turns"] == conv.N_POST_TURNS
    assert h2["roster"] == list(PHASE1D_MODELS)
    assert (h1["post_turns"], h1["roster"]) != (h2["post_turns"], h2["roster"]), (
        "las dos tandas tienen que poder distinguirse por la cabecera"
    )
    # Y el plantel declarado es el que de verdad corrió, no el de la constante.
    assert {r["model_id"] for r in filas1} == set(h1["roster"])
    assert {r["model_id"] for r in filas2} == set(h2["roster"])


def test_el_presupuesto_declarado_es_el_que_la_parte_1_gasta(harness, tmp_path):
    """El presupuesto de la cabecera se cuenta contra las llamadas de verdad.

    `call_budget` cablea los dos turnos de D7, así que una parte 1 heredaría el
    presupuesto del protocolo completo —CELLS x (1 + N_POST_TURNS) llamadas al
    modelo evaluado y CELLS x N_POST_TURNS al usuario simulado— para una tanda
    que hace CELLS y ninguna. El número que se lee antes de pagar es justo el que
    no puede mentir.
    """
    from tests.test_run_phase1a import CALLS, _reset_calls

    path = _part1(tmp_path)
    header, _ = _rows(path)
    presupuesto = header["call_budget"]
    evaluadas = len(CALLS["paste"]) + sum(n for _, n in CALLS["post"])
    assert presupuesto["evaluated_model_calls"] == evaluadas == CELLS
    assert presupuesto["simulated_user_calls"] == 0
    assert presupuesto["post_turns"] == 0

    _reset_calls()
    completa = rp.main(seed=1, out=tmp_path / "completa.jsonl", measure=False)
    header2, _ = _rows(completa)
    evaluadas2 = len(CALLS["paste"]) + sum(n for _, n in CALLS["post"])
    assert header2["call_budget"]["evaluated_model_calls"] == evaluadas2 == CELLS * (1 + conv.N_POST_TURNS)
    # El usuario simulado habla una vez por turno posterior y por celda: derivado
    # de CELLS y no clavado, que es lo que hizo que este test dijera 2.304
    # cuando la tanda ya pedía 2.688.
    assert header2["call_budget"]["simulated_user_calls"] == CELLS * conv.N_POST_TURNS


def test_el_camino_por_defecto_sigue_dando_exactamente_lo_de_antes(harness, tmp_path):
    """Los parámetros nuevos, puestos a su valor por defecto, no mueven nada.

    Se compara fila a fila la tirada por defecto contra la que declara los dos
    parámetros a mano: si `post_turns` o `models` hubieran cambiado el plan, el
    muestreo o el reparto de modelos, las dos tandas dejarían de ser la misma.
    Lo que se ignora es el reloj y el nombre del fichero, que es lo único que
    tiene que cambiar entre dos ejecuciones.
    """
    from tests.test_run_phase1a import CALLS

    a = rp.main(seed=1, out=tmp_path / "a.jsonl", measure=False)
    assert CALLS["post"] == [
        (r["model_id"], conv.N_POST_TURNS) for _, r in enumerate(_rows(a)[1])
    ]
    b = rp.main(
        seed=1,
        out=tmp_path / "b.jsonl",
        measure=False,
        post_turns=conv.N_POST_TURNS,
        models=PHASE1D_MODELS,
    )
    reloj = {"run_id", "started_at", "ended_at", "latency_ms"}

    ha, filas_a = _rows(a)
    hb, filas_b = _rows(b)
    assert len(filas_a) == len(filas_b) == CELLS
    for x, y in zip(filas_a, filas_b):
        assert {k: v for k, v in x.items() if k not in reloj} == {
            k: v for k, v in y.items() if k not in reloj
        }, x["conversation_id"]
    assert {k: v for k, v in ha.items() if k not in reloj | {"path"}} == {
        k: v for k, v in hb.items() if k not in reloj | {"path"}
    }
    # Y los turnos posteriores siguen ahí, que es lo que la parte 1 quita.
    for r in filas_a:
        assert len(r["post_indices"]) == 2 * conv.N_POST_TURNS
        assert [m["tag"] for m in r["transcript"]][-4:] == [
            "post",
            "assistant",
            "post",
            "assistant",
        ]


# --- el plantel es un parámetro, y el reparto tiene que seguir cuadrando -----


def test_con_un_solo_modelo_el_plan_sigue_siendo_el_cruce_completo():
    """Solo Opus, pero el cruce entero: 1.152 celdas y 1.152 estímulos distintos.

    Que el plantel encoja no puede encoger el diseño. El modelo se asigna a una
    celda **después** de que el cruce esté hecho, así que quitar dos modelos
    tiene que dejar las mismas 1.152 celdas con los mismos (prefijo, banda,
    slot) — y por tanto los mismos 1.152 pegotes distintos, que es el recuento
    de estímulos independientes por el que se rediseñó la fase.
    """
    plan = plan_phase1d(1, models=["claude-opus-5"])
    assert len(plan) == CELLS
    assert {c["model_id"] for c in plan} == {"claude-opus-5"}
    assert len({c["conversation_id"] for c in plan}) == CELLS

    r = _ranking()
    pares = [
        (
            (c["topic_id"], c["n_turns"]),
            choose_band_artifact(
                r, c["band"], c["artifact_slot"], f"pfx-{c['topic_id']}-{c['n_turns']}"
            )[0].id,
        )
        for c in plan
    ]
    assert len(set(pares)) == CELLS, collections.Counter(pares).most_common(3)

    # Y son los MISMOS estímulos que con el plantel de tres: lo único que cambia
    # es quién los ve.
    completo = plan_phase1d(1)
    for a, b in zip(plan, completo):
        assert (a["topic_id"], a["n_turns"], a["band"], a["artifact_slot"]) == (
            b["topic_id"],
            b["n_turns"],
            b["band"],
            b["artifact_slot"],
        )


@pytest.mark.parametrize(
    "plantel",
    [
        list(PHASE1D_MODELS),
        ["claude-opus-5"],
        # Un plantel de DOS no entra aquí a propósito: 21 slots no se reparten
        # entre dos, y la guarda lo rechaza. El caso vive en
        # `test_un_plantel_que_no_reparte_la_banda_por_igual_revienta`.
    ],
)
def test_el_reparto_de_modelos_es_exacto_con_cualquier_plantel(plantel):
    """Equilibrado en las tres particiones, sea el plantel de 3, de 1 o de 2.

    Es la misma propiedad que el plan declara para los tres modelos, y la razón
    es la misma: si un modelo cayera más en las bandas altas, el contraste entre
    modelos mediría la banda y el contraste entre bandas mediría el modelo. Con
    un plantel de uno es trivial, pero el test tiene que seguir mirándolo: el
    día que la tanda vuelva a correr dos modelos, el reparto no puede depender
    de que alguien se acuerde de comprobarlo.
    """
    plan = plan_phase1d(1, models=plantel)
    m = len(plantel)
    assert collections.Counter(c["model_id"] for c in plan) == {
        modelo: CELLS // m for modelo in plantel
    }
    por_banda = collections.Counter((c["model_id"], c["band"]) for c in plan)
    assert set(por_banda.values()) == {CELLS_PER_BAND // m}
    assert len(por_banda) == m * BANDS
    por_largo = collections.Counter(
        (c["model_id"], c["band"], c["n_turns"]) for c in plan
    )
    assert set(por_largo.values()) == {CELLS // (BANDS * len(LENGTHS)) // m}
    assert len(por_largo) == m * BANDS * len(LENGTHS)


def test_un_plantel_que_no_reparte_la_banda_por_igual_revienta():
    """Con 2 o 4 modelos, 21 slots no se reparten por igual.

    El desequilibrio es pequeño y por eso es peligroso: el plan seguiría
    teniendo CELLS celdas bien formadas y nadie miraría el recuento. Lo que
    quedaría dentro es el modelo correlacionado con la banda, que son las dos
    preguntas de la tanda a la vez.
    """
    for plantel in (
        [*PHASE1D_MODELS, "claude-opus-5-bis"],
        ["claude-opus-5", "gpt-5.6-sol-tst"],
    ):
        with pytest.raises(ValueError, match=str(ARTIFACTS_PER_BAND)):
            plan_phase1d(1, models=plantel)
    with pytest.raises(ValueError, match="plantel"):
        plan_phase1d(1, models=[])


def test_la_tirada_de_un_solo_modelo_escribe_sus_1152_filas(harness, tmp_path):
    """La tanda que se va a pagar, de punta a punta: Opus, parte 1, 1.152 filas."""
    path = _part1(tmp_path, models=["claude-opus-5"])
    header, rows = _rows(path)
    assert header["planned_cells"] == CELLS and len(rows) == CELLS
    assert {r["model_id"] for r in rows} == {"claude-opus-5"}
    pares = {(r["prefix_id"], r["artifact_id"]) for r in rows}
    assert len(pares) == CELLS
    assert {r["stratum"] for r in rows} == set(range(BANDS))
    assert all(r["post_indices"] == [] for r in rows)


# --- la parte 2: retomar la transcripción sin volver a pagar la reacción -----


def test_resume_post_turns_completa_las_filas_sin_repagar_la_reaccion(
    harness, tmp_path
):
    """Lo que hará la Fase 2, y la razón de que la parte 1 se pueda pagar sola.

    Existe ahora —con sus tests, sin correrse— para que la parte 1 no se compre
    sin saber que la parte 2 es posible. Lo que no puede hacer, y es la mitad
    del valor de la función, es volver a llamar al modelo para la reacción: eso
    ya está pagado y además cambiaría el estímulo que la fila declara.
    """
    from tests.test_run_phase1a import CALLS, _reset_calls

    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    _, antes = _rows(parte1)

    _reset_calls()
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    header, despues = _rows(completa)

    assert CALLS["paste"] == [], "la reacción ya estaba pagada"
    assert CALLS["post"] == [("claude-opus-5", conv.N_POST_TURNS)] * CELLS
    assert len(despues) == CELLS

    por_id = {r["conversation_id"]: r for r in antes}
    for r in despues:
        viejo = por_id[r["conversation_id"]]
        # Lo de la parte 1 se conserva intacto: es lo que ya se pagó.
        assert r["reaction"] == viejo["reaction"]
        assert r["artifact_id"] == viejo["artifact_id"]
        assert r["similarity_user"] == viejo["similarity_user"]
        assert r["stratum"] == viejo["stratum"]
        assert r["transcript"][: len(viejo["transcript"])] == viejo["transcript"]
        # Y lo de la parte 2 aparece, con sus índices y sus etiquetas.
        assert len(r["post_indices"]) == 2 * conv.N_POST_TURNS
        assert [m["tag"] for m in r["transcript"]][-4:] == [
            "post",
            "assistant",
            "post",
            "assistant",
        ]
        assert [r["transcript"][i]["tag"] for i in r["post_indices"]] == [
            "post",
            "assistant",
            "post",
            "assistant",
        ]
        assert r["status"] == "ok"
        assert len(r["stop_reasons"]) == 1 + conv.N_POST_TURNS

    assert header["post_turns"] == conv.N_POST_TURNS
    assert header["roster"] == ["claude-opus-5"]
    assert header["post_turns_run"]["source_run_id"] == "parte1"


def test_resume_post_turns_no_inventa_turnos_sobre_las_filas_rotas(
    harness, tmp_path, monkeypatch
):
    """Una fila que falló en la parte 1 no tiene reacción que continuar.

    Se copia tal cual —las pérdidas por banda son un dato del diseño y no se
    pueden perder por el camino— y no se le gasta ni una llamada: continuar una
    conversación rota produciría turnos posteriores colgando de un agujero, y la
    fila seguiría contando como completa en la Fase 2.
    """
    from tests.test_run_phase1a import CALLS, _reset_calls, fake_inject_paste

    def revienta(
        model_id,
        transcript,
        artifact,
        request_params_out=None,
        max_tokens=conv.MAX_TOKENS,
    ):
        if artifact.id.endswith("3"):
            raise RuntimeError("se cayó el gateway")
        return fake_inject_paste(
            model_id,
            transcript,
            artifact,
            request_params_out=request_params_out,
            max_tokens=max_tokens,
        )

    monkeypatch.setattr(rp, "inject_paste", revienta)
    parte1 = _part1(tmp_path)
    _, antes = _rows(parte1)
    rotas = {r["conversation_id"] for r in antes if r["status"] != "ok"}
    assert rotas, "el doble tenía que romper algunas celdas"

    _reset_calls()
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    _, despues = _rows(completa)

    assert len(despues) == len(antes), "una fila rota no se descarta en silencio"
    assert len(CALLS["post"]) == len(antes) - len(rotas)
    por_id = {r["conversation_id"]: r for r in antes}
    for r in despues:
        if r["conversation_id"] in rotas:
            assert r == por_id[r["conversation_id"]], "se copia tal cual"
            assert r["post_indices"] == []
        else:
            assert len(r["post_indices"]) == 2 * conv.N_POST_TURNS


def test_resume_post_turns_se_niega_sobre_una_tanda_que_ya_los_tiene(
    harness, tmp_path
):
    """Retomar una tirada completa le añadiría DOS turnos más a cada fila.

    La transcripción se puede continuar siempre —esa es la propiedad que hace
    posible la partición—, así que nada revienta: quedarían filas de cuatro
    turnos posteriores bajo una cabecera que declara dos, y el recuento de fuga
    de la Fase 2 saldría sobre el doble de texto en la mitad de las tandas. Lo
    que decide es lo que la cabecera declara (D5), no la forma de las filas.
    """
    completa = rp.main(seed=1, out=tmp_path / "completa.jsonl", measure=False)
    with pytest.raises(ValueError, match="post_turns"):
        rp.resume_post_turns(completa, out=tmp_path / "otra.jsonl")
    assert not (tmp_path / "otra.jsonl").exists()


def test_resume_post_turns_se_reanuda_sin_duplicar_ni_repagar(harness, tmp_path):
    """Los turnos posteriores también cuestan, así que tampoco se pagan dos veces."""
    from tests.test_run_phase1a import CALLS, _reset_calls

    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    completa = tmp_path / "completa.jsonl"
    rp.resume_post_turns(parte1, out=completa)
    header, filas = _rows(completa)
    completa.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *filas[:300]])
        + "\n",
        encoding="utf-8",
    )

    _reset_calls()
    rp.resume_post_turns(parte1, out=completa)
    _, filas2 = _rows(completa)
    assert len(filas2) == CELLS
    assert len({r["conversation_id"] for r in filas2}) == CELLS
    assert len(CALLS["post"]) == CELLS - 300, "las 300 ya hechas no se vuelven a pagar"
    assert all(len(r["post_indices"]) == 2 * conv.N_POST_TURNS for r in filas2)
    # Y el estado de la parte 1 sobrevive a la reanudación, que reescribe el
    # fichero (`compact_resume_file`): sin él, las filas de un fichero reanudado
    # volverían a depender de `status` para entrar en el denominador de G.
    assert all(r["part1_status"] == "ok" for r in filas2)
    assert len(rp.reaction_judgeable(filas2)) == CELLS


def test_no_se_reanuda_una_parte_1_como_si_fuera_el_protocolo_completo(
    harness, tmp_path
):
    """`main` tampoco puede mezclar las dos mitades dentro de un mismo fichero.

    Es el mismo fallo silencioso que el del barrido viejo: los identificadores
    de las celdas no llevan ni el plantel ni los turnos, así que reanudar una
    parte 1 con el protocolo completo saltaría las 1.152 celdas hechas, no
    correría ninguna y dejaría un fichero de filas sin turnos posteriores bajo
    una cabecera que declara dos. Y al revés —otro plantel— no saltaría ninguna
    celda, porque el modelo sí va en el identificador: se pagaría la tanda
    entera otra vez.
    """
    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    with pytest.raises(ValueError, match="post_turns"):
        rp.main(seed=1, out=parte1, measure=False, models=["claude-opus-5"])
    with pytest.raises(ValueError, match="plantel"):
        rp.main(seed=1, out=parte1, measure=False, post_turns=0)


# --- la precedencia del `status`, que ahora aplican dos caminos --------------
#
# `run_cell` y `resume_post_turns` deciden el mismo `status` sobre la misma
# conversación en dos momentos separados por meses. La precedencia vive en una
# función compartida por eso, y estos tests existen porque sin ellos no había
# ninguno: quitarle la rama de `refusal` dejaba la suite entera en verde, con
# las negativas del modelo entrando en el denominador de G como reacciones
# normales.


def _resp(text: str):
    """Algo con `.text`, que es todo lo que mira `_has_empty_response`."""
    import types

    return types.SimpleNamespace(text=text)


def test_la_precedencia_del_status_es_la_de_siempre():
    """Las cuatro reglas, con los empates que las ordenan.

    `refusal` manda sobre todo porque es lo único de aquí que es conducta;
    `empty` sobre `truncated` porque es la afirmación más fuerte que se puede
    hacer de una respuesta; `truncated` sobre `ok` porque una reacción cortada a
    media frase se lee luego como «lo ignoró y siguió», que es justo la
    categoría que el experimento mide.
    """
    assert rp.conversation_status(["end_turn"], [_resp("hola")]) == "ok"
    assert rp.conversation_status(["max_tokens"], [_resp("hola")]) == "truncated"
    assert rp.conversation_status(["end_turn"], [_resp("  ")]) == "empty"
    # Empates:
    assert rp.conversation_status(["max_tokens", "refusal"], [_resp("x")]) == "refusal"
    assert rp.conversation_status(["refusal"], [_resp("")]) == "refusal"
    assert rp.conversation_status(["max_tokens"], [_resp("")]) == "empty"
    # Y mira TODAS las respuestas, no solo la última: la reacción al pegote de
    # una fila de parte 1 sigue contando cuando se le añaden los turnos.
    assert rp.conversation_status(["end_turn", "end_turn"], [_resp(""), _resp("ya")]) == (
        "empty"
    )


def _continue_con_stop(stop_reason: str):
    """Doble de `continue_after_paste` cuyos turnos vuelven con ese `stop_reason`."""
    from tests.test_run_phase1a import _fake_reply
    from wrongpaste.clients import Reply

    def doble(
        model_id,
        transcript,
        topic,
        n_post=conv.N_POST_TURNS,
        request_params_out=None,
        max_tokens=conv.MAX_TOKENS,
        user_max_tokens=su.USER_MAX_TOKENS,
        user_replies_out=None,
    ):
        indices, usages, replies = [], [], []
        for _ in range(n_post):
            if user_replies_out is not None:
                user_replies_out.append(
                    Reply("sigo", {"total_tokens": 2}, {}, stop_reason="stop")
                )
            indices.append(len(transcript))
            transcript.append({"role": "user", "content": "sigo", "tag": "post"})
            indices.append(len(transcript))
            transcript.append({"role": "assistant", "content": "ya", "tag": "assistant"})
            usages.append({"total_tokens": 3})
            replies.append(_fake_reply(model_id, "ya", stop_reason=stop_reason))
        return indices, usages, replies

    return doble


@pytest.mark.parametrize(
    ("stop_reason", "esperado"), [("max_tokens", "truncated"), ("refusal", "refusal")]
)
def test_la_parte_2_no_deja_pasar_por_ok_un_turno_posterior_roto(
    harness, tmp_path, monkeypatch, stop_reason, esperado
):
    """Los turnos que añade la parte 2 pasan por la misma precedencia que los de
    la parte 1: un turno cortado por el tope no es `ok`, y una negativa es una
    negativa aunque llegue meses después de la reacción que sí se pagó."""
    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    monkeypatch.setattr(rp, "continue_after_paste", _continue_con_stop(stop_reason))
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    _, filas = _rows(completa)
    assert {r["status"] for r in filas} == {esperado}
    # Y la fila conserva de dónde viene: la reacción de la parte 1 sigue ahí.
    assert all(r["reaction"] and r["stop_reasons"][0] == "end_turn" for r in filas)


def test_la_parte_2_deja_las_filas_que_revientan_como_estaban(
    harness, tmp_path, monkeypatch
):
    """Una continuación que se cae no puede dejar la transcripción a medias.

    `continue_after_paste` escribe el turno del usuario **antes** de llamar al
    modelo evaluado, así que cuando el gateway se cae el mensaje ya está dentro
    del transcript. Si la fila se guardara así, el reintento —que es todo el
    sentido de que esto se reanude— continuaría encima de un mensaje colgado y
    la conversación acabaría con dos turnos de usuario seguidos y un turno
    posterior de más: una fila que la Fase 2 contaría como buena.
    """
    from tests.test_run_phase1a import fake_continue_after_paste

    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    _, antes = _rows(parte1)
    estado = {"n": 0}

    def revienta(
        model_id,
        transcript,
        topic,
        n_post=conv.N_POST_TURNS,
        request_params_out=None,
        max_tokens=conv.MAX_TOKENS,
        user_max_tokens=su.USER_MAX_TOKENS,
        user_replies_out=None,
    ):
        estado["n"] += 1
        if estado["n"] % 3 == 0:
            transcript.append({"role": "user", "content": "sigo", "tag": "post"})
            raise RuntimeError("se cayó el gateway")
        return fake_continue_after_paste(
            model_id,
            transcript,
            topic,
            n_post=n_post,
            request_params_out=request_params_out,
            max_tokens=max_tokens,
            user_max_tokens=user_max_tokens,
            user_replies_out=user_replies_out,
        )

    completa = tmp_path / "completa.jsonl"
    monkeypatch.setattr(rp, "continue_after_paste", revienta)
    rp.resume_post_turns(parte1, out=completa)
    _, primeras = _rows(completa)

    viejas = {r["conversation_id"]: r for r in antes}
    rotas = [r for r in primeras if r["status"] != "ok"]
    assert len(rotas) == CELLS // 3
    for r in rotas:
        assert r["status"] == "http_error"
        assert r["post_indices"] == []
        assert r["transcript"] == viejas[r["conversation_id"]]["transcript"], (
            "el turno de usuario que se quedó sin respuesta no puede quedarse"
        )
        # Una caída del gateway en la parte 2 tampoco dice nada del estímulo: la
        # reacción al pegote es la misma que el juez clasificó, así que la fila
        # no puede salir del denominador de G por esto.
        assert r["part1_status"] == "ok"

    assert len(rp.reaction_judgeable(primeras)) == CELLS
    resumen = json.loads(
        (completa.parent / f"summary-{completa.stem}.json").read_text(encoding="utf-8")
    )
    # Y las degradaciones se cuentan **también** cuando vienen de una excepción,
    # no solo de un `stop_reason` feo: las dos sacan la fila de `judgeable`.
    assert resumen["degraded"] == CELLS // 3
    assert (
        sum(resumen["by_band"][str(b)]["degraded"] for b in range(BANDS))
        == CELLS // 3
    )

    # Y el reintento las completa bien, que es lo que la restauración compra.
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue_after_paste)
    rp.resume_post_turns(parte1, out=completa)
    _, segundas = _rows(completa)
    assert len(segundas) == CELLS
    assert len({r["conversation_id"] for r in segundas}) == CELLS
    for r in segundas:
        assert r["status"] == "ok", r["conversation_id"]
        assert [m["tag"] for m in r["transcript"]] == [
            "opening",
            "assistant",
            "paste",
            "assistant",
            "post",
            "assistant",
            "post",
            "assistant",
        ], r["conversation_id"]


# --- lo que la parte 2 NO puede quitarle al denominador de G -----------------
#
# La categoría G la decide la reacción al pegote, y la reacción al pegote la
# paga y la clasifica la parte 1. Los turnos posteriores ocurren DESPUÉS de ese
# estímulo, así que nada de lo que pase en ellos puede cambiar qué filas entran
# en el contraste de H4/H5 — y sin embargo `status` es uno solo por fila y baja
# a `truncated` con un turno `post` cortado, que es lo que `rates` y
# `run_judging` miran para juzgar.


def _continue_cortando_una_de_cada(n: int, stop_reason: str = "max_tokens"):
    """Doble de `continue_after_paste` que corta una continuación de cada `n`.

    Con `n=4` las roturas caen **todas en la misma banda**: el plan lleva la
    banda en el bucle más interno, así que las filas salen b0, b1, b2, b3, b0...
    y una de cada cuatro es siempre el mismo tramo del ranking. No es un caso
    rebuscado, es el peor caso realista —un hueco sistemático en un extremo de
    la variable independiente— y es el que hace falta para que el recuento por
    banda tenga algo que decir.
    """
    from tests.test_run_phase1a import _fake_reply
    from wrongpaste.clients import Reply

    estado = {"n": 0}

    def doble(
        model_id,
        transcript,
        topic,
        n_post=conv.N_POST_TURNS,
        request_params_out=None,
        max_tokens=conv.MAX_TOKENS,
        user_max_tokens=su.USER_MAX_TOKENS,
        user_replies_out=None,
    ):
        estado["n"] += 1
        corta = estado["n"] % n == 0
        indices, usages, replies = [], [], []
        for _ in range(n_post):
            if user_replies_out is not None:
                user_replies_out.append(
                    Reply("sigo", {"total_tokens": 2}, {}, stop_reason="stop")
                )
            indices.append(len(transcript))
            transcript.append({"role": "user", "content": "sigo", "tag": "post"})
            indices.append(len(transcript))
            transcript.append({"role": "assistant", "content": "ya", "tag": "assistant"})
            usages.append({"total_tokens": 3})
            replies.append(
                _fake_reply(model_id, "ya", stop_reason=stop_reason if corta else "end_turn")
            )
        return indices, usages, replies

    return doble


def test_la_parte_2_no_retira_del_denominador_de_g_lo_que_la_parte_1_pago(
    harness, tmp_path, monkeypatch
):
    """Una fila juzgable en la parte 1 sigue siéndolo en el fichero completo.

    El caso concreto: una de cada cuatro continuaciones vuelve con
    `max_tokens`. `status` baja a `truncated` —y tiene que bajar, porque la
    conversación entera ya no es material limpio para el recuento de fuga de la
    Fase 2— pero la reacción al pegote es **byte a byte la que el juez ya
    clasificó**. Sin este arreglo, `run_judging.judgeable` pasaba de 1.152 filas
    en la parte 1 a 864 en la completa: CELLS_PER_BAND filas pagadas, juzgadas y retiradas
    del contraste por algo que ocurrió después del estímulo que mide G.
    """
    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    _, antes = _rows(parte1)
    monkeypatch.setattr(rp, "continue_after_paste", _continue_cortando_una_de_cada(4))
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    _, despues = _rows(completa)

    # El escenario es el que se dice que es: la conversación entera sí se
    # degrada, y `judgeable` —que mira `status`— pierde filas por ello.
    degradadas = [r for r in despues if r["status"] != "ok"]
    assert len(degradadas) == CELLS_PER_BAND
    assert {r["status"] for r in degradadas} == {"truncated"}
    assert len(run_judging.judgeable(despues)) == CELLS - CELLS_PER_BAND

    # Y lo que el arreglo compra: el denominador de G del fichero completo es el
    # mismo que el de la parte 1, fila a fila.
    assert {r["conversation_id"] for r in rp.reaction_judgeable(despues)} == {
        r["conversation_id"] for r in run_judging.judgeable(antes)
    }
    assert len(rp.reaction_judgeable(despues)) == CELLS

    # El estímulo de las CELLS_PER_BAND es idéntico al que ya se clasificó: no es que se
    # readmitan filas dudosas, es que su reacción nunca cambió.
    por_id = {r["conversation_id"]: r for r in antes}
    for r in degradadas:
        viejo = por_id[r["conversation_id"]]
        assert r["reaction"] == viejo["reaction"]
        assert r["stop_reasons"][0] == viejo["stop_reasons"][0] == "end_turn"
        assert r["part1_status"] == "ok"


def test_reaction_judgeable_no_readmite_lo_que_la_parte_1_ya_habia_roto(
    harness, tmp_path, monkeypatch
):
    """El denominador de G se hereda de la parte 1, y eso corta en los dos lados.

    Una fila que salió rota de la parte 1 no tiene reacción que clasificar, y la
    parte 2 la copia tal cual sin gastar una llamada. `reaction_judgeable` no
    puede colarla de vuelta por el hecho de que la parte 2 no la haya tocado:
    sin `part1_status` cae a `status`, que en esas filas sigue siendo el suyo.
    """
    from tests.test_run_phase1a import fake_inject_paste

    def revienta(
        model_id, transcript, artifact, request_params_out=None, max_tokens=conv.MAX_TOKENS
    ):
        if artifact.id.endswith("3"):
            raise RuntimeError("se cayó el gateway")
        return fake_inject_paste(
            model_id,
            transcript,
            artifact,
            request_params_out=request_params_out,
            max_tokens=max_tokens,
        )

    monkeypatch.setattr(rp, "inject_paste", revienta)
    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    _, antes = _rows(parte1)
    rotas = {r["conversation_id"] for r in antes if r["status"] != "ok"}
    assert rotas, "el doble tenía que romper algunas celdas"

    from tests.test_run_phase1a import fake_continue_after_paste

    monkeypatch.setattr(rp, "continue_after_paste", fake_continue_after_paste)
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    _, despues = _rows(completa)

    ids = {r["conversation_id"] for r in rp.reaction_judgeable(despues)}
    assert ids == {r["conversation_id"] for r in run_judging.judgeable(antes)}
    assert not (ids & rotas), "una fila sin reacción no vuelve al denominador"


def test_el_resumen_de_la_parte_2_cuenta_por_banda_lo_que_degrada(
    harness, tmp_path, monkeypatch
):
    """Si el hueco cae todo en un extremo del eje, el resumen tiene que decirlo.

    `main` escribe `by_band` justo por esto —las pérdidas por banda dicen si el
    hueco es sistemático o aleatorio— y la parte 2 no lo hacía: su resumen solo
    tenía totales, así que CELLS_PER_BAND degradaciones apiladas en una sola banda salían
    del fichero sin que nada las contara.
    """
    parte1 = _part1(tmp_path, models=["claude-opus-5"])
    monkeypatch.setattr(rp, "continue_after_paste", _continue_cortando_una_de_cada(4))
    completa = rp.resume_post_turns(parte1, out=tmp_path / "completa.jsonl")
    _, despues = _rows(completa)

    resumen = json.loads(
        (completa.parent / f"summary-{completa.stem}.json").read_text(encoding="utf-8")
    )
    por_banda = resumen["by_band"]
    assert sum(por_banda[str(b)]["degraded"] for b in range(BANDS)) == CELLS_PER_BAND
    # Y están todas en la misma banda, que es el caso que hace daño.
    con_hueco = [b for b in range(BANDS) if por_banda[str(b)]["degraded"]]
    assert len(con_hueco) == 1
    assert por_banda[str(con_hueco[0])]["degraded"] == CELLS_PER_BAND
    # El recuento del resumen es el del fichero, no un contador aparte.
    reales = collections.Counter(
        r["stratum"] for r in despues if r.get("part1_status") == "ok" and r["status"] != "ok"
    )
    assert reales[con_hueco[0]] == CELLS_PER_BAND
    # Y el denominador de G que sobrevive, también por banda: CELLS_PER_BAND en cada una.
    assert collections.Counter(
        r["stratum"] for r in rp.reaction_judgeable(despues)
    ) == {b: CELLS_PER_BAND for b in range(BANDS)}
