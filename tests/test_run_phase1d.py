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

# El banco N1 de verdad tiene 72 artefactos, que es lo que hace que una banda
# traiga exactamente los 18 slots del diseño. Los tests que cuentan estímulos se
# escriben contra ese tamaño: con otro, los números del plan (4 x 18 x 16)
# dejarían de ser los que se van a correr.
BANK_N1 = 72

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
    assert BANDS * 8 * len(LENGTHS) * ARTIFACTS_PER_BAND == 1152
    assert len(plan_phase1d(1)) == 1152


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
    """384 por modelo, 96 por (modelo, banda) y 48 por (modelo, banda, longitud).

    El equilibrio por banda no es cosmético: si un modelo cayera más en las
    bandas altas, el contraste entre modelos mediría la banda, y el contraste
    entre bandas mediría el modelo. Los dos son las preguntas de la tanda.
    """
    plan = plan_phase1d(1)
    assert collections.Counter(c["model_id"] for c in plan) == {
        m: 384 for m in PHASE1D_MODELS
    }
    por_banda = collections.Counter((c["model_id"], c["band"]) for c in plan)
    assert set(por_banda.values()) == {96}
    assert len(por_banda) == len(PHASE1D_MODELS) * BANDS
    por_largo = collections.Counter(
        (c["model_id"], c["band"], c["n_turns"]) for c in plan
    )
    assert set(por_largo.values()) == {48}
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
    assert len(pares) == 1152
    assert len(set(pares)) == 1152, collections.Counter(pares).most_common(3)


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
    mismo modelo. El equilibrio de 384/96/48 no lo notaría, porque sigue saliendo
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
    assert len(set(ids)) == 1152
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
    solo contara 288 celdas no vería un cambio de semillas, de longitudes o de
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
    assert header["planned_cells"] == 1152
    assert header["bands"] == BANDS
    assert header["artifacts_per_band"] == ARTIFACTS_PER_BAND
    assert "sweep_positions" not in header, "1d no barre posiciones: barre bandas"
    assert len(rows) == 1152
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
    assert len(pares) == 1152
    assert len(set(pares)) == 1152, collections.Counter(pares).most_common(3)
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
    assert len(rows) == 1152, "una celda rota no se descarta en silencio"
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
    assert len(rows) == 1152
    assert len({r["conversation_id"] for r in rows}) == 1152
    # Y reanudar no cambia el pegote de ninguna celda: el muestreo depende de
    # (prefijo, banda, slot) y de nada más.
    pares = {(r["prefix_id"], r["artifact_id"]) for r in rows}
    assert len(pares) == 1152


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
    288 identificadores `...-s05` del fichero viejo está entre los 1.152
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
# fase, pero describe el barrido de doce posiciones —288 celdas, banco de 44,
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
    """
    bajo = texto.lower()
    return (
        "banda" in bajo
        and "1.152" in texto
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
        "describe 288 celdas sobre un banco de 44 y sin runner nuevo"
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
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, _como_hipotesis

    h4 = _como_hipotesis(HYPOTHESIS_FAMILIES["1d"]["H4 contempla el error"])
    assert h4.axis == "stratum", "es el campo en el que `run_cell` escribe la banda"
    # Y los niveles son las bandas de ESTE runner: si alguien sube `BANDS` a 6,
    # una familia clavada en 4 dejaría dos bandas fuera del contraste.
    assert h4.levels == tuple(range(BANDS))


def test_el_analisis_de_1d_cuenta_las_1152_filas_que_escribe_el_runner(
    harness, tmp_path
):
    """De punta a punta: tirada real, veredictos encima, familia declarada.

    Las categorías se ponen aquí porque el juez es otro paso y no se llama sin
    red; lo que importa es que las FILAS son las del runner, con su `stratum` y
    su `sweep_position` a `None`. La caída va metida en la banda y es enorme, así
    que si H4 mirase el eje viejo no saldría discreta: saldría n = 0.
    """
    from wrongpaste.curve import HYPOTHESIS_FAMILIES, primary_family_report

    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    assert len(rows) == 1152
    assert all(r["sweep_position"] is None for r in rows), "el runner no barre"

    # G cae con la banda, de forma determinista: 75 % en la 0 y 25 % en la 3.
    g_por_banda = {0: 3, 1: 2, 2: 1, 3: 0}
    for i, r in enumerate(rows):
        r["judge_category"] = "G" if i % 4 <= g_por_banda[r["stratum"]] else "B"

    rep = primary_family_report(rows, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    assert h4["axis"] == "stratum"
    assert sum(c["n"] for c in h4["pooled"]["curve"]) == 1152, (
        "las 1.152 conversaciones que se pagan tienen que entrar en el contraste"
    )
    assert [c["n"] for c in h4["pooled"]["curve"]] == [288] * BANDS
    for modelo in PHASE1D_MODELS:
        t = h4["by"][modelo]["trend"]
        assert t["n"] == 384, "1.152 celdas entre 3 modelos"
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
    grupos de 48 filas exactas: con la misma cuota en las cuatro bandas, la tasa
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
    # 12 de 48 en las cortas y 36 de 48 en las largas: 25 % contra 75 %.
    filas = _con_veredicto_por_longitud(rows, {2: 12, 10: 36})

    rep = primary_family_report(filas, family=HYPOTHESIS_FAMILIES["1d"])
    h4 = rep["hypotheses"]["H4 contempla el error"]
    h5 = rep["hypotheses"]["H5 contempla el error en las largas"]
    assert h5["axis"] == "n_turns" and h5["levels"] == [2, 10]
    for modelo in PHASE1D_MODELS:
        # H4 sigue midiendo sus 384 filas —no es que se haya quedado sin
        # denominador— y sale plana porque en la banda no se plantó nada.
        assert h4["by"][modelo]["trend"]["n"] == 384
        assert h4["by"][modelo]["trend"]["z"] == pytest.approx(0.0, abs=1e-9)
        largo = h5["by"][modelo]["trend"]
        assert largo["n"] == 384 and largo["slope_sign"] == 1
        assert largo["significant_holm"] is True
    # 144 de 576 en las cortas y 432 de 576 en las largas: los dos puntos de la
    # curva de H5 salen de las 1.152 filas y de ninguna otra parte.
    assert [c["n"] for c in h5["pooled"]["curve"]] == [576, 576]
    assert [c["k"] for c in h5["pooled"]["curve"]] == [144, 432]


# --- la potencia del diseño -------------------------------------------------
#
# El rediseño entero existe por un número —la potencia del 16 % de la Fase 1b— y
# hasta aquí ese número solo vivía en la prosa del módulo. Estos tests lo ponen
# en código: cuánta potencia tiene ESTE plan, medida con el estimador del propio
# repositorio y con las tasas de G medidas en la Fase 1a, y qué hace el runner
# cuando la respuesta es «no la suficiente».


def test_la_potencia_del_diseno_se_calcula_desde_el_plan_y_no_se_supone():
    """Las seis pruebas de la familia declarada, con su potencia, antes de gastar.

    Los números son el resultado, no la promesa: con 96 celdas por (modelo,
    banda) y las tasas base de N1 de la Fase 1a, la caída de 9 puntos que la
    puerta declara relevante se vería el 9 % de las veces en Opus y el 35 % en
    `sol`. Van clavados —no como desigualdad— porque son lo que decide si esta
    tanda se puede leer: una versión que devolviera «0,8 y pico» para todo
    pasaría cualquier `>` y volvería a certificar lo que este módulo vino a
    arreglar.
    """
    rep = rp.design_power(plan_phase1d())

    assert rep["family_size"] == 6, "2 hipótesis declaradas x 3 modelos"
    assert rep["alpha_effective"] == pytest.approx(0.05 / 6)
    assert rep["min_power"] == 0.80 and rep["declared_drop"] == pytest.approx(0.09)

    h4 = rep["tests"]["H4 contempla el error"]
    h5 = rep["tests"]["H5 contempla el error en las largas"]
    assert h4["claude-opus-5"]["power"] == pytest.approx(0.094, abs=5e-3)
    assert h4["gpt-5.6-sol-tst"]["power"] == pytest.approx(0.351, abs=5e-3)
    assert h4["gpt-5.6-luna-tst"]["power"] == 0.0
    assert h5["claude-opus-5"]["power"] == pytest.approx(0.194, abs=5e-3)
    assert h5["gpt-5.6-sol-tst"]["power"] == pytest.approx(0.651, abs=5e-3)
    assert h5["gpt-5.6-luna-tst"]["power"] == 0.0

    # El denominador de cada prueba, que es lo que hace comparables los números:
    # 96 celdas por (modelo, banda) en H4 y 192 por (modelo, longitud) en H5.
    assert h4["claude-opus-5"]["counts"] == [[0, 96], [1, 96], [2, 96], [3, 96]]
    assert h5["claude-opus-5"]["counts"] == [[2, 192], [10, 192]]

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
    assert mas["counts"] == [[0, 768], [1, 768], [2, 768], [3, 768]]


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
    assert len(rows) == 1152
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
    ] == pytest.approx(0.094, abs=5e-3)
    assert header["underpowered_acknowledged"] is False
