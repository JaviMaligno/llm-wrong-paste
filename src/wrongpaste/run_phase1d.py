"""Runner de la Fase 1d: el eje de similaridad por BANDAS, un modelo por estímulo.

Este módulo existe porque el barrido de la Fase 1b, corrido tal cual sobre N1,
no respondía a la pregunta. Conviene dejar escrito por qué, porque la tanda
anterior salió verde y publicada:

**El diseño viejo compraba conversaciones, no información.** `plan_phase1b`
muestreaba **doce puestos exactos** del ranking: para cada prefijo, la posición
`p` daba el artefacto del puesto `sweep_index(p, len(banco))`. Con 8 temas y 2
longitudes hay 16 prefijos, así que cada posición tenía **16 estímulos
distintos** — y los tres modelos veían los mismos 16. Las 288 conversaciones de
la tanda eran 192 pares (prefijo, artefacto) repetidos tres veces. Reconstruida
sobre las tasas de G medidas en la Fase 1a, la potencia para la caída de 9 puntos
que el propio plan declara relevante salía del **16 %**: el desenlace casi seguro
era un nulo que no se puede leer.

**El rediseño cambia tres cosas, y ninguna más.**

1. **Bandas en vez de puestos.** El ranking se parte en `BANDS` tramos contiguos
   por rango —`band_window`, que es `stratum_window` de la Fase 0 sin tocar— y de
   cada banda se sortean `ARTIFACTS_PER_BAND` artefactos **distintos** por
   prefijo, sin reemplazo. Con el banco N1 de 72, cada banda trae 18 y cada
   (prefijo, banda) la agota entera. El eje pierde resolución —cuatro puntos en
   vez de doce— y gana estímulos: 288 distintos por banda en vez de 16 por
   posición. Es el intercambio que hace falta: la curva de 1b no fue plana por
   falta de puntos, fue ilegible por falta de denominador.
2. **Un modelo por estímulo.** Cada (prefijo, artefacto) lo ve UN solo modelo.
   Con las mismas conversaciones se triplican los estímulos independientes,
   porque dos modelos sobre el mismo pegote están correlacionados por el pegote.
3. **El tamaño.** 4 bandas x 8 temas x 2 longitudes x 18 artefactos = **1.152
   celdas**, todas con (prefijo, artefacto) distinto.

**Lo que hizo falta para que el rediseño alcanzara, porque triplicar los
estímulos no bastó.** Pasado el estimador del repo por el plan (`design_power`)
con las tasas de G medidas en N1 y el alfa de Holm dentro de la familia
declarada, el plan de **tres modelos** sale con 0,23 en Opus, 0,74 en `sol` y
**0,00 en `luna`** —que no produce G ni una vez en 96 conversaciones, así que ahí
no hay nada que medir— contra un mínimo declarado de 0,80. Ninguna llega.

Dos cambios lo arreglan, y los dos están en el plan:

1. **El plantel baja a `claude-opus-5`.** Es el único donde la conducta existe
   (55 % de G en N1, frente al 9 % de `sol` y el 0 % de `luna`). La familia baja
   de 6 pruebas a 2, así que el alfa de Holm pasa de 0,0083 a 0,025, y las celdas
   por prueba se triplican.
2. **21 artefactos por banda y no 18.** Con 18 el diseño se quedaba en 0,797
   contra el 0,80 declarado.

Con eso, H4 y H5 salen a **0,859** y `null_is_informative` pasa a `True`: un nulo
de esta tanda sí se podrá leer como «no hay efecto» y no como «no lo habríamos
visto». Con los tres modelos, no; por eso el plantel es parte del diseño y no una
preferencia.

Se escribe en el docstring y además en una **puerta** (`check_design_power`),
porque la Fase 1b enseñó que un número que solo vive en la prosa no frena a
nadie: la tirada no empieza sin que quien la paga firme que la conoce, y la
cabecera del JSONL se la lleva escrita. Lo que sube a 0,81 es el agregado de los
tres modelos —el rediseño quita la razón por la que `pooled` estaba degradado,
porque ahora cada (prefijo, artefacto) aparece una sola vez—, pero `pooled` es lo
que los planes declaran secundario y no decide nada, y al alfa de Holm tampoco
llegaría (0,72). Pasarse a él después de mirar cuál de los dos da 0,80 sería
elegir el análisis por su resultado; si el primario tiene que cambiar, se cambia
en el plan y antes de correr.

**Lo que NO cambia**: los prefijos, el pegado, los turnos posteriores, la
reanudación, el fallo como dato y la puerta del eje (D12) se importan de la
Fase 0 y de la 1b y se usan tal cual.

**Por qué un módulo nuevo y no otro parámetro de `run_phase1b`.** La Fase 1b está
pagada y publicada, y su plan se reanuda por identificador: `plan_phase1b` tiene
que seguir dando exactamente lo mismo, semillas incluidas. Un `design="bands"`
dentro de aquel runner pondría los dos muestreos a compartir bucle, y el día que
alguien arregle uno moverá el otro. Lo que sí se comparte —todo lo que no es el
muestreo— se importa; lo que se copia es `run_cell` y `main`, y se copia a
sabiendas: la línea que cambia es la de elegir el artefacto.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from wrongpaste import config
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.conversation import (
    MAX_TOKENS,
    N_POST_TURNS,
    continue_after_paste,
    inject_paste,
    reply_traces,
)
from wrongpaste.curve import (
    ALPHA,
    DECLARED_DROP,
    HYPOTHESIS_FAMILIES,
    MIN_POWER,
    fold_band,
    trend_power,
)
from wrongpaste.judging import JUDGES
from wrongpaste.prefixes import PREFIX_MODEL, ensure_prefix
from wrongpaste.rates import JUDGEABLE_STATUSES
from wrongpaste.records import (
    ARM_NO_REPAIR,
    ConversationRecord,
    RunHeader,
    run_header_line,
)
from wrongpaste.rubric import RUBRIC_VERSION
from wrongpaste.run_judging import load_rows

# La maquinaria de las fases anteriores, tal cual. Lo que se importa con guion
# bajo es privado de aquel módulo y se toma prestado a propósito: dos copias de
# la clasificación de fallos (D6) divergen en cuanto alguien arregle una.
from wrongpaste.run_phase0 import (
    CORRECTIONS_PATH,
    DONE_STATUSES,
    ERROR_BODY_CHARS,
    HARNESS_ERROR,
    LENGTHS,
    STAGE_PASTE,
    STAGE_POST,
    STAGE_PREFIX,
    STAGE_RANK,
    USER_SYSTEM_PROMPT,
    Phase0Record,
    _attempts,
    _has_empty_response,
    _model_label,
    _response_model,
    _status_for_error,
    _system_prompt,
    bank_sha,
    code_sha,
    compact_resume_file,
    failure_origin,
    is_resumable,
    rank_for_prefix,
    read_run_header,
    resume_state,
    status_for_stop_reasons,
    stratum_window,
    summary_path,
    topics_sha,
    user_call_traces,
    user_side_problems,
)
from wrongpaste.run_phase0 import failed_record as phase0_failed_record
from wrongpaste.run_phase1a import call_budget
from wrongpaste.run_phase1b import (
    PHASE1B_MODELS,
    SPEC_PATH,
    check_axis,
    check_resume_level,
)
from wrongpaste.simulated_user import USER_MODEL, SimulatedUserError
from wrongpaste.topics import Topic, load_topics

# Los mismos tres modelos que 1a y 1b, para que las tandas se puedan comparar.
PHASE1D_MODELS = list(PHASE1B_MODELS)

# Cuatro bandas. Es un compromiso declarado antes de correr: menos bandas dejan
# el eje sin forma —con dos solo hay «cerca» y «lejos»— y más bandas vuelven al
# problema de 1b, que era quedarse sin observaciones por punto. Con 4 y el banco
# N1 de 72 salen 18 artefactos por banda, que es un número entero de slots y
# múltiplo de los 3 modelos: el reparto sale exacto sin redondeos.
BANDS = 4

# Cuántos artefactos distintos se sortean de cada banda para cada prefijo. Es
# `len(banco) // BANDS` con el banco N1 de hoy (72), y se escribe como constante
# —no se deriva del banco— porque es el TAMAÑO DEL PLAN: el presupuesto de la
# tanda se declara antes de tocar el disco, y un plan que encogiera solo porque
# alguien quitó un artefacto del banco sería un presupuesto que cambia solo. Si
# el banco crece o mengua, esto se sube o se baja a mano y se vuelve a mirar el
# recuento.
#
# Que sea constante obliga a atarla al banco en otro sitio, y ese sitio es
# `check_bank_size`, que corre en `main` antes de escribir nada. No basta con
# que `choose_band_artifact` reviente: revienta DENTRO de una celda, y D6 —que
# es lo que hace que una celda rota no tumbe la tirada— la convierte en una fila
# fallida y sigue. Con un banco de 71, eso era la tanda entera pagada para
# terminar con 1.136 estímulos de los 1.152 y las 16 pérdidas todas en la banda
# 0: un hueco sistemático en un extremo de la variable independiente.
# 21 y no 18: con 18 el diseño se quedaba en potencia 0,797 contra el mínimo
# declarado de 0,80 —tres milésimas— y la tirada habría necesitado firmar
# `acknowledge_underpowered`. Con 21 la tanda va a 1.344 conversaciones y 0,859.
#
# Y 21 y no 19 o 20, que también cruzaban, porque **tiene que dividir entre 3**:
# la guarda de `plan_phase1d` exige que el plantel reparta por igual los slots de
# cada (prefijo, banda), y con un número primo ningún plantel de tres modelos lo
# hace. Subir la potencia a costa de perder la vía de tres modelos habría sido un
# mal cambio: la tanda actual corre solo Opus, pero el diseño no debería cerrarse
# esa puerta por 0,02 de potencia.
ARTIFACTS_PER_BAND = 21

MASTER_SEED = 20260916

# Esta tanda corre un solo brazo: el banco con señal. El pegote neutro ya se
# barrió en 1b y salió plano, y la pregunta que queda es la del brazo donde los
# modelos SÍ dudan.
PASTE_LEVEL = "N1"
PHASE = "1d"

OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase1d"

# La procedencia de D5. El plan vive en el repo del blog, no en este.
#
# Y **no** es `PLAN_PATH_BY_LEVEL["N1"]` de 1b, aunque aquel documento también se
# llame «Fase 1d» y declare este mismo brazo: describe el barrido de doce
# posiciones —288 celdas, banco de 44, los tres modelos viendo cada pegote y
# «no hay runner nuevo»—, que es justo el diseño que esta tanda NO corre.
# Heredarlo dejaría en la cabecera una procedencia falsa en el único campo que
# existe para no tener que reconstruir el diseño: quien abriera el JSONL dentro
# de seis meses leería 1.152 filas de 4 bandas x 18 artefactos como si fueran
# 288 de doce puestos. El plan del barrido se conserva como registro del diseño
# descartado, y este apunta al que describe el muestreo por bandas.
PLAN_PATH = "docs/superpowers/plans/2026-09-16-pegado-accidental-fase-1d-bandas.md"


# --- las bandas del ranking ------------------------------------------------


def band_window(
    ranked: list[tuple[Artifact, float]], band: int, n_bands: int = BANDS
) -> list[tuple[Artifact, float]]:
    """El tramo del ranking que corresponde a una banda.

    Es `run_phase0.stratum_window` **sin una línea propia**: aquella función ya
    parte el ranking por rango —no por valor de coseno, que es lo que dejaría
    ventanas vacías porque la distribución real está muy concentrada—, en tramos
    contiguos, sin huecos ni solapes y sin devolver nunca una ventana vacía. Una
    banda es exactamente eso, así que reimplementarla dejaría dos definiciones de
    «tramo del ranking» que se pueden separar en los bordes sin que nadie lo
    note.

    Lo que aporta este envoltorio es el vocabulario: en esta tanda la partición
    no es un control (el estrato de 1a, que servía para que el nivel del pegote
    no se confundiera con el parecido) sino la VARIABLE INDEPENDIENTE, y llamarla
    banda en el código es lo que hace que el análisis no la lea como un estrato
    más.
    """
    return stratum_window(ranked, band, n_bands)


def band_order(prefix_id: str, band: int, size: int) -> list[int]:
    """Orden en que un (prefijo, banda) recorre su ventana: una permutación.

    El slot `s` de una celda no es el puesto `s` de la banda, sino el puesto
    `band_order(...)[s]`. Lo que el barajado compra es que **el slot no diga nada
    del puesto dentro de la banda**: sin él, `slot` sería el rango fino dentro
    del tramo, el mismo en los 16 prefijos, y como el modelo se asigna por slot
    quedaría asignado por la similaridad de grano fino — de forma determinista y
    repetida en todas las bandas y todos los prefijos. El sesgo que eso mete es
    pequeño (dentro de una banda de 18 puestos, la media de rango de los tres
    modelos se separaría en 2 puestos como mucho) pero es sistemático, y no hay
    ninguna razón para pagarlo: barajar cuesta una permutación.

    Lo que el barajado **no** arregla, y conviene no atribuírselo: que cada
    modelo vea los 18 artefactos de cada banda ya sale del desplazamiento por
    tema y longitud de `plan_phase1d`, y sigue saliendo con la ventana en su
    orden natural. Comprobado rompiendo esta función a propósito: devolver
    `ventana[slot]` deja verde todo menos el test que mira precisamente el puesto
    (`test_el_slot_no_es_el_puesto_dentro_de_la_banda`).

    **La semilla sale de `blake2b`, no de `hash()`.** El `hash()` de Python está
    aleatorizado por proceso (PYTHONHASHSEED), así que sembrar con él daría un
    barajado distinto en cada arranque: la misma celda reintentada mañana
    recibiría otro pegote y la tirada dejaría de ser reproducible — y lo haría en
    silencio, porque el plan seguiría teniendo 1.152 celdas bien formadas.

    **Y no entra la semilla maestra de la tirada.** Es a propósito: el artefacto
    de una fila tiene que poder reconstruirse desde lo que la fila declara
    —prefijo, banda y slot— sin saber con qué `seed` se corrió, y una reanudación
    no puede depender de que quien la lance recuerde la semilla. La aleatoriedad
    que aquí hace falta es la de romper la correspondencia slot-artefacto entre
    prefijos, y para eso el prefijo ya es una fuente suficiente.
    """
    if size < 1:
        raise ValueError(f"una banda vacía no se puede barajar: size={size}")
    semilla = int.from_bytes(
        hashlib.blake2b(
            f"{prefix_id}|{band}".encode("utf-8"), digest_size=8
        ).digest(),
        "big",
    )
    return [int(i) for i in np.random.default_rng(semilla).permutation(size)]


def choose_band_artifact(
    ranking: list[tuple[Artifact, float]],
    band: int,
    slot: int,
    prefix_id: str,
    n_bands: int = BANDS,
) -> tuple[Artifact, float]:
    """El artefacto que le toca a un (prefijo, banda, slot): sin reemplazo.

    Los `ARTIFACTS_PER_BAND` slots de un (prefijo, banda) dan los artefactos de
    esa banda **sin repetir ninguno**, que es la garantía entera del rediseño:
    así cada prefijo agota su banda y las 1.152 celdas son 1.152 estímulos
    distintos.

    Sin estado, como en 1b: (prefijo, banda, slot) determina el artefacto por
    completo, así que reanudar no arrastra nada y una celda reintentada recibe el
    pegote que le tocaba.

    Un slot fuera de la ventana **revienta**. Es la diferencia deliberada con
    `sweep_index`, que con un banco pequeño dejaba que dos posiciones cayeran en
    el mismo artefacto para que una tanda de prueba pudiera correr: allí repetir
    era un defecto del tamaño del banco, aquí sería la ruina del diseño — y sería
    invisible, porque el plan seguiría teniendo 1.152 celdas.
    """
    ventana = band_window(ranking, band, n_bands)
    if not 0 <= slot < len(ventana):
        raise ValueError(
            f"slot {slot} fuera de la banda {band}, que tiene {len(ventana)} "
            f"artefactos: el muestreo es SIN REEMPLAZO y un banco de "
            f"{len(ranking)} no da para {ARTIFACTS_PER_BAND} por banda. Ajusta "
            "ARTIFACTS_PER_BAND o completa el banco; repetir pegotes dejaría el "
            "recuento de estímulos independientes inflado, que es el fallo que "
            "esta fase viene a arreglar."
        )
    return ventana[band_order(prefix_id, band, len(ventana))[slot]]


def make_conversation_id(
    model_id: str, topic_id: str, n_turns: int, band: int, slot: int
) -> str:
    """Identidad de una celda: legible y única sin necesitar el índice.

    Lleva banda y slot donde la de 1b llevaba la posición (`-s07`), así que los
    identificadores de los dos diseños no se pueden confundir aunque los dos
    vivan bajo el prefijo `p1d-`: `-b1-a07` no es `-s07`. Importa porque la
    reanudación va por identificador.
    """
    return f"p{PHASE}-{model_id}-{topic_id}-{n_turns}-b{band}-a{slot:02d}"


def plan_phase1d(
    seed: int = MASTER_SEED,
    models: tuple[str, ...] | list[str] = tuple(PHASE1D_MODELS),
) -> list[dict]:
    """1.152 celdas: 4 bandas x 8 temas x 2 longitudes x 18 artefactos.

    **El plantel es un parámetro, y el cruce no depende de él.** La tanda que se
    paga corre SOLO `claude-opus-5`: es el único modelo que produce G en N1
    —55 % frente al 9 % de `sol` y el 0 % de `luna`, medido en la Fase 1a con
    n = 96 por modelo—, así que en los otros dos no hay curva que medir y las
    conversaciones que se les compraran no entrarían en ningún contraste. Lo que
    NO cambia al encoger el plantel es el diseño: el modelo se asigna a una celda
    **después** de que el cruce esté hecho, así que las 1.152 celdas, sus
    (prefijo, banda, slot) y por tanto sus 1.152 pegotes distintos son los
    mismos con tres modelos que con uno. Lo único que cambia es quién los ve.

    **El modelo no se sortea.** Se asigna con `(slot + t + longitud + banda) %
    len(plantel)`, y el reparto sale exacto sin depender de la suerte: cada
    (prefijo, banda) tiene 18 slots y el plantel tiene que dividir a 18, así que
    cada modelo se lleva 18/len de cada (prefijo, banda) —con tres: 6, o sea 384
    por modelo, 96 por (modelo, banda) y 48 por (modelo, banda, longitud)—.
    Dejarlo al azar sería el error que más caro sale aquí: si un modelo cayera
    más en las bandas altas, el contraste entre modelos mediría la banda y el
    contraste entre bandas mediría el modelo, que son justo las dos preguntas de
    la tanda.

    Un plantel que **no** divida a 18 revienta en vez de repartir de más a unos
    y de menos a otros. Con 4 modelos, dos verían 5 artefactos de cada banda y
    dos verían 4: un desequilibrio pequeño, sistemático y perfectamente
    invisible —el plan seguiría teniendo 1.152 celdas bien formadas— que mete
    justo la correlación entre modelo y banda que el párrafo anterior existe
    para evitar.

    El desplazamiento por tema, longitud y banda no cambia el equilibrio —con 18
    slots cualquier desplazamiento reparte 6, 6 y 6—, y hay que decir qué sí
    hace, porque es fácil atribuirle de más: lo que hace es que **los modelos
    roten celda a celda en el orden de la tirada**. Sin el término de la banda,
    las cuatro celdas seguidas que comparten slot —las cuatro bandas— irían al
    mismo modelo, y una tirada de horas contra un gateway compartido tendría el
    modelo correlacionado con la hora de reloj en bloques de cuatro. Es la misma
    razón por la que 1a y 1b ponían el modelo en round-robin.

    Los términos de tema y longitud, en cambio, no cambian ninguna propiedad
    medible mientras el barajado esté en pie: comprobado quitándolos, la suite se
    queda verde entera. Se mantienen porque son quienes sostienen —solos, sin
    depender del barajado— que cada modelo vea los 18 artefactos de cada banda, y
    esa es de las propiedades que conviene no dejar colgando de una sola línea.

    **El orden del bucle es el orden de la tirada**, y por eso la banda va en el
    bucle más interno: una tirada dura horas contra un gateway compartido, y con
    la banda por fuera el primer cuarto de la tanda se correría entero en la
    banda 0. La banda es la variable independiente; confundirla con la hora de
    reloj sería el peor sitio donde meter esa correlación. El prefijo va por
    fuera porque el ranking se cachea por prefijo (`rank_for_prefix`).

    El `seed` de la celda no elige el artefacto —eso lo hace (prefijo, banda,
    slot)—: se conserva porque identifica la tirada y porque el runner lo pasa a
    los turnos posteriores.
    """
    plantel = list(models)
    if not plantel:
        raise ValueError(
            "el plantel está vacío: eso no es una tanda pequeña, son 1.152 "
            "celdas sin nadie que las corra"
        )
    if ARTIFACTS_PER_BAND % len(plantel):
        raise ValueError(
            f"un plantel de {len(plantel)} modelos no reparte por igual los "
            f"{ARTIFACTS_PER_BAND} slots de cada (prefijo, banda): unos verían "
            f"{ARTIFACTS_PER_BAND // len(plantel) + 1} artefactos de cada banda "
            f"y otros {ARTIFACTS_PER_BAND // len(plantel)}. El desequilibrio es "
            "pequeño, sistemático e invisible —el plan seguiría teniendo 1.152 "
            "celdas bien formadas— y deja el modelo correlacionado con la "
            f"banda, que son las dos preguntas de la tanda. Usa un plantel que "
            f"divida a {ARTIFACTS_PER_BAND}."
        )
    topics = load_topics()
    rng = np.random.default_rng(seed)
    plan: list[dict] = []
    for t, topic in enumerate(topics):
        for li, n_turns in enumerate(LENGTHS):
            for slot in range(ARTIFACTS_PER_BAND):
                for band in range(BANDS):
                    model_id = plantel[(slot + t + li + band) % len(plantel)]
                    plan.append(
                        {
                            "cell_index": len(plan),
                            "model_id": model_id,
                            "topic_id": topic.id,
                            "n_turns": n_turns,
                            "band": band,
                            "n_bands": BANDS,
                            "artifact_slot": slot,
                            "paste_level": PASTE_LEVEL,
                            "condition": "paste",
                            "seed": int(rng.integers(0, 2**31)),
                            "conversation_id": make_conversation_id(
                                model_id, topic.id, n_turns, band, slot
                            ),
                        }
                    )
    return plan


# --- ejecución de una celda -----------------------------------------------


def conversation_status(stop_reasons: list[str | None], replies: list[Any]) -> str:
    """La precedencia de `status` de una conversación (D6), en un solo sitio.

    Es la de las tandas anteriores, sin cambiarle nada: `refusal` es lo único de
    aquí que es conducta y manda; `empty` es la afirmación más fuerte que se
    puede hacer sobre una respuesta; `truncated` es un corte por el tope que
    mandamos nosotros; y si no hay nada de eso, `ok`.

    Vive en una función porque ahora la aplican DOS caminos —`run_cell`, cuando
    la celda se corre entera, y `resume_post_turns`, cuando los turnos
    posteriores se generan meses después sobre una transcripción guardada— y dos
    copias de esta precedencia divergen en cuanto alguien arregle una: bastaría
    con que la de la parte 2 leyera un `truncated` como `ok` para que una
    reacción cortada a media frase entrase en el denominador de G.

    `replies` es cualquier cosa con `.text` (los `Reply` del cliente, o el
    envoltorio con el que la parte 2 vuelve a meter la reacción ya pagada), y
    tienen que estar TODAS las respuestas del modelo evaluado de la
    conversación: mirar solo las nuevas dejaría de ver una reacción vacía.

    **Lo que este `status` NO es: el denominador de G.** Es el estado de la
    CONVERSACIÓN ENTERA, turnos posteriores incluidos, y así tiene que ser —un
    turno `post` cortado falsea el recuento de fuga igual que una reacción
    cortada falsea la categoría—. Pero la categoría G la decide solo la reacción
    al pegote, y en una tanda partida esa reacción se paga y se clasifica meses
    antes de que existan los turnos posteriores. O sea que la precedencia mira
    también hacia el otro lado: aplicada tal cual en `resume_post_turns`, un
    turno `post` roto baja la fila de `ok` a `truncated` y la saca de
    `JUDGEABLE_STATUSES` —que es exactamente `{"ok"}`— con la reacción intacta.
    El fichero completo tendría entonces menos filas en el denominador de H4 y
    H5 que el de la parte 1, y la diferencia la decidiría algo ocurrido DESPUÉS
    del estímulo. Por eso la parte 2 escribe además `part1_status`, y el
    denominador de G se lee con `reaction_judgeable`.
    """
    declared = status_for_stop_reasons(stop_reasons)
    if declared == "refusal":
        return "refusal"
    if _has_empty_response(replies):
        return "empty"
    if declared is not None:
        return declared
    return "ok"


def reaction_judgeable(rows: list[dict]) -> list[dict]:
    """Las filas cuya REACCIÓN al pegote se puede clasificar: el denominador de G.

    `run_judging.judgeable` mira `status`, que es el estado de la conversación
    entera, y sobre una tanda de una sola pieza las dos cosas coinciden. Sobre
    una tanda partida no: el `status` de un fichero de parte 2 ya lleva dentro
    los turnos posteriores, así que una fila cuya reacción el juez clasificó en
    la parte 1 sale de `judgeable` si meses después un turno `post` volvió
    cortado. Medido con una de cada cuatro continuaciones en `max_tokens`:
    `judgeable` baja de 1.152 filas a 864, y las 288 que se van conservan su
    `reaction` byte a byte.

    Eso no es una pérdida aceptable, es una pérdida **de información nueva que
    crea la partición**: la variable de H4 y H5 es la categoría G, la categoría G
    la decide el estímulo (prefijo + pegote + reacción) y ese estímulo terminó
    antes de que el primer turno posterior existiera. Un filtro que lo deshace
    después estaría metiendo en el eje de la similaridad un hueco decidido por
    algo posterior al estímulo — y, como las bandas se recorren en el bucle más
    interno, un hueco que ni siquiera tiene por qué repartirse igual entre ellas.

    Por eso se lee `part1_status` —el estado que la parte 1 declaró y con el que
    el juez trabajó— y solo se cae a `status` cuando no está: una fila sin el
    campo es de una tanda de una sola pieza, o una fila que la parte 2 copió tal
    cual porque ya venía rota, y en los dos casos su `status` **es** el de su
    reacción. Caer a `status` no readmite nada: lo que la parte 1 rompió sigue
    fuera.

    No toca `run_judging.judgeable` ni `JUDGEABLE_STATUSES`: los usan las Fases
    1a y 1b, que están pagadas y publicadas, y allí no hay nada que arreglar
    porque no hay partición. `JUDGEABLE_STATUSES` se importa en vez de
    reescribirse para que el día que cambie la definición de «juzgable» cambie
    en los dos sitios a la vez.
    """
    return [
        r
        for r in rows
        if str(r.get("part1_status", r.get("status", "ok"))) in JUDGEABLE_STATUSES
    ]


def run_cell(
    cell: dict,
    topic: Topic,
    bank: list[Artifact],
    rank_caches: dict[str, tuple],
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
    post_turns: int = N_POST_TURNS,
) -> ConversationRecord:
    """Una conversación: prefijo, pegote de la banda, turnos posteriores.

    Calcada de `run_phase1b.run_cell` salvo en una línea —el artefacto sale de
    `choose_band_artifact`, que necesita el `prefix_id` porque el barajado se
    siembra con él— y en lo que la fila declara. Se copia en vez de
    parametrizarse porque la de 1b corre una tanda ya publicada.

    **La banda viaja en la fila como `stratum`/`n_strata`.** No es un apaño: una
    banda es literalmente un estrato por rango del ranking —la misma partición
    que `stratum_window` implementa— y el esquema ya tiene ese par de campos con
    su denominador al lado. Escribirla en `sweep_position` sería peor que un
    apaño: ese campo significa «uno de los doce puestos del barrido de 1b», y
    ponerle un 3 haría que un análisis conjunto leyera las dos tandas como si
    hubieran muestreado igual. Por eso `sweep_position` se queda en `None`, que
    es lo que significa «aquí no hubo barrido».

    **`post_turns=0` corre la parte 1 y solo la parte 1**: prefijo, pegote y
    reacción. Es lo único que decide la categoría G, que es la variable de H4 y
    H5, así que es lo único que hay que comprar para responder la pregunta de la
    fase; los dos turnos posteriores de D7 alimentan el recuento de fuga de la
    Fase 2 y se generan más adelante con `resume_post_turns`, retomando la
    transcripción que esta fila guarda. Con 0 **no se llama** a
    `continue_after_paste` —que es de donde sale el ahorro: son dos tercios de
    las llamadas al modelo evaluado y todas las del usuario simulado— y la fila
    queda con `post_indices` vacía y la transcripción terminada en la reacción.
    Lo demás no cambia: `reaction`, `status`, las dos similaridades, la banda y
    las etiquetas del transcript salen igual, así que la fila se clasifica y se
    analiza como cualquier otra.
    """
    model_id = cell["model_id"]

    partial["stage"] = STAGE_PREFIX
    prefix = ensure_prefix(topic, cell["n_turns"])
    partial["prefix_id"] = prefix["prefix_id"]
    partial["prefix_user_traces"] = list(prefix.get("user_reply_traces", []))

    # Copia: el prefijo lo comparten las 72 celdas que cuelgan de él, y no se
    # muta.
    transcript = [dict(m) for m in prefix["transcript"]]
    partial["transcript"] = transcript

    request_params: dict = {}
    partial["request_params"] = request_params
    replies: list = []
    partial["replies"] = replies
    user_replies: list = []
    partial["user_replies"] = user_replies

    partial["stage"] = STAGE_RANK
    ranking, full_by_id, truncated_user, truncated_full = rank_for_prefix(
        prefix, bank, rank_caches
    )
    artifact, similarity_user = choose_band_artifact(
        ranking, cell["band"], cell["artifact_slot"], prefix["prefix_id"]
    )
    partial["artifact"] = artifact
    similarity_full = full_by_id.get(artifact.id)
    ids = [art.id for art, _ in ranking]
    similarity_rank = ids.index(artifact.id)
    similarity_pct = similarity_rank / (len(ids) - 1) if len(ids) > 1 else 0.0
    ranking_rows = [{"artifact_id": art.id, "similarity": s} for art, s in ranking]

    partial["stage"] = STAGE_PASTE
    reaction, paste_usage, paste_index, paste_reply = inject_paste(
        model_id, transcript, artifact, request_params_out=request_params
    )
    usages = [paste_usage]
    replies.append(paste_reply)

    post_indices: list[int] = []
    if post_turns:
        partial["stage"] = STAGE_POST
        post_indices, post_usages, post_replies = continue_after_paste(
            model_id,
            transcript,
            topic,
            n_post=post_turns,
            request_params_out=request_params,
            user_replies_out=user_replies,
        )
        usages.extend(post_usages)
        replies.extend(post_replies)

    traces = reply_traces(replies)
    stop_reasons = [trace["stop_reason"] for trace in traces]

    # La precedencia de siempre, ahora compartida con la parte 2 (D6).
    status = conversation_status(stop_reasons, replies)

    # Y por encima de todo, el lado del usuario simulado: si uno de sus turnos
    # salió vacío o cortado, la conversación tiene un agujero y ninguna categoría
    # de la rúbrica se puede leer de ahí. Aquí pesa más que en 1b, no menos: cada
    # celda es un estímulo único, así que no hay otra fila que lo promedie.
    user_traces = user_call_traces(partial["prefix_user_traces"], user_replies)
    problemas = user_side_problems(
        user_traces, transcript, prefix_len=len(prefix["transcript"])
    )
    error_code: int | str | None = None
    error_body: str | None = None
    if problemas:
        status = HARNESS_ERROR
        error_code = SimulatedUserError.__name__
        detalle = "; ".join(problemas)
        if any(f"etapa {STAGE_PREFIX}" in problema for problema in problemas):
            detalle += (
                f". El turno roto está en el prefijo compartido "
                f"{prefix['prefix_id']}, que se reutiliza tal cual (D1): "
                f"reintentar la celda lo vuelve a leer igual. Borra "
                f"runs/prefixes/{prefix['prefix_id']}.json para que se regenere"
            )
        error_body = detalle[:ERROR_BODY_CHARS]

    ended_at = time.time()
    return Phase0Record(
        run_id=run_id,
        conversation_id=cell["conversation_id"],
        cell_index=cell["cell_index"],
        model_id=model_id,
        model_label=_model_label(model_id),
        response_model=_response_model(traces),
        topic_id=topic.id,
        n_turns=cell["n_turns"],
        stratum=cell["band"],
        n_strata=cell["n_bands"],
        prefix_id=prefix["prefix_id"],
        condition=cell["condition"],
        arm=ARM_NO_REPAIR,
        artifact_id=artifact.id,
        artifact_kind=artifact.kind,
        artifact_signal=artifact.signal,
        paste_level=cell["paste_level"],
        artifact_text=artifact.text,
        artifact_entities=list(artifact.entities),
        similarity_user=similarity_user,
        similarity_full=similarity_full,
        similarity_rank=similarity_rank,
        similarity_pct=similarity_pct,
        ranking=ranking_rows,
        similarity_user_truncated=bool(truncated_user),
        similarity_full_truncated=bool(truncated_full),
        paste_index=paste_index,
        post_indices=post_indices,
        transcript=transcript,
        reaction=reaction,
        request_params=dict(request_params),
        system_prompt=_system_prompt(request_params, transcript),
        user_model=USER_MODEL,
        prefix_model=prefix.get("prefix_model", PREFIX_MODEL),
        max_tokens=MAX_TOKENS,
        stop_reasons=stop_reasons,
        usages=usages,
        seed=cell["seed"],
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=int((ended_at - started_at) * 1000),
        status=status,
        error_code=error_code,
        error_body=error_body,
        attempts=_attempts(traces),
        user_stop_reasons=[trace.get("stop_reason") for trace in user_traces],
        user_reply_traces=user_traces,
    )


def failed_record(
    cell: dict,
    topic_id: str,
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
    exc: BaseException,
) -> ConversationRecord:
    """La fila de una celda que revienta (D6), con su banda y su nivel.

    Se apoya entera en la de la Fase 0 —la que sabe distinguir un fallo del arnés
    de una negativa del modelo evaluado— y le añade lo que aquel runner no puede
    saber: de qué banco salía el pegote y en qué banda estaba.

    La banda se escribe **aunque la celda haya fallado**: no es un resultado sino
    un dato del diseño, conocido antes de llamar a nadie. Sin ella, el recuento
    de pérdidas por banda —el que dice si el hueco es sistemático o aleatorio— no
    se puede hacer. El `n_strata` hay que ponerlo a mano porque la fila de la
    Fase 0 lo cablea a sus 8 estratos, y una banda de 4 escrita bajo un
    denominador de 8 sería una banda mal situada en el eje.
    """
    rec = phase0_failed_record(
        {"replicate_idx": 0, "stratum": cell["band"], **cell},
        topic_id,
        run_id,
        started_at,
        partial,
        exc,
    )
    rec.paste_level = cell["paste_level"]
    rec.stratum = cell["band"]
    rec.n_strata = cell["n_bands"]
    return rec


# --- tirada ----------------------------------------------------------------


# Tasa de G por modelo en el brazo N1, medida en la Fase 1a con el juez validado
# (`JUDGES[0]`) sobre los veredictos `ok`: es `(k, n)` y no un decimal para que
# el denominador viaje al lado —29 y 32 no son 300, y una potencia calculada
# sobre una tasa de 16/29 hereda esa incertidumbre—. Se recuentan del disco en
# `test_las_tasas_base_estan_medidas_en_la_fase_1a_y_no_supuestas`: tres
# constantes plausibles escritas a mano volverían aritmética correcta sobre una
# entrada inventada todo lo que cuelga de aquí, que es la forma que tiene este
# repositorio de quedarse verde certificando algo falso.
N1_G_RATE_BY_MODEL: dict[str, tuple[int, int]] = {
    "gpt-5.6-sol-tst": (3, 32),
    "gpt-5.6-luna-tst": (0, 32),
    "claude-opus-5": (16, 29),
}

# Qué columna del PLAN corresponde a cada eje que declara el análisis. El plan no
# tiene filas todavía, así que la potencia se cuenta sobre las celdas, y para eso
# hay que saber dónde vive cada eje: `stratum` —la banda— se planifica como
# `band`, y `n_turns` se llama igual en los dos sitios.
#
# Un eje que no esté aquí **revienta** en vez de contar cero. Es la diferencia
# entre las dos formas de no tener potencia: una hipótesis declarada sobre un eje
# que esta tanda no muestrea no es una hipótesis débil, es una hipótesis que no
# se puede correr, y devolver 0,0 la disfrazaría de lo primero.
PLAN_FIELD_BY_AXIS: dict[str, str] = {"stratum": "band", "n_turns": "n_turns"}


def design_power(
    plan: list[dict],
    family: dict | None = None,
    base_rates: dict[str, tuple[int, int]] | None = None,
    alpha: float = ALPHA,
) -> dict:
    """Qué vería este plan antes de correrlo: la potencia de las seis pruebas.

    **Por qué existe.** Este módulo entero se escribió porque la Fase 1b compró
    288 conversaciones con una potencia del 16 %, y la escribió en su docstring
    como el motivo del rediseño. Pero el rediseño solo se declaraba a sí mismo:
    triplicaba los estímulos independientes y nadie volvía a pasar el estimador
    del repo por el plan nuevo. Hecha la cuenta —las celdas que salgan del plan,
    las tasas de G medidas en N1 y el alfa que de verdad hay que batir dentro de
    la familia declarada— el plan de tres modelos no llega al `MIN_POWER` de
    0,80: con `luna` a cero de tasa base, dos de las seis pruebas no tienen nada
    que medir y arrastran la familia entera. El plan de un solo modelo sí llega.

    Esa diferencia es el motivo de que esta función exista y de que corra
    **antes** del gasto: es lo que separa pagar una tanda cuyo nulo significará
    algo de pagar una cuyo nulo habrá que escribir como «no lo habríamos visto».

    La cuenta se hace **sobre el plan**, sin datos y sin red: las celdas por
    nivel las da el propio plan y la tasa base la Fase 1a. Por eso puede ir
    delante del gasto, que es el único sitio donde sirve de algo.

    **El reparto es el declarado, no el que conviene.** `by="model_id"` con la
    familia de la Fase 1d: 2 hipótesis x 3 modelos = 6 pruebas y Holm sobre las
    seis, que es lo que `curve.primary_family_report` corre en el Paso 8. Agregar
    los tres modelos sube H4 a 0,81 —el rediseño quita la razón por la que
    `pooled` estaba degradado, porque ahora cada (prefijo, artefacto) aparece una
    sola vez—, pero `pooled` es justo lo que los planes declaran secundario y no
    decide nada; y al alfa de Holm tampoco llegaría (0,72). Cambiar el
    estadístico primario después de mirar cuál da 0,80 sería elegir el análisis
    por su resultado. Si hay que cambiarlo, se cambia en el plan y antes.

    Devuelve un dict JSON-able: viaja en la cabecera del JSONL (D5), que es donde
    tiene que estar para que dentro de seis meses se pueda leer qué habría podido
    ver esta tanda sin reconstruir el plan.
    """
    familia = HYPOTHESIS_FAMILIES[PHASE] if family is None else family
    tasas = N1_G_RATE_BY_MODEL if base_rates is None else base_rates

    modelos = sorted({str(celda["model_id"]) for celda in plan})
    faltan = [m for m in modelos if m not in tasas]
    if faltan:
        raise ValueError(
            f"el plan corre {faltan} y no hay tasa base medida para esos "
            "modelos: sin tasa base no hay potencia que calcular, y ponerle una "
            "plausible sería inventar el número que autoriza a concluir una "
            "ausencia. Mídela en la Fase 1a o quítalos del plan."
        )

    # El tamaño de la familia sale de contar lo que se va a correr —hipótesis
    # declaradas x modelos con celdas—, igual que en `primary_family_report`: es
    # el multiplicador que paga el `p` más pequeño, y con él el alfa que la
    # potencia tiene que batir.
    family_size = len(familia) * len(modelos)
    alfa_efectivo = alpha / family_size if family_size else alpha

    pruebas: dict[str, dict[str, dict]] = {}
    sin_potencia: list[str] = []
    for nombre, hipotesis in familia.items():
        eje = getattr(hipotesis, "axis", "sweep_position")
        niveles = getattr(hipotesis, "levels", None)
        campo = PLAN_FIELD_BY_AXIS.get(eje)
        if campo is None or niveles is None:
            raise ValueError(
                f"la hipótesis {nombre!r} se declara sobre el eje {eje!r}, que "
                f"este diseño no muestrea (muestrea {sorted(PLAN_FIELD_BY_AXIS)}"
                "). No es una hipótesis con poca potencia: es una hipótesis que "
                "no se puede correr sobre esta tanda, y la curva saldría bien "
                "formada con n = 0 en todos los niveles."
            )
        # **Contar el plan por nivel exige plegar igual que pliega el análisis.**
        # H4 declara `levels=(0, 1)` y `fold_from=BANDS`: la celda trae una de
        # las CUATRO bandas del muestreo y el contraste va sobre DOS, agrupando
        # 0+1 y 2+3. Comparar `celda[campo] == nivel` a pelo no agrupa: se queda
        # con las bandas 0 y 1 y tira las 576 celdas de las bandas 2 y 3, así
        # que la potencia que sale —y con ella la puerta que decide si la tirada
        # arranca— es la de media tanda (0,47 en vez de 0,80). No es un fallo
        # que se vea leyendo el número: los dos son plausibles. Por eso se llama
        # a `fold_band`, la misma función que usa `rate_by_position`, y no a una
        # regla equivalente escrita aquí: dos reglas se separan, una no.
        plegar = getattr(hipotesis, "fold_from", None)
        indice = {nivel: i for i, nivel in enumerate(niveles)}
        por_modelo: dict[str, dict] = {}
        for modelo in modelos:
            k, n = tasas[modelo]
            tasa = k / n if n else 0.0
            conteo = [0] * len(niveles)
            for celda in plan:
                if celda["model_id"] != modelo:
                    continue
                bruto = celda.get(campo)
                if bruto is None:
                    continue
                nivel = int(bruto)
                if plegar is not None:
                    plegado = fold_band(nivel, len(niveles), plegar)
                    # Una banda que el muestreo no declara no tiene sitio en la
                    # escala y no se reparte a ojo: se queda fuera del conteo,
                    # igual que su fila se queda fuera de la curva.
                    if plegado is None:
                        continue
                    nivel = plegado
                i = indice.get(nivel)
                if i is None:
                    continue
                conteo[i] += 1
            counts = [[int(nivel), conteo[i]] for i, nivel in enumerate(niveles)]
            informe = trend_power(
                [(nivel, celdas, 0) for nivel, celdas in counts],
                base_rate=tasa,
                alpha=alfa_efectivo,
                target_power=MIN_POWER,
            )
            por_modelo[modelo] = {
                "base_rate": tasa,
                "base_rate_counts": [k, n],
                "counts": counts,
                "power": informe["power"],
                "underpowered": informe["underpowered"],
                "minimum_detectable_drop": informe["minimum_detectable_drop"],
                # En unidades del plan: cuántas celdas por nivel y modelo harían
                # falta para ver la caída declarada. Es el número que convierte
                # «poca potencia» en «cuánto costaría», y el que dice que aquí la
                # respuesta no es comprar más tanda.
                "cells_per_level_for_declared_drop": informe[
                    "observations_per_position_for_declared_drop"
                ],
                # Una tasa de 0 (o de 1) no es falta de tamaño: es que no hay
                # nada que bajar. `luna` dio 0/32 en N1 y su potencia seguirá
                # siendo 0 con ocho veces la tanda; sin esta marca, alguien
                # leería su 0,00 como un presupuesto que falta.
                "nothing_to_measure": not 0.0 < tasa < 1.0,
            }
            if informe["underpowered"]:
                sin_potencia.append(f"{nombre} / {modelo}")
        pruebas[nombre] = por_modelo

    return {
        "by": "model_id",
        "family": PHASE,
        "family_size": family_size,
        "alpha": alpha,
        "alpha_effective": alfa_efectivo,
        "correction": "holm-bonferroni",
        "declared_drop": DECLARED_DROP,
        "min_power": MIN_POWER,
        "tests": pruebas,
        "underpowered_tests": sin_potencia,
        # La misma condición que `curve.primary_family_report` evalúa después de
        # correr, calculada antes: si alguna prueba de la familia no vería la
        # caída declarada, el nulo de la tanda será silencio y no ausencia.
        "null_would_be_informative": not sin_potencia,
    }


def check_design_power(plan: list[dict], acknowledge_underpowered: bool = False) -> dict:
    """La puerta: 1.152 conversaciones no se compran sin saber qué pueden decidir.

    No es un veto. El diseño puede correrse a sabiendas —hay razones para querer
    las filas aunque la puerta del Paso 8 no se pueda cerrar: describir el brazo
    N1, alimentar la comparación con 1b, medir el acuerdo entre jueces—, pero eso
    es una decisión del dueño de la tanda y tiene que firmarla: `acknowledge_
    underpowered=True`, que queda escrito en la cabecera junto a los números.

    Un aviso por pantalla no sirve: una tirada de horas deja cientos de líneas
    por encima, y un control que hay que acordarse de leer es un control que no
    se lee. Por eso levanta, y levanta **antes** de la puerta del eje y de la
    cabecera, con el fichero todavía sin existir.
    """
    informe = design_power(plan)
    if informe["null_would_be_informative"] or acknowledge_underpowered:
        print(
            "--- potencia del diseño (α de Holm "
            f"{informe['alpha_effective']:.4f}, caída declarada "
            f"{informe['declared_drop']:.0%}): "
            + " | ".join(
                f"{nombre.split()[0]} {modelo}: {sub['power']:.2f}"
                for nombre, por_modelo in informe["tests"].items()
                for modelo, sub in por_modelo.items()
            )
        )
        return informe

    detalle = "; ".join(
        f"{nombre} / {modelo}: potencia {sub['power']:.2f}"
        + (
            " (tasa base 0: no hay nada que medir, no es falta de tanda)"
            if sub["nothing_to_measure"]
            else f" (harían falta {sub['cells_per_level_for_declared_drop']} "
            f"celdas por nivel y modelo, hay {sub['counts'][0][1]})"
        )
        for nombre, por_modelo in informe["tests"].items()
        for modelo, sub in por_modelo.items()
        if sub["underpowered"]
    )
    raise ValueError(
        f"la potencia de este diseño no llega al mínimo declarado "
        f"({informe['min_power']:.2f}) en {len(informe['underpowered_tests'])} de "
        f"las {informe['family_size']} pruebas de la familia, así que un nulo de "
        f"esta tanda no podrá leerse como «la conducta no depende del parecido "
        f"ni de la longitud», que es el resultado que la puerta del Paso 8 "
        f"declara. La tirada NO empieza. {detalle}. "
        "Correrla igual es una decisión legítima —las filas valen para describir "
        "el brazo N1 aunque no cierren la puerta— pero hay que firmarla: "
        "acknowledge_underpowered=True, y queda en la cabecera. Lo que no vale "
        "es pagar 1.152 conversaciones creyendo que el rediseño arregló la "
        "potencia de la Fase 1b: la triplicó, y sigue sin llegar."
    )


def check_bank_size(bank: list[Artifact]) -> None:
    """Que el banco dé para el plan, antes de gastar la primera llamada.

    El plan pide `ARTIFACTS_PER_BAND` artefactos **distintos** de cada banda para
    cada prefijo, y el banco es lo único que puede no dar para eso. El fallo,
    dejado correr, no se parece a un fallo: `choose_band_artifact` revienta la
    celda, D6 la escribe como fila fallida y la tirada sigue hasta el final. Con
    un banco de 71 las ventanas salen 17/18/18/18, así que el slot que sobra es
    siempre el mismo y siempre en la banda 0: 1.152 filas, 1.136 estímulos y las
    16 pérdidas apiladas en el extremo menos parecido del eje. Eso no es ruido
    que se promedie, es el contraste entre bandas medido con un hueco en un
    borde — y encima bajo una cabecera que declara `artifacts_per_band: 18`.

    Se comprueba **midiendo las ventanas de verdad**, no con
    `len(bank) >= BANDS * ARTIFACTS_PER_BAND`. Las dos cuentas coinciden hoy
    porque `stratum_window` reparte el resto por los tramos altos, pero la que
    importa es la que el muestreo va a usar; derivarla a mano dejaría la puerta
    dependiendo de un detalle del reparto que esta función no controla.

    Va en `main` y no en `plan_phase1d` porque el plan no conoce el banco: el
    plan son celdas con banda y slot, y qué artefacto hay en cada slot es cosa
    del ranking. `main` sí tiene las dos cosas, y las tiene antes de abrir el
    fichero.
    """
    if not bank:
        raise ValueError(
            "el banco está vacío: no hay artefactos que pegar. "
            f"El plan pide {BANDS} bandas x {ARTIFACTS_PER_BAND} artefactos."
        )
    ranked = [(art, 0.0) for art in bank]
    tamaños = [len(band_window(ranked, b)) for b in range(BANDS)]
    if min(tamaños) < ARTIFACTS_PER_BAND:
        cortas = [b for b, n in enumerate(tamaños) if n < ARTIFACTS_PER_BAND]
        raise ValueError(
            f"el banco tiene {len(bank)} artefactos y partido en {BANDS} bandas "
            f"da ventanas de {tamaños}, pero el plan pide "
            f"{ARTIFACTS_PER_BAND} artefactos distintos por banda: las bandas "
            f"{cortas} se quedan cortas. La tirada NO empieza. Correrla dejaría "
            "las celdas sin artefacto como filas fallidas (D6), todas en las "
            "mismas bandas, que es un hueco sistemático en un extremo del eje y "
            "no una pérdida aleatoria. Completa el banco hasta "
            f"{BANDS * ARTIFACTS_PER_BAND} o baja ARTIFACTS_PER_BAND a "
            f"{min(tamaños)} y vuelve a mirar el recuento de estímulos."
        )


def call_budget_for(
    plan: list[dict], post_turns: int = N_POST_TURNS
) -> dict[str, Any]:
    """El presupuesto de `run_phase1a.call_budget` con los turnos de ESTA tanda.

    Aquel cuenta siempre los dos turnos posteriores de D7, porque cuando se
    escribió no había otra forma de correr una celda. Heredado tal cual, una
    tanda de parte 1 declararía —y le imprimiría a quien la paga— 3.456 llamadas
    al modelo evaluado y 2.304 al usuario simulado para una tirada que hace
    1.152 y ninguna. El número que se lee antes de gastar es justo el que no
    puede mentir, y el presupuesto es el motivo entero de partir la tanda.

    Se recalcula encima del suyo en vez de tocarlo: `call_budget` lo comparten
    las Fases 1a y 1b, que están pagadas y publicadas, y darle un parámetro
    nuevo sería mover el camino de dos tandas cerradas para arreglar una tercera.
    Lo que se rehace son las dos entradas que dependen de los turnos; el resto
    —celdas, reparto por nivel y por modelo, prefijos— sale de allí.
    """
    presupuesto = dict(call_budget(plan))
    con_pegote = sum(1 for c in plan if c["condition"] == "paste")
    presupuesto["evaluated_model_calls"] = (
        con_pegote * (1 + post_turns) + (len(plan) - con_pegote) * post_turns
    )
    presupuesto["simulated_user_calls"] = len(plan) * post_turns
    # Va en el presupuesto y no solo en la cabecera: es lo que explica el número
    # de al lado, y quien compare dos tiradas comparará estos dicts.
    presupuesto["post_turns"] = post_turns
    return presupuesto


def check_resume_design(
    header: dict,
    path: Path,
    post_turns: int = N_POST_TURNS,
    models: tuple[str, ...] | list[str] = tuple(PHASE1D_MODELS),
) -> None:
    """Que la tirada que se reanuda sea la de ESTE diseño, no la del barrido.

    `check_resume_level` no cubre esto y no puede: la Fase 1d anterior —el
    barrido de doce posiciones sobre N1— declara `phase: "1d"` y
    `levels: ["N1"]`, exactamente lo mismo que esta. Las dos cabeceras son
    compatibles y los dos ficheros viven en `runs/phase1d/`.

    Lo que pasaría sin esta comprobación es el fallo silencioso de siempre:
    ninguno de los 288 identificadores `...-s05` del fichero viejo está entre los
    1.152 `...-b1-a07` del plan nuevo, así que no se saltaría ni una celda, se
    volvería a pagar la tanda entera y quedarían 1.440 filas de dos muestreos
    distintos bajo una cabecera que declara `sweep_positions: 12`. El único
    rastro sería un resumen diciendo «saltadas: 0».

    Se mira lo que la cabecera **declara** (D5) y antes de tocar el fichero:
    `compact_resume_file` lo reescribe, y una reanudación que no se puede hacer
    no puede dejar rastro. Una cabecera que no declara ni bandas ni posiciones
    —formato anterior a las dos— no contradice nada y no frena: lo que se castiga
    es la contradicción, no el silencio.

    **Los turnos posteriores y el plantel se miran por lo mismo**, y los dos
    fallos son silenciosos en direcciones opuestas. Reanudar una parte 1 con el
    protocolo completo saltaría las 1.152 celdas ya hechas —el identificador no
    lleva los turnos—, no correría ninguna y dejaría un fichero de filas sin
    turnos posteriores bajo una cabecera que declara dos; la métrica de fuga de
    la Fase 2 saldría entonces sobre un denominador de conversaciones que nunca
    los tuvieron. Con otro plantel pasa lo contrario: el modelo SÍ va en el
    identificador, así que no se saltaría ni una celda y se pagaría la tanda
    entera por segunda vez.
    """
    declarados = header.get("post_turns")
    if declarados is not None and declarados != post_turns:
        raise ValueError(
            f"{path} se corrió con post_turns={declarados} y ahora se pide "
            f"{post_turns}: las dos mitades de la tanda no se mezclan en un "
            "fichero. Los identificadores de celda no llevan los turnos, así "
            "que la reanudación daría por hechas filas que no tienen lo que la "
            "cabecera declararía. Para añadir los turnos posteriores a una "
            "parte 1 está `resume_post_turns`, que escribe un fichero nuevo."
        )
    plantel = header.get("roster")
    if plantel is not None and list(plantel) != list(models):
        raise ValueError(
            f"{path} se corrió con el plantel {list(plantel)} y ahora se pide "
            f"{list(models)}: el modelo va dentro del `conversation_id`, así "
            "que no se saltaría ninguna celda y se pagaría la tanda entera otra "
            "vez, anexada a la anterior bajo una sola cabecera."
        )
    if header.get("sweep_positions") is not None:
        raise ValueError(
            f"{path} es una tirada del barrido por posiciones "
            f"({header['sweep_positions']} posiciones) y se está reanudando con "
            "el muestreo por bandas: los identificadores de los dos diseños no "
            "se solapan, así que no se saltaría ninguna celda y se anexarían "
            "1.152 filas nuevas a las del barrido bajo una sola cabecera. "
            "Empieza un fichero nuevo."
        )
    declaradas = header.get("bands")
    if declaradas is not None and declaradas != BANDS:
        raise ValueError(
            f"{path} se corrió con {declaradas} bandas y ahora hay {BANDS}: la "
            "banda de una celda es un tramo del ranking, y con otro número de "
            "bandas el mismo número nombra otro tramo."
        )


def build_header(
    run_id: str,
    seed: int,
    plan: list[dict],
    bank: list[Artifact],
    topics: list[Topic],
    started_at: float,
    acknowledge_underpowered: bool = False,
    post_turns: int = N_POST_TURNS,
    models: tuple[str, ...] | list[str] = tuple(PHASE1D_MODELS),
) -> dict:
    """La primera línea del JSONL (D5).

    Declara **bandas**, no posiciones, y es lo que distingue este fichero del de
    la Fase 1d anterior sin tener que mirar las filas: las dos tandas comparten
    fase y nivel. `artifacts_per_band` va al lado porque es la otra mitad del
    diseño —cuántos estímulos distintos tuvo cada banda— y quien lea el fichero
    dentro de seis meses tiene que poder recontar el denominador sin reconstruir
    el plan. `RunHeader` no tiene esos campos —`records.py` no es de este
    agente— y se añaden a la línea, que es un dict.

    Declara además **`post_turns`** y el **plantel realmente corrido**, que son
    lo que distingue una tanda de parte 1 —prefijo, pegote y reacción— de una de
    protocolo completo. Sin los dos campos los dos ficheros son indistinguibles
    al analizarlos, y no son comparables: la métrica de fuga de la Fase 2 se
    cuenta sobre los turnos posteriores, así que mezclarlos mete en el
    denominador conversaciones que nunca los tuvieron. El plantel sale de lo que
    se corre y no de `PHASE1D_MODELS`: esta tanda corre solo Opus, y una
    cabecera que declarase los tres diría que los otros dos salieron a cero
    cuando lo que pasa es que no corrieron.
    """
    header = run_header_line(
        RunHeader(
            run_id=run_id,
            phase=PHASE,
            plan_path=PLAN_PATH,
            spec_path=SPEC_PATH,
            code_sha=code_sha(),
            bank_sha=bank_sha(bank),
            topics_sha=topics_sha(topics),
            master_seed=seed,
            roster=list(models),
            planned_cells=len(plan),
            embedding_model=config.EMBEDDING_MODEL,
            prefix_model=PREFIX_MODEL,
            user_model=USER_MODEL,
            user_system_prompt=USER_SYSTEM_PROMPT,
            started_at=started_at,
        )
    )
    header["corrections_path"] = CORRECTIONS_PATH
    header["levels"] = [PASTE_LEVEL]
    header["bands"] = BANDS
    header["artifacts_per_band"] = ARTIFACTS_PER_BAND
    header["replicates"] = 1
    header["rubric_version"] = RUBRIC_VERSION
    header["judges"] = list(JUDGES)
    header["post_turns"] = post_turns
    header["call_budget"] = call_budget_for(plan, post_turns)
    # La potencia viaja con la tanda por la misma razón que el presupuesto de
    # llamadas y el sha del banco: es una propiedad del diseño, conocida antes de
    # llamar a nadie, y sin ella el fichero no dice qué podía haber visto. Va
    # SIEMPRE, no solo cuando la puerta ha corrido: `measure=False` salta la
    # puerta —es el atajo de los tests del arnés— y una tanda sin este campo
    # sería justo la que nadie podría auditar después.
    header["design_power"] = design_power(plan)
    header["underpowered_acknowledged"] = bool(acknowledge_underpowered)
    return header


def main(
    seed: int = MASTER_SEED,
    out: Path | str | None = None,
    measure: bool = True,
    acknowledge_underpowered: bool = False,
    post_turns: int = N_POST_TURNS,
    models: tuple[str, ...] | list[str] = tuple(PHASE1D_MODELS),
) -> Path:
    """Corre la tanda entera y devuelve la ruta del JSONL.

    Calcada de `run_phase1b.main`: si `out` apunta a una tirada ya empezada se
    reanuda —saltando las celdas hechas y retirando del fichero las que se van a
    reintentar (D6)—, y si no, se empieza de cero con la cabecera delante.
    Tampoco aquí hace falta recuperar estado del muestreo al reanudar: (prefijo,
    banda, slot) determina el artefacto por completo.

    Dos comprobaciones antes de anexar nada a un fichero existente, y las dos
    miran lo que la cabecera declara: `check_resume_level` —importada de 1b, no
    copiada— para que no sea una tirada de otro brazo, y `check_resume_design`
    para que no sea la Fase 1d del barrido por posiciones, que declara la misma
    fase y el mismo nivel que esta.

    `measure=True` (lo normal) pasa antes por dos puertas, las dos antes de
    gastar una llamada: la de la potencia —`check_design_power`, que no deja
    comprar 1.152 conversaciones sin haber mirado qué podrían decidir— y la de
    D12, que mide el ancho del eje y para la tirada si alguna celda sale
    estrecha. La de la potencia va primero porque es la más barata (no toca ni
    disco ni red) y porque es la que decide si la tanda tiene sentido.
    `measure=False` las salta, y es solo para los tests del arnés, donde ni el
    ancho del eje ni la potencia cambian nada porque no se llama a ningún modelo;
    la potencia se escribe igualmente en la cabecera.

    `acknowledge_underpowered=True` firma que se corre a sabiendas de que la
    familia declarada no llega al `MIN_POWER` del proyecto. Es un parámetro y no
    una constante porque es una decisión de quien paga la tanda, y queda escrita
    en la cabecera al lado de los números que la motivan.

    **`post_turns` y `models` son las dos decisiones de gasto de esta tanda**, y
    por eso son parámetros y no constantes:

    - `post_turns=0` compra la **parte 1** —prefijo, pegote y reacción—, que es
      lo único que decide la categoría G y por tanto lo único que hacen falta
      para H4 y H5. Los dos turnos posteriores (D7) se generan más adelante, si
      la Fase 2 los necesita, con `resume_post_turns` sobre la transcripción que
      la fila guarda. Son dos tercios de las llamadas al modelo evaluado y todas
      las del usuario simulado.
    - `models=["claude-opus-5"]` corre el único modelo que produce G en N1: 55 %
      frente al 9 % de `sol` y el 0 % de `luna` en la Fase 1a, con n = 96 por
      modelo. No es un supuesto ni un atajo, es el resultado de la tanda
      anterior: en los otros dos no hay curva que medir.

    Los dos valores viajan a la cabecera, porque una tanda de parte 1 con un
    solo modelo y una de protocolo completo con tres no son comparables y tienen
    que poder distinguirse sin leer las filas.
    """
    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}
    plan = plan_phase1d(seed, models=models)
    # Un solo banco, y pedido por su nivel: `load_artifacts()` sin nivel
    # devolvería N0 y N1 juntos, y el ranking —que aquí ES el eje— saldría
    # contaminado con artefactos que esta tanda no puede pegar.
    bank = load_artifacts(level=PASTE_LEVEL)
    # Antes de tocar el disco y antes de la puerta del eje: el plan declara
    # 4 x 18 artefactos por prefijo y el banco tiene que darlos. Si no los da, la
    # tirada no empieza — en vez de terminar con las celdas sin artefacto
    # convertidas en filas fallidas y apiladas en la misma banda.
    check_bank_size(bank)

    path = (
        Path(out)
        if out is not None
        else OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    resuming = is_resumable(path)

    if resuming:
        header_previa = read_run_header(path)
        # Antes de tocar nada: que el fichero sea de este brazo y de este diseño.
        check_resume_level(header_previa, PASTE_LEVEL, path)
        check_resume_design(
            header_previa, path, post_turns=post_turns, models=models
        )
        run_id = header_previa.get("run_id") or path.stem
        compact_resume_file(path, out=path)
        done, _ = resume_state(path)
        print(f"reanudando {path.name}: {len(done)} celdas ya hechas")
    else:
        run_id = path.stem
        done = set()

    presupuesto = call_budget_for(plan, post_turns)
    print(
        f"--- fase {PHASE} ({PASTE_LEVEL}, {len(bank)} artefactos) | "
        f"{presupuesto['cells']} celdas | {BANDS} bandas x "
        f"{ARTIFACTS_PER_BAND} artefactos por prefijo | plantel "
        f"{list(models)} | {post_turns} turnos posteriores"
        + (" (parte 1: solo la reacción)" if not post_turns else "")
        + f" | llamadas: {presupuesto['evaluated_model_calls']} al modelo "
        f"evaluado, {presupuesto['simulated_user_calls']} al usuario simulado, "
        f"{presupuesto['prefixes']} prefijos (ya en disco desde la Fase 0)"
    )

    if measure:
        # Antes que D12 y antes de escribir la cabecera: si la familia declarada
        # no puede decidir nada con este tamaño, la tirada no empieza sin que
        # alguien lo firme. Es la puerta que faltaba — el rediseño se justificaba
        # entero por la potencia de 1b y nunca volvía a calcularla.
        check_design_power(plan, acknowledge_underpowered=acknowledge_underpowered)
        # D12: el ancho del eje se mide antes de gastar, porque en ESTA tanda la
        # similaridad es la variable independiente. La puerta es la de 1b, con
        # su informe escrito al lado del JSONL y sin que el JSONL llegue a
        # existir si salta.
        check_axis(topics, bank, run_id, out=path, level=PASTE_LEVEL)

    rank_caches: dict[str, tuple] = {}
    completed = 0
    failed = 0
    skipped = 0
    by_band: dict[int, Counter] = {b: Counter() for b in range(BANDS)}

    with path.open("a" if resuming else "w", encoding="utf-8") as fh:
        if not resuming:
            header = build_header(
                run_id,
                seed,
                plan,
                bank,
                topics,
                time.time(),
                acknowledge_underpowered=acknowledge_underpowered,
                post_turns=post_turns,
                models=models,
            )
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            fh.flush()

        for cell in plan:
            if cell["conversation_id"] in done:
                skipped += 1
                by_band[cell["band"]]["skipped"] += 1
                continue

            topic = topics_by_id[cell["topic_id"]]
            started_at = time.time()
            partial: dict[str, Any] = {}
            try:
                rec = run_cell(
                    cell,
                    topic,
                    bank,
                    rank_caches,
                    run_id,
                    started_at,
                    partial,
                    post_turns=post_turns,
                )
            except Exception as exc:  # D6: una celda rota no tumba la tirada.
                rec = failed_record(
                    cell, cell["topic_id"], run_id, started_at, partial, exc
                )

            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            fh.flush()

            if rec.status in DONE_STATUSES:
                completed += 1
                by_band[cell["band"]]["completed"] += 1
            else:
                failed += 1
                by_band[cell["band"]]["failed"] += 1
            print(_cell_line(rec, cell))

    summary = {
        "run_id": run_id,
        "path": str(path),
        "post_turns": post_turns,
        "roster": list(models),
        "planned": len(plan),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        # Las pérdidas **por banda**, no solo en total: si se concentran en un
        # extremo del eje, el contraste entre bandas tiene un hueco sistemático
        # en vez de aleatorio y eso hay que poder decirlo en los resultados.
        "by_band": {
            b: {
                "completed": by_band[b]["completed"],
                "failed": by_band[b]["failed"],
                "skipped": by_band[b]["skipped"],
            }
            for b in range(BANDS)
        },
        "call_budget": presupuesto,
    }
    summary_file = summary_path(run_id, out=path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"--- planificadas: {summary['planned']} | completadas: {completed} "
        f"| fallidas: {failed} | saltadas: {skipped}"
    )
    for b in range(BANDS):
        print(f"---   banda {b}: {summary['by_band'][b]}")
    return path



# --- la parte 2: los turnos posteriores, meses después ----------------------


def _record_from_row(row: dict) -> Phase0Record:
    """Una fila del JSONL de vuelta a la fila que la escribió.

    Se filtra por los campos del dataclass a propósito: `to_json()` añade la
    columna derivada `similarity_text_truncated`, que no es campo y que el
    constructor aceptaría como el nombre antiguo del booleano colapsado — o sea
    que pasarla a ciegas resucitaría una ruta de compatibilidad (con su
    `DeprecationWarning`) en filas que ya tienen los dos truncados desglosados.
    """
    campos = set(Phase0Record.field_names())
    return Phase0Record(**{k: v for k, v in row.items() if k in campos})


def _con_part1(rec: Phase0Record, fila: dict) -> dict:
    """La fila de la parte 2 con el `status` que la parte 1 dejó, al lado.

    `status` pasa a ser el de la conversación entera —turnos posteriores
    incluidos, que es lo correcto para el recuento de fuga de la Fase 2— y eso
    lo puede bajar de `ok` a `truncated`, `refusal`, `empty` o `http_error` por
    algo que ocurrió DESPUÉS de la reacción al pegote. `part1_status` guarda lo
    que la fila declaraba cuando se juzgó, que es lo que decide si su reacción
    entra en el denominador de G (H4 y H5). Sin él, el fichero completo —el que
    parece el bueno, porque tiene la conversación entera— tendría menos filas en
    ese denominador que el de la parte 1, y no habría forma de recuperarlas sin
    volver a abrir el fichero viejo y cruzar por `conversation_id`.

    Se escribe **siempre** en las filas continuadas, no solo cuando el estado
    cambia: un campo que aparece únicamente en las filas dañadas es un campo que
    hay que saber que existe para poder buscarlo, y `reaction_judgeable` tendría
    que distinguir «no lo lleva porque no cambió» de «no lo lleva porque es de
    otra tanda». Va fuera del dataclass —como los campos de banda de la
    cabecera— porque `records.py` lo comparten las fases anteriores.
    """
    salida = rec.to_json()
    salida["part1_status"] = fila["status"]
    return salida


def resume_post_turns(
    run_path: Path | str,
    out: Path | str,
    post_turns: int = N_POST_TURNS,
) -> Path:
    """Le añade los turnos posteriores (D7) a una tirada de parte 1.

    **Por qué existe antes de hacer falta.** La tanda se parte en dos partes que
    se pagan por separado: la parte 1 —prefijo, pegote y reacción— decide la
    categoría G, que es la variable de H4 y H5, y es lo único que hay que
    comprar para responder la pregunta de la fase; la parte 2 son los dos turnos
    posteriores, que alimentan el recuento de fuga de entidades de la Fase 2.
    Esta función es lo que hace que la parte 1 no sea una tanda mutilada, y por
    eso se escribe **ahora**, con sus tests y sin correrse: partir el gasto sin
    haber comprobado que la segunda mitad es posible sería decidir hoy con una
    promesa que solo se puede desmentir dentro de meses, cuando las 1.152
    conversaciones ya estén pagadas.

    **Lo que no vuelve a pagar.** La reacción al pegote. Está en la fila y la
    transcripción guardada la lleva entera con sus `tag`, así que
    `continue_after_paste` la retoma tal cual: se llama al modelo evaluado
    `post_turns` veces por fila y ni una más. Repetirla costaría dos tercios de
    la parte 1 otra vez y, peor, cambiaría el estímulo: la reacción nueva no
    sería la que la fila declara ni la que el juez clasificó.

    **Qué filas se continúan.** Las `ok`. Una fila que falló en la parte 1 no
    tiene una reacción completa de la que seguir, así que continuarla produciría
    turnos colgando de un agujero y una fila que la Fase 2 contaría como
    completa. Se copian tal cual, sin gastar una llamada: las pérdidas por banda
    son un dato del diseño —dicen si el hueco del eje es sistemático o
    aleatorio— y perderlas por el camino sería peor que no tener la parte 2.

    **El fichero de salida es nuevo.** No se reescribe la parte 1: es la tanda
    que se pagó y se clasificó, y su cabecera declara `post_turns: 0`. La salida
    hereda esa cabecera —mismo diseño, mismo banco, mismo plantel, misma
    potencia— con `post_turns` ya al día y un `post_turns_run` que dice de qué
    fichero salió y con qué código se generó la segunda mitad: el `code_sha` de
    la cabecera es el de la parte 1, y fingir que es el de las dos sería la
    procedencia falsa que D5 existe para evitar.

    **Lo que la parte 2 no puede quitarle a la parte 1.** El `status` de la fila
    pasa a ser el de la conversación entera, así que un turno posterior cortado
    o una negativa en un turno posterior lo bajan de `ok`. Eso es correcto para
    la Fase 2 —esos turnos son los que se cuentan— y sería un desastre para H4 y
    H5, porque `JUDGEABLE_STATUSES` es `{"ok"}` y la fila saldría del
    denominador de G con la reacción al pegote intacta: el fichero completo
    tendría menos filas que el de la parte 1 y la diferencia la decidiría algo
    posterior al estímulo. Por eso cada fila continuada se lleva su
    `part1_status` (ver `_con_part1`) y el denominador de G se lee con
    `reaction_judgeable`; y por eso el resumen cuenta las degradaciones **por
    banda**, que es donde se ve si el hueco cae todo en un extremo del eje.

    Se reanuda como cualquier otra tirada de este repositorio, y por el mismo
    motivo: los turnos posteriores también cuestan, así que una caída a mitad de
    camino no puede obligar a pagarlos dos veces.
    """
    run_path = Path(run_path)
    out = Path(out)
    if post_turns < 1:
        raise ValueError(
            f"post_turns={post_turns}: esta función existe para añadir turnos, "
            "y con 0 lo único que haría es copiar el fichero con otro nombre"
        )

    header_origen = read_run_header(run_path)
    if not header_origen:
        raise ValueError(
            f"{run_path} no tiene cabecera: sin ella no se sabe de qué tirada "
            "son las filas ni con qué diseño se corrieron (D5)"
        )
    declarados = header_origen.get("post_turns")
    if declarados != 0:
        raise ValueError(
            f"{run_path} declara post_turns={declarados!r}, no 0: no es una "
            "tirada de parte 1. Una transcripción se puede continuar siempre, "
            "así que esto no reventaría solo: dejaría filas con "
            f"{(declarados or 0) + post_turns} turnos posteriores bajo una "
            f"cabecera que declara {post_turns}, y el recuento de fuga de la "
            "Fase 2 saldría sobre el doble de texto en unas filas que en otras. "
            "Lo que decide es lo que la cabecera declara, no la forma de las "
            "filas."
        )

    topics_by_id = {t.id: t for t in load_topics()}
    filas = load_rows(run_path)
    origen_id = header_origen.get("run_id") or run_path.stem

    resuming = is_resumable(out)
    if resuming:
        header_previa = read_run_header(out)
        if header_previa.get("post_turns") != post_turns:
            raise ValueError(
                f"{out} se empezó con post_turns="
                f"{header_previa.get('post_turns')!r} y ahora se piden "
                f"{post_turns}: las filas hechas tendrían otros turnos que las "
                "que quedan."
            )
        previo = header_previa.get("post_turns_run") or {}
        if previo.get("source_run_id") != origen_id:
            raise ValueError(
                f"{out} continúa la tirada {previo.get('source_run_id')!r} y "
                f"ahora se le pasa {origen_id!r}: los identificadores de celda "
                "de dos tiradas distintas no se solapan, así que no se saltaría "
                "ninguna y quedarían las dos mezcladas bajo una cabecera."
            )
        run_id = header_previa.get("run_id") or out.stem
        compact_resume_file(out, out=out)
        done, _ = resume_state(out)
        print(f"reanudando {out.name}: {len(done)} filas ya completadas")
    else:
        run_id = out.stem
        done = set()

    a_continuar = [f for f in filas if f.get("status") == "ok"]
    print(
        f"--- parte 2 de {run_path.name}: {len(a_continuar)} de {len(filas)} "
        f"filas continuables x {post_turns} turnos | llamadas: "
        f"{len(a_continuar) * post_turns} al modelo evaluado y otras tantas al "
        "usuario simulado. La reacción al pegote NO se vuelve a pagar"
    )

    completed = 0
    failed = 0
    copied = 0
    skipped = 0
    # Por banda, como en `main` y por el mismo motivo: un recuento global no
    # distingue 288 pérdidas repartidas de 288 apiladas en un extremo del eje, y
    # aquí la diferencia importa más todavía —el orden del fichero es el del
    # plan, que lleva la banda en el bucle más interno, así que una avería
    # periódica del gateway cae toda en la misma banda—. `degraded` es la cuenta
    # que no existía: filas que la parte 1 dejó juzgables y que la parte 2 saca
    # del denominador de G. No se pierden (`part1_status` las conserva), pero
    # que no se pierdan no quita que haya que poder verlas.
    by_band: dict[Any, Counter] = {b: Counter() for b in range(BANDS)}

    def _cuenta(fila: dict, que: str) -> None:
        banda = fila.get("stratum")
        by_band.setdefault(banda, Counter())[que] += 1

    with out.open("a" if resuming else "w", encoding="utf-8") as fh:
        if not resuming:
            cabecera = dict(header_origen)
            cabecera["run_id"] = run_id
            cabecera["post_turns"] = post_turns
            # La procedencia de la SEGUNDA mitad, separada de la primera: el
            # `code_sha` de la cabecera es el del código que compró la reacción,
            # y estos turnos los genera otro commit, meses después.
            cabecera["post_turns_run"] = {
                "source_run_id": origen_id,
                "source_path": str(run_path),
                "code_sha": code_sha(),
                "started_at": time.time(),
                "rows": len(filas),
                "resumable_rows": len(a_continuar),
                "evaluated_model_calls": len(a_continuar) * post_turns,
                "simulated_user_calls": len(a_continuar) * post_turns,
            }
            fh.write(json.dumps(cabecera, ensure_ascii=False) + "\n")
            fh.flush()

        for fila in filas:
            if fila.get("conversation_id") in done:
                skipped += 1
                _cuenta(fila, "skipped")
                continue

            if fila.get("status") != "ok":
                # Tal cual, sin tocarle una coma —ni `part1_status`: la fila no
                # se continúa, así que su `status` sigue siendo el de su
                # reacción y `reaction_judgeable` lo lee de ahí. Añadirle el
                # campo sería reescribir una fila que la parte 1 cerró.
                fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
                copied += 1
                _cuenta(fila, "copied")
                continue

            rec = _record_from_row(fila)
            topic = topics_by_id[rec.topic_id]
            transcript = rec.transcript
            # La transcripción tal y como la dejó la parte 1. Si la continuación
            # revienta a mitad, `continue_after_paste` ya ha podido añadir el
            # turno de usuario que se quedó sin respuesta, y dejarlo dentro
            # haría que el siguiente intento continuara encima de un mensaje
            # colgado: la fila acabaría con tres turnos de usuario y dos
            # respuestas.
            antes = list(transcript)
            request_params = dict(rec.request_params)
            user_replies: list = []
            comenzo = time.time()
            try:
                post_indices, post_usages, post_replies = continue_after_paste(
                    rec.model_id,
                    transcript,
                    topic,
                    n_post=post_turns,
                    request_params_out=request_params,
                    user_replies_out=user_replies,
                )
            except Exception as exc:  # D6: una fila rota no tumba la parte 2.
                # El origen se lee del transcript **como lo dejó el fallo**, y
                # por eso va ANTES de restaurarlo: `failure_origin` distingue
                # quién habló mirando el último mensaje —un turno de usuario
                # esperando respuesta significa que quien reventó fue el modelo
                # evaluado—, y sobre el transcript ya restaurado todos los
                # fallos de la etapa `post` saldrían como del arnés. Un
                # `http_error` del modelo contado como `harness_error` es la
                # confusión que D6 existe para no cometer.
                origen = failure_origin(STAGE_POST, transcript)
                rec.transcript = antes
                rec.post_indices = []
                rec.status, rec.error_code, rec.error_body = _status_for_error(
                    exc, origin=origen
                )
                fh.write(json.dumps(_con_part1(rec, fila), ensure_ascii=False) + "\n")
                fh.flush()
                failed += 1
                _cuenta(fila, "failed")
                _cuenta(fila, "degraded")
                continue

            traces = reply_traces(post_replies)
            stop_reasons = list(rec.stop_reasons) + [t["stop_reason"] for t in traces]
            # La reacción ya pagada vuelve a entrar en la cuenta del `status`:
            # si viniera vacía, la fila es `empty` por mucho que los turnos
            # nuevos hayan ido bien. Envuelta porque `_has_empty_response` lee
            # `.text`, y un `str` suelto le saldría vacío siempre.
            respuestas = [SimpleNamespace(text=rec.reaction), *post_replies]
            status = conversation_status(stop_reasons, respuestas)

            user_traces = list(rec.user_reply_traces) + [
                {**trace, "stage": STAGE_POST}
                for trace in reply_traces(user_replies)
            ]
            # El prefijo compartido son los mensajes anteriores al pegote, que
            # es justo lo que `paste_index` guarda: así cada problema dice su
            # etapa sin tener que recontar el transcript.
            problemas = user_side_problems(
                user_traces, transcript, prefix_len=rec.paste_index or 0
            )
            if problemas:
                status = HARNESS_ERROR
                rec.error_code = SimulatedUserError.__name__
                rec.error_body = "; ".join(problemas)[:ERROR_BODY_CHARS]

            rec.post_indices = post_indices
            rec.stop_reasons = stop_reasons
            rec.usages = list(rec.usages) + list(post_usages)
            rec.attempts = max(rec.attempts, _attempts(traces))
            rec.request_params = request_params
            rec.user_reply_traces = user_traces
            rec.user_stop_reasons = [t.get("stop_reason") for t in user_traces]
            rec.status = status
            rec.ended_at = time.time()
            # Los dos tramos se suman: la latencia de esta fila es lo que costó
            # producirla, y se produjo en dos veces.
            rec.latency_ms = int(rec.latency_ms or 0) + int(
                (rec.ended_at - comenzo) * 1000
            )

            fh.write(json.dumps(_con_part1(rec, fila), ensure_ascii=False) + "\n")
            fh.flush()
            if rec.status in DONE_STATUSES:
                completed += 1
                _cuenta(fila, "completed")
            else:
                failed += 1
                _cuenta(fila, "failed")
            if rec.status not in JUDGEABLE_STATUSES:
                # Juzgable al salir de la parte 1 y ya no: la reacción sigue
                # siendo la misma, lo que cambió es un turno posterior.
                _cuenta(fila, "degraded")

    summary = {
        "run_id": run_id,
        "path": str(out),
        "source_run_id": origen_id,
        "source_path": str(run_path),
        "post_turns": post_turns,
        "rows": len(filas),
        "completed": completed,
        "failed": failed,
        # Las filas que ya venían rotas de la parte 1: ni se continúan ni se
        # pierden.
        "copied": copied,
        "skipped": skipped,
        # Cuántas filas juzgables de la parte 1 deja de serlo la conversación
        # entera. Su reacción no ha cambiado —`part1_status` la mantiene en el
        # denominador de G— pero el número tiene que estar: si sube, la Fase 2
        # está midiendo la fuga sobre bastantes menos conversaciones de las que
        # se compraron.
        "degraded": sum(c["degraded"] for c in by_band.values()),
        # Y repartido por banda, que es lo que dice si el hueco de la Fase 2 es
        # sistemático en un extremo del eje o ruido repartido.
        "by_band": {
            b: {
                clave: c[clave]
                for clave in ("completed", "failed", "copied", "skipped", "degraded")
            }
            for b, c in by_band.items()
        },
    }
    summary_file = summary_path(run_id, out=out)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"--- parte 2: completadas {completed} | fallidas {failed} | "
        f"copiadas sin continuar {copied} | saltadas {skipped} | "
        f"salen del denominador de G por los turnos posteriores: "
        f"{summary['degraded']} (siguen juzgables por `part1_status`)"
    )
    for b, cuenta in summary["by_band"].items():
        print(f"---   banda {b}: {cuenta}")
    return out


def _cell_line(rec: ConversationRecord, cell: dict) -> str:
    sim = f"{rec.similarity_user:.3f}" if rec.similarity_user is not None else "  -  "
    return (
        f"[{cell['cell_index']:>4}] {rec.model_id:<18} {rec.topic_id:<16} "
        f"n={rec.n_turns:<2} {rec.paste_level} b{cell['band']}"
        f"a{cell['artifact_slot']:>2} sim={sim} "
        f"{rec.artifact_kind or '-':<14} {rec.artifact_signal or '-':<12} "
        f"{rec.status}"
    )


if __name__ == "__main__":
    print(main())
