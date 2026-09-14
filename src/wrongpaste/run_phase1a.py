"""Runner de la Fase 1a: los tres niveles de pegote sobre los tres modelos.

288 celdas = 3 niveles × 8 temas × 2 longitudes × 2 réplicas × 3 modelos.

Casi todo lo que hace este fichero lo hace ya `run_phase0.py`, y por eso lo
importa en vez de copiarlo: la cabecera de tirada (D5), el `try`/`except` por
celda con `status` (D6), la reanudación por `conversation_id`, el resumen final
y `measure_axis` (D12) son exactamente los mismos. Lo que cambia es esto y solo
esto:

1. **El plan lleva `paste_level` y `replicate_idx`.** El nivel es el factor del
   experimento (§5 del spec de la Fase 1), así que viaja en el dict del plan y
   en la fila; no se deriva del `artifact_id` ni del banco del que salió.
2. **La elección del artefacto depende del nivel.** N0 y N1 se muestrean de su
   banco por estrato de similaridad, como en la Fase 0. N2 no sale de ningún
   banco: lo fabrica `contradictions.ensure_contradiction` contra el prefijo ya
   construido, y se persiste.
3. **Las dos réplicas de una celda comparten prefijo Y artefacto** (D1). La
   semilla del artefacto se deriva de `(prefix_id, paste_level, stratum)` —lo
   que las dos réplicas tienen en común— y **no** del `seed` de la celda, que es
   justo lo que las distingue. El docstring de `plan_phase0` dejó esto apuntado
   («al añadir réplicas, el artefacto tiene que pasar a ser función de
   `(prefix_id, stratum)`»); aquí es donde se cumple, porque aquí es donde hay
   réplicas. Una réplica que pega otro artefacto no mide el muestreo de la
   respuesta: mide otra cosa y no se sabe cuál.

**N2 no se compara nunca con N0 ni con N1** (§5 del spec de la Fase 1). Se
fabrica sabiendo de qué va la conversación, así que rompe a propósito la
independencia del §4.1 del spec original. Corre en la misma tirada porque
comparte prefijos, arnés y jueces; no porque entre en las mismas tablas.

**El ancho del eje (D12) aquí se mide pero no frena.** En la Fase 0 —y en la
Fase 1b— la similaridad **es** la variable independiente, y un eje estrecho deja
la tirada sin pregunta que responder: por eso allí la puerta para la tirada. En
la Fase 1a la variable independiente es el **nivel**; el estrato solo reparte la
elección del artefacto a lo largo del banco para que el brazo N0 sea comparable
con la Fase 0 y con 1b. Frenar la Fase 1a porque el banco N1 tenga un rango de
coseno corto sería aplicar un criterio que no gobierna esta pregunta. Se mide,
se escribe en `axis-<run_id>.json` y se avisa en voz alta; quien planifique 1b
lo mirará ahí, que es para quien está.
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wrongpaste import config, records
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.contradictions import CONTRADICTION_MODEL, ensure_contradiction
from wrongpaste.conversation import (
    MAX_TOKENS,
    N_POST_TURNS,
    continue_after_paste,
    inject_paste,
    reply_traces,
)
from wrongpaste.judging import JUDGES
from wrongpaste.prefixes import PREFIX_MODEL, ensure_prefix
from wrongpaste.records import (
    ARM_NO_REPAIR,
    ConversationRecord,
    RunHeader,
    run_header_line,
)
from wrongpaste.rubric import RUBRIC_VERSION

# La maquinaria de la Fase 0, reutilizada tal cual. Lo que se importa con
# guion bajo es privado de aquel módulo y se toma prestado a propósito: la
# alternativa era copiarlo, y dos copias de la clasificación de fallos (D6)
# divergen en cuanto alguien arregle una de las dos.
from wrongpaste.run_phase0 import (
    CORRECTIONS_PATH,
    DONE_STATUSES,
    ERROR_BODY_CHARS,
    HARNESS_ERROR,
    NARROW_RANGE,
    STAGE_PASTE,
    STAGE_POST,
    STAGE_PREFIX,
    STAGE_RANK,
    STRATA,
    STRATUM_SHIFTS,
    USER_SYSTEM_PROMPT,
    Phase0Record,
    _attempts,
    _has_empty_response,
    _model_label,
    _response_model,
    _system_prompt,
    axis_path,
    bank_sha,
    code_sha,
    compact_resume_file,
    is_resumable,
    measure_axis,
    pick_artifact,
    rank_for_prefix,
    read_run_header,
    resume_state,
    status_for_stop_reasons,
    summary_path,
    topics_sha,
    user_call_traces,
    user_side_problems,
)
from wrongpaste.run_phase0 import failed_record as phase0_failed_record
from wrongpaste.simulated_user import USER_MODEL, SimulatedUserError
from wrongpaste.topics import Topic, load_topics

PHASE1A_MODELS = ["gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5"]
LEVELS: tuple[str, ...] = ("N0", "N1", "N2")
LENGTHS = [2, 10]
REPLICATES = 2
MASTER_SEED = 20260914

# Los niveles que salen de un banco en disco. N2 no está: se fabrica contra
# cada prefijo (ver `contradictions.py`), así que no hay banco del que
# muestrear ni estrato que asignar.
BANK_LEVELS: tuple[str, ...] = ("N0", "N1")
CONTRADICTION_LEVEL = "N2"

# Estrato de una fila que no muestreó nada de ningún banco. Es la misma
# convención que usa el brazo de control de la Fase 0 (D11): escribir un número
# de estrato en una fila N2 sería inventar un dato, porque ninguna ventana del
# ranking intervino en elegir ese pegote.
NO_STRATUM = -1

# La etapa en la que se fabrica la contradicción. Se llama `rank` a propósito:
# es una etapa del **arnés** —la paga `CONTRADICTION_MODEL`, no el modelo
# evaluado— y `run_phase0.failure_origin` decide eso por el nombre de la etapa
# contra una lista cerrada (`HARNESS_STAGES`). Inventar aquí un nombre nuevo
# haría que un 400 del generador se escribiera como `refusal` del modelo
# evaluado: una negativa que nadie dio, y justo en la variable que mide el
# experimento. La etapa no viaja a la fila: solo decide el origen del fallo.
STAGE_CONTRADICTION = STAGE_RANK

OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase1a"

# Rutas declaradas en la cabecera (D5). Viven en el repo del blog, no en este.
PLAN_PATH = "docs/superpowers/plans/2026-09-14-pegado-accidental-fase-1a.md"
SPEC_PATH = "docs/superpowers/specs/2026-09-14-pegado-accidental-fase-1-design.md"

# El vocabulario de niveles lo cierra `records`, no este módulo: una fila con
# `"n1"` en minúscula se agruparía aparte y partiría el brazo en dos sin que
# nadie lo notara hasta el análisis.
assert set(LEVELS) == set(records.PASTE_LEVELS)
# §9 del spec de la Fase 1: nadie se puntúa a sí mismo, y el plantel de 1a es
# un subconjunto del evaluado.
assert set(PHASE1A_MODELS) <= set(config.EVALUATED)
assert not set(PHASE1A_MODELS) & set(JUDGES)


# --- planificación ---------------------------------------------------------


def make_conversation_id(
    model_id: str, topic_id: str, n_turns: int, paste_level: str, replicate_idx: int
) -> str:
    """Identificador determinista y legible de una conversación de la Fase 1a.

    Formato: ``p1a-{model}-{topic}-{n_turns}-{nivel}-r{replica}``. No se
    reutiliza `records.make_conversation_id` porque el suyo no tiene sitio para
    el nivel, y sin el nivel las tres celdas que solo se diferencian en eso
    colisionarían: la reanudación (D6) daría por hecha la de N1 al ver escrita
    la de N0, y la tirada acabaría con un tercio de las filas.
    """
    return f"p1a-{model_id}-{topic_id}-{n_turns}-{paste_level}-r{replicate_idx}"


def plan_phase1a(seed: int) -> list[dict]:
    """3 niveles × 8 temas × 2 longitudes × 2 réplicas × 3 modelos = 288.

    A diferencia de la Fase 0, la longitud **no rota**: cada (tema, modelo,
    nivel) se corre en las dos. Es lo que impide confundir nivel con longitud,
    que es la confusión que arruinaría el único contraste que esta tanda mide.
    El estrato sí conserva la rotación de D3 —`(t + STRATUM_SHIFTS[m]) % 8`—
    para que el brazo N0 siga siendo comparable con la Fase 0 y con la 1b.

    El `seed` de cada celda **ya no elige el artefacto**: eso lo hace
    `artifact_seed`, a partir de lo que las réplicas comparten (D1). Se conserva
    en la fila porque es lo único que declara, por escrito, que las dos réplicas
    de una celda son dos tiradas distintas de la misma celda y no un duplicado.

    El bucle exterior es el tema y el siguiente el modelo, o sea round-robin de
    modelos: con cinco o siete horas de tirada contra un gateway compartido, el
    orden modelo-mayor confunde el modelo con la hora de reloj.
    """
    if len(STRATUM_SHIFTS) != len(PHASE1A_MODELS):
        raise ValueError(
            "hay que dar un desplazamiento de estrato por modelo: "
            f"{len(STRATUM_SHIFTS)} desplazamientos para "
            f"{len(PHASE1A_MODELS)} modelos"
        )

    topics = load_topics()
    rng = np.random.default_rng(seed)
    plan: list[dict] = []
    for t, topic in enumerate(topics):
        for m, model_id in enumerate(PHASE1A_MODELS):
            for nivel in LEVELS:
                for n_turns in LENGTHS:
                    for rep in range(REPLICATES):
                        plan.append(
                            {
                                "cell_index": len(plan),
                                "model_id": model_id,
                                "topic_id": topic.id,
                                "n_turns": n_turns,
                                "paste_level": nivel,
                                "replicate_idx": rep,
                                "stratum": (t + STRATUM_SHIFTS[m]) % STRATA,
                                "condition": "paste",
                                "seed": int(rng.integers(0, 2**31)),
                                "conversation_id": make_conversation_id(
                                    model_id, topic.id, n_turns, nivel, rep
                                ),
                            }
                        )
    return plan


def call_budget(plan: list[dict]) -> dict[str, Any]:
    """Cuántas llamadas cuesta la tirada, por papel. Se dice ANTES de gastar.

    No convierte a dinero: los precios no viven en este repositorio y un número
    inventado es peor que ninguno. Lo que da es el recuento exacto, que es lo
    que hay que multiplicar por la tarifa de cada proveedor.
    """
    con_pegote = [c for c in plan if c["condition"] == "paste"]
    claves_n2 = {
        (c["topic_id"], c["n_turns"], c["stratum"])
        for c in plan
        if c["paste_level"] == CONTRADICTION_LEVEL
    }
    return {
        "cells": len(plan),
        "by_level": dict(sorted(Counter(c["paste_level"] for c in plan).items())),
        "by_model": dict(sorted(Counter(c["model_id"] for c in plan).items())),
        # Por celda: la del pegote más los dos turnos posteriores de D7.
        "evaluated_model_calls": len(con_pegote) * (1 + N_POST_TURNS)
        + (len(plan) - len(con_pegote)) * N_POST_TURNS,
        "simulated_user_calls": len(plan) * N_POST_TURNS,
        # Una por (prefijo, estrato), no una por celda: las réplicas comparten.
        "contradiction_calls": len(claves_n2),
        # Los 16 ya están en disco de la Fase 0 y no se vuelven a pagar.
        "prefixes": len({(c["topic_id"], c["n_turns"]) for c in plan}),
    }


# --- la semilla del artefacto (D1) ----------------------------------------


def artifact_seed(prefix_id: str, paste_level: str, stratum: int) -> int:
    """Semilla con la que se elige el pegote de una celda.

    Sale de lo que las **dos réplicas comparten** —el prefijo, el nivel y el
    estrato— y de nada más. En particular NO ve el `seed` de la celda, que es lo
    único que distingue una réplica de la otra: si lo viera, las dos réplicas
    pegarían artefactos distintos y su variación mezclaría el muestreo de la
    respuesta con el cambio de estímulo (D1).

    Es también lo que hace la tirada reproducible a trozos: la misma celda
    elegida hoy y dentro de un mes da el mismo artefacto, con el mismo banco.
    """
    blob = f"{prefix_id}|{paste_level}|{int(stratum)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**31)


def choice_key(prefix_id: str, cell: dict) -> tuple[str, str, int]:
    """La clave que comparten las dos réplicas de una celda."""
    return (prefix_id, cell["paste_level"], int(cell["stratum"]))


def choose_bank_artifact(
    prefix_id: str,
    cell: dict,
    ranking: list[tuple[Artifact, float]],
    kind_counts: Counter,
    chosen: dict[tuple[str, str, int], str],
    signal_counts: Counter,
) -> tuple[Artifact, float]:
    """Elige el pegote de una celda N0/N1, y lo recuerda para su réplica (D1).

    La primera réplica que llega elige dentro de su estrato cubriendo géneros
    (D4, `pick_artifact`) con el `rng` de `artifact_seed`; la segunda no vuelve
    a elegir: lee lo que eligió la primera. El memo no es una optimización —el
    recuento de géneros va cambiando a lo largo de la tirada, así que dos
    llamadas a `pick_artifact` con el mismo `rng` pero distinto `kind_counts`
    pueden devolver cosas distintas—, es la garantía.

    `chosen` se rellena además al reanudar (`resume_choices`), para que la
    garantía sobreviva a una tirada partida en dos ejecuciones.

    `signal_counts` hace con las señales de N1 lo que `kind_counts` con los
    géneros. Cubrir géneros no cubre señales: el banco N1 tiene 11 géneros y 4
    señales repartidas por ellos, así que empatar por género deja la señal al
    azar. Con el muestreo real eso dejaba 14 celdas de 96 (mediana) entre la
    señal más y la menos vista. No sesga la comparación N0/N1 —la que decide la
    puerta—, pero sí la tasa de N1 celda a celda, y deja sin potencia la
    pregunta de qué señal hace preguntar.
    """
    key = choice_key(prefix_id, cell)
    if key in chosen:
        for art, sim in ranking:
            if art.id == chosen[key]:
                return art, sim
        raise ValueError(
            f"la réplica anterior de {key} pegó el artefacto {chosen[key]!r}, "
            "que ya no está en el banco de este nivel. El banco ha cambiado "
            "entre ejecuciones y las dos réplicas dejarían de compartir pegote "
            "(D1): corre la tirada en un fichero nuevo en vez de reanudar esta."
        )
    rng = np.random.default_rng(artifact_seed(*key))
    art, sim = pick_artifact(
        ranking,
        cell["stratum"],
        kind_counts,
        rng,
        n_strata=STRATA,
        signal_counts=signal_counts,
    )
    kind_counts[art.kind] += 1
    if art.signal:
        signal_counts[art.signal] += 1
    chosen[key] = art.id
    return art, sim


def resume_choices(
    path: Path,
) -> tuple[dict[str, Counter], dict[tuple[str, str, int], str], dict[str, Counter]]:
    """Qué se eligió ya: géneros y señales gastados por banco, y artefacto por celda.

    Dos cosas que la reanudación de la Fase 0 no necesitaba:

    - **Los géneros, por nivel.** D4 garantiza cobertura dentro de un banco, y
      aquí hay dos: contarlos juntos haría que lo que ya se pegó de N0 apartara
      géneros de N1, que es otro banco con otra mezcla.
    - **Las señales, por nivel.** Mismo motivo que los géneros, sobre el banco
      N1. Si la reanudación no las trajera, la segunda ejecución repartiría
      señales desde cero y el equilibrio que busca `choose_bank_artifact` valdría
      solo dentro de cada trozo, no en la tirada.
    - **El artefacto de cada (prefijo, nivel, estrato).** Si la réplica 0 ya está
      escrita y la 1 no, la 1 tiene que pegar exactamente lo mismo (D1). Dentro
      de una ejecución eso lo garantiza el memo de `choose_bank_artifact`; entre
      ejecuciones, esto.

    Solo cuentan las filas hechas (`DONE_STATUSES`), que son las mismas que
    `resume_state` da por hechas: contar una celda que se va a reintentar
    gastaría su género dos veces.
    """
    kind_counts: dict[str, Counter] = {lvl: Counter() for lvl in BANK_LEVELS}
    signal_counts: dict[str, Counter] = {lvl: Counter() for lvl in BANK_LEVELS}
    chosen: dict[tuple[str, str, int], str] = {}
    if not path.exists():
        return kind_counts, chosen, signal_counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "run_header":
            continue
        if row.get("status", "ok") not in DONE_STATUSES:
            continue
        level = row.get("paste_level")
        if level not in BANK_LEVELS or not row.get("artifact_id"):
            continue
        if row.get("artifact_kind"):
            kind_counts[level][row["artifact_kind"]] += 1
        if row.get("artifact_signal"):
            signal_counts[level][row["artifact_signal"]] += 1
        key = (row.get("prefix_id", ""), level, int(row.get("stratum", NO_STRATUM)))
        anterior = chosen.setdefault(key, row["artifact_id"])
        if anterior != row["artifact_id"]:
            raise ValueError(
                f"{path.name} ya trae dos pegotes distintos para {key}: "
                f"{anterior!r} y {row['artifact_id']!r}. Las réplicas de esa "
                "celda no comparten artefacto (D1), así que el fichero no se "
                "puede reanudar sin decidir a mano cuál vale."
            )
    return kind_counts, chosen, signal_counts


# --- ancho del eje (D12, informativo en 1a) -------------------------------


def measure_axes(
    banks: dict[str, list[Artifact]],
    topics: list[Topic],
    run_id: str,
    out: Path | str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Mide el ancho del eje de **cada banco** y escribe el informe.

    Aquí no hay puerta: en la Fase 1a la variable independiente es el nivel del
    pegote, no la similaridad (ver el docstring del módulo). Un banco estrecho
    se dice en voz alta y queda escrito, porque es el dato que necesita la Fase
    1b —cuyo eje sí es la similaridad— para decidir si se puede correr.
    """
    informe: dict[str, Any] = {
        "run_id": run_id,
        "narrow_threshold": NARROW_RANGE,
        "gate": (
            "informativo: en la Fase 1a la variable independiente es el nivel "
            "del pegote, no la similaridad. La puerta GO/NO-GO de D12 gobierna "
            "la Fase 1b."
        ),
        "levels": {},
    }
    estrechas: list[str] = []
    for level in BANK_LEVELS:
        print(f"--- banco {level}: {len(banks[level])} artefactos")
        entrada = measure_axis(
            topics=topics, arts=banks[level], lengths=LENGTHS, run_id=run_id
        )
        informe["levels"][level] = entrada
        estrechas += [
            f"{level}/{c['topic_id']}(n={c['n_turns']})"
            for c in entrada.get("narrow_cells", [])
        ]
    informe["narrow_cells"] = estrechas
    if estrechas:
        print(
            f"--- OJO: eje ESTRECHO (rango < {NARROW_RANGE}) en "
            f"{len(estrechas)} celdas: {', '.join(estrechas)}. La Fase 1a "
            "sigue: su factor es el nivel, no la similaridad. Quien planifique "
            "la 1b tiene que mirar esto antes."
        )

    if write:
        path = axis_path(run_id, out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        informe["path"] = str(path)
    return informe


# --- ejecución de una celda -----------------------------------------------


def run_cell(
    cell: dict,
    topic: Topic,
    banks: dict[str, list[Artifact]],
    rank_caches: dict[str, dict[str, tuple]],
    kind_counts: dict[str, Counter],
    chosen: dict[tuple[str, str, int], str],
    signal_counts: dict[str, Counter],
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
) -> ConversationRecord:
    """Corre una celda entera y devuelve su fila.

    Es la de la Fase 0 con el nivel dentro. La secuencia es la misma —prefijo
    compartido (D1), pegote, dos turnos más sin reparación (D7)— y `partial` se
    va rellenando sobre la marcha para que una celda que reviente a mitad deje
    igualmente una fila con lo que se supiera (D6).

    Lo único que cambia es de dónde sale el pegote:

    - **N0 y N1**: del banco de su nivel, dentro del estrato del plan y con
      cobertura de géneros (D3/D4). El ranking se cachea por (prefijo, nivel):
      el mismo prefijo contra dos bancos distintos da dos rankings distintos, y
      un caché que no distinguiera el nivel serviría el de N0 a las celdas de
      N1 sin que nada lo dijera.
    - **N2**: lo escribe `CONTRADICTION_MODEL` contra el prefijo, y se persiste.
      No tiene estrato, ni ranking, ni similaridad: no se muestreó de ninguna
      parte, así que esos campos van a `None` en vez de a un número inventado.
    """
    model_id = cell["model_id"]
    level = cell["paste_level"]

    partial["stage"] = STAGE_PREFIX
    prefix = ensure_prefix(topic, cell["n_turns"])
    partial["prefix_id"] = prefix["prefix_id"]
    partial["prefix_user_traces"] = list(prefix.get("user_reply_traces", []))

    # Copia: el prefijo lo comparten los tres modelos y las dos réplicas, y no
    # se muta.
    transcript = [dict(m) for m in prefix["transcript"]]
    partial["transcript"] = transcript

    request_params: dict = {}
    partial["request_params"] = request_params
    replies: list = []
    partial["replies"] = replies
    user_replies: list = []
    partial["user_replies"] = user_replies

    similarity_user: float | None = None
    similarity_full: float | None = None
    similarity_rank: int | None = None
    similarity_pct: float | None = None
    ranking_rows: list[dict] | None = None
    truncated_user = False
    truncated_full = False
    stratum = int(cell["stratum"])

    if level in BANK_LEVELS:
        partial["stage"] = STAGE_RANK
        ranking, full_by_id, truncated_user, truncated_full = rank_for_prefix(
            prefix, banks[level], rank_caches[level]
        )
        artifact, similarity_user = choose_bank_artifact(
            prefix["prefix_id"],
            cell,
            ranking,
            kind_counts[level],
            chosen,
            signal_counts[level],
        )
        partial["artifact"] = artifact
        similarity_full = full_by_id.get(artifact.id)
        ids = [art.id for art, _ in ranking]
        similarity_rank = ids.index(artifact.id)
        similarity_pct = similarity_rank / (len(ids) - 1) if len(ids) > 1 else 0.0
        ranking_rows = [
            {"artifact_id": art.id, "similarity": sim} for art, sim in ranking
        ]
    else:
        partial["stage"] = STAGE_CONTRADICTION
        # La semilla se calcula con `cell["stratum"]`, el del plan, y no con la
        # variable local, que acaba de ponerse a NO_STRATUM: es la clave que
        # comparten las dos réplicas (D1), no un dato de la fila.
        #
        # Consecuencia declarada de que el estrato entre en la clave: como el
        # estrato depende del modelo (D3), los tres modelos reciben TRES
        # contradicciones distintas para el mismo prefijo, en vez de una
        # compartida. Dentro de una celda las réplicas comparten pegote, que es
        # lo que D1 exige; entre modelos, el brazo N2 compara respuestas a
        # estímulos parecidos pero no idénticos. Se acepta porque N2 es un
        # distractor declarado que no entra en ninguna tabla con N0 ni N1 (§5),
        # y queda escrito para que el reparto por familia dentro de N2 se lea
        # sabiéndolo.
        stratum = NO_STRATUM
        artifact = ensure_contradiction(
            prefix["prefix_id"],
            topic,
            prefix["transcript"],
            artifact_seed(prefix["prefix_id"], level, cell["stratum"]),
        )
        partial["artifact"] = artifact

    partial["stage"] = STAGE_PASTE
    reaction, paste_usage, paste_index, paste_reply = inject_paste(
        model_id, transcript, artifact, request_params_out=request_params
    )
    usages = [paste_usage]
    replies.append(paste_reply)

    partial["stage"] = STAGE_POST
    post_indices, post_usages, post_replies = continue_after_paste(
        model_id,
        transcript,
        topic,
        n_post=N_POST_TURNS,
        request_params_out=request_params,
        user_replies_out=user_replies,
    )
    usages.extend(post_usages)
    replies.extend(post_replies)

    traces = reply_traces(replies)
    stop_reasons = [trace["stop_reason"] for trace in traces]

    # Precedencia del `status`, igual que en la Fase 0 y por los mismos motivos:
    # `refusal` es lo único que es conducta y manda; `empty` es la afirmación
    # más fuerte que se puede hacer sobre una respuesta; `truncated` es un corte
    # por el tope que mandamos nosotros; y si no hay nada de eso, `ok`.
    declared = status_for_stop_reasons(stop_reasons)
    if declared == "refusal":
        status = "refusal"
    elif _has_empty_response(replies):
        status = "empty"
    elif declared is not None:
        status = declared
    else:
        status = "ok"

    # Y por encima de todo, el lado del usuario simulado: si uno de sus turnos
    # salió vacío o cortado, la conversación que se acaba de medir tiene un
    # agujero y ninguna categoría de la rúbrica se puede leer de ahí.
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
        replicate_idx=cell["replicate_idx"],
        model_id=model_id,
        model_label=_model_label(model_id),
        response_model=_response_model(traces),
        topic_id=topic.id,
        n_turns=cell["n_turns"],
        stratum=stratum,
        n_strata=STRATA,
        prefix_id=prefix["prefix_id"],
        condition=cell["condition"],
        arm=ARM_NO_REPAIR,
        artifact_id=artifact.id,
        artifact_kind=artifact.kind,
        artifact_signal=artifact.signal,
        paste_level=level,
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
    """La fila de una celda que revienta (D6), con el nivel del pegote dentro.

    Se apoya entera en la de la Fase 0 —que es la que sabe distinguir un fallo
    del arnés de una negativa del modelo evaluado— y solo le añade lo que aquel
    runner no podía saber: el nivel, y que una celda N2 no tiene estrato.
    """
    rec = phase0_failed_record(cell, topic_id, run_id, started_at, partial, exc)
    rec.paste_level = cell["paste_level"]
    if cell["paste_level"] == CONTRADICTION_LEVEL:
        rec.stratum = NO_STRATUM
    return rec


# --- tirada ----------------------------------------------------------------


def build_header(
    run_id: str,
    seed: int,
    plan: list[dict],
    banks: dict[str, list[Artifact]],
    topics: list[Topic],
    started_at: float,
) -> dict:
    """La primera línea del JSONL (D5).

    Además de lo de la Fase 0 lleva lo que hace falta para leer esta tanda
    dentro de seis meses: los tres niveles, las réplicas, el sha de **cada**
    banco por separado (un sha conjunto no diría cuál de los dos cambió), la
    versión de la rúbrica y quiénes son los jueces. `RunHeader` no tiene esos
    campos —`records.py` no es de este agente— y se añaden a la línea, que es un
    dict; queda anotado para que `records` los adopte.
    """
    todos = sorted(banks["N0"] + banks["N1"], key=lambda a: a.id)
    header = run_header_line(
        RunHeader(
            run_id=run_id,
            phase="1a",
            plan_path=PLAN_PATH,
            spec_path=SPEC_PATH,
            code_sha=code_sha(),
            bank_sha=bank_sha(todos),
            topics_sha=topics_sha(topics),
            master_seed=seed,
            roster=list(PHASE1A_MODELS),
            planned_cells=len(plan),
            embedding_model=config.EMBEDDING_MODEL,
            prefix_model=PREFIX_MODEL,
            user_model=USER_MODEL,
            user_system_prompt=USER_SYSTEM_PROMPT,
            started_at=started_at,
        )
    )
    header["corrections_path"] = CORRECTIONS_PATH
    header["levels"] = list(LEVELS)
    header["replicates"] = REPLICATES
    header["bank_sha_by_level"] = {lvl: bank_sha(banks[lvl]) for lvl in BANK_LEVELS}
    header["contradiction_model"] = CONTRADICTION_MODEL
    header["rubric_version"] = RUBRIC_VERSION
    header["judges"] = list(JUDGES)
    header["call_budget"] = call_budget(plan)
    return header


def main(
    seed: int = MASTER_SEED,
    out: Path | str | None = None,
    measure: bool = True,
) -> Path:
    """Corre la Fase 1a entera y devuelve la ruta del JSONL.

    Misma mecánica que la Fase 0: si `out` apunta a una tirada ya empezada se
    reanuda —saltando las celdas hechas y retirando del fichero las que se van a
    reintentar (D6)—, y si no, se empieza de cero con la cabecera delante. Los
    ficheros laterales (ejes, resumen) se escriben al lado del JSONL.

    `measure=False` salta la medición del ancho del eje (D12), que aquí es
    informativa: no hay puerta que frene esta tanda por un eje estrecho, porque
    el eje de la Fase 1a es el nivel del pegote. Ver el docstring del módulo.
    """
    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}
    # Cada banco por separado y pedido por su nivel: `load_artifacts()` sin
    # nivel devolvería los dos juntos, y estratificar sobre la mezcla haría que
    # el estrato de una celda N0 dependiera de cuántos N1 hay escritos.
    banks = {lvl: load_artifacts(level=lvl) for lvl in BANK_LEVELS}
    plan = plan_phase1a(seed)

    path = (
        Path(out)
        if out is not None
        else OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    resuming = is_resumable(path)

    if resuming:
        run_id = read_run_header(path).get("run_id") or path.stem
        compact_resume_file(path, out=path)
        done, _ = resume_state(path)
        kind_counts, chosen, signal_counts = resume_choices(path)
        print(f"reanudando {path.name}: {len(done)} celdas ya hechas")
    else:
        run_id = path.stem
        done = set()
        kind_counts = {lvl: Counter() for lvl in BANK_LEVELS}
        signal_counts = {lvl: Counter() for lvl in BANK_LEVELS}
        chosen = {}

    presupuesto = call_budget(plan)
    print(
        f"--- {presupuesto['cells']} celdas | llamadas: "
        f"{presupuesto['evaluated_model_calls']} al modelo evaluado, "
        f"{presupuesto['simulated_user_calls']} al usuario simulado, "
        f"{presupuesto['contradiction_calls']} al generador de N2 "
        f"({CONTRADICTION_MODEL}), {presupuesto['prefixes']} prefijos "
        "(ya en disco desde la Fase 0)"
    )

    if measure:
        measure_axes(banks, topics, run_id, out=path)

    rank_caches: dict[str, dict[str, tuple]] = {lvl: {} for lvl in BANK_LEVELS}
    completed = 0
    failed = 0
    skipped = 0
    by_level: dict[str, Counter] = {lvl: Counter() for lvl in LEVELS}

    with path.open("a" if resuming else "w", encoding="utf-8") as fh:
        if not resuming:
            header = build_header(run_id, seed, plan, banks, topics, time.time())
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            fh.flush()

        for cell in plan:
            if cell["conversation_id"] in done:
                skipped += 1
                by_level[cell["paste_level"]]["skipped"] += 1
                continue

            topic = topics_by_id[cell["topic_id"]]
            started_at = time.time()
            partial: dict[str, Any] = {}
            try:
                rec = run_cell(
                    cell,
                    topic,
                    banks,
                    rank_caches,
                    kind_counts,
                    chosen,
                    signal_counts,
                    run_id,
                    started_at,
                    partial,
                )
            except Exception as exc:  # D6: una celda rota no tumba la tirada.
                rec = failed_record(
                    cell, cell["topic_id"], run_id, started_at, partial, exc
                )

            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            fh.flush()

            if rec.status in DONE_STATUSES:
                completed += 1
                by_level[cell["paste_level"]]["completed"] += 1
            else:
                failed += 1
                by_level[cell["paste_level"]]["failed"] += 1
            print(_cell_line(rec, cell))

    summary = {
        "run_id": run_id,
        "path": str(path),
        "planned": len(plan),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "by_level": {
            lvl: {
                "completed": by_level[lvl]["completed"],
                "failed": by_level[lvl]["failed"],
                "skipped": by_level[lvl]["skipped"],
            }
            for lvl in LEVELS
        },
        "kinds": {lvl: dict(sorted(kind_counts[lvl].items())) for lvl in BANK_LEVELS},
        "signals": {
            lvl: dict(sorted(signal_counts[lvl].items())) for lvl in BANK_LEVELS
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
    for lvl in LEVELS:
        print(f"---   {lvl}: {summary['by_level'][lvl]}")
    return path


def _cell_line(rec: ConversationRecord, cell: dict) -> str:
    sim = f"{rec.similarity_user:.3f}" if rec.similarity_user is not None else "  -  "
    return (
        f"[{cell['cell_index']:>3}] {rec.model_id:<18} {rec.topic_id:<16} "
        f"n={rec.n_turns:<2} {rec.paste_level} r{rec.replicate_idx} "
        f"s={rec.stratum:<2} sim={sim} {rec.artifact_kind or '-':<14} {rec.status}"
    )


if __name__ == "__main__":
    print(main())
