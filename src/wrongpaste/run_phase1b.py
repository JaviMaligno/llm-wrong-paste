"""Runner del barrido de similaridad: Fase 1b sobre N0, Fase 1d sobre N1.

La Fase 1a usaba la similaridad como **estrato**: ocho ventanas, y dentro de cada
ventana se elegía cubriendo géneros. Servía para que el nivel del pegote no se
confundiera con el parecido, pero no para dibujar nada contra el parecido, porque
cada (tema, modelo) veía un solo estrato.

Aquí la similaridad es la variable independiente, así que el muestreo cambia:
cada (tema, modelo) recorre **las doce posiciones** del ranking de su prefijo, de
la menos parecida a la más parecida. Lo demás —prefijos, pegado, turnos
posteriores, reanudación, fallo como dato— se importa de la Fase 1a sin tocarlo.

**Un solo runner para las dos tandas.** La Fase 1b barrió el brazo limpio (N0) y
no encontró nada: con un pegote sin señal, el parecido no predice la conducta.
La pregunta que quedaba —en el brazo donde los modelos SÍ dudan, ¿el parecido
ayuda o estorba?— es la Fase 1d, y es este mismo barrido sobre el banco N1. La
única diferencia real es de qué banco sale el pegote, así que el nivel entra por
parámetro (`main(level="N1")`) en vez de por un runner copiado: dos copias del
mismo bucle divergen en cuanto alguien arregle una de las dos.

Lo que el nivel cambia son tres cosas, y ninguna más: el banco, el prefijo del
`conversation_id` —para que las dos tandas no compartan identificadores, que
haría que una reanudación diera por hecha una celda de la otra— y el directorio
de salida. El plan de celdas es idéntico, semillas incluidas, porque 1d se lee
celda a celda contra 1b.

Dos cosas que NO se hacen aquí, y constan en el plan:

1. **No hay réplicas.** El acuerdo entre réplicas ya se midió en 1a —0,73 sobre
   N0, 0,87 sobre N1— y viaja en el informe como suelo de ruido. Para estimar
   una curva, 288 estímulos distintos valen más que 144 repetidos: dos tiradas
   del mismo pegote están correlacionadas.
2. **No entra N2.** No es un banco: se fabrica contra cada conversación (ver
   `contradictions.py`), así que no hay ranking que barrer.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
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
from wrongpaste.judging import JUDGES
from wrongpaste.prefixes import PREFIX_MODEL, ensure_prefix
from wrongpaste.records import (
    ARM_NO_REPAIR,
    ConversationRecord,
    RunHeader,
    run_header_line,
)
from wrongpaste.rubric import RUBRIC_VERSION

# La maquinaria de las fases anteriores, reutilizada tal cual. Lo que se importa
# con guion bajo es privado de aquel módulo y se toma prestado a propósito: dos
# copias de la clasificación de fallos (D6) divergen en cuanto alguien arregle
# una de las dos.
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
    NarrowAxisError,
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
from wrongpaste.run_phase1a import PHASE1A_MODELS, call_budget
from wrongpaste.simulated_user import USER_MODEL, SimulatedUserError
from wrongpaste.topics import Topic, load_topics

# Los mismos tres modelos que 1a, para que las dos tandas se puedan comparar.
PHASE1B_MODELS = list(PHASE1A_MODELS)

# Doce posiciones sobre el ranking de 64 artefactos: una de cada ~5,7 puestos.
# Más posiciones dejarían menos observaciones por punto y la curva es una
# estimación por punto, no una interpolación.
SWEEP_POSITIONS = 12

MASTER_SEED = 20260915

# Cada tirada corre UN solo brazo, y el nivel entra por parámetro porque la
# Fase 1d es este mismo barrido sobre el banco N1. No hay runner nuevo: la
# única diferencia real es de qué banco sale el pegote, y dos copias del mismo
# bucle divergen en cuanto alguien arregle una de las dos. `PASTE_LEVEL` se
# queda como el valor por defecto —el de la tanda ya pagada— para que llamar a
# `main()` sin argumentos siga siendo exactamente la Fase 1b.
PASTE_LEVEL = "N0"

# Qué tanda es cada brazo. El nivel no basta como identidad: las dos tandas se
# analizan juntas y viven en el mismo repositorio, así que la fase es lo que
# separa los identificadores, los directorios de salida y el `phase` de la
# cabecera (D5).
PHASE_BY_LEVEL = {"N0": "1b", "N1": "1d"}

PHASE = PHASE_BY_LEVEL[PASTE_LEVEL]

OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase1b"

# Rutas declaradas en la cabecera (D5). Viven en el repo del blog, no en este.
# El plan es distinto en cada tanda y la cabecera tiene que decir cuál se
# siguió: apuntar la tirada de 1d al plan de 1b sería declarar una procedencia
# falsa justo en el campo que existe para no tener que reconstruirla.
PLAN_PATH_BY_LEVEL = {
    "N0": "docs/superpowers/plans/2026-09-15-pegado-accidental-fase-1b.md",
    "N1": "docs/superpowers/plans/2026-09-16-pegado-accidental-fase-1d.md",
}
PLAN_PATH = PLAN_PATH_BY_LEVEL[PASTE_LEVEL]
SPEC_PATH = "docs/superpowers/specs/2026-09-14-pegado-accidental-fase-1-design.md"


def out_dir_for(level: str) -> Path:
    """Dónde cae la tirada de cada brazo cuando no se le da `out`.

    N1 va a `runs/phase1d/` y no a `runs/phase1b/`: el análisis de una tanda
    lista su directorio, y mezclarlas ahí haría que 1b leyera filas de 1d sin
    que nadie lo hubiera pedido. Se deriva de `OUT_DIR` en vez de ser una tabla
    aparte para que quede un solo sitio que apunte a `runs/` —y para que los
    tests, que redirigen `OUT_DIR` a un `tmp_path`, redirijan los dos brazos.
    """
    if level == PASTE_LEVEL:
        return OUT_DIR
    return OUT_DIR.parent / f"phase{PHASE_BY_LEVEL[level]}"


def check_resume_level(header: dict, level: str, path: Path) -> None:
    """Que la tirada que se reanuda sea la del brazo que se está corriendo.

    El prefijo `p1b-`/`p1d-` de `make_conversation_id` impide que los
    identificadores de las dos tandas choquen, y eso es justo lo que hace que
    equivocarse de nivel al reanudar sea **silencioso** en vez de ruidoso:
    reanudar sin `level` un fichero de la Fase 1d no da por hecha ninguna celda
    —ninguno de los 288 identificadores `p1b-` está entre los `p1d-` del
    fichero—, así que no salta ninguna, vuelve a pagar la tirada entera y anexa
    288 filas `paste_level: "N0"` bajo una cabecera que declara `phase: "1d"` y
    `levels: ["N1"]`. El análisis de 1d lista su directorio y lee el fichero, con
    lo que mezclaría los dos brazos; lo único que se vería sería un resumen
    diciendo «saltadas: 0».

    Se comprueba contra lo que la cabecera **declara** (D5), que es el campo que
    existe para no tener que reconstruir la procedencia, y antes de tocar el
    fichero: `compact_resume_file` lo reescribe, y una reanudación que no se
    puede hacer no puede dejar rastro. Una cabecera que no declara nivel ni fase
    —formato anterior— no contradice nada y no frena: lo que se castiga es la
    contradicción, no el silencio.
    """
    declarados = header.get("levels")
    fase = header.get("phase")
    esperada = PHASE_BY_LEVEL[level]
    if declarados is not None and list(declarados) != [level]:
        raise ValueError(
            f"{path} es una tirada de {declarados} y se está reanudando con "
            f"level={level!r}: las celdas de un brazo no aparecen entre las "
            f"hechas del otro (los identificadores llevan el prefijo de su "
            f"fase), así que la reanudación no saltaría ninguna y anexaría "
            f"{level} bajo una cabecera que dice {declarados}. Pasa "
            f"level={declarados[0]!r} si querías continuar esa tirada."
        )
    if fase is not None and fase != esperada:
        raise ValueError(
            f"{path} es una tirada de la fase {fase} y se está reanudando como "
            f"fase {esperada} (level={level!r}): mezclar las dos tandas en un "
            f"fichero deja filas de dos brazos bajo una sola cabecera."
        )


def sweep_index(
    position: int, n_artifacts: int, n_positions: int = SWEEP_POSITIONS
) -> int:
    """Qué puesto del ranking le toca a una posición del barrido.

    Reparte las posiciones **uniformemente por rango**, extremos incluidos: la
    posición 0 es el artefacto menos parecido del banco y la última el más
    parecido. Estratificar por valor de coseno en vez de por rango dejaría
    ventanas vacías, porque la distribución real está muy concentrada (es la
    misma razón que `stratum_window` documenta para la Fase 0).

    Con un banco más pequeño que el número de posiciones, dos posiciones pueden
    caer en el mismo artefacto. Es preferible a reventar: el banco es un dato de
    entrada, y una tanda de prueba con cinco artefactos tiene que poder correr.
    """
    if n_positions < 2:
        raise ValueError(f"un barrido necesita al menos 2 posiciones, no {n_positions}")
    if not 0 <= position < n_positions:
        raise ValueError(
            f"posición {position} fuera del barrido de {n_positions} posiciones"
        )
    if n_artifacts < 1:
        raise ValueError("el ranking está vacío: no hay de dónde muestrear")
    return round(position * (n_artifacts - 1) / (n_positions - 1))


def make_conversation_id(
    model_id: str,
    topic_id: str,
    n_turns: int,
    position: int,
    level: str = PASTE_LEVEL,
) -> str:
    """Identidad de una celda: legible y única sin necesitar el índice.

    El prefijo es el de la **fase**, no el del nivel, y es lo único que impide
    que las dos tandas choquen: sin él, `p1b-claude-opus-5-viaje-japon-2-s11`
    nombraría una celda de N0 y otra de N1, y una reanudación de 1d daría por
    hecha una celda que en realidad corrió 1b contra el otro banco.
    """
    return f"p{PHASE_BY_LEVEL[level]}-{model_id}-{topic_id}-{n_turns}-s{position:02d}"


def plan_phase1b(seed: int, level: str = PASTE_LEVEL) -> list[dict]:
    """12 posiciones × 8 temas × 3 modelos = 288 celdas, sin réplicas.

    La longitud **no rota por modelo** como en la Fase 0, sino que se alterna con
    la posición: `LENGTHS[(pos + t + m) % 2]`. Eso deja las dos longitudes
    empatadas dentro de cada posición (12 y 12 de las 24 celdas) y también dentro
    de cada (tema, modelo) (6 y 6 de las 12). Si una posición cayera entera en
    conversaciones cortas, la curva mediría la longitud disfrazada de parecido.

    El `seed` de la celda no elige nada: el artefacto lo determina la posición,
    que es el punto del diseño. Se conserva en la fila porque identifica la
    tirada y porque el runner lo pasa a los turnos posteriores.

    El bucle exterior es el tema y el siguiente el modelo —round-robin de
    modelos— por lo mismo que en 1a: con horas de tirada contra un gateway
    compartido, el orden modelo-mayor confunde el modelo con la hora de reloj.

    `level` elige el banco del que saldrá el pegote y **no cambia nada más**:
    las mismas 288 celdas, las mismas longitudes y las mismas semillas. Es a
    propósito: la Fase 1d se compara celda a celda con la 1b, y si el plan de N1
    sorteara distinto la comparación dejaría de serlo.
    """
    if level not in PHASE_BY_LEVEL:
        raise KeyError(
            f"nivel {level!r} sin fase declarada: el barrido solo corre sobre "
            f"un banco en disco ({sorted(PHASE_BY_LEVEL)}). N2 no es un banco, "
            "se fabrica contra cada conversación."
        )
    topics = load_topics()
    rng = np.random.default_rng(seed)
    plan: list[dict] = []
    for t, topic in enumerate(topics):
        for m, model_id in enumerate(PHASE1B_MODELS):
            for pos in range(SWEEP_POSITIONS):
                n_turns = LENGTHS[(pos + t + m) % len(LENGTHS)]
                plan.append(
                    {
                        "cell_index": len(plan),
                        "model_id": model_id,
                        "topic_id": topic.id,
                        "n_turns": n_turns,
                        "sweep_position": pos,
                        "paste_level": level,
                        "condition": "paste",
                        "seed": int(rng.integers(0, 2**31)),
                        "conversation_id": make_conversation_id(
                            model_id, topic.id, n_turns, pos, level
                        ),
                    }
                )
    return plan


# --- el muestreo por posición ---------------------------------------------


def choose_sweep_artifact(
    ranking: list[tuple[Artifact, float]], position: int
) -> tuple[Artifact, float]:
    """El artefacto que le toca a una posición del barrido.

    **Sin estado, a diferencia de la Fase 1a.** Allí la elección dependía del
    recuento de géneros ya gastados, y por eso hacía falta memo y reanudación
    cuidadosa. Aquí la posición determina el artefacto por completo: la misma
    celda elegida hoy y dentro de un mes da lo mismo, y reanudar no arrastra nada.

    El precio es que el género queda a merced del ranking, que es exactamente lo
    que D15 avisa. No se corrige forzando géneros —eso rompería el eje— sino que
    se mide después, con `curve.by_kind`.
    """
    return ranking[sweep_index(position, len(ranking))]


# --- la puerta del eje (D12) ----------------------------------------------


# Lo que midió la Fase 1a sobre cada banco, para que el mensaje de NO-GO diga
# contra qué se compara. Son rangos distintos —el eje de N1 es suficiente pero
# más estrecho que el de N0— y enseñar el del otro brazo convertiría la pista en
# una pista falsa justo cuando hace falta.
AXIS_REFERENCE_BY_LEVEL = {
    "N0": "0,236-0,502 (mediana 0,334)",
    "N1": "0,219-0,369 (mediana 0,290)",
}


def check_axis(
    topics: list[Topic],
    bank: list[Artifact],
    run_id: str,
    out: Path | str | None = None,
    level: str = PASTE_LEVEL,
) -> dict:
    """Mide el ancho del eje y decide si merece la pena barrerlo (D12).

    D12 —«por debajo de 0,15 de rango de coseno el tema no separa nada y la
    estratificación es decorativa»— gobierna **esta** tanda y no la 1a: allí la
    variable independiente era el nivel del pegote y un eje estrecho solo se
    anotaba; aquí el eje ES la variable, así que un eje estrecho no deja una
    tanda peor, deja una tanda que no responde a nada. 288 conversaciones contra
    posiciones que no se distinguen entre sí producen una curva imposible de
    interpretar, y eso es peor que no tenerla.

    Se levanta `NarrowAxisError` —la misma clase que usa la puerta de la Fase 0,
    no una copia— **antes** de abrir el JSONL de la tirada, así que no queda un
    fichero a medias. Lo que sí queda escrito es el informe del eje: la evidencia
    de por qué no se corrió es justo lo que hay que mirar para completar el
    banco, y es la misma convención que `run_phase0.main`.

    Lo que esta puerta ahorra y lo que no, dicho igual que en la Fase 0: ahorra
    las 288 celdas contra los modelos evaluados, que son el grueso del
    presupuesto; **no** ahorra los 16 prefijos, porque para embeber el lado de
    usuario hay que tenerlos y `measure_axis` se los pide a `ensure_prefix`. Esos
    quedan en disco y la tirada siguiente los reutiliza, pero se han pagado.

    Medido en la Fase 1a, este mismo eje dio 0,236-0,502 (mediana 0,334) sobre
    N0 y 0,219-0,369 (mediana 0,290) sobre N1, sin una sola celda estrecha en
    ninguno de los dos. Si esto salta, lo que ha cambiado es el banco o los
    prefijos, no el umbral.
    """
    informe = measure_axis(
        topics=topics, arts=bank, lengths=tuple(LENGTHS), run_id=run_id
    )
    ruta = axis_path(run_id, out=out)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    informe["path"] = str(ruta)

    estrechas = informe.get("narrow_cells") or []
    if estrechas:
        print(
            f"--- NO-GO: {len(estrechas)} celdas con el eje estrecho en los "
            f"temas {sorted(informe.get('narrow_topics') or [])}. El barrido de "
            f"la Fase {PHASE_BY_LEVEL[level]} mediría posiciones que no se "
            f"distinguen entre sí. En la Fase 1a el eje de {level} medía "
            f"{AXIS_REFERENCE_BY_LEVEL[level]}, así que si esto salta ha "
            "cambiado el banco o los prefijos."
        )
        raise NarrowAxisError(estrechas, report_path=ruta)
    return informe


# --- ejecución de una celda -----------------------------------------------


def run_cell(
    cell: dict,
    topic: Topic,
    bank: list[Artifact],
    rank_caches: dict[str, tuple],
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
) -> ConversationRecord:
    """Una conversación: prefijo, pegote de la posición, turnos posteriores.

    Idéntica a `run_phase1a.run_cell` salvo en tres puntos: el banco es el del
    brazo que corre esta tanda —uno solo—, el pegote sale de
    `choose_sweep_artifact` (sin estado compartido) y la fila lleva
    `sweep_position` en vez de `stratum`.

    El nivel se lee de la **celda** y no de la constante del módulo: la celda es
    lo único que sabe de qué banco salió su pegote, y con el nivel cableado una
    tirada de 1d escribiría 288 filas diciendo que son de N0.

    `partial` se va rellenando sobre la marcha, igual que en 1a: una celda que
    reviente a mitad tiene que dejar una fila con todo lo que se supiera hasta
    ahí (D6).
    """
    model_id = cell["model_id"]

    partial["stage"] = STAGE_PREFIX
    prefix = ensure_prefix(topic, cell["n_turns"])
    partial["prefix_id"] = prefix["prefix_id"]
    partial["prefix_user_traces"] = list(prefix.get("user_reply_traces", []))

    # Copia: el prefijo lo comparten los tres modelos y las doce posiciones, y
    # no se muta.
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
    artifact, similarity_user = choose_sweep_artifact(ranking, cell["sweep_position"])
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

    # Precedencia del `status`, igual que en la Fase 0 y en la 1a y por los
    # mismos motivos: `refusal` es lo único que es conducta y manda; `empty` es
    # la afirmación más fuerte que se puede hacer sobre una respuesta;
    # `truncated` es un corte por el tope que mandamos nosotros; y si no hay
    # nada de eso, `ok`.
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
    # agujero y ninguna categoría de la rúbrica se puede leer de ahí. En 1b esto
    # pesa más que en 1a, no menos: cada celda es un punto único de la curva, y
    # un turno roto que se colara como conducta movería el punto entero.
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
    # `stratum`, `n_strata` y `replicate_idx` NO se rellenan: esta tanda no
    # estratificó ni replicó nada, y escribirles un número sería decir que sí.
    # Se quedan en el valor por defecto del esquema, que es lo que significa
    # «aquí no hubo».
    return Phase0Record(
        run_id=run_id,
        conversation_id=cell["conversation_id"],
        cell_index=cell["cell_index"],
        model_id=model_id,
        model_label=_model_label(model_id),
        response_model=_response_model(traces),
        topic_id=topic.id,
        n_turns=cell["n_turns"],
        prefix_id=prefix["prefix_id"],
        condition=cell["condition"],
        arm=ARM_NO_REPAIR,
        artifact_id=artifact.id,
        artifact_kind=artifact.kind,
        artifact_signal=artifact.signal,
        paste_level=cell["paste_level"],
        sweep_position=cell["sweep_position"],
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
    """La fila de una celda que revienta (D6), con su posición del barrido.

    Se apoya entera en la de la Fase 0 —que es la que sabe distinguir un fallo
    del arnés de una negativa del modelo evaluado— y solo le añade lo que aquel
    runner no podía saber: de qué banco salía el pegote y en qué punto del eje
    estaba. El nivel sale de la celda por lo mismo que en `run_cell`: si aquí
    quedara cableado, las pérdidas de 1d se contarían como pérdidas de 1b.

    La posición se escribe **aunque la celda haya fallado**: no es un resultado
    sino un dato del diseño, conocido antes de llamar a nadie. Sin ella, el
    recuento de pérdidas por posición —el que dice si el hueco de la curva es
    sistemático o aleatorio— no se puede hacer.

    El `cell` de 1b no trae `replicate_idx` ni `stratum` y la fila de la Fase 0
    los espera, así que se le pasan los valores por defecto del esquema: los
    mismos que llevan las filas buenas de esta tanda.
    """
    rec = phase0_failed_record(
        {"replicate_idx": 0, "stratum": -1, **cell},
        topic_id,
        run_id,
        started_at,
        partial,
        exc,
    )
    rec.paste_level = cell["paste_level"]
    rec.sweep_position = cell["sweep_position"]
    return rec


# --- tirada ----------------------------------------------------------------


def build_header(
    run_id: str,
    seed: int,
    plan: list[dict],
    bank: list[Artifact],
    topics: list[Topic],
    started_at: float,
    level: str = PASTE_LEVEL,
) -> dict:
    """La primera línea del JSONL (D5).

    Un solo banco y un solo nivel —el del brazo que corre—, así que un solo
    `bank_sha`. La cabecera declara la fase y el plan de ESA tanda: es el campo
    por el que se sabe, dentro de seis meses, cuál de los dos barridos es este
    fichero sin tener que mirar las filas. Lo que sí lleva
    de más que la de 1a es el **número de posiciones del barrido**: es la
    variable independiente de la tanda y quien lea el fichero dentro de seis
    meses tiene que poder saber sobre cuántos puntos se dibujó la curva sin
    recontarlos. `RunHeader` no tiene esos campos —`records.py` no es de este
    agente— y se añaden a la línea, que es un dict.
    """
    header = run_header_line(
        RunHeader(
            run_id=run_id,
            phase=PHASE_BY_LEVEL[level],
            plan_path=PLAN_PATH_BY_LEVEL[level],
            spec_path=SPEC_PATH,
            code_sha=code_sha(),
            bank_sha=bank_sha(bank),
            topics_sha=topics_sha(topics),
            master_seed=seed,
            roster=list(PHASE1B_MODELS),
            planned_cells=len(plan),
            embedding_model=config.EMBEDDING_MODEL,
            prefix_model=PREFIX_MODEL,
            user_model=USER_MODEL,
            user_system_prompt=USER_SYSTEM_PROMPT,
            started_at=started_at,
        )
    )
    header["corrections_path"] = CORRECTIONS_PATH
    header["levels"] = [level]
    header["sweep_positions"] = SWEEP_POSITIONS
    header["replicates"] = 1
    header["rubric_version"] = RUBRIC_VERSION
    header["judges"] = list(JUDGES)
    header["call_budget"] = call_budget(plan)
    return header


def main(
    seed: int = MASTER_SEED,
    out: Path | str | None = None,
    measure: bool = True,
    level: str = PASTE_LEVEL,
) -> Path:
    """Corre el barrido entero sobre un brazo y devuelve la ruta del JSONL.

    Misma mecánica que las tandas anteriores: si `out` apunta a una tirada ya
    empezada se reanuda —saltando las celdas hechas y retirando del fichero las
    que se van a reintentar (D6)—, y si no, se empieza de cero con la cabecera
    delante.

    Lo que aquí **no** hace falta es recuperar el estado del muestreo al
    reanudar. En 1a había que releer el fichero para saber qué géneros se habían
    gastado; aquí la posición determina el artefacto por completo, así que una
    celda reintentada dentro de un mes recibe el mismo pegote que habría
    recibido hoy. Es la ventaja práctica de que el eje sea la variable.

    `measure=True` (lo normal) pasa antes por la puerta de D12: se mide el ancho
    del eje y, si alguna celda sale estrecha, la tirada **no empieza** y se
    levanta `NarrowAxisError` sin haber escrito el JSONL. `measure=False` la
    salta, y es solo para los tests del arnés, donde el ancho del eje da igual
    porque no se llama a ningún modelo.

    `level="N0"` es la Fase 1b, la tanda ya pagada, y llamar sin argumentos
    sigue siendo exactamente eso. `level="N1"` es la Fase 1d: el mismo barrido
    sobre el banco con señal, con sus propios identificadores y su propio
    directorio de salida. Lo que cambia son esas tres cosas y nada más; si
    cambiara el plan de celdas, las dos tandas dejarían de compararse.
    """
    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}
    # Un solo banco, y pedido por su nivel: `load_artifacts()` sin nivel
    # devolvería N0 y N1 juntos, y el ranking —que aquí ES el eje— saldría
    # contaminado con artefactos que esta tanda no puede pegar.
    plan = plan_phase1b(seed, level)  # valida el nivel antes de tocar el disco
    bank = load_artifacts(level=level)

    path = (
        Path(out)
        if out is not None
        else out_dir_for(level) / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    resuming = is_resumable(path)

    if resuming:
        header_previa = read_run_header(path)
        # Antes de tocar nada: si el fichero es del otro brazo, esto para aquí.
        check_resume_level(header_previa, level, path)
        run_id = header_previa.get("run_id") or path.stem
        compact_resume_file(path, out=path)
        done, _ = resume_state(path)
        print(f"reanudando {path.name}: {len(done)} celdas ya hechas")
    else:
        run_id = path.stem
        done = set()

    presupuesto = call_budget(plan)
    print(
        f"--- fase {PHASE_BY_LEVEL[level]} ({level}, {len(bank)} artefactos) | "
        f"{presupuesto['cells']} celdas | {SWEEP_POSITIONS} posiciones del "
        f"barrido | llamadas: {presupuesto['evaluated_model_calls']} al modelo "
        f"evaluado, {presupuesto['simulated_user_calls']} al usuario simulado, "
        f"{presupuesto['prefixes']} prefijos (ya en disco desde la Fase 0)"
    )

    if measure:
        # D12: el ancho del eje se mide antes de gastar, porque en ESTA tanda la
        # similaridad es la variable independiente. Y no solo se mide: si sale
        # estrecho, `check_axis` para la tirada aquí, con el informe ya escrito
        # al lado del JSONL y sin que el JSONL llegue a existir.
        check_axis(topics, bank, run_id, out=path, level=level)

    rank_caches: dict[str, tuple] = {}
    completed = 0
    failed = 0
    skipped = 0
    by_position: dict[int, Counter] = {p: Counter() for p in range(SWEEP_POSITIONS)}

    with path.open("a" if resuming else "w", encoding="utf-8") as fh:
        if not resuming:
            header = build_header(
                run_id, seed, plan, bank, topics, time.time(), level=level
            )
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            fh.flush()

        for cell in plan:
            if cell["conversation_id"] in done:
                skipped += 1
                by_position[cell["sweep_position"]]["skipped"] += 1
                continue

            topic = topics_by_id[cell["topic_id"]]
            started_at = time.time()
            partial: dict[str, Any] = {}
            try:
                rec = run_cell(
                    cell, topic, bank, rank_caches, run_id, started_at, partial
                )
            except Exception as exc:  # D6: una celda rota no tumba la tirada.
                rec = failed_record(
                    cell, cell["topic_id"], run_id, started_at, partial, exc
                )

            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            fh.flush()

            if rec.status in DONE_STATUSES:
                completed += 1
                by_position[cell["sweep_position"]]["completed"] += 1
            else:
                failed += 1
                by_position[cell["sweep_position"]]["failed"] += 1
            print(_cell_line(rec, cell))

    summary = {
        "run_id": run_id,
        "path": str(path),
        "planned": len(plan),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        # Las pérdidas **por posición**, no solo en total: si se concentran en
        # un extremo del barrido, la curva tiene un hueco sistemático en vez de
        # aleatorio y eso hay que poder decirlo en los resultados.
        "by_position": {
            p: {
                "completed": by_position[p]["completed"],
                "failed": by_position[p]["failed"],
                "skipped": by_position[p]["skipped"],
            }
            for p in range(SWEEP_POSITIONS)
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
    for p in range(SWEEP_POSITIONS):
        print(f"---   posición {p:>2}: {summary['by_position'][p]}")
    return path


def _cell_line(rec: ConversationRecord, cell: dict) -> str:
    sim = f"{rec.similarity_user:.3f}" if rec.similarity_user is not None else "  -  "
    return (
        f"[{cell['cell_index']:>3}] {rec.model_id:<18} {rec.topic_id:<16} "
        f"n={rec.n_turns:<2} {rec.paste_level} s{rec.sweep_position:>2} "
        f"sim={sim} {rec.artifact_kind or '-':<14} {rec.status}"
    )


if __name__ == "__main__":
    print(main())
