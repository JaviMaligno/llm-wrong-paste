"""Runner de la Fase 0: planifica las celdas, las corre y las escribe en JSONL.

Lo que este fichero garantiza, decisión por decisión:

- **D3** — rotación de ejes. El estrato de similaridad va *en el plan*, calculado
  como `(t + 3·m) % 8` sobre el índice de tema y el de modelo, y la longitud
  alterna como `LENGTHS[(t + m) % 2]`. Así ningún eje queda confundido con otro:
  cada estrato cae en tres temas distintos y cada tema aparece en ambas
  longitudes. El orden de ejecución es round-robin de modelos dentro de cada
  tema, porque una tirada de dos horas contra un gateway compartido en orden
  modelo-mayor confunde el modelo con la hora de reloj.
- **D4** — cobertura de `kind`. Dentro del estrato asignado se elige el artefacto
  cuyo género esté menos representado hasta ese momento en la tirada, de modo que
  la rúbrica no se derive sin haber visto nunca, por ejemplo, un `prompt` pegado.
- **D5** — identidad. La primera línea del fichero es el `run_header`, y cada
  fila lleva un `conversation_id` determinista y legible.
- **D6** — los fallos son datos. Cada celda va en su `try/except` y **siempre**
  escribe fila, con `status`; la tirada se puede reanudar sobre un fichero ya
  empezado y al final se imprime el recuento de planificadas, completadas y
  fallidas.
- **D7** — tras la reacción se siguen dos turnos con el usuario simulado, sin
  reparación.
- **D11** — tres celdas de control sin pegote, para que el formato, el runner y
  el verificador demuestren que lo soportan.
- **D12** — antes de gastar nada se mide el ancho del eje: se embeben los ocho
  prefijos contra el banco y se imprime min/mediana/máx de coseno por tema.

El prefijo **no** se construye aquí: viene de `prefixes.ensure_prefix`, que lo
fabrica una sola vez por (tema, longitud) y se lo sirve idéntico a los tres
modelos (D1). Llamar a `build_prefix` desde el runner devolvería el eje x a
depender de quién conteste.
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from wrongpaste import config, simulated_user
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.conversation import (
    MAX_TOKENS,
    N_POST_TURNS,
    continue_after_paste,
    conversation_text,
    inject_paste,
)
from wrongpaste.prefixes import PREFIX_MODEL, ensure_prefix
from wrongpaste.records import (
    ARM_NO_REPAIR,
    N_STRATA,
    ConversationRecord,
    RunHeader,
    make_conversation_id,
    run_header_line,
)
from wrongpaste.similarity import rank_artifacts, rank_stats, user_text
from wrongpaste.simulated_user import USER_MODEL
from wrongpaste.topics import Topic, load_topics

PHASE0_MODELS = ["gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5"]
LENGTHS = [2, 10]
STRATA = N_STRATA

# Salto de estrato entre modelos consecutivos (D3). Con 8 estratos y 3 modelos,
# 3 es coprimo con 8: los tres modelos de un mismo tema caen siempre en estratos
# distintos y el estrato deja de ser una función del tema.
STRATUM_STEP = 3

# D11: tres celdas sin pegote. No pretenden dar una tasa base creíble —eso es de
# la Fase 2—, sino demostrar que el formato, el runner y el verificador soportan
# el brazo de control.
N_CONTROL_CELLS = 3

# D12: por debajo de este rango de coseno, el tema no separa nada y la
# estratificación es decorativa. Es criterio explícito del GO/NO-GO.
NARROW_RANGE = 0.15

# Cuántos artefactos de la cola alta se listan en el informe de ejes, para que
# quien lo lea pueda juzgar si la cola alta está vacía (D12).
AXIS_TOP_N = 3

MASTER_SEED = 20260913

OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase0"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Rutas declaradas en la cabecera (D5). Viven en el repo del blog, no en este.
PLAN_PATH = (
    "docs/superpowers/plans/2026-09-13-pegado-accidental-fase-0.md"
)
SPEC_PATH = "docs/superpowers/specs/2026-09-13-pegado-accidental-design.md"

# El system prompt del usuario simulado es privado de su módulo; se lee así para
# poder registrarlo en la cabecera sin tocar un fichero que no es de este.
USER_SYSTEM_PROMPT = getattr(simulated_user, "_SYSTEM", "")

# Cuánto del cuerpo de error se guarda en la fila cuando una celda falla (D6).
ERROR_BODY_CHARS = 2000


# --- planificación (D3, D11) ----------------------------------------------


def plan_phase0(seed: int) -> list[dict]:
    """Devuelve las celdas de la Fase 0, en el orden en que se van a correr.

    Con `t` el índice del tema y `m` el del modelo (D3)::

        stratum = (t + 3 * m) % 8
        n_turns = LENGTHS[(t + m) % 2]

    El estrato viaja **en el dict**: no se deriva del orden de iteración, que es
    precisamente lo que hacía que estrato, tema y longitud fueran el mismo eje.

    El bucle exterior es el tema y el interior el modelo, o sea round-robin de
    modelos: si el gateway se degrada a mitad de tirada, la degradación se
    reparte entre los tres modelos en vez de caer entera sobre el último.

    Al final se añaden las tres celdas de control sin pegote (D11). Cada una
    reutiliza el prefijo de la longitud que **no** usa ese mismo modelo en ese
    mismo tema: así no genera ningún prefijo nuevo (cada tema se fabrica en las
    dos longitudes de todos modos) y su `conversation_id`, que lleva la longitud
    dentro, no puede chocar con el de la celda con pegote.
    """
    topics = load_topics()
    rng = np.random.default_rng(seed)
    plan: list[dict] = []

    for t, topic in enumerate(topics):
        for m, model_id in enumerate(PHASE0_MODELS):
            n_turns = LENGTHS[(t + m) % len(LENGTHS)]
            plan.append(
                _cell(
                    cell_index=len(plan),
                    model_id=model_id,
                    topic_id=topic.id,
                    n_turns=n_turns,
                    stratum=(t + STRATUM_STEP * m) % STRATA,
                    condition="paste",
                    seed=int(rng.integers(0, 2**31)),
                )
            )

    for i in range(N_CONTROL_CELLS):
        t = i % len(topics)
        m = i % len(PHASE0_MODELS)
        # La longitud contraria a la de la celda con pegote de ese (tema, modelo).
        n_turns = LENGTHS[(t + m + 1) % len(LENGTHS)]
        plan.append(
            _cell(
                cell_index=len(plan),
                model_id=PHASE0_MODELS[m],
                topic_id=topics[t].id,
                n_turns=n_turns,
                # Sin pegote no hay estrato que asignar.
                stratum=-1,
                condition="no_paste",
                seed=int(rng.integers(0, 2**31)),
            )
        )

    return plan


def _cell(
    cell_index: int,
    model_id: str,
    topic_id: str,
    n_turns: int,
    stratum: int,
    condition: str,
    seed: int,
    replicate_idx: int = 0,
) -> dict:
    """Una celda del plan, con su `conversation_id` ya resuelto (D5)."""
    return {
        "cell_index": cell_index,
        "model_id": model_id,
        "topic_id": topic_id,
        "n_turns": n_turns,
        "stratum": stratum,
        "condition": condition,
        "replicate_idx": replicate_idx,
        "seed": seed,
        "conversation_id": make_conversation_id(
            model_id, topic_id, n_turns, replicate_idx
        ),
    }


# --- muestreo del artefacto (D4) ------------------------------------------


def stratum_window(
    ranked: list[tuple[Artifact, float]], stratum: int, n_strata: int = STRATA
) -> list[tuple[Artifact, float]]:
    """El trozo del ranking que corresponde a un estrato.

    Se estratifica por **posición** en el ranking, no por valor de coseno: la
    distribución real está muy concentrada y los estratos por valor saldrían
    vacíos. La ventana nunca es vacía, aunque el banco sea más pequeño que el
    número de estratos.
    """
    if not ranked:
        raise ValueError("el ranking está vacío: no hay de dónde muestrear")
    if not 0 <= stratum < n_strata:
        raise ValueError(f"estrato fuera de rango: {stratum} (de {n_strata})")
    lo = len(ranked) * stratum // n_strata
    hi = len(ranked) * (stratum + 1) // n_strata
    return ranked[lo : max(hi, lo + 1)]


def pick_artifact(
    ranked: list[tuple[Artifact, float]],
    stratum: int,
    kind_counts: Counter,
    rng: np.random.Generator,
    n_strata: int = STRATA,
) -> tuple[Artifact, float]:
    """Elige artefacto dentro del estrato, cubriendo géneros (D4).

    De la ventana del estrato se queda con los artefactos cuyo `kind` esté menos
    representado **hasta ese momento en la tirada**, y desempata al azar con el
    `rng` de la celda. Con 24 celdas y 11 géneros, dejarlo a suerte permitía
    derivar la rúbrica sin haber visto nunca un `prompt` pegado, que es justo el
    caso donde el pegote es una instrucción ejecutable.

    No toca `kind_counts`: la contabilidad la lleva quien llama, que es quien
    sabe si la celda llegó a correrse.
    """
    window = stratum_window(ranked, stratum, n_strata)
    fewest = min(kind_counts[art.kind] for art, _ in window)
    candidates = [pair for pair in window if kind_counts[pair[0].kind] == fewest]
    return candidates[int(rng.integers(0, len(candidates)))]


# --- ancho del eje (D12) ---------------------------------------------------


def axis_path(run_id: str) -> Path:
    """Dónde se guarda el informe de ejes de una tirada."""
    return OUT_DIR / f"axis-{run_id}.json"


def summary_path(run_id: str) -> Path:
    """Dónde se guarda el resumen de planificadas/completadas/fallidas (D6)."""
    return OUT_DIR / f"summary-{run_id}.json"


def measure_axis(
    topics: list[Topic] | None = None,
    arts: list[Artifact] | None = None,
    n_turns: int = LENGTHS[0],
    run_id: str = "",
) -> dict[str, Any]:
    """Embebe los ocho prefijos contra el banco y mide el rango por tema (D12).

    Se corre **antes** de la tirada y cuesta céntimos. Sirve para no gastar el
    presupuesto en un eje que no separa: si un tema tiene un rango de coseno por
    debajo de `NARROW_RANGE`, o su cola alta está vacía, la estratificación de
    ese tema es decorativa y hay que completar el banco antes de seguir.

    Usa el prefijo corto de cada tema, que es el que `ensure_prefix` ya va a
    fabricar para la tirada: no genera trabajo extra, solo lo adelanta.

    Devuelve el informe (que `main` guarda en `runs/phase0/axis-<run_id>.json`)
    y lo imprime por pantalla de camino.
    """
    topics = load_topics() if topics is None else topics
    arts = load_artifacts() if arts is None else arts

    per_topic: list[dict[str, Any]] = []
    print(f"--- ancho del eje (D12): {len(arts)} artefactos, n_turns={n_turns}")
    for topic in topics:
        prefix = ensure_prefix(topic, n_turns)
        ranking, truncated = rank_artifacts(user_text(prefix["transcript"]), arts)
        stats = rank_stats(ranking)
        entry = {
            "topic_id": topic.id,
            "prefix_id": prefix["prefix_id"],
            "n_turns": int(n_turns),
            "min": stats["min"],
            "median": stats["median"],
            "max": stats["max"],
            "range": stats["max"] - stats["min"],
            "narrow": (stats["max"] - stats["min"]) < NARROW_RANGE,
            "truncated": bool(truncated),
            # La cola alta, para poder juzgar si está vacía de dominio.
            "top": [
                {"artifact_id": art.id, "kind": art.kind, "similarity": sim}
                for art, sim in ranking[-AXIS_TOP_N:][::-1]
            ],
        }
        per_topic.append(entry)
        print(
            f"  {topic.id:<16} min={entry['min']:.3f} "
            f"med={entry['median']:.3f} max={entry['max']:.3f} "
            f"rango={entry['range']:.3f}"
            + ("  <-- ESTRECHO" if entry["narrow"] else "")
        )

    narrow = [e["topic_id"] for e in per_topic if e["narrow"]]
    if narrow:
        print(f"  OJO: temas con rango < {NARROW_RANGE}: {', '.join(narrow)}")

    return {
        "run_id": run_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "prefix_model": PREFIX_MODEL,
        "n_turns": int(n_turns),
        "n_artifacts": len(arts),
        "narrow_threshold": NARROW_RANGE,
        "narrow_topics": narrow,
        "topics": per_topic,
    }


# --- identidad de la tirada (D5) ------------------------------------------


def _sha_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def bank_sha(arts: list[Artifact]) -> str:
    """Huella del banco de artefactos: si cambia, las tiradas no son comparables."""
    return _sha_of([[a.id, a.kind, a.text, list(a.entities)] for a in arts])


def topics_sha(topics: list[Topic]) -> str:
    """Huella del banco de temas, por el mismo motivo."""
    return _sha_of([[t.id, t.opening, list(t.goals)] for t in topics])


def code_sha() -> str:
    """SHA del commit con el que se corrió; cadena vacía si no se puede saber.

    Se lee del `.git` a pelo en vez de con `git rev-parse` porque el runner no
    tiene por qué correr con un git utilizable delante, y una tirada de dos
    horas no se cae por no saber el sha.
    """
    try:
        head = (REPO_ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = (REPO_ROOT / ".git" / head[5:]).read_text(encoding="utf-8")
            return ref.strip()
        return head
    except OSError:
        return ""


# --- reanudación (D6) ------------------------------------------------------


def resume_state(path: Path) -> tuple[set[str], Counter]:
    """Lee un JSONL a medias: qué celdas ya están y qué géneros se han gastado.

    Los `conversation_id` presentes se saltan al reanudar, y los `artifact_kind`
    ya escritos vuelven al contador de D4: sin eso, reanudar rompería la
    cobertura de géneros justo en las celdas que quedan por correr.

    Una línea ilegible se ignora en vez de tumbar la reanudación: un fichero
    truncado a mitad de escritura es exactamente el caso que esto tiene que
    sobrevivir.
    """
    done: set[str] = set()
    kinds: Counter = Counter()
    if not path.exists():
        return done, kinds
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "run_header":
            continue
        cid = row.get("conversation_id")
        if cid:
            done.add(cid)
        if row.get("artifact_kind"):
            kinds[row["artifact_kind"]] += 1
    return done, kinds


def read_run_header(path: Path) -> dict:
    """La primera línea del fichero, si es una cabecera; `{}` si no."""
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return {}
        return row if row.get("kind") == "run_header" else {}
    return {}


def _status_for_error(exc: BaseException) -> tuple[str, int | str | None, str]:
    """Traduce una excepción al vocabulario cerrado de `status` (D6).

    `refusal` no se decide aquí: una negativa del modelo llega con HTTP 200 y
    texto, así que es `ok` en el fichero y la marca quien anota. Lo que sí es
    `refusal` es que el proveedor corte por filtro de contenido, y eso llega
    como error del proveedor con su propio código.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:ERROR_BODY_CHARS]
        if exc.response.status_code == 400 and "content" in body.lower():
            return "refusal", exc.response.status_code, body
        return "http_error", exc.response.status_code, body
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", exc.__class__.__name__, str(exc)[:ERROR_BODY_CHARS]
    # Cualquier otra cosa entra como `http_error` porque el vocabulario está
    # cerrado; el tipo real de la excepción queda en `error_code`.
    return "http_error", exc.__class__.__name__, str(exc)[:ERROR_BODY_CHARS]


# --- ejecución de una celda -----------------------------------------------


def rank_for_prefix(
    prefix: dict,
    arts: list[Artifact],
    cache: dict[str, tuple],
) -> tuple[list[tuple[Artifact, float]], dict[str, float], bool]:
    """Ranking del banco contra un prefijo, cacheado por `prefix_id`.

    Devuelve el ranking primario (D2: solo el lado del usuario, ascendente por
    coseno), el diccionario de similaridades secundarias (conversación entera)
    por `artifact_id`, y si hubo que recortar algún texto antes de embeber.

    El caché no es una optimización cualquiera: como el prefijo es el mismo para
    los tres modelos (D1), el eje x de una celda **tiene** que salir idéntico
    para los tres. Recalcularlo por celda lo dejaría a merced del ruido del
    embedder.
    """
    pid = prefix["prefix_id"]
    if pid not in cache:
        transcript = prefix["transcript"]
        ranking, truncated_user = rank_artifacts(user_text(transcript), arts)
        full, truncated_full = rank_artifacts(conversation_text(transcript), arts)
        cache[pid] = (
            ranking,
            {art.id: sim for art, sim in full},
            bool(truncated_user or truncated_full),
        )
    return cache[pid]


def run_cell(
    cell: dict,
    topic: Topic,
    arts: list[Artifact],
    rank_cache: dict[str, tuple],
    kind_counts: Counter,
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
) -> ConversationRecord:
    """Corre una celda entera y devuelve su fila.

    `partial` se va rellenando sobre la marcha con lo que ya se sabe (prefijo,
    artefacto, transcripción), para que si esto revienta a mitad, la fila de
    fallo que escribe `main` no salga vacía (D6).

    Secuencia: prefijo compartido (D1), muestreo del artefacto dentro del
    estrato del plan (D3/D4), pegote literal, y dos turnos más con el usuario
    simulado sin reparación (D7). En las celdas de control (D11) se salta el
    pegote y se va directo a los turnos posteriores.
    """
    model_id = cell["model_id"]
    prefix = ensure_prefix(topic, cell["n_turns"])
    partial["prefix_id"] = prefix["prefix_id"]

    # Copia: el prefijo cacheado lo comparten los tres modelos y no se muta.
    transcript = [dict(m) for m in prefix["transcript"]]
    partial["transcript"] = transcript

    artifact: Artifact | None = None
    similarity_user: float | None = None
    similarity_full: float | None = None
    similarity_rank: int | None = None
    similarity_pct: float | None = None
    ranking_rows: list[dict] | None = None
    truncated = False
    paste_index: int | None = None
    reaction: str | None = None
    usages: list[dict] = []

    if cell["condition"] == "paste":
        rng = np.random.default_rng(cell["seed"])
        ranking, full_by_id, truncated = rank_for_prefix(prefix, arts, rank_cache)
        artifact, similarity_user = pick_artifact(
            ranking, cell["stratum"], kind_counts, rng
        )
        kind_counts[artifact.kind] += 1
        partial["artifact"] = artifact

        similarity_full = full_by_id.get(artifact.id)
        ids = [art.id for art, _ in ranking]
        similarity_rank = ids.index(artifact.id)
        # Percentil dentro del banco, 0 = el más lejano, 1 = el más parecido.
        similarity_pct = (
            similarity_rank / (len(ids) - 1) if len(ids) > 1 else 0.0
        )
        ranking_rows = [
            {"artifact_id": art.id, "similarity": sim} for art, sim in ranking
        ]

        reaction, paste_usage, paste_index = inject_paste(
            model_id, transcript, artifact
        )
        usages.append(paste_usage)

    post_indices, post_usages = continue_after_paste(
        model_id, transcript, topic, n_post=N_POST_TURNS
    )
    usages.extend(post_usages)

    # Una celda con pegote que devuelve reacción vacía no es un `ok`: es texto
    # que no está, y el análisis tiene que poder descartarla sin leerla.
    status = "ok"
    if cell["condition"] == "paste" and not (reaction or "").strip():
        status = "empty"

    ended_at = time.time()
    return ConversationRecord(
        run_id=run_id,
        conversation_id=cell["conversation_id"],
        cell_index=cell["cell_index"],
        replicate_idx=cell["replicate_idx"],
        model_id=model_id,
        model_label=_model_label(model_id),
        topic_id=topic.id,
        n_turns=cell["n_turns"],
        stratum=cell["stratum"],
        n_strata=STRATA,
        prefix_id=prefix["prefix_id"],
        condition=cell["condition"],
        arm=ARM_NO_REPAIR,
        artifact_id=artifact.id if artifact else None,
        artifact_kind=artifact.kind if artifact else None,
        artifact_text=artifact.text if artifact else None,
        artifact_entities=list(artifact.entities) if artifact else None,
        similarity_user=similarity_user,
        similarity_full=similarity_full,
        similarity_rank=similarity_rank,
        similarity_pct=similarity_pct,
        ranking=ranking_rows,
        similarity_text_truncated=truncated,
        paste_index=paste_index,
        post_indices=post_indices,
        transcript=transcript,
        reaction=reaction,
        user_model=USER_MODEL,
        prefix_model=prefix.get("prefix_model", PREFIX_MODEL),
        max_tokens=MAX_TOKENS,
        # Solo las llamadas pagadas por esta celda: las del prefijo se pagaron
        # una vez y viven en `runs/prefixes/<prefix_id>.json` (D1).
        usages=usages,
        seed=cell["seed"],
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=int((ended_at - started_at) * 1000),
        status=status,
    )


def _model_label(model_id: str) -> str:
    model = config.MODELS.get(model_id)
    return model.label if model else ""


def failed_record(
    cell: dict,
    topic_id: str,
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
    exc: BaseException,
) -> ConversationRecord:
    """La fila que se escribe cuando una celda revienta (D6).

    Lleva todo lo que se llegó a saber antes del fallo. Un fichero con 19 filas
    y ninguna marca es indistinguible de uno completo; con esto, no.
    """
    status, error_code, error_body = _status_for_error(exc)
    artifact: Artifact | None = partial.get("artifact")
    ended_at = time.time()
    return ConversationRecord(
        run_id=run_id,
        conversation_id=cell["conversation_id"],
        cell_index=cell["cell_index"],
        replicate_idx=cell["replicate_idx"],
        model_id=cell["model_id"],
        model_label=_model_label(cell["model_id"]),
        topic_id=topic_id,
        n_turns=cell["n_turns"],
        stratum=cell["stratum"],
        n_strata=STRATA,
        prefix_id=partial.get("prefix_id", ""),
        condition=cell["condition"],
        arm=ARM_NO_REPAIR,
        artifact_id=artifact.id if artifact else None,
        artifact_kind=artifact.kind if artifact else None,
        artifact_text=artifact.text if artifact else None,
        artifact_entities=list(artifact.entities) if artifact else None,
        transcript=partial.get("transcript", []),
        user_model=USER_MODEL,
        prefix_model=PREFIX_MODEL,
        max_tokens=MAX_TOKENS,
        seed=cell["seed"],
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=int((ended_at - started_at) * 1000),
        status=status,
        error_code=error_code,
        error_body=error_body,
        attempts=int(getattr(exc, "attempts", 1)),
    )


# --- tirada ----------------------------------------------------------------


def build_header(
    run_id: str,
    seed: int,
    plan: list[dict],
    arts: list[Artifact],
    topics: list[Topic],
    started_at: float,
) -> dict:
    """La primera línea del JSONL (D5)."""
    return run_header_line(
        RunHeader(
            run_id=run_id,
            phase="0",
            plan_path=PLAN_PATH,
            spec_path=SPEC_PATH,
            code_sha=code_sha(),
            bank_sha=bank_sha(arts),
            topics_sha=topics_sha(topics),
            master_seed=seed,
            roster=list(PHASE0_MODELS),
            planned_cells=len(plan),
            embedding_model=config.EMBEDDING_MODEL,
            prefix_model=PREFIX_MODEL,
            user_model=USER_MODEL,
            user_system_prompt=USER_SYSTEM_PROMPT,
            started_at=started_at,
        )
    )


def main(
    seed: int = MASTER_SEED,
    out: Path | str | None = None,
    measure: bool = True,
) -> Path:
    """Corre la Fase 0 entera y devuelve la ruta del JSONL.

    Si `out` apunta a un fichero que ya existe, la tirada **se reanuda**: se
    saltan los `conversation_id` que ya están dentro y se sigue escribiendo al
    final del mismo fichero (D6). Si no, se crea uno nuevo con la cabecera.

    `measure=False` salta el paso de ancho de eje (D12); solo para pruebas, la
    tirada de verdad lo quiere delante.
    """
    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}
    arts = load_artifacts()
    plan = plan_phase0(seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resuming = out is not None and Path(out).exists()
    path = (
        Path(out)
        if out is not None
        else OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )

    if resuming:
        run_id = read_run_header(path).get("run_id") or path.stem
        done, kind_counts = resume_state(path)
        print(f"reanudando {path.name}: {len(done)} celdas ya escritas")
    else:
        run_id = path.stem
        done, kind_counts = set(), Counter()

    if measure:
        axis = measure_axis(topics=topics, arts=arts, run_id=run_id)
        axis_path(run_id).parent.mkdir(parents=True, exist_ok=True)
        axis_path(run_id).write_text(
            json.dumps(axis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    rank_cache: dict[str, tuple] = {}
    completed = 0
    failed = 0
    skipped = 0

    with path.open("a" if resuming else "w", encoding="utf-8") as fh:
        if not resuming:
            header = build_header(run_id, seed, plan, arts, topics, time.time())
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            fh.flush()

        for cell in plan:
            if cell["conversation_id"] in done:
                skipped += 1
                continue

            topic = topics_by_id[cell["topic_id"]]
            started_at = time.time()
            partial: dict[str, Any] = {}
            try:
                rec = run_cell(
                    cell,
                    topic,
                    arts,
                    rank_cache,
                    kind_counts,
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

            if rec.status == "ok":
                completed += 1
            else:
                failed += 1
            print(_cell_line(rec, cell))

    summary = {
        "run_id": run_id,
        "path": str(path),
        "planned": len(plan),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "kinds": dict(sorted(kind_counts.items())),
    }
    summary_path(run_id).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"--- planificadas: {summary['planned']} | completadas: {completed} "
        f"| fallidas: {failed} | saltadas: {skipped}"
    )
    print(f"--- géneros cubiertos: {len(kind_counts)} -> {summary['kinds']}")
    return path


def _cell_line(rec: ConversationRecord, cell: dict) -> str:
    sim = f"{rec.similarity_user:.3f}" if rec.similarity_user is not None else "  -  "
    return (
        f"[{cell['cell_index']:>2}] {rec.model_id:<18} {rec.topic_id:<16} "
        f"n={rec.n_turns:<2} s={rec.stratum:<2} {rec.condition:<8} "
        f"sim={sim} {rec.artifact_kind or '-':<14} {rec.status}"
    )


if __name__ == "__main__":
    print(main())
