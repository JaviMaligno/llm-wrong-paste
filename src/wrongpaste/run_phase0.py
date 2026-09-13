"""Runner de la Fase 0: planifica las celdas, las corre y las escribe en JSONL.

Lo que este fichero garantiza, decisión por decisión:

- **D3** — rotación de ejes. El estrato de similaridad va *en el plan*, calculado
  con un **desplazamiento por modelo** (`STRATUM_SHIFTS`), y la longitud alterna
  como `LENGTHS[(t + m) % 2]`. **Ningún mapa lineal `(t + c·m) % 8` sirve**: con
  `c` impar la paridad del estrato ES el índice de longitud (`c·m ≡ m mod 2`) y
  con `c` par los tres temas de cada estrato (`t = s − c·m`) tienen todos la
  misma paridad, así que el estrato vuelve a decir de qué tema se trata. Los
  desplazamientos `(0, 3, 5)` rompen las dos cosas a la vez: cada estrato cae en
  los temas `{s, s−3, s−5}`, de paridades `{s, s+1, s+1}` —mixtas—, y en las
  longitudes `{s, s, s+1}` —mixtas también—. El orden de ejecución es
  round-robin de modelos dentro de cada tema, porque una tirada de dos horas
  contra un gateway compartido en orden modelo-mayor confunde el modelo con la
  hora de reloj.
- **D4** — cobertura de `kind`. Dentro del estrato asignado se elige el artefacto
  cuyo género esté menos representado hasta ese momento en la tirada, de modo que
  la rúbrica no se derive sin haber visto nunca, por ejemplo, un `prompt` pegado.
- **D5** — identidad. La primera línea del fichero es el `run_header`, y cada
  fila lleva un `conversation_id` determinista y legible, más lo que se envió
  (`request_params`, `system_prompt`) y lo que contestó el proveedor
  (`response_model`, `stop_reasons`, `attempts`).
- **D6** — los fallos son datos. Cada celda va en su `try/except` y **siempre**
  escribe fila, con `status`; la tirada se puede reanudar sobre un fichero ya
  empezado —y solo se dan por hechas las celdas con `status` en `DONE_STATUSES`,
  para que una que murió por un 429 transitorio se reintente— y al final se
  imprime el recuento de planificadas, completadas y fallidas. Dos cosas que la
  primera versión hacía mal y aquí ya no:
  · **El origen del fallo viaja en el `status`.** Las llamadas del usuario
    simulado, de la fabricación del prefijo y del embedder NO son conducta del
    modelo evaluado: si revientan, la fila sale como `harness_error` y nunca
    como `refusal` (ver `failure_origin`).
  · **Los fallos del usuario simulado no son invisibles.** Un turno suyo vacío o
    cortado por el tope rompe la conversación que se mide, y antes la fila decía
    `ok`: `_has_empty_response` y `status_for_stop_reasons` solo miraban los
    `Reply` del modelo evaluado. Ahora sus turnos también se traen —los del
    prefijo desde `user_reply_traces`, los de después del pegote pasando
    `user_replies_out` a `continue_after_paste`—, viajan en la fila
    (`user_stop_reasons`, `user_reply_traces`) y, si alguno está roto, la celda
    sale `harness_error` (ver `user_side_problems`). Nunca `ok` ni `refusal`:
    quien se calló no es el modelo que se está midiendo.
  · **Un corte por `max_tokens` no es un `ok`.** Sale como `truncated`: es un
    artefacto del arnés disfrazado de «el modelo lo ignora» (ver
    `status_for_stop_reasons`).
  · **Reanudar no duplica filas.** Al reanudar se reescribe el fichero sin las
    filas de las celdas que se van a reintentar, y esas filas viejas se guardan
    en `superseded-<run_id>.jsonl`: así el recuento final cuadra con
    `planned_cells` (ver `compact_resume_file`).
- **D7** — tras la reacción se siguen dos turnos con el usuario simulado, sin
  reparación.
- **D10** — sonda de caché. `probe_cache()` repite dos veces la misma llamada
  (prefijo + pegote) contra un modelo de cada proveedor y reporta el
  `cache_read` de cada una. Es el único sitio donde el criterio de D10 se puede
  comprobar: en el plan de la Fase 0 no hay dos celdas del mismo proveedor
  Anthropic que compartan `prefix_id`, así que el criterio «dos celdas con el
  mismo `prefix_id` leen caché» no se puede verificar sobre la tirada.
- **D11** — tres celdas de control sin pegote, repartidas entre las dos
  longitudes, para que el formato, el runner y el verificador demuestren que lo
  soportan.
- **D12** — antes de la tirada se mide el ancho del eje: se embeben los ocho
  prefijos **en las dos longitudes** contra el banco y se imprime min/mediana/máx
  de coseno por (tema, longitud). Medir solo la longitud corta dejaría el
  criterio GO/NO-GO ciego a la mitad de la tirada. Y el criterio **frena**: si
  alguna celda sale estrecha, `main()` levanta `NarrowAxisError` y no corre
  ninguna celda; para seguir a propósito hay que pedirlo (`allow_narrow=True`,
  `--force`). Medirlo, imprimir «ESTRECHO» y seguir gastando era tener el
  criterio escrito y no aplicarlo.
  **Qué ahorra exactamente la puerta, dicho sin adornos.** Medir el eje **no es
  gratis**: para medirlo hay que tener los 16 prefijos (8 temas × 2 longitudes),
  y `measure_axis` los pide a `ensure_prefix`, que genera los que falten
  llamando de verdad a `PREFIX_MODEL` y al usuario simulado. La puerta salta
  *después* de ese gasto. Lo que ahorra son las **27 celdas** contra los tres
  modelos evaluados —tres llamadas cada una (pegote + los dos turnos de D7)—,
  que es el grueso del presupuesto. Y lo que se gasta en prefijos no se tira: se
  persisten en `runs/prefixes/` y los reutiliza tal cual la tirada siguiente,
  incluida la que se lance después de completar el banco (D12 no cambia los
  temas, solo los artefactos). El informe de ejes dice cuántos prefijos se
  generaron y cuántos se reutilizaron, para que el coste sea auditable y no
  haya que creerse este párrafo.

El prefijo **no** se construye aquí: viene de `prefixes.ensure_prefix`, que lo
fabrica una sola vez por (tema, longitud) y se lo sirve idéntico a los tres
modelos (D1). Llamar a `build_prefix` desde el runner devolvería el eje x a
depender de quién conteste.
"""

import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from wrongpaste import config, records, simulated_user
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.clients import chat
from wrongpaste.conversation import (
    MAX_TOKENS,
    N_POST_TURNS,
    continue_after_paste,
    conversation_text,
    inject_paste,
    reply_traces,
)
from wrongpaste.prefixes import PREFIX_MODEL, ensure_prefix, find_prefix
from wrongpaste.records import (
    ARM_NO_REPAIR,
    N_STRATA,
    ConversationRecord,
    RunHeader,
    make_conversation_id,
    run_header_line,
)
from wrongpaste.similarity import rank_artifacts, rank_stats, user_text
from wrongpaste.simulated_user import (
    USER_MODEL,
    SimulatedUserError,
)
from wrongpaste.topics import Topic, load_topics

PHASE0_MODELS = ["gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5"]
LENGTHS = [2, 10]
STRATA = N_STRATA

# Desplazamiento de estrato **por modelo** (D3): `stratum = (t + SHIFTS[m]) % 8`.
#
# No es un salto constante, y no puede serlo: con `stratum = (t + c·m) % 8`
# ningún `c` sirve. Si `c` es impar, `c·m ≡ m (mod 2)` y la paridad del estrato
# es literalmente el índice de longitud `(t + m) % 2`: los estratos pares salen
# todos en conversaciones de 2 turnos y los impares en las de 10. Si `c` es par,
# los tres temas que tocan un estrato `s` son `t = s − c·m`, todos de la misma
# paridad que `s`: saber el estrato vuelve a ser saber de qué tema va la
# conversación, que es la confusión que D3 venía a deshacer.
#
# Con desplazamientos (0, 3, 5) las dos confusiones caen a la vez. Comprobado a
# mano: los temas de un estrato `s` son `{s, s−3, s−5}`, de paridades
# `{s, s+1, s+1}` —mixtas—, y sus longitudes son `LENGTHS[(t + m) % 2]` con
# `t = s − SHIFTS[m]`, o sea `{s, s, s+1}` —mixtas también—. Siguen valiendo las
# propiedades que ya se exigían: tres temas distintos por estrato (los tres
# desplazamientos son distintos módulo 8), cada modelo recorre los ocho estratos
# (para `m` fijo, `t` barre 0..7) y cada tema aparece en las dos longitudes
# (para `t` fijo, `(t + m) % 2` toma los dos valores).
STRATUM_SHIFTS: tuple[int, ...] = (0, 3, 5)

# D11: tres celdas sin pegote. No pretenden dar una tasa base creíble —eso es de
# la Fase 2—, sino demostrar que el formato, el runner y el verificador soportan
# el brazo de control. Se reparten entre las dos longitudes: un control que solo
# existe en conversaciones largas no sirve de tasa base para las cortas.
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
# El documento de diseño: de dónde salieron los temas, el banco y las cinco
# categorías de conducta.
SPEC_PATH = "docs/superpowers/specs/2026-09-13-pegado-accidental-design.md"
# El documento que **gobierna esta tirada**: las decisiones D1..D17 que corrigen
# el diseño. Van los dos en la cabecera a propósito: quien lea un JSONL dentro
# de seis meses necesita saber no solo de qué experimento es, sino con qué
# versión de las decisiones se corrió. Registrar solo el diseño dejaba fuera
# justo el documento que fija la rotación de ejes, el vocabulario de `status` y
# el criterio GO/NO-GO.
CORRECTIONS_PATH = (
    "docs/superpowers/specs/2026-09-13-pegado-accidental-correcciones.md"
)

# El system prompt del usuario simulado es privado de su módulo; se lee así para
# poder registrarlo en la cabecera sin tocar un fichero que no es de este.
USER_SYSTEM_PROMPT = getattr(simulated_user, "_SYSTEM", "")

# Cuánto del cuerpo de error se guarda en la fila cuando una celda falla (D6).
ERROR_BODY_CHARS = 2000

# --- vocabulario de `status` que añade este runner (D6) --------------------
#
# `records.STATUSES` cerraba el vocabulario en `ok | http_error | timeout |
# refusal | empty`, y con ese vocabulario dos cosas distintas salían disfrazadas
# de otra: un fallo de una llamada **del arnés** (usuario simulado, prefijo,
# embedder) acababa clasificado como conducta del modelo evaluado, y un corte
# por tope de tokens salía como `ok`.
#
# D6: estados que usa el runner. El vocabulario vive en `records.STATUSES`, que
# es donde `ConversationRecord` lo valida; aquí solo se les pone nombre para no
# repetir literales. Antes se registraban desde aquí mutando `records.STATUSES`
# al importar este módulo, lo que hacía que un análisis que importara solo
# `records` rechazara filas válidas de su propio JSONL.
HARNESS_ERROR = "harness_error"
TRUNCATED = "truncated"
assert HARNESS_ERROR in records.STATUSES and TRUNCATED in records.STATUSES

# D6: qué cuenta como celda hecha al reanudar. `ok` es el resultado normal y
# `refusal` es **un resultado** —que un modelo se niegue es conducta, no avería—,
# así que ninguno de los dos se repite. `truncated` tampoco se repite: el corte
# lo produce el `max_tokens` que mandamos nosotros, así que reintentar la celda
# tal cual vuelve a cortarla en el mismo sitio y solo gasta dinero; para
# arreglarla hay que subir el tope y volver a correrla a propósito.
# `http_error`, `timeout`, `harness_error` y `empty` son averías del arnés o del
# proveedor: se vuelven a intentar, porque si no, una celda que murió por un 429
# transitorio no se corre jamás y el hueco queda en la zona interesante del
# diseño (conversaciones largas, cosenos altos).
DONE_STATUSES: frozenset[str] = frozenset({"ok", "refusal", TRUNCATED})


# --- la fila de la Fase 0 (D5/D6) ------------------------------------------


@dataclass
class Phase0Record(ConversationRecord):
    """La fila de `records` más las dos columnas del **lado del usuario**.

    En una celda hablan dos modelos: el evaluado y el usuario simulado. La fila
    de `records` solo tenía sitio para el primero (`stop_reasons`,
    `response_model`, `attempts`), así que el `stop_reason` de los turnos del
    usuario simulado no llegaba a ningún fichero: si uno salía cortado por el
    tope, la conversación medida tenía un agujero y el JSONL no lo decía en
    ninguna parte.

    - `user_stop_reasons`: un `stop_reason` por llamada del usuario simulado, en
      orden. Es el equivalente exacto de `stop_reasons`, para el otro papel.
    - `user_reply_traces`: la traza entera de esas llamadas (`stop_reason`,
      `response_model`, `attempts`, `latency_ms`) más un `stage` que dice si el
      turno es del prefijo compartido (D1) o de los turnos posteriores al pegote
      (D7). Los del prefijo se pagan una vez y se sirven a los tres modelos, así
      que aparecen —idénticos— en las tres filas que comparten `prefix_id`.

    Es la misma costura declarada que `EXTRA_STATUSES`: `records.py` no es de
    este agente y las columnas nuevas se añaden aquí, en una subclase que hereda
    la validación y el `to_json()` del original. Queda anotado para que `records`
    las adopte como campos propios de `ConversationRecord`.
    """

    user_stop_reasons: list[str | None] = field(default_factory=list)
    user_reply_traces: list[dict] = field(default_factory=list)


# --- planificación (D3, D11) ----------------------------------------------


def plan_phase0(seed: int) -> list[dict]:
    """Devuelve las celdas de la Fase 0, en el orden en que se van a correr.

    Con `t` el índice del tema y `m` el del modelo (D3)::

        stratum = (t + STRATUM_SHIFTS[m]) % 8      # SHIFTS = (0, 3, 5)
        n_turns = LENGTHS[(t + m) % 2]

    El estrato viaja **en el dict**: no se deriva del orden de iteración, que es
    precisamente lo que hacía que estrato, tema y longitud fueran el mismo eje.
    Los desplazamientos no son intercambiables por un salto lineal `c·m`:
    cualquier `c` impar pega el estrato a la longitud y cualquier `c` par lo pega
    a la paridad del tema. El porqué, con la comprobación a mano, está en
    `STRATUM_SHIFTS`.

    **Réplicas (D1).** D1 dice que las réplicas comparten prefijo *y* artefacto,
    porque lo que una réplica debe medir es el muestreo de la respuesta y nada
    más. **La Fase 0 no tiene réplicas**: `replicate_idx` es siempre 0, hay una
    sola celda por (tema, modelo) y el mecanismo no está implementado aquí ni
    hace falta fingir que lo está. Lo que sí queda apuntado para quien planifique
    la Fase 1: hoy el artefacto se elige con un `rng` sembrado por el `seed` **de
    la celda**, así que dos réplicas de la misma celda con semillas distintas
    elegirían artefactos distintos y dejarían de ser réplicas. Al añadirlas, el
    artefacto tiene que pasar a ser función de `(prefix_id, stratum)` —que es lo
    que comparten las réplicas— y no del seed de la celda; el seed se queda para
    lo que sí debe variar.

    El bucle exterior es el tema y el interior el modelo, o sea round-robin de
    modelos: si el gateway se degrada a mitad de tirada, la degradación se
    reparte entre los tres modelos en vez de caer entera sobre el último.

    Al final se añaden las tres celdas de control sin pegote (D11). Cada una
    coge la longitud **contraria** a la de la celda con pegote de ese mismo
    (tema, modelo): así no genera ningún prefijo nuevo (cada tema se fabrica en
    las dos longitudes de todos modos) y su `conversation_id`, que lleva la
    longitud dentro, no puede chocar con el de la celda con pegote. Los tres
    (tema, modelo) se eligen con `t = 2·i` para que `t + m` cambie de paridad
    entre controles y los tres no acaben en la misma longitud.
    """
    if len(STRATUM_SHIFTS) != len(PHASE0_MODELS):
        raise ValueError(
            "hay que dar un desplazamiento de estrato por modelo: "
            f"{len(STRATUM_SHIFTS)} desplazamientos para "
            f"{len(PHASE0_MODELS)} modelos"
        )

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
                    stratum=(t + STRATUM_SHIFTS[m]) % STRATA,
                    condition="paste",
                    seed=int(rng.integers(0, 2**31)),
                )
            )

    for i in range(N_CONTROL_CELLS):
        t = (2 * i) % len(topics)
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


# --- ficheros laterales de una tirada --------------------------------------


def _sidecar_path(name: str, run_id: str, out: Path | str | None = None) -> Path:
    """Ruta de un fichero lateral (ejes, resumen, sonda) de una tirada.

    Sigue **al fichero de salida**, no a `OUT_DIR`: si alguien corre la tirada
    con `out=/otro/sitio/tirada.jsonl`, el informe de ejes y el resumen tienen
    que quedar al lado de sus datos. Escribirlos siempre en `OUT_DIR` los dejaba
    huérfanos —o, peor, pisando los de otra tirada con el mismo `run_id`.
    """
    base = Path(out).parent if out is not None else OUT_DIR
    return base / f"{name}-{run_id}.json"


def axis_path(run_id: str, out: Path | str | None = None) -> Path:
    """Dónde se guarda el informe de ejes de una tirada (D12)."""
    return _sidecar_path("axis", run_id, out)


def summary_path(run_id: str, out: Path | str | None = None) -> Path:
    """Dónde se guarda el resumen de planificadas/completadas/fallidas (D6)."""
    return _sidecar_path("summary", run_id, out)


def cache_probe_path(run_id: str, out: Path | str | None = None) -> Path:
    """Dónde se guarda el resultado de la sonda de caché (D10)."""
    return _sidecar_path("cache-probe", run_id, out)


def superseded_path(run_id: str, out: Path | str | None = None) -> Path:
    """Dónde van las filas que una reanudación deja fuera del JSONL.

    JSONL, no JSON: se van apilando tal cual, tirada tras tirada. Ver
    `compact_resume_file`.
    """
    base = Path(out).parent if out is not None else OUT_DIR
    return base / f"superseded-{run_id}.jsonl"


# --- ancho del eje (D12) ---------------------------------------------------


class NarrowAxisError(RuntimeError):
    """El eje de similaridad no separa: criterio GO/NO-GO de D12 incumplido.

    D12 dice que «el ancho de rango observado entra como criterio explícito del
    GO/NO-GO». Un criterio que se mide, se imprime y no frena nada no es un
    criterio: la primera versión escribía `axis-<run_id>.json`, imprimía
    «ESTRECHO» y seguía gastando el presupuesto en una estratificación
    decorativa. Ahora `main()` para aquí.

    **Lo que esta puerta ahorra y lo que no.** Ahorra las 27 celdas contra los
    tres modelos evaluados, que son tres llamadas cada una y el grueso del
    presupuesto. **No** ahorra los prefijos: para medir el eje hay que tenerlos,
    y `measure_axis` los pide a `ensure_prefix`, que genera los que falten
    llamando de verdad a `PREFIX_MODEL` y al usuario simulado. Ese gasto ya está
    hecho cuando esto se levanta —y no se tira: los prefijos quedan en
    `runs/prefixes/` y la tirada siguiente los reutiliza—. El informe dice
    cuántos se generaron.

    Para correr de todos modos —por ejemplo para una prueba de humo del arnés,
    donde el ancho del eje da igual— hay que pedirlo explícitamente:
    `main(allow_narrow=True)` o `--force` en la línea de órdenes. Y entonces
    queda dicho en voz alta que se corrió sabiéndolo.
    """

    def __init__(self, narrow_cells: list[dict], report_path: Path | None = None):
        self.narrow_cells = list(narrow_cells)
        self.report_path = report_path
        detalle = ", ".join(
            f"{c['topic_id']}(n={c['n_turns']})" for c in self.narrow_cells
        )
        super().__init__(
            f"rango de coseno < {NARROW_RANGE} en {len(self.narrow_cells)} "
            f"celdas: {detalle}. El eje de similaridad no separa ahí, así que la "
            "estratificación sería decorativa y la tirada no respondería la "
            "pregunta (D12). Completa el banco con artefactos de esos dominios, "
            "o corre a propósito con allow_narrow=True / --force."
            + (f" Informe: {report_path}" if report_path else "")
        )


def measure_axis(
    topics: list[Topic] | None = None,
    arts: list[Artifact] | None = None,
    lengths: Iterable[int] | int = tuple(LENGTHS),
    run_id: str = "",
) -> dict[str, Any]:
    """Embebe los prefijos contra el banco y mide el rango por (tema, longitud).

    Se corre **antes** de la tirada. Sirve para no gastar el presupuesto en un
    eje que no separa: si un tema tiene un rango de coseno por debajo de
    `NARROW_RANGE`, o su cola alta está vacía, la estratificación de ese tema es
    decorativa y hay que completar el banco antes de seguir.

    **Cuánto cuesta medir, dicho claro.** Dos cosas, no una. La barata son los
    embeddings (el banco entero y un texto por celda del diseño: céntimos). La
    otra es que para embeber el lado de usuario de un prefijo hay que **tener**
    el prefijo, y `ensure_prefix` genera el que falte con llamadas de verdad a
    `PREFIX_MODEL` y al usuario simulado: 16 prefijos (8 temas × 2 longitudes),
    de 2 y 10 turnos. No es trabajo extra —la tirada los iba a fabricar de todos
    modos y quedan guardados en `runs/prefijos/` para la siguiente—, pero es
    gasto, y ocurre **antes** de que la puerta de D12 pueda frenar nada. El
    informe lleva `prefixes_generated` y `prefixes_reused` para que ese coste se
    pueda auditar en vez de suponerlo.

    La alternativa considerada era medir el eje solo sobre la apertura de cada
    tema, que no cuesta ninguna llamada. Se descarta porque el eje x del
    experimento es el lado de usuario del prefijo **entero** (D2), y en el brazo
    de 10 turnos eso son diez mensajes de los que la apertura es uno: medir el
    GO/NO-GO sobre otro texto distinto del que luego estratifica sería certificar
    una cosa y correr otra.

    Mide **las dos longitudes** por defecto. El ancho del eje es criterio
    GO/NO-GO (D12) y la mitad de las celdas corren sobre el prefijo largo: un
    prefijo de diez turnos habla de bastantes más cosas que uno de dos, así que
    su rango de cosenos no tiene por qué parecerse. Medir solo el corto
    certificaba media tirada sin haberla mirado.

    Usa los prefijos que `ensure_prefix` ya va a fabricar para la tirada (cada
    tema se construye en las dos longitudes): no genera trabajo extra, solo lo
    adelanta.

    Devuelve el informe (que `main` guarda en `runs/phase0/axis-<run_id>.json`)
    y lo imprime por pantalla de camino.
    """
    topics = load_topics() if topics is None else topics
    arts = load_artifacts() if arts is None else arts
    lengths = [int(lengths)] if isinstance(lengths, int) else [int(n) for n in lengths]

    entries: list[dict[str, Any]] = []
    generated = 0
    reused = 0
    print(
        f"--- ancho del eje (D12): {len(arts)} artefactos, "
        f"longitudes={lengths}"
    )
    for topic in topics:
        for n_turns in lengths:
            # Se mira antes de pedirlo para poder decir después cuánto costó
            # medir: `ensure_prefix` no distingue entre generar y reutilizar.
            ya_estaba = find_prefix(topic.id, n_turns) is not None
            prefix = ensure_prefix(topic, n_turns)
            generated += 0 if ya_estaba else 1
            reused += 1 if ya_estaba else 0
            ranking, truncated = rank_artifacts(
                user_text(prefix["transcript"]), arts
            )
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
            entries.append(entry)
            print(
                f"  {topic.id:<16} n={n_turns:<3} min={entry['min']:.3f} "
                f"med={entry['median']:.3f} max={entry['max']:.3f} "
                f"rango={entry['range']:.3f}"
                + ("  <-- ESTRECHO" if entry["narrow"] else "")
            )

    narrow_cells = [
        {"topic_id": e["topic_id"], "n_turns": e["n_turns"]}
        for e in entries
        if e["narrow"]
    ]
    narrow_topics = sorted({c["topic_id"] for c in narrow_cells})
    if narrow_cells:
        detalle = ", ".join(f"{c['topic_id']}(n={c['n_turns']})" for c in narrow_cells)
        print(f"  OJO: rango < {NARROW_RANGE} en: {detalle}")
    # El coste de medir, en voz alta: medir el eje NO es gratis, y la puerta de
    # D12 frena después de esto, no antes.
    print(
        f"--- prefijos: {generated} generados (llamadas reales a "
        f"{PREFIX_MODEL} y al usuario simulado), {reused} reutilizados de disco"
    )

    return {
        "run_id": run_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "prefix_model": PREFIX_MODEL,
        "lengths": lengths,
        "n_artifacts": len(arts),
        # Lo que costó medir el eje, para que no haya que creerse el docstring.
        "prefixes_generated": generated,
        "prefixes_reused": reused,
        "narrow_threshold": NARROW_RANGE,
        "narrow_cells": narrow_cells,
        "narrow_topics": narrow_topics,
        "entries": entries,
    }


# --- sonda de caché (D10) --------------------------------------------------

# Un modelo por proveedor: el criterio de D10 se declara «desglosado por
# proveedor», y lo que cambia entre proveedores es justo el mecanismo (Claude
# necesita el `cache_control` explícito que pone `clients._mark_cacheable_prefix`;
# el gateway cachea el prefijo solo).
CACHE_PROBE_MODELS = ["claude-opus-5", "gpt-5.6-sol-tst"]

# Cuántas veces se repite la misma llamada. Dos es el mínimo que responde a la
# pregunta: la primera escribe la caché, la segunda tiene que leerla.
CACHE_PROBE_REPEATS = 2


def _cache_read_tokens(usage: dict) -> int | None:
    """Tokens servidos de caché en una llamada, o None si el proveedor no lo dice.

    Dos formas: Anthropic lo pone en `cache_read_input_tokens`, y los cuerpos
    estilo OpenAI en `usage.prompt_tokens_details.cached_tokens`.
    """
    if not isinstance(usage, dict):
        return None
    if usage.get("cache_read_input_tokens") is not None:
        return int(usage["cache_read_input_tokens"])
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details["cached_tokens"])
    return None


def _cache_write_tokens(usage: dict) -> int | None:
    """Tokens escritos a caché en una llamada, o None si no se informa."""
    if not isinstance(usage, dict):
        return None
    if usage.get("cache_creation_input_tokens") is not None:
        return int(usage["cache_creation_input_tokens"])
    return None


def probe_cache(
    models: Iterable[str] | None = None,
    topic: Topic | None = None,
    artifact: Artifact | None = None,
    n_turns: int = LENGTHS[0],
    repeats: int = CACHE_PROBE_REPEATS,
    run_id: str = "",
    out: Path | str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Repite dos veces la misma llamada y reporta el `cache_read` de cada una.

    **Por qué existe** (D10). El criterio escrito es «dos celdas con el mismo
    `prefix_id` deben mostrar `cache_read > 0` en la llamada del pegote,
    desglosado por proveedor». Sobre el plan de la Fase 0 ese criterio no se
    puede comprobar nunca: cada (tema, longitud) tiene un `prefix_id` distinto y
    los tres modelos que lo comparten son de proveedores distintos, así que no
    hay **ninguna** pareja de celdas del mismo proveedor con el mismo prefijo.
    Un chequeo que no puede fallar tampoco puede verificar nada.

    Lo que hace: monta prefijo + pegote una sola vez, y manda **exactamente esa
    misma petición** `repeats` veces a cada modelo. La primera llamada escribe la
    caché y la segunda tiene que leerla. Son cuatro llamadas de céntimos y
    responden la pregunta de verdad —¿el breakpoint está donde creemos y el
    proveedor lo respeta?— antes de presupuestar la Fase 1 contando con el
    ahorro.

    No la llama `main()`: se lanza a mano (`python -m wrongpaste.run_phase0
    probe-cache`) porque gasta dinero y su respuesta vale para toda la fase, no
    para una tirada.
    """
    models = list(CACHE_PROBE_MODELS) if models is None else list(models)
    topic = load_topics()[0] if topic is None else topic
    artifact = load_artifacts()[0] if artifact is None else artifact

    prefix = ensure_prefix(topic, n_turns)
    # Prefijo compartido + el pegote, etiquetados: el breakpoint de caché de
    # `clients` se coloca leyendo los `tag`, no contando posiciones.
    base = [dict(m) for m in prefix["transcript"]]
    base.append({"role": "user", "content": artifact.text, "tag": "paste"})

    results: list[dict[str, Any]] = []
    for model_id in models:
        model = config.MODELS.get(model_id)
        calls: list[dict[str, Any]] = []
        for index in range(repeats):
            request_params: dict = {}
            try:
                reply = chat(
                    model_id,
                    [dict(message) for message in base],
                    max_tokens=MAX_TOKENS,
                    request_params_out=request_params,
                )
            except Exception as exc:  # la sonda tampoco tumba nada (D6).
                status, error_code, error_body = _status_for_error(exc)
                calls.append(
                    {
                        "call": index,
                        "status": status,
                        "error_code": error_code,
                        "error_body": error_body,
                        "cache_read_input_tokens": None,
                        "cache_creation_input_tokens": None,
                        "usage": {},
                        "request_params": dict(request_params),
                    }
                )
                continue
            calls.append(
                {
                    "call": index,
                    "status": "ok",
                    "cache_read_input_tokens": _cache_read_tokens(reply.usage),
                    "cache_creation_input_tokens": _cache_write_tokens(reply.usage),
                    "usage": reply.usage,
                    "response_model": reply.response_model,
                    "stop_reason": reply.stop_reason,
                    "attempts": reply.attempts,
                    "latency_ms": reply.latency_ms,
                    "request_params": dict(request_params),
                }
            )

        after_first = [c["cache_read_input_tokens"] for c in calls[1:]]
        entry = {
            "model_id": model_id,
            "model_label": _model_label(model_id),
            "provider": model.provider if model else "",
            "calls": calls,
            "cache_reads": [c["cache_read_input_tokens"] for c in calls],
            # El criterio de D10, hecho comprobable: a partir de la segunda
            # llamada, el proveedor tiene que decir que ha leído caché.
            "criterion_met": bool(after_first)
            and all(value is not None and value > 0 for value in after_first),
        }
        results.append(entry)
        print(
            f"  {model_id:<18} {entry['provider']:<16} "
            f"cache_read={entry['cache_reads']} "
            f"criterio={'OK' if entry['criterion_met'] else 'NO'}"
        )

    report = {
        "run_id": run_id,
        "topic_id": topic.id,
        "prefix_id": prefix["prefix_id"],
        "artifact_id": artifact.id,
        "n_turns": int(n_turns),
        "repeats": int(repeats),
        "models": results,
        "by_provider": {
            entry["provider"]: entry["criterion_met"] for entry in results
        },
        "criterion_met": bool(results) and all(e["criterion_met"] for e in results),
    }

    if write:
        path = cache_probe_path(run_id or "manual", out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["path"] = str(path)
        print(f"--- sonda de caché escrita en {path}")

    return report


# --- identidad de la tirada (D5) ------------------------------------------


def _sha_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def bank_sha(arts: list[Artifact]) -> str:
    """Huella del banco de artefactos: si cambia, las tiradas no son comparables."""
    return _sha_of([[a.id, a.kind, a.text, list(a.entities)] for a in arts])


def topics_sha(topics: list[Topic]) -> str:
    """Huella del banco de temas, por el mismo motivo.

    Entran también `task`, `expected` y `verifier` (D13): son parte del tema y
    lo que la Fase 2 va a usar para verificar. Sin ellos, dos tiradas con temas
    distintos —misma apertura, tareas distintas— declararían el mismo sha, que
    es justo lo que el sha existe para impedir.
    """
    return _sha_of(
        [
            [t.id, t.opening, list(t.goals), t.task, t.expected, t.verifier]
            for t in topics
        ]
    )


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

    **Solo cuentan como hechas las filas con `status` en `DONE_STATUSES`.** Una
    fila `http_error` o `timeout` es una celda que no llegó a correrse: darla por
    hecha significaba que una celda muerta por un 429 transitorio no se
    reintentaba jamás, y el hueco no cae al azar —fallan más las conversaciones
    largas y los cosenos altos, que es la zona interesante—. `empty` tampoco
    cuenta: es una reacción que no está. Una fila sin `status` es de un formato
    anterior a D6 y se trata como `ok`, que es el valor por defecto del esquema.

    Los géneros que vuelven al contador de D4 son los de esas mismas filas
    hechas: contar el género de una celda que se va a reintentar lo gastaría dos
    veces y rompería la cobertura en las celdas que quedan.

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
        if row.get("status", "ok") not in DONE_STATUSES:
            continue
        cid = row.get("conversation_id")
        if cid:
            done.add(cid)
        if row.get("artifact_kind"):
            kinds[row["artifact_kind"]] += 1
    return done, kinds


def compact_resume_file(path: Path, out: Path | str | None = None) -> list[dict]:
    """Deja el JSONL con **una sola fila por conversación** antes de reanudar.

    **El problema.** Al reanudar, las celdas que no cuentan como hechas se
    vuelven a correr y la fila nueva se añadía al final, con la vieja todavía
    dentro. Una tirada de 27 celdas con una en `http_error` acababa con 28 filas
    y dos filas con el mismo `conversation_id`. Eso rompe la única comprobación
    barata de integridad que da la cabecera (D5): «¿hay tantas filas como
    `planned_cells`?» dejaba de valer, y cualquier recuento —de conductas, de
    fugas, de negativas— salía sesgado hacia las celdas que fallaron y se
    reintentaron, que no son una muestra al azar.

    **La decisión, de las dos que había sobre la mesa: se reescribe el fichero.**
    La alternativa era dejar las dos filas y marcar la nueva con `supersedes`,
    obligando a todo lector a desduplicar antes de contar. Se descarta porque
    pone la corrección en el lector: cualquier análisis, cuaderno o script que
    olvide la regla cuenta mal y no se entera. Reescribiendo, el invariante
    «filas == `planned_cells`, un `conversation_id` cada una» vuelve a ser cierto
    en el fichero mismo, que es donde se puede comprobar.

    **Las filas viejas no se tiran.** Se apilan en `superseded-<run_id>.jsonl`,
    al lado del JSONL: el 429 que mató la celda sigue estando, con su cuerpo de
    error y su hora, para quien quiera saber cuánto peleó la tirada. Lo que no
    hacen es contarse como una conversación del experimento.

    Devuelve las filas retiradas. El fichero se reescribe de forma atómica
    (temporal + `replace`), para que un corte a mitad no se lleve por delante lo
    que ya estaba.
    """
    if not path.exists():
        return []

    header_line: str | None = None
    kept: list[str] = []
    kept_at: dict[str, int] = {}
    dropped: list[dict] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Última línea a medio escribir: no es una fila, no cuenta como
            # nada y no se conserva (`resume_state` ya la ignora).
            continue
        if row.get("kind") == "run_header":
            if header_line is None:
                header_line = line
            continue
        if row.get("status", "ok") not in DONE_STATUSES:
            dropped.append(row)
            continue
        cid = row.get("conversation_id") or ""
        if cid and cid in kept_at:
            # Duplicado heredado de una reanudación anterior: se queda la
            # última, que es la que escribió la ejecución más reciente.
            dropped.append(json.loads(kept[kept_at[cid]]))
            kept[kept_at[cid]] = line
            continue
        if cid:
            kept_at[cid] = len(kept)
        kept.append(line)

    if not dropped:
        return []

    lines = ([header_line] if header_line else []) + kept
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    tmp.replace(path)

    run_id = (json.loads(header_line).get("run_id") if header_line else "") or path.stem
    trash = superseded_path(run_id, out=out if out is not None else path)
    trash.parent.mkdir(parents=True, exist_ok=True)
    with trash.open("a", encoding="utf-8") as fh:
        for row in dropped:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"reanudando: {len(dropped)} filas retiradas del JSONL (se reintentan) "
        f"y guardadas en {trash.name}"
    )
    return dropped


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


def is_resumable(path: Path) -> bool:
    """¿Ese fichero es una tirada empezada que se puede continuar?

    Que el fichero **exista** no basta: un fichero vacío —lo crea cualquier
    `touch`, o una ejecución anterior que murió antes de escribir nada— se tomaba
    por reanudación, se abría en modo `a` y la tirada entera quedaba sin
    `run_header`. Sin cabecera no hay `run_id`, ni shas del banco, ni forma de
    saber si el fichero está completo: el JSONL nace inservible.

    Reanudable = existe, tiene contenido y su primera línea es una cabecera
    válida. Un fichero con contenido y **sin** cabecera válida no es ninguna de
    las dos cosas: ni se reanuda (no se sabe de qué tirada es) ni se pisa (tiene
    datos de alguien), así que se levanta un error en vez de decidir por el
    usuario.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not path.read_text(encoding="utf-8").strip():
        return False
    if read_run_header(path).get("kind") == "run_header":
        return True
    raise ValueError(
        f"{path} tiene contenido pero su primera línea no es un run_header: "
        "no se puede reanudar (no se sabe de qué tirada es) ni empezar encima "
        "(se perderían las filas que ya tiene). Muévelo o bórralo a mano."
    )


# --- clasificación de fallos (D6) ------------------------------------------


def _token(value: Any) -> str:
    """Normaliza un código de proveedor: minúsculas y solo alfanuméricos.

    Así `content_filter`, `CONTENT-FILTER` y `ResponsibleAIPolicyViolation`
    entran todos por la misma puerta.
    """
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


# Códigos con los que un proveedor dice, **sin ambigüedad**, que ha cortado por
# filtro de contenido. Es una lista cerrada a propósito: cualquier heurística
# más ancha acaba clasificando una petición malformada como conducta del modelo.
REFUSAL_CODES: frozenset[str] = frozenset(
    _token(code)
    for code in (
        "content_filter",
        "content_filtered",
        "content_policy_violation",
        "responsible_ai_policy_violation",
        "prohibited_content",
        "blocked_by_content_filter",
        "blocked_reason_content",
        "safety",
        "safety_filter",
        "refusal",
    )
)

# Claves cuyo valor es un código de proveedor y no prosa. `message` queda fuera
# a propósito: es texto libre y en él la palabra «content» aparece a diario en
# errores de petición corrientes.
_CODE_KEYS: frozenset[str] = frozenset(
    _token(key)
    for key in (
        "code",
        "type",
        "status",
        "reason",
        "blockreason",
        "block_reason",
        "finishreason",
        "finish_reason",
        "stopreason",
        "stop_reason",
        "filterreason",
        "errorcode",
    )
)

# `stop_reason` / `finish_reason` que significan que el proveedor se negó. Llegan
# con HTTP 200, así que los mira `status_for_stop_reasons`, no el traductor de
# excepciones.
REFUSAL_STOP_REASONS: frozenset[str] = frozenset(
    _token(reason) for reason in ("refusal", "content_filter", "safety")
)

# `stop_reason` / `finish_reason` que significan que la respuesta se cortó por
# el tope de tokens: `length` en los cuerpos estilo OpenAI, `max_tokens` en
# Claude. **No es conducta**: el tope lo mandamos nosotros (`MAX_TOKENS`), así
# que una respuesta cortada es un artefacto del arnés, y dejarla pasar como `ok`
# la disfraza de «el modelo lo ignoró y siguió a lo suyo» — que es exactamente
# una de las cinco categorías que el experimento quiere medir.
#
# Sale de `simulated_user.TRUNCATION_STOP_REASONS` a propósito: el usuario
# simulado tiene que plantarse ante exactamente los mismos códigos ante los que
# esta fila marca `truncated`. Con dos listas separadas, un código añadido en un
# sitio y no en el otro deja un turno de usuario cortado pasando por bueno.
TRUNCATION_STOP_REASONS: frozenset[str] = frozenset(
    _token(reason) for reason in simulated_user.TRUNCATION_STOP_REASONS
)

# Hasta dónde se baja buscando códigos dentro del JSON de error.
_MAX_ERROR_DEPTH = 6


def _code_like_values(data: Any, depth: int = 0) -> Iterable[str]:
    """Los valores de las claves «de código» del cuerpo, normalizados."""
    if depth > _MAX_ERROR_DEPTH:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                yield from _code_like_values(value, depth + 1)
            elif value is not None and _token(key) in _CODE_KEYS:
                yield _token(value)
    elif isinstance(data, list):
        for item in data:
            yield from _code_like_values(item, depth + 1)


def looks_like_content_filter(body: str) -> bool:
    """¿El cuerpo de error dice, sin ambigüedad, que fue un filtro de contenido?

    Se exige que el proveedor lo declare en un **campo de código** (`code`,
    `type`, `blockReason`, `finish_reason`...) con uno de los valores de
    `REFUSAL_CODES`. La versión anterior se conformaba con que la subcadena
    `content` apareciera en cualquier parte del cuerpo de un 400 — y esa
    subcadena sale en errores de petición malformada corrientes («invalid
    content type», «messages: content must be a string»). Contar un fallo del
    arnés como una negativa del modelo es inventar conducta: contamina justo la
    variable que mide el experimento.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return any(value in REFUSAL_CODES for value in _code_like_values(data))


def status_for_stop_reasons(stop_reasons: Iterable[str | None]) -> str | None:
    """Lo que dicen los `stop_reason` de la celda: `refusal`, `truncated` o None.

    Dos cosas distintas se leen aquí, y en este orden de precedencia:

    1. **`refusal`** — alguna llamada acabó en una negativa declarada. Es la vía
       normal de una negativa: llega con HTTP 200 y el proveedor la marca en
       `stop_reason` (`refusal` en Claude, `content_filter` en los cuerpos estilo
       OpenAI). Que un modelo se niegue **es un resultado** (D6), no una avería,
       y por eso cuenta como celda hecha al reanudar. Manda sobre lo demás
       porque es lo único de aquí que es conducta.
    2. **`truncated`** — alguna llamada se cortó por el tope de tokens. El tope
       es nuestro, así que esto es un artefacto del arnés; lo que no puede es
       salir como `ok`, porque una reacción cortada a media frase se lee luego
       como «el modelo lo ignoró y siguió». Cuenta también los turnos `post`
       (D7): son los que alimentan el recuento de entidades, y uno cortado
       falsea ese recuento igual que una reacción cortada falsea la categoría.

    Si no hay ni una cosa ni la otra devuelve None, y quien llama decide entre
    `empty` y `ok`.
    """
    reasons = [_token(r) for r in stop_reasons if r]
    if any(reason in REFUSAL_STOP_REASONS for reason in reasons):
        return "refusal"
    if any(reason in TRUNCATION_STOP_REASONS for reason in reasons):
        return TRUNCATED
    return None


# --- origen del fallo: arnés o modelo evaluado (D6) ------------------------
#
# En una celda hablan tres modelos distintos: el evaluado, el usuario simulado
# (`USER_MODEL`) y el que fabricó el prefijo (`PREFIX_MODEL`), más el embedder
# del ranking. Solo el primero produce conducta. Si revienta cualquiera de los
# otros y la fila lo cuenta como `refusal`, el experimento se inventa una
# negativa que nadie dio — y la inventa justo en la variable que mide.
ORIGIN_MODEL = "model"
ORIGIN_HARNESS = "harness"

# Etapas de una celda, en el orden en que ocurren. Viajan en `partial["stage"]`
# para que la fila de fallo sepa quién estaba hablando cuando se rompió.
STAGE_PREFIX = "prefix"  # `ensure_prefix`: usuario simulado + PREFIX_MODEL
STAGE_RANK = "rank"  # embeddings del banco contra el prefijo
STAGE_PASTE = "paste"  # la llamada del pegote: SOLO el modelo evaluado
STAGE_POST = "post"  # D7: alterna usuario simulado y modelo evaluado

# Etapas en las que el modelo evaluado todavía no ha abierto la boca.
HARNESS_STAGES: frozenset[str] = frozenset({STAGE_PREFIX, STAGE_RANK})


def failure_origin(stage: str, transcript: list[dict] | None) -> str:
    """¿La llamada que reventó era del modelo evaluado o del arnés?

    No se adivina mirando la excepción —el proveedor no dice qué modelo era, y
    en Claude el `model` ni siquiera viaja en el cuerpo—: se deduce del estado
    que este runner controla.

    - `prefix` y `rank`: el modelo evaluado aún no ha sido llamado. Arnés.
    - `paste`: la única llamada de esa etapa es la del modelo evaluado.
    - `post`: la etapa alterna usuario simulado y modelo evaluado, y el
      transcript dice en cuál de los dos se paró. `continue_after_paste` añade
      el mensaje de usuario **después** de que `next_user_turn` haya devuelto,
      así que si el último mensaje es un turno de usuario esperando respuesta,
      quien falló fue el modelo evaluado; si el último mensaje sigue siendo del
      asistente (la reacción al pegote, o la respuesta al turno anterior), el
      usuario simulado ni siquiera llegó a escribir: fue su llamada la que
      reventó.

    Sin etapa —un fallo antes de que la celda empiece— se responde `harness`:
    ahí seguro que no hay conducta que atribuir.
    """
    if stage in HARNESS_STAGES or not stage:
        return ORIGIN_HARNESS
    if stage == STAGE_POST:
        last = transcript[-1] if transcript else None
        if last is not None and last.get("role") == "user":
            return ORIGIN_MODEL
        return ORIGIN_HARNESS
    return ORIGIN_MODEL


# --- el lado del usuario simulado (D6) -------------------------------------
#
# `_has_empty_response` y `status_for_stop_reasons` solo miran los `Reply` del
# modelo evaluado, que es lo correcto: son las funciones que deciden si una fila
# describe conducta. Pero el usuario simulado también habla, y sus fallos eran
# invisibles: un turno suyo vacío o cortado por el tope deja la conversación que
# se mide rota, y la fila salía `ok`. Lo que sigue es la comprobación equivalente
# para el otro papel, con un status —`harness_error`— que no se puede leer como
# conducta del modelo evaluado.

# Etiquetas de los mensajes que escribe el usuario simulado. La apertura queda
# fuera: la escribimos nosotros desde `topics.py`, no la genera ningún modelo.
USER_SIM_TAGS: frozenset[str] = frozenset({"user_sim", "post"})


def user_call_traces(
    prefix_traces: Iterable[dict], replies: Iterable[Any]
) -> list[dict]:
    """Las trazas de las llamadas del usuario simulado de una celda (D5/D6).

    Dos tandas, cada una con su `stage`:

    - `prefix` — los turnos del prefijo compartido, que paga
      `prefixes.generate_prefix` una sola vez y guarda en
      `runs/prefixes/<prefix_id>.json` bajo `user_reply_traces`. Se copian a la
      fila porque forman parte de la conversación que esta celda mide: los tres
      modelos que comparten `prefix_id` los llevan idénticos.
    - `post` — los turnos posteriores al pegote (D7), que los paga esta celda y
      llegan como `Reply` vivos por `user_replies_out`.

    Un prefijo escrito antes de que esto existiera no trae `user_reply_traces`;
    entonces la tanda `prefix` sale vacía y la comprobación de contenido sobre el
    transcript es la que cubre ese caso.
    """
    traces = [
        {**trace, "stage": STAGE_PREFIX} for trace in (prefix_traces or [])
    ]
    traces += [
        {**trace, "stage": STAGE_POST} for trace in reply_traces(list(replies))
    ]
    return traces


def user_side_problems(
    traces: list[dict], transcript: list[dict], prefix_len: int = 0
) -> list[str]:
    """Qué hay roto en el lado del usuario simulado, si hay algo (D6).

    Devuelve una lista de problemas en prosa (vacía = nada que objetar), que es
    lo que acaba en `error_body` de la fila. Mira dos cosas, que son las mismas
    que se miran del modelo evaluado pero para el otro papel:

    1. **Turnos cortados por el tope** (`stop_reason` en
       `TRUNCATION_STOP_REASONS`), de las dos tandas. El tope lo mandamos
       nosotros, así que es un artefacto del arnés.
    2. **Turnos vacíos**, leídos del transcript y no de los `Reply`, porque así
       se cubre también el prefijo que viene de disco: sus `Reply` ya no existen,
       pero su texto sí. Un mensaje de usuario vacío además revienta con 400 en
       `vertex_anthropic`, y ese 400 estalla en la llamada siguiente, que es la
       del modelo evaluado.

    Desde que `next_user_turn` levanta `SimulatedUserError`, un turno roto
    normalmente ni llega hasta aquí. Esto es la red de abajo, y cubre lo que la
    guarda no puede ver: un prefijo fabricado por una versión anterior del
    código, o reutilizado desde `runs/prefixes/` (D1), que ya venga con un turno
    de usuario cortado o vacío dentro.

    `prefix_len` es la longitud del prefijo compartido dentro de `transcript`, y
    sirve para que cada problema diga su **etapa**: los mensajes anteriores a ese
    índice son del prefijo —que se reutiliza y hay que borrar para arreglarlo— y
    los posteriores, de esta celda. Con 0 (el valor por defecto, para quien solo
    quiera la comprobación) todos salen como `post`.
    """
    problems: list[str] = []
    for trace in traces:
        stop_reason = trace.get("stop_reason")
        if stop_reason and _token(stop_reason) in TRUNCATION_STOP_REASONS:
            problems.append(
                f"un turno del usuario simulado (etapa {trace.get('stage', '?')}) "
                f"se cortó por el tope de tokens (stop_reason={stop_reason!r})"
            )
    for index, message in enumerate(transcript):
        tag = message.get("tag")
        if tag in USER_SIM_TAGS and not (message.get("content") or "").strip():
            stage = STAGE_PREFIX if index < prefix_len else STAGE_POST
            problems.append(
                f"el mensaje {index} del transcript (etapa {stage}), escrito por "
                f"el usuario simulado (tag={tag!r}), está vacío"
            )
    return problems


def _status_for_error(
    exc: BaseException, origin: str = ORIGIN_MODEL
) -> tuple[str, int | str | None, str]:
    """Traduce una excepción al vocabulario de `status` (D6).

    Regla de oro: **un fallo del arnés nunca se cuenta como conducta del
    modelo**. Tiene dos mitades, y antes solo estaba la primera:

    - Ante la duda dentro de una llamada del modelo evaluado, `http_error`; solo
      se marca `refusal` cuando el proveedor lo declara en un campo de código
      del cuerpo (`looks_like_content_filter`). Una negativa normal ni siquiera
      pasa por aquí, porque llega con HTTP 200 y la ve `status_for_stop_reasons`.
    - Si la llamada que reventó **no era del modelo evaluado** (`origin` =
      `harness`), el status es `harness_error` y punto. Un filtro de contenido
      sobre el prompt del usuario simulado es un 400 con `content_filter` en el
      cuerpo, exactamente igual que uno sobre el modelo evaluado: sin el origen,
      la fila decía «este modelo se negó al pegote» cuando el que se negó fue
      otro modelo a otra pregunta. El qué pasó no se pierde: sigue en
      `error_code` y `error_body`.

    `SimulatedUserError` no depende del origen: la levanta el usuario simulado
    sobre su propia respuesta, así que es `harness_error` venga de la etapa que
    venga. Su mensaje lleva dentro el `stop_reason`, que es el dato que explica
    el fallo, y acaba entero en `error_body`.
    """
    if isinstance(exc, SimulatedUserError):
        return HARNESS_ERROR, exc.__class__.__name__, str(exc)[:ERROR_BODY_CHARS]
    if isinstance(exc, httpx.HTTPStatusError):
        # Se analiza el cuerpo entero y se guarda recortado: recortar antes
        # rompería el JSON y dejaría el análisis ciego justo en los errores
        # largos.
        full_body = exc.response.text
        body = full_body[:ERROR_BODY_CHARS]
        if origin == ORIGIN_HARNESS:
            return HARNESS_ERROR, exc.response.status_code, body
        if looks_like_content_filter(full_body):
            return "refusal", exc.response.status_code, body
        return "http_error", exc.response.status_code, body
    if isinstance(exc, httpx.TimeoutException):
        status = HARNESS_ERROR if origin == ORIGIN_HARNESS else "timeout"
        return status, exc.__class__.__name__, str(exc)[:ERROR_BODY_CHARS]
    # Cualquier otra cosa entra como `http_error` porque el vocabulario está
    # cerrado; el tipo real de la excepción queda en `error_code`.
    status = HARNESS_ERROR if origin == ORIGIN_HARNESS else "http_error"
    return status, exc.__class__.__name__, str(exc)[:ERROR_BODY_CHARS]


# --- ejecución de una celda -----------------------------------------------


def rank_for_prefix(
    prefix: dict,
    arts: list[Artifact],
    cache: dict[str, tuple],
) -> tuple[list[tuple[Artifact, float]], dict[str, float], bool, bool]:
    """Ranking del banco contra un prefijo, cacheado por `prefix_id`.

    Devuelve el ranking primario (D2: solo el lado del usuario, ascendente por
    coseno), el diccionario de similaridades secundarias (conversación entera)
    por `artifact_id`, y **los dos truncados por separado**: el del texto de
    usuario (que es el eje x, el que estratifica) y el de la conversación
    entera (que solo afecta a la métrica secundaria). Colapsarlos en un booleano
    obligaba al análisis a tirar filas perfectamente buenas, porque en la
    práctica el único texto que se pasa de los 24.000 caracteres es el completo.

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
            bool(truncated_user),
            bool(truncated_full),
        )
    return cache[pid]


def _system_prompt(request_params: dict, transcript: list[dict]) -> str | None:
    """El system prompt que vio el modelo evaluado, o None si no hubo.

    Se lee de lo que se envió de verdad: los cuerpos de Claude lo llevan en la
    clave `system`, y los de estilo OpenAI como un mensaje `role: "system"`
    dentro de la conversación. En la Fase 0 el modelo evaluado corre **sin**
    system prompt a propósito (el único que hay es el del usuario simulado, que
    va en la cabecera de la tirada), así que aquí sale None — pero sale de mirar
    la petición, no de una constante: el día que la Fase 1 añada uno, la fila lo
    dirá sola.
    """
    system = request_params.get("system")
    if isinstance(system, str) and system.strip():
        return system
    inline = " ".join(
        message["content"]
        for message in transcript
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ).strip()
    return inline or None


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
    artefacto, transcripción, cuerpo enviado, respuestas recibidas), para que si
    esto revienta a mitad, la fila de fallo que escribe `main` no salga vacía
    (D6).

    Secuencia: prefijo compartido (D1), muestreo del artefacto dentro del
    estrato del plan (D3/D4), pegote literal, y dos turnos más con el usuario
    simulado sin reparación (D7). En las celdas de control (D11) se salta el
    pegote y se va directo a los turnos posteriores.

    Trazabilidad (D5/D9): el mismo dict `request_params` viaja a las tres
    puertas de `conversation`, que se lo ceden a `chat()`; y los `Reply`
    completos que devuelven esas puertas son los que rellenan `stop_reasons`,
    `response_model` y `attempts`. Antes se descartaban y las 27 filas salían
    con `{}`, `[]`, `""` y `1`: la fila decía que ninguna llamada había
    reintentado nunca y que no se sabía con qué parámetros se había corrido.

    **El usuario simulado también deja fila (D6).** `continue_after_paste` recibe
    `user_replies_out`, y del prefijo se leen sus `user_reply_traces`: las dos
    tandas viajan a `user_stop_reasons` y `user_reply_traces` de la fila. Si
    alguno de esos turnos salió vacío o cortado por el tope, la celda sale
    `harness_error` —nunca `ok` ni `refusal`—, porque quien se calló no es el
    modelo que se está midiendo y la conversación medida tiene un agujero.
    """
    model_id = cell["model_id"]
    # La etapa viaja en `partial` para que, si esto revienta, la fila sepa quién
    # estaba hablando: las llamadas del prefijo y del embedder no son conducta
    # del modelo evaluado (ver `failure_origin`).
    partial["stage"] = STAGE_PREFIX
    prefix = ensure_prefix(topic, cell["n_turns"])
    partial["prefix_id"] = prefix["prefix_id"]
    # Las trazas de los turnos que escribió el usuario simulado dentro del
    # prefijo compartido. Van a `partial` antes de nada para que también las
    # lleve la fila de una celda que reviente más adelante.
    partial["prefix_user_traces"] = list(prefix.get("user_reply_traces", []))

    # Copia: el prefijo cacheado lo comparten los tres modelos y no se muta.
    transcript = [dict(m) for m in prefix["transcript"]]
    partial["transcript"] = transcript

    # `chat()` lo vacía y lo rellena antes de cada POST, así que queda con el
    # cuerpo de la última llamada —el mismo turno a turno— y ya está puesto
    # aunque la llamada acabe en error.
    request_params: dict = {}
    partial["request_params"] = request_params
    replies: list = []
    partial["replies"] = replies
    # Los `Reply` de los turnos del usuario simulado de **esta** celda (los de
    # después del pegote, D7). `continue_after_paste` los va apilando aquí, y
    # `next_user_turn` apila el suyo antes de levantar `SimulatedUserError`, así
    # que la traza del turno que falló llega igualmente a la fila.
    user_replies: list = []
    partial["user_replies"] = user_replies

    artifact: Artifact | None = None
    similarity_user: float | None = None
    similarity_full: float | None = None
    similarity_rank: int | None = None
    similarity_pct: float | None = None
    ranking_rows: list[dict] | None = None
    truncated_user = False
    truncated_full = False
    paste_index: int | None = None
    reaction: str | None = None
    usages: list[dict] = []

    if cell["condition"] == "paste":
        rng = np.random.default_rng(cell["seed"])
        partial["stage"] = STAGE_RANK
        ranking, full_by_id, truncated_user, truncated_full = rank_for_prefix(
            prefix, arts, rank_cache
        )
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

        partial["stage"] = STAGE_PASTE
        reaction, paste_usage, paste_index, paste_reply = inject_paste(
            model_id, transcript, artifact, request_params_out=request_params
        )
        usages.append(paste_usage)
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

    # Precedencia del `status`, de más informativo a menos:
    #
    # 1. `refusal` — lo único de aquí que es conducta, y manda sobre todo lo
    #    demás: un filtro de contenido suele devolver además texto vacío, y
    #    marcarlo `empty` escondería la conducta que el experimento mide.
    # 2. `empty` — no hay texto que leer. Es la afirmación más fuerte que se
    #    puede hacer sobre una respuesta, y por eso va antes que `truncated`:
    #    una respuesta cortada que además viene vacía es, sobre todo, una
    #    respuesta que no está.
    # 3. `truncated` — hay texto, pero se cortó por el tope de tokens.
    # 4. `ok`.
    declared = status_for_stop_reasons(stop_reasons)
    if declared == "refusal":
        status = "refusal"
    elif _has_empty_response(replies):
        status = "empty"
    elif declared is not None:
        status = declared
    else:
        status = "ok"

    # Y por encima de todo lo anterior, el lado del usuario simulado. Va el
    # último y manda sobre el resto porque no habla del modelo evaluado sino de
    # si la conversación que acabamos de medir existe: con un turno de usuario
    # vacío o cortado, el contexto que vio el modelo tiene un agujero y ninguna
    # de las cinco categorías del spec §5 se puede leer de ahí. `harness_error`
    # es además el único status que nadie puede confundir con conducta, y no
    # cuenta como celda hecha, así que al reanudar se vuelve a intentar.
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
            # `harness_error` no cuenta como celda hecha, así que al reanudar se
            # reintenta; si el turno roto está en el prefijo compartido, el
            # reintento lo vuelve a leer igual de roto. La fila dice qué hay que
            # borrar para salir del bucle.
            detalle += (
                f". El turno roto está en el prefijo compartido "
                f"{prefix['prefix_id']}, que se reutiliza tal cual (D1): "
                f"reintentar la celda lo vuelve a leer igual. Borra "
                f"runs/prefixes/{prefix['prefix_id']}.json para que se "
                f"regenere"
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
        # Solo las llamadas pagadas por esta celda: las del prefijo se pagaron
        # una vez y viven en `runs/prefixes/<prefix_id>.json` (D1).
        usages=usages,
        seed=cell["seed"],
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=int((ended_at - started_at) * 1000),
        status=status,
        error_code=error_code,
        error_body=error_body,
        attempts=_attempts(traces),
        # El otro papel de la conversación (D6): sus `stop_reason` y sus trazas,
        # separados de los del modelo evaluado para que nadie los confunda.
        user_stop_reasons=[trace.get("stop_reason") for trace in user_traces],
        user_reply_traces=user_traces,
    )


def _has_empty_response(replies: Iterable[Any]) -> bool:
    """¿Alguna respuesta del modelo evaluado en esta celda vino vacía? (D6)

    Mira **todas** las llamadas de la celda: la del pegote y las dos de D7. La
    versión anterior solo miraba la reacción al pegote y además lo condicionaba
    a `condition == "paste"`, así que en el brazo de control —que no tiene
    pegote y cuyas únicas respuestas son las de los turnos posteriores— una
    conversación entera de respuestas vacías salía como `ok`. El control existe
    para dar la tasa base de fuga de entidades (D11): una tasa base calculada
    sobre respuestas que no existen es cero por construcción, y nadie se entera
    al leer el fichero.

    Se marca con que **una** venga vacía, no con que lo vengan todas: una fila
    con un turno sin texto ya no es material limpio para el recuento, y el
    análisis tiene que poder descartarla sin releer el transcript.
    """
    return any(not (getattr(r, "text", "") or "").strip() for r in replies)


def _model_label(model_id: str) -> str:
    model = config.MODELS.get(model_id)
    return model.label if model else ""


def _response_model(traces: list[dict]) -> str:
    """Lo que el proveedor dice haber ejecutado, de la primera llamada que lo diga.

    Todas las llamadas de una celda van al mismo modelo, así que basta con una;
    se recorre por si la primera viene sin el campo.
    """
    for trace in traces:
        if trace.get("response_model"):
            return str(trace["response_model"])
    return ""


def _attempts(traces: list[dict]) -> int:
    """Intentos del **peor** turno de la celda (1 = ninguno necesitó reintento).

    No es la suma: lo que interesa saber al leer una fila es si el proveedor
    obligó a reintentar, no cuántas llamadas tuvo la celda —que ya se sabe por
    el plan.
    """
    return max((int(trace.get("attempts") or 1) for trace in traces), default=1)


def failed_record(
    cell: dict,
    topic_id: str,
    run_id: str,
    started_at: float,
    partial: dict[str, Any],
    exc: BaseException,
) -> ConversationRecord:
    """La fila que se escribe cuando una celda revienta (D6).

    Lleva todo lo que se llegó a saber antes del fallo — incluido **el cuerpo
    que se envió** (`chat()` rellena `request_params` antes de hacer el POST) y
    la trazabilidad de los turnos que sí llegaron a contestar. Un fichero con 19
    filas y ninguna marca es indistinguible de uno completo; con esto, no.

    Y lleva el **origen**: `partial["stage"]` más el estado del transcript dicen
    si la llamada que reventó era del modelo evaluado o del arnés (usuario
    simulado, prefijo, embedder). Si era del arnés, el status es
    `harness_error`, jamás `refusal`: atribuirle a un modelo una negativa que
    dio otro modelo a otra pregunta contamina la única variable que mide el
    experimento.
    """
    artifact: Artifact | None = partial.get("artifact")
    request_params: dict = partial.get("request_params", {}) or {}
    transcript: list[dict] = partial.get("transcript", [])
    origin = failure_origin(partial.get("stage", ""), transcript)
    status, error_code, error_body = _status_for_error(exc, origin=origin)
    traces = reply_traces(partial.get("replies", []))
    # El turno del usuario simulado que reventó también deja traza: su `Reply`
    # se apila antes de que `next_user_turn` levante la excepción.
    user_traces = user_call_traces(
        partial.get("prefix_user_traces", []), partial.get("user_replies", [])
    )
    ended_at = time.time()
    return Phase0Record(
        run_id=run_id,
        conversation_id=cell["conversation_id"],
        cell_index=cell["cell_index"],
        replicate_idx=cell["replicate_idx"],
        model_id=cell["model_id"],
        model_label=_model_label(cell["model_id"]),
        response_model=_response_model(traces),
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
        transcript=transcript,
        request_params=dict(request_params),
        system_prompt=_system_prompt(request_params, transcript),
        user_model=USER_MODEL,
        prefix_model=PREFIX_MODEL,
        max_tokens=MAX_TOKENS,
        stop_reasons=[trace["stop_reason"] for trace in traces],
        seed=cell["seed"],
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=int((ended_at - started_at) * 1000),
        status=status,
        error_code=error_code,
        error_body=error_body,
        # El recuento del intento que falló manda; si el fallo llegó después de
        # turnos que sí contestaron, se queda el peor de todos.
        attempts=max(int(getattr(exc, "attempts", 1) or 1), _attempts(traces)),
        user_stop_reasons=[trace.get("stop_reason") for trace in user_traces],
        user_reply_traces=user_traces,
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
    """La primera línea del JSONL (D5).

    Lleva **las dos rutas**: `spec_path`, el documento de diseño, y
    `corrections_path`, el de decisiones que gobierna de verdad esta tirada.
    `corrections_path` no es campo de `RunHeader` —`records.py` no es de este
    agente— y se añade a la línea, que es un dict; queda anotado para que
    `records` lo adopte.
    """
    header = run_header_line(
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
    header["corrections_path"] = CORRECTIONS_PATH
    return header


def main(
    seed: int = MASTER_SEED,
    out: Path | str | None = None,
    measure: bool = True,
    allow_narrow: bool = False,
) -> Path:
    """Corre la Fase 0 entera y devuelve la ruta del JSONL.

    Si `out` apunta a una tirada ya empezada —fichero con contenido y con
    `run_header` válido en la primera línea—, la tirada **se reanuda**: se saltan
    las celdas ya hechas (`status` en `DONE_STATUSES`) y se sigue escribiendo al
    final del mismo fichero (D6). Si el fichero no existe o está vacío, se
    empieza de cero escribiendo la cabecera; si tiene contenido pero no cabecera,
    `is_resumable` levanta un error en vez de pisarlo.

    Los ficheros laterales (informe de ejes, resumen) se escriben **al lado del
    JSONL**, no en `OUT_DIR`, para que una tirada dirigida a otro directorio no
    deje sus datos separados del informe que los explica.

    `measure=False` salta el paso de ancho de eje (D12); solo para pruebas, la
    tirada de verdad lo quiere delante. Si se mide y alguna celda sale estrecha,
    **la tirada se para** con `NarrowAxisError` sin correr ni una celda: el ancho
    del eje es criterio GO/NO-GO de D12, y un criterio que imprime «ESTRECHO» y
    sigue no es un criterio. `allow_narrow=True` (o `--force` en la línea de
    órdenes) corre de todos modos, a propósito y dejando constancia.

    **Qué se ha gastado ya cuando esa puerta salta.** Los 16 prefijos (8 temas ×
    2 longitudes), que `measure_axis` necesita tener para poder embeberlos y que
    `ensure_prefix` fabrica con llamadas reales a `PREFIX_MODEL` y al usuario
    simulado. Decir que la puerta frena «antes de gastar un céntimo en modelos»
    era falso. Lo que frena es lo caro: las 27 celdas contra los tres modelos
    evaluados, a tres llamadas por celda. Y los prefijos no se pierden —quedan
    en `runs/prefixes/` y la tirada que se lance después de completar el banco
    los reutiliza—, pero se han pagado.
    """
    topics = load_topics()
    topics_by_id = {t.id: t for t in topics}
    arts = load_artifacts()
    plan = plan_phase0(seed)

    path = (
        Path(out)
        if out is not None
        else OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    resuming = is_resumable(path)

    if resuming:
        run_id = read_run_header(path).get("run_id") or path.stem
        # Antes de nada, fuera las filas de las celdas que se van a reintentar:
        # si no, el fichero acaba con más filas que `planned_cells` y con dos
        # filas para la misma conversación.
        compact_resume_file(path, out=path)
        done, kind_counts = resume_state(path)
        print(f"reanudando {path.name}: {len(done)} celdas ya hechas")
    else:
        run_id = path.stem
        done, kind_counts = set(), Counter()

    if measure:
        axis = measure_axis(topics=topics, arts=arts, lengths=LENGTHS, run_id=run_id)
        axis_file = axis_path(run_id, out=path)
        axis_file.parent.mkdir(parents=True, exist_ok=True)
        axis_file.write_text(
            json.dumps(axis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # D12 como criterio, no como adorno. El informe queda escrito **antes**
        # de parar: la evidencia de por qué no se corrió es justo lo que hay que
        # mirar para completar el banco.
        if axis["narrow_cells"] and not allow_narrow:
            raise NarrowAxisError(axis["narrow_cells"], report_path=axis_file)
        if axis["narrow_cells"]:
            print(
                f"--- OJO: se corre con {len(axis['narrow_cells'])} celdas "
                "estrechas porque se ha pedido explícitamente (allow_narrow)"
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

            if rec.status in DONE_STATUSES:
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
    summary_file = summary_path(run_id, out=path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
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
    # `probe-cache` se pide a mano y a propósito (D10): gasta dinero y su
    # respuesta vale para toda la fase, no para una tirada.
    if "probe-cache" in sys.argv[1:]:
        print(json.dumps(probe_cache(), ensure_ascii=False, indent=2))
    else:
        # `--force` es la única forma de correr con el eje estrecho (D12), y hay
        # que teclearla: así nadie se salta el GO/NO-GO sin querer.
        print(main(allow_narrow="--force" in sys.argv[1:]))
