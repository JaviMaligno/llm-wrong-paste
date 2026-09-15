"""El análisis de la Fase 1b: cómo cambia la mezcla de la taxonomía con el coseno.

Se escribe **antes** de correr la tanda. Las tres hipótesis del plan quedan en
código y con test antes de ver un dato, que es la diferencia entre contrastar y
buscar.

No hay `scipy` en este proyecto y no se añade por una prueba: Cochran-Armitage
tiene forma cerrada y la cola normal sale de `math.erfc`. Una dependencia nueva
para esto costaría más de mantener que las quince líneas que sustituye.
"""

from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass

from .rates import CATEGORY_FIELD
from .rubric import MENTIONS_JUMP

# Las tres hipótesis del plan, escritas con fecha antes de ver un dato. Viven
# aquí y no en el guion de análisis por lo mismo que las tasas viven en
# `rates.py`: una familia de pruebas que solo existe en la prosa del plan es una
# familia que nadie cuenta a la hora de corregir.
PRIMARY_HYPOTHESES: dict[str, frozenset[str]] = {
    "H1 menciona el salto": MENTIONS_JUMP,
    "H2 puente confabulado": frozenset({"E"}),
    "H3 ejecuta en silencio": frozenset({"A"}),
}

# Nivel de la familia entera, no de cada prueba suelta.
ALPHA = 0.05

# Acuerdo entre las dos réplicas de una misma celda N0 en la Fase 1a: mismo
# prefijo, mismo pegote, otra tirada, 45 pares. Es el suelo de ruido de este
# experimento y viaja con cada informe: una pendiente que mueve la tasa menos de
# lo que la mueve volver a tirar el mismo estímulo no es un hallazgo, por
# pequeño que salga el p.
#
# **Hay dos suelos distintos y confundirlos infla el umbral casi al triple.** El
# acuerdo sobre la ETIQUETA de siete categorías es 33/45 = 0,73, y es el número
# que se cita cuando se habla de lo estable que es la clasificación entera. Pero
# cada hipótesis del plan es una tasa BINARIA —«¿está en este conjunto?»—, y ahí
# la mayoría de los desacuerdos de categoría no cruzan la frontera: de los 12
# pares que cambian de letra, solo 4 cambian de lado en `MENTIONS_JUMP`. El suelo
# que le toca a una curva es el de SU conjunto, no el de la etiqueta.
REPLICATE_AGREEMENT_N0 = 0.73  # etiqueta A-G; para contexto, no para umbrales

# Acuerdo binario por conjunto, medido sobre los mismos 45 pares con el juez
# validado (`gpt-5.5-tst`). La clave es el conjunto ordenado, que es como
# `trend_report` identifica lo que está midiendo.
REPLICATE_AGREEMENT_BY_MEMBER: dict[tuple[str, ...], float] = {
    tuple(sorted(MENTIONS_JUMP)): 0.911,  # 41/45 — H1
    ("E",): 0.933,  # 42/45 — H2
    ("A",): 0.867,  # 39/45 — H3
}


def replicate_agreement(member: set[str]) -> float | None:
    """El suelo de ruido que le toca a esta curva, o `None` si no se ha medido.

    `None` no es un cero ni un valor por defecto: dice que para ese conjunto
    nadie ha medido cuánto se mueve la tasa al volver a tirar el mismo estímulo,
    así que no hay con qué comparar la pendiente. Inventar aquí un número
    plausible es justo lo que convierte un umbral en un adorno.
    """
    return REPLICATE_AGREEMENT_BY_MEMBER.get(tuple(sorted(member)))

# Rango de coseno mínimo para que la pendiente dentro de un género signifique
# algo. Medido sobre el brazo N0 de la Fase 1a: los géneros anchos (`job_ad`
# 0,312; `recipe` 0,254; `meeting_notes` y `shopping_list` 0,204) llegan; los
# estrechos (`prompt` 0,108; `sql` 0,125) no. El umbral se declara, no se elige
# después de ver los datos.
MIN_KIND_SPAN = 0.20

# El barrido de la Fase 1b. Se importa desde `run_phase1b` en producción; el
# valor por defecto está aquí para que el análisis se pueda usar sobre una
# tirada con otro número de posiciones sin tocar el módulo.
SWEEP_POSITIONS = 12


@dataclass(frozen=True)
class PositionRate:
    """La tasa de una posición del barrido, con su denominador a la vista."""

    position: int
    n: int
    k: int

    @property
    def rate(self) -> float | None:
        """`None` cuando `n == 0`: cero de cero no es cero, es «no hay dato»."""
        return None if self.n == 0 else self.k / self.n


@dataclass(frozen=True)
class TrendTest:
    """Cochran-Armitage: ¿la proporción se mueve de forma monótona con la posición?

    `slope_sign` es +1, -1 o 0. Vale 0 exactamente cuando no hay tendencia que
    medir (varianza nula), no cuando la tendencia es pequeña.
    """

    z: float
    p: float
    slope_sign: int
    n: int
    k: int


def cochran_armitage(counts: list[tuple[int, int, int]]) -> TrendTest:
    """Prueba de tendencia sobre proporciones ordenadas.

    `counts` son tripletes `(posición, n, k)`. La posición hace de puntuación,
    que es lo correcto aquí: el barrido está espaciado uniformemente **por rango**
    en el ranking, no por valor de coseno, así que las posiciones son
    equidistantes por construcción y usarlas como puntuación no impone nada.
    """
    total_n = sum(n for _, n, _ in counts)
    total_k = sum(k for _, _, k in counts)
    if total_n == 0:
        return TrendTest(0.0, 1.0, 0, 0, 0)
    p_barra = total_k / total_n
    if p_barra in (0.0, 1.0):
        # Todas iguales: la varianza es cero y no hay tendencia que medir.
        return TrendTest(0.0, 1.0, 0, total_n, total_k)

    t = sum(x * (k - n * p_barra) for x, n, k in counts)
    suma_nx = sum(n * x for x, n, _ in counts)
    suma_nx2 = sum(n * x * x for x, n, _ in counts)
    var = p_barra * (1 - p_barra) * (suma_nx2 - suma_nx**2 / total_n)
    if var <= 0:
        return TrendTest(0.0, 1.0, 0, total_n, total_k)

    z = t / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))  # dos colas
    signo = 0 if z == 0 else (1 if z > 0 else -1)
    return TrendTest(z, p, signo, total_n, total_k)


def rate_by_position(
    rows: list[dict],
    member: set[str],
    positions: int = SWEEP_POSITIONS,
) -> list[PositionRate]:
    """Tasa de pertenencia a `member` en cada posición del barrido.

    Devuelve **una entrada por posición**, incluidas las vacías: un hueco en la
    curva es un dato sobre la tanda, y borrarlo dibuja otra curva.

    Solo cuentan las filas con categoría. Una fila sin veredicto no es una
    respuesta que no hizo lo que se mide: es una respuesta que no se clasificó.
    """
    n = [0] * positions
    k = [0] * positions
    for row in rows:
        cat = row.get(CATEGORY_FIELD)
        pos = row.get("sweep_position")
        if cat is None or pos is None or not 0 <= int(pos) < positions:
            continue
        n[int(pos)] += 1
        if cat in member:
            k[int(pos)] += 1
    return [PositionRate(p, n[p], k[p]) for p in range(positions)]


def _curva(rows: list[dict], member: set[str], positions: int) -> dict:
    tasas = rate_by_position(rows, member, positions)
    prueba = cochran_armitage([(t.position, t.n, t.k) for t in tasas])
    return {
        "curve": [asdict(t) | {"rate": t.rate} for t in tasas],
        "trend": asdict(prueba),
    }


def trend_report(
    rows: list[dict],
    member: set[str],
    by: str | None = "model_id",
    positions: int = SWEEP_POSITIONS,
) -> dict:
    """Curva y prueba de tendencia, por grupo y agregada.

    `by="model_id"` es el reparto **primario** del plan: dentro de un modelo, las
    96 celdas son 96 estímulos distintos. Agregando los tres, el mismo
    (prefijo, artefacto) puede aparecer dos veces, así que `pooled` se reporta
    pero no manda.

    Los `p` que salen de aquí están **sin corregir**, y no pueden estar de otra
    manera: esto es una hipótesis de las tres, o sea un tercio de la familia, y
    Holm necesita verla entera. Quien contrasta las tres hipótesis del plan llama
    a `primary_family_report`. Lo que este informe sí lleva es `n_tests`, el
    número de pruebas que acaba de correr, para que nadie lea tres pruebas al
    5 % como si fueran una.
    """
    informe: dict = {
        "member": sorted(member),
        "positions": positions,
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # El que manda para juzgar ESTA curva: el binario de su conjunto.
        "replicate_agreement_member": replicate_agreement(member),
        "pooled": _curva(rows, member, positions),
    }
    if by:
        grupos: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            grupos[str(row.get(by))].append(row)
        informe["by"] = {g: _curva(v, member, positions) for g, v in sorted(grupos.items())}
        informe["n_tests"] = len(informe["by"])
    return informe


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni sobre una familia de pruebas. Devuelve los `p` ajustados.

    Holm y no Bonferroni porque controla el mismo error de familia sin regalar
    potencia: solo el `p` más pequeño paga el tamaño entero de la familia, y el
    siguiente paga uno menos. Con el diseño de 1b —que ya anda justo de potencia—
    esa diferencia no es cosmética.

    El máximo arrastrado impone la monotonía: un `p` ajustado nunca queda por
    debajo de otro cuyo `p` crudo era menor, que es como se lee una lista
    ordenada sin tener que mirar dos columnas.
    """
    ordenados = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordenados)
    ajustados: dict = {}
    arrastre = 0.0
    for i, (clave, p) in enumerate(ordenados):
        arrastre = min(1.0, max(arrastre, p * (m - i)))
        ajustados[clave] = arrastre
    return ajustados


def primary_family_report(
    rows: list[dict],
    by: str | None = "model_id",
    positions: int = SWEEP_POSITIONS,
    alpha: float = ALPHA,
) -> dict:
    """Las TRES hipótesis por los TRES modelos, corregidas como lo que son: nueve.

    El plan reporta «las tres hipótesis con su z y su p, por modelo» y la puerta
    de la fase se abre si alguna de ellas baja de 0,05 en algún modelo. Nueve
    pruebas al 5 % no hacen una familia al 5 %: simulando 288 filas con las tasas
    base de N0 y la categoría SIN ninguna dependencia de la posición —o sea, con
    las nueve nulas ciertas por construcción— al menos una baja de 0,05 en el
    32 % de las tandas. Uno de cada tres experimentos planos publicaría un
    «este modelo sí mueve la mezcla y ese otro no».

    Así que la corrección no es un adorno del informe: es lo que hace que el
    número que se reporta signifique lo que dice. `p` crudo se conserva al lado,
    porque es el que permite recalcular con otra familia si alguien la define
    distinta.

    `pooled` NO entra en la familia: el plan lo declara secundario y no decide
    nada. Las curvas por género de `by_kind` tampoco — son exploración de D15, y
    se reportan por signo, no por `p`.
    """
    informes = {
        nombre: trend_report(rows, set(member), by=by, positions=positions)
        for nombre, member in PRIMARY_HYPOTHESES.items()
    }
    crudos = {
        (nombre, grupo): sub["trend"]["p"]
        for nombre, informe in informes.items()
        for grupo, sub in informe.get("by", {}).items()
    }
    for (nombre, grupo), ajustado in holm(crudos).items():
        prueba = informes[nombre]["by"][grupo]["trend"]
        prueba["p_holm"] = ajustado
        prueba["significant_holm"] = ajustado < alpha
    return {
        "alpha": alpha,
        "family_size": len(crudos),
        "correction": "holm-bonferroni",
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # Aquí no hay UN conjunto sino tres, y cada uno tiene su propio suelo
        # binario: va dentro de su informe, en `replicate_agreement_member`.
        "hypotheses": informes,
    }


def by_kind(
    rows: list[dict],
    member: set[str],
    min_span: float = MIN_KIND_SPAN,
    positions: int = SWEEP_POSITIONS,
) -> dict:
    """La mitad medible de D15: ¿se mueve la tasa DENTRO de un mismo género?

    Registro y similaridad son colineales por construcción (D15) y este diseño no
    los separa. Lo que sí puede hacer es mirar dentro de cada género: si en
    `job_ad`, que recorre 0,3 de coseno, la tasa sigue moviéndose, parte del
    efecto es de la similaridad y no solo del registro. Los géneros que no
    recorren eje se listan aparte **con su rango**, para que se vea por qué no
    dicen nada.
    """
    grupos: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row.get("artifact_kind"):
            grupos[row["artifact_kind"]].append(row)

    anchos: dict[str, dict] = {}
    estrechos: dict[str, dict] = {}
    for kind, filas in sorted(grupos.items()):
        sims = [r["similarity_user"] for r in filas if r.get("similarity_user") is not None]
        span = (max(sims) - min(sims)) if sims else 0.0
        base = {"n": len(filas), "span": span}
        if span >= min_span:
            anchos[kind] = base | _curva(filas, member, positions)
        else:
            estrechos[kind] = base
    # `n_tests` por lo mismo que en `trend_report`: aquí se corre una prueba por
    # género ancho, y son pruebas exploratorias que NO están en la familia
    # primaria. Se reportan por signo de pendiente, no por `p`; el recuento viaja
    # para que se vea cuántas son si alguien decide mirar sus `p`.
    return {
        "min_span": min_span,
        "wide": anchos,
        "narrow": estrechos,
        "n_tests": len(anchos),
    }
