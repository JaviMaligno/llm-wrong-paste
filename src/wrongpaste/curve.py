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
from .rubric import ENTERTAINS_ERROR, MENTIONS_JUMP

# Las hipótesis de cada tanda, escritas con fecha antes de ver un dato. Viven
# aquí y no en el guion de análisis por lo mismo que las tasas viven en
# `rates.py`: una familia de pruebas que solo existe en la prosa del plan es una
# familia que nadie cuenta a la hora de corregir.
#
@dataclass(frozen=True)
class Hypothesis:
    """Una hipótesis declarada: qué conjunto se mide y sobre qué eje ordenado.

    Hasta la Fase 1d el eje era implícito —siempre `sweep_position`— porque
    todas las hipótesis barrían el mismo. La Fase 1d declara dos hipótesis sobre
    el MISMO conjunto (G) y ejes distintos: H4 lo contrasta contra la posición
    del barrido y H5 contra la longitud de la conversación. Sin el eje dentro de
    la declaración, H5 no se puede correr, y una hipótesis que no se puede
    correr no entra en Holm: la familia se queda en 3 pruebas donde el plan
    declaró 6, el `p` más pequeño paga x3 en vez de x6, y un 0,012 crudo sale
    0,036 —significativo— en lugar de 0,072 —no—.

    `levels` son los valores ordenados del eje, y hacen de puntuación de
    Cochran-Armitage. `None` significa `range(positions)`, que es el barrido.
    Con dos niveles —(2, 10) turnos— Cochran-Armitage es exactamente la prueba z
    de dos proporciones con varianza agrupada, así que no hace falta una prueba
    nueva: hace falta poder decirle sobre qué columna mirar.
    """

    member: frozenset[str]
    axis: str = "sweep_position"
    levels: tuple[int, ...] | None = None


def _como_hipotesis(valor: Hypothesis | frozenset[str] | set[str]) -> Hypothesis:
    """Un conjunto pelado sigue valiendo: es una hipótesis sobre el barrido.

    La familia de la Fase 1b está publicada como nombre -> conjunto y se queda
    así. Normalizar aquí es lo que permite que `primary_family_report` sin
    argumentos siga dando byte a byte lo mismo que cuando se publicó.
    """
    return valor if isinstance(valor, Hypothesis) else Hypothesis(frozenset(valor))


# El eje de la Fase 1d: cuatro BANDAS por rango del ranking, no doce puestos.
#
# Viaja en la fila como `stratum`/`n_strata` porque una banda es literalmente un
# estrato por rango —la misma partición que `stratum_window` de la Fase 0—, y el
# runner deja `sweep_position` en `None`: escribir la banda ahí haría que un
# análisis conjunto leyera las dos tandas como si hubieran muestreado igual.
#
# Se escribe aquí y no se importa de `run_phase1d` para no atar el análisis al
# runner —`curve` no sabe de tiradas—, y la suite del runner comprueba que los
# dos números sigan siendo el mismo. Duplicar el 4 sin ese test sería la manera
# de que el día que alguien mueva las bandas el análisis siga leyendo cuatro.
BANDS = 4
BAND_LEVELS: tuple[int, ...] = tuple(range(BANDS))

# **Cada tanda declara SU familia.** Hasta la Fase 1d bastaba una constante
# única porque solo había una tanda que corregir; con dos, una familia global
# corrige por pruebas que esta tanda no ha corrido —infla el `p` ajustado— y
# deja fuera las que sí —lo desinfla—. Holm necesita ver exactamente el conjunto
# que se contrastó, ni uno más ni uno menos.
HYPOTHESIS_FAMILIES: dict[str, dict[str, Hypothesis | frozenset[str]]] = {
    # Fase 1b, brazo neutro N0: las tres quedaron planas tras Holm.
    "1b": {
        "H1 menciona el salto": MENTIONS_JUMP,
        "H2 puente confabulado": frozenset({"E"}),
        "H3 ejecuta en silencio": frozenset({"A"}),
    },
    # Fase 1d, brazo con señal N1. DOS hipótesis sobre el mismo conjunto —G,
    # `contempla que sea un error`— y ejes distintos, que es como el plan las
    # declara antes de mirar: «2 hipótesis x 3 modelos = 6 pruebas», y la puerta
    # del Paso 8 dice «H4 o H5 se sostienen si sobreviven a Holm». H5 tiene que
    # estar AQUÍ y no en la prosa: una hipótesis que solo existe en el plan es
    # una prueba que nadie cuenta al corregir, y la puerta acabaría declarando
    # sostenida una H5 que no pasó por Holm, mientras H4 paga x3 en vez de x6.
    # Las tres de la Fase 1b se reportan sobre este brazo como SECUNDARIAS y por
    # eso no están en esta familia: describir no es contrastar, y meterlas
    # dentro le costaría potencia a H4 por pruebas que no deciden nada.
    "1d": {
        # H4 va sobre la BANDA (`stratum`), que es el eje que esta tanda muestrea
        # y el único que sus filas escriben. Declararla sobre `sweep_position`
        # —el eje de la 1b— la dejaba midiendo un campo que el runner de la 1d
        # pone a `None` en las 1.152 filas, y el informe salía bien formado con
        # la curva a n = [0] * 12 y p = 1,0: la hipótesis primaria de la tanda
        # certificando una llanura sobre cero conversaciones.
        "H4 contempla el error": Hypothesis(
            ENTERTAINS_ERROR, axis="stratum", levels=BAND_LEVELS
        ),
        # H5: más G en las conversaciones largas. Mismo conjunto, otro eje. Va
        # por modelo como H4 porque así la declara la familia (2 x 3 = 6); el
        # agregado de ~137 por longitud que menciona el texto del plan sale en
        # `pooled`, que se reporta y no entra en la familia (D: `pooled` nunca
        # decide, ya era así para 1b).
        "H5 contempla el error en las largas": Hypothesis(
            ENTERTAINS_ERROR, axis="n_turns", levels=(2, 10)
        ),
    },
}

# La familia de la Fase 1b sigue siendo el valor por defecto de
# `primary_family_report`: los `p_holm` de esa tanda están publicados y tienen
# que seguir saliendo idénticos, o las dos tandas dejan de ser comparables.
PRIMARY_HYPOTHESES: dict[str, Hypothesis | frozenset[str]] = HYPOTHESIS_FAMILIES["1b"]

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

# Acuerdo binario por conjunto, medido con el juez validado (`gpt-5.5-tst`).
# La clave es el conjunto ordenado, que es como `trend_report` identifica lo que
# está midiendo. **La clave no lleva el brazo**, así que el brazo va en el
# comentario de cada entrada: las tres primeras salen de los 45 pares N0 y la de
# G de los 46 pares N1, que es el brazo donde G se disputa. Poner el suelo N0 de
# G aquí sería inútil: en el brazo neutro G casi no aparece y su acuerdo binario
# sería el de «ninguna de las dos réplicas dijo G».
REPLICATE_AGREEMENT_BY_MEMBER: dict[tuple[str, ...], float] = {
    tuple(sorted(MENTIONS_JUMP)): 0.911,  # 41/45, brazo N0 — H1 de la Fase 1b
    ("E",): 0.933,  # 42/45, brazo N0 — H2 de la Fase 1b
    ("A",): 0.867,  # 39/45, brazo N0 — H3 de la Fase 1b
    # H4 de la Fase 1d. 43/46 = 0,935, medido sobre las réplicas del brazo N1 de
    # la Fase 1a (`runs/phase1a/20260914T135814.jsonl` + sus veredictos): 96
    # filas N1, 93 `ok`, 47 celdas con réplica y 46 pares completos — material
    # comparable a los 45 pares con que se midieron los suelos de N0.
    #
    # No es el artefacto de «ambas réplicas dijeron que no»: la tabla de pares es
    # TT=8, TF=1, FT=2, FF=35, o sea 8 pares que coinciden en G POSITIVAMENTE,
    # con prevalencia 0,207 —el 20,4 % que cita el plan para N1— y kappa 0,80.
    # De control, sobre esos mismos 46 pares el acuerdo en la etiqueta A-G da
    # 40/46 = 0,870, idéntico al 0,87 ya publicado de la Fase 1a: los pares se
    # agrupan igual que en las tandas anteriores. `MENTIONS_JUMP` en este brazo
    # da 45/46 = 0,978, que no se guarda aquí porque su clave ya la ocupa el
    # valor N0 de la Fase 1b y mezclar brazos bajo una misma clave es justo lo
    # que este comentario evita.
    ("G",): 0.935,
}


def replicate_agreement(member: set[str]) -> float | None:
    """El suelo de ruido que le toca a esta curva, o `None` si no se ha medido.

    `None` no es un cero ni un valor por defecto: dice que para ese conjunto
    nadie ha medido cuánto se mueve la tasa al volver a tirar el mismo estímulo,
    así que no hay con qué comparar la pendiente. Inventar aquí un número
    plausible es justo lo que convierte un umbral en un adorno.
    """
    return REPLICATE_AGREEMENT_BY_MEMBER.get(tuple(sorted(member)))

# La caída de tasa que la puerta de la Fase 1d declara relevante. No es un
# número redondo elegido a ojo: los brazos N0 de las Fases 1a y 1b son la misma
# condición y `menciona el salto` dio 25,3 % en una y 16,4 % en la otra, o sea
# **8,9 puntos de diferencia por volver a tirar**, que la puerta redondea al
# alza. Un efecto menor que la variación entre tandas no se distingue de haber
# vuelto a tirar; por eso es también la caída para la que hay que calcular
# potencia, y no otra.
DECLARED_DROP = 0.09

# Potencia mínima para que un resultado plano signifique «no hay efecto» en vez
# de «este diseño no lo habría visto». 0,80 es la convención, y aquí importa más
# que de costumbre: la puerta de la Fase 1d convierte el nulo en el resultado
# que cierra la fase.
MIN_POWER = 0.80


def _phi(z: float) -> float:
    """Φ(z), la normal acumulada. `erfc` otra vez, que es lo que hay sin scipy."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _z_critico(alpha: float) -> float:
    """El |z| que deja `alpha` en las dos colas. Bisección sobre `erfc`.

    Invertir la normal a mano tiene aproximaciones racionales conocidas, pero
    aquí se llama un puñado de veces por informe y la bisección es exacta hasta
    donde importa sin traer una tabla de coeficientes que nadie podría revisar.
    """
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Φ⁻¹(0,80): el cuantil que pide la potencia objetivo en el efecto mínimo
# detectable. Clavado porque `_z_critico` invierte la cola de DOS colas y esta es
# de una.
_Z_POTENCIA_80 = 0.8416212335729143


def trend_power(
    counts: list[tuple[int, int, int]],
    drop: float = DECLARED_DROP,
    alpha: float = ALPHA,
    base_rate: float | None = None,
    target_power: float = MIN_POWER,
) -> dict:
    """Qué caída habría visto este diseño, y con qué probabilidad.

    **Por qué existe.** La Fase 1b publicó tres curvas planas y las escribió como
    planas. La puerta de la Fase 1d quiere más: si ninguna hipótesis se sostiene,
    el resultado declarado es *la conducta no depende del parecido ni de la
    longitud*. Eso es afirmar una ausencia, y una ausencia solo se afirma si el
    diseño habría visto la presencia. Con 288 celdas repartidas entre 3 modelos y
    12 posiciones quedan **8 observaciones por punto y por modelo**, y el reparto
    primario es por modelo (`trend_report(by="model_id")`); `pooled` no manda. Sin
    esta cifra al lado, un nulo de este tamaño se lee como evidencia cuando es
    silencio.

    **La cuenta.** Bajo la alternativa `p_i = p̄ + b·(x_i − x̄)`, el numerador de
    Cochran-Armitage tiene esperanza `b·S` con `S = Σn_i x_i² − (Σn_i x_i)²/N`
    —exactamente el mismo `S` que entra en la varianza de `cochran_armitage`—, así
    que `E[z] = |b|·raíz(S / (p̄(1−p̄)))` y la potencia exigiendo además el signo
    predicho es `Φ(E[z] − z_crítico)`. `b = drop / span`, con `span` el recorrido
    real de posiciones con filas. Contrastado contra Monte Carlo sobre la
    implementación real en `test_la_potencia_esta_calculada_y_no_supuesta`.

    `base_rate` permite calcular antes de tener datos; por defecto es la tasa
    observada. Una tasa de 0 o de 1 no da potencia ni efecto mínimo detectable:
    una hipótesis de bajada sobre una tasa que ya es cero no está refutada, está
    **sin nada que medir**, como le pasó a H2 en la Fase 1b.
    """
    total_n = sum(n for _, n, _ in counts)
    total_k = sum(k for _, _, k in counts)
    p_barra = base_rate if base_rate is not None else (total_k / total_n if total_n else 0.0)
    con_filas = [(x, n) for x, n, _ in counts if n > 0]
    span = (max(x for x, _ in con_filas) - min(x for x, _ in con_filas)) if con_filas else 0
    suma_nx = sum(n * x for x, n, _ in counts)
    suma_nx2 = sum(n * x * x for x, n, _ in counts)
    s_xx = suma_nx2 - (suma_nx**2 / total_n if total_n else 0.0)

    informe = {
        "alpha": alpha,
        "declared_drop": drop,
        "base_rate": p_barra,
        "power": 0.0,
        "minimum_detectable_drop": None,
        "observations_per_position_for_declared_drop": None,
        "underpowered": True,
    }
    if total_n == 0 or span == 0 or s_xx <= 0 or not 0.0 < p_barra < 1.0:
        # Sin varianza, sin eje o sin filas no hay potencia que calcular, y
        # devolver un número aquí sería inventarlo.
        return informe

    escala = math.sqrt(s_xx / (p_barra * (1 - p_barra)))
    lam = abs(drop) / span * escala
    z_crit = _z_critico(alpha)
    informe["power"] = _phi(lam - z_crit)
    informe["underpowered"] = informe["power"] < target_power

    # Efecto mínimo detectable: la caída que alcanzaría `target_power`.
    informe["minimum_detectable_drop"] = (z_crit + _Z_POTENCIA_80) / escala * span
    # Y lo que costaría ver la caída declarada: `E[z]` crece con la raíz de n, así
    # que multiplicar todas las celdas por f multiplica `E[z]` por raíz(f). Se
    # expresa en observaciones por posición —la unidad en la que se compra la
    # tanda— sobre la media actual.
    if lam > 0:
        factor = ((z_crit + _Z_POTENCIA_80) / lam) ** 2
        media_actual = total_n / len(con_filas)
        informe["observations_per_position_for_declared_drop"] = math.ceil(factor * media_actual)
    return informe


# Rango de coseno mínimo para que la pendiente dentro de un género signifique
# algo. Medido sobre el brazo N0 de la Fase 1a: los géneros anchos (`job_ad`
# 0,312; `recipe` 0,254; `meeting_notes` y `shopping_list` 0,204) llegan; los
# estrechos (`prompt` 0,108; `sql` 0,125) no. El umbral se declara, no se elige
# después de ver los datos.
MIN_KIND_SPAN = 0.20

# Cuánto puede mover la tasa la MEZCLA DE SEÑALES por sí sola antes de que el
# informe la marque al lado de la curva. Dos puntos de tasa, declarados antes de
# mirar: la puerta de la Fase 1d pide efectos de más de 9 puntos, así que una
# composición que mueve 2 ya es la cuarta parte del listón — y en un modelo
# suelto multiplica. Reconstruido sobre el plan real de 1d (`plan_phase1b(
# 20260916, level="N1")` + `sweep_index(p, 44)` sobre los 16 rankings N1 de la
# Fase 1a) y con las tasas de G por señal de aquella tanda, la composición SOLA
# mueve 2,2 puntos agregados y 7,4 en `gpt-5.6-sol-tst` entre la posición 0 y la
# 11. No es un umbral de validez: por debajo se mide y se reporta igual, solo
# deja de marcarse `confounded`.
MAX_COMPOSITION_SPAN = 0.02

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
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
) -> list[PositionRate]:
    """Tasa de pertenencia a `member` en cada nivel del eje ordenado.

    Devuelve **una entrada por nivel**, incluidas las vacías: un hueco en la
    curva es un dato sobre la tanda, y borrarlo dibuja otra curva.

    `axis` y `levels` existen porque la Fase 1d contrasta el mismo conjunto
    sobre dos ejes: `sweep_position` con los doce niveles del barrido (H4) y
    `n_turns` con los dos de la longitud (H5). El eje cableado dejaba a H5 sin
    poder correrse, y una prueba que no se corre no entra en Holm.

    `position` lleva el VALOR del nivel, no su índice: la curva de H5 se lee
    «2 turnos / 10 turnos». Usarlo también como puntuación de Cochran-Armitage
    es inocuo porque la z es invariante a transformaciones afines de la
    puntuación, y con dos niveles la prueba es la z de dos proporciones se
    escriba (0, 1) o (2, 10).

    Solo cuentan las filas con categoría. Una fila sin veredicto no es una
    respuesta que no hizo lo que se mide: es una respuesta que no se clasificó.
    """
    escala = tuple(range(positions)) if levels is None else tuple(levels)
    indice = {nivel: i for i, nivel in enumerate(escala)}
    n = [0] * len(escala)
    k = [0] * len(escala)
    for row in rows:
        cat = row.get(CATEGORY_FIELD)
        bruto = row.get(axis)
        if cat is None or bruto is None:
            continue
        i = indice.get(int(bruto))
        if i is None:
            continue
        n[i] += 1
        if cat in member:
            k[i] += 1
    return [PositionRate(nivel, n[i], k[i]) for i, nivel in enumerate(escala)]


def _curva(
    rows: list[dict],
    member: set[str],
    positions: int,
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
    alpha: float = ALPHA,
) -> dict:
    tasas = rate_by_position(rows, member, positions, axis=axis, levels=levels)
    counts = [(t.position, t.n, t.k) for t in tasas]
    prueba = cochran_armitage(counts)
    return {
        "curve": [asdict(t) | {"rate": t.rate} for t in tasas],
        "trend": asdict(prueba),
        # La potencia viaja pegada a la curva porque es la mitad que falta para
        # leerla: una pendiente plana sin el efecto mínimo detectable al lado no
        # dice si no hay efecto o si no había con qué verlo. Aquí va con el alfa
        # suelto; dentro de una familia, `primary_family_report` la recalcula con
        # el que de verdad hay que batir.
        "power": trend_power(counts, alpha=alpha),
    }


def trend_report(
    rows: list[dict],
    member: set[str],
    by: str | None = "model_id",
    positions: int = SWEEP_POSITIONS,
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
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
    # Los niveles de VERDAD, que con `levels` no son `positions`. El informe
    # declaraba 12 puntos para la curva de dos de H5 —y habría declarado 12 para
    # las cuatro bandas de H4— porque copiaba el argumento en vez de la escala
    # que se acaba de usar. Un `positions` que no es el largo de `curve` es la
    # misma clase de mentira que este módulo está corrigiendo, en pequeño.
    niveles = _niveles(positions, levels)
    informe: dict = {
        "member": sorted(member),
        "positions": len(niveles),
        # El eje viaja en el informe porque dos hipótesis de la misma familia
        # pueden medir el MISMO conjunto sobre ejes distintos (H4 y H5 de la
        # Fase 1d): sin esto, las dos curvas se leen como la misma repetida.
        "axis": axis,
        "levels": niveles,
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # El que manda para juzgar ESTA curva: el binario de su conjunto.
        "replicate_agreement_member": replicate_agreement(member),
        "pooled": _curva(rows, member, positions, axis=axis, levels=levels),
        # La curva NO se puede leer sola cuando el pegote lleva señal dentro:
        # ver `signal_confound`. Viaja dentro del informe y no en un guion
        # aparte porque esto es lo que lee quien contrasta la hipótesis, y un
        # control que hay que acordarse de correr es un control que no se corre.
        # En el brazo neutro sale `None` —N0 no tiene señales— y por eso el
        # informe de la Fase 1b no se mueve ni un decimal.
        "signal_confound": signal_confound(
            rows, member, positions, axis=axis, levels=levels
        ),
    }
    if by:
        grupos: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            grupos[str(row.get(by))].append(row)
        informe["by"] = {
            g: _curva(v, member, positions, axis=axis, levels=levels)
            for g, v in sorted(grupos.items())
        }
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
    family: dict[str, Hypothesis | frozenset[str]] = PRIMARY_HYPOTHESES,
) -> dict:
    """Las hipótesis DECLARADAS por la tanda, por modelo, corregidas juntas.

    `family` es la familia de ESA tanda (`HYPOTHESIS_FAMILIES["1b"]` por defecto,
    que es lo ya publicado) y el tamaño de la familia sale de contar lo que se
    acaba de correr —hipótesis declaradas x grupos con filas—, no de multiplicar
    constantes: con la familia de la Fase 1b son nueve pruebas y con la de la 1d
    son tres, y el multiplicador que paga el `p` más pequeño cambia con ello.

    El plan de la Fase 1b reporta «las tres hipótesis con su z y su p, por
    modelo» y la puerta de la fase se abre si alguna de ellas baja de 0,05 en
    algún modelo. Nueve pruebas al 5 % no hacen una familia al 5 %: simulando
    288 filas con las tasas base de N0 y la categoría SIN ninguna dependencia de
    la posición —o sea, con las nueve nulas ciertas por construcción— al menos
    una baja de 0,05 en el 32 % de las tandas. Uno de cada tres experimentos planos publicaría un
    «este modelo sí mueve la mezcla y ese otro no».

    Así que la corrección no es un adorno del informe: es lo que hace que el
    número que se reporta signifique lo que dice. `p` crudo se conserva al lado,
    porque es el que permite recalcular con otra familia si alguien la define
    distinta.

    `pooled` NO entra en la familia: los planes lo declaran secundario y no
    decide nada. Las curvas por género de `by_kind` tampoco — son exploración de
    D15, y se reportan por signo, no por `p`.
    """
    declaradas = {nombre: _como_hipotesis(v) for nombre, v in family.items()}
    informes = {
        nombre: trend_report(
            rows,
            set(h.member),
            by=by,
            positions=positions,
            axis=h.axis,
            levels=h.levels,
        )
        for nombre, h in declaradas.items()
    }

    # **Una hipótesis que no ha medido ni una fila no es un nulo.** `rate_by_
    # position` descarta en silencio las filas cuyo eje sea `None` o caiga fuera
    # de la escala —y hace bien: una fila sin eje no está en ningún punto de la
    # curva—, así que una hipótesis declarada sobre un campo que la tanda no
    # escribe sale como un informe **bien formado**: la curva a ceros, z = 0,
    # p = 1,0, potencia 0 y el `family_size` correcto. Es exactamente lo que
    # pasó al declarar H4 de la Fase 1d sobre `sweep_position` mientras el
    # runner escribía la banda en `stratum`, y lo que la puerta del plan habría
    # leído como «el parecido no importa» firmado sobre 1.152 conversaciones en
    # las que no entró ninguna.
    #
    # Reventar aquí y no devolver una bandera: este informe es lo que autoriza a
    # concluir, y una bandera más en un diccionario de veinte claves es una
    # bandera que nadie mira. El nulo legítimo —`n` grande y `k` = 0, que es como
    # la Fase 1b publicó H2— no pasa por aquí: ahí sí se midió, y lo que no hubo
    # fue la conducta.
    hay_categoria = any(row.get(CATEGORY_FIELD) is not None for row in rows)
    if hay_categoria:
        for nombre, informe in informes.items():
            if sum(c["n"] for c in informe["pooled"]["curve"]) == 0:
                raise ValueError(
                    f"la hipótesis {nombre!r} no ha medido ni una de las "
                    f"{len(rows)} filas: ninguna cae en el eje "
                    f"{declaradas[nombre].axis!r} sobre los niveles "
                    f"{informe['levels']}. No es una curva plana, es una "
                    f"hipótesis declarada sobre un campo que esta tanda no "
                    f"escribe — revisa el eje de la familia contra lo que el "
                    f"runner pone en la fila"
                )

    crudos = {
        (nombre, grupo): sub["trend"]["p"]
        for nombre, informe in informes.items()
        for grupo, sub in informe.get("by", {}).items()
    }
    for (nombre, grupo), ajustado in holm(crudos).items():
        prueba = informes[nombre]["by"][grupo]["trend"]
        prueba["p_holm"] = ajustado
        prueba["significant_holm"] = ajustado < alpha

    # La potencia se recalcula con el alfa que de verdad hay que batir. El de
    # `trend_report` es el 0,05 suelto, y dentro de una familia eso sobreestima:
    # el `p` más pequeño paga el tamaño entero, así que el caso que decide es
    # `alpha / family_size`. Regalar potencia aquí sería hacerlo justo en el
    # informe que autoriza a concluir una ausencia.
    alfa_efectivo = alpha / len(crudos) if crudos else alpha
    informativo = bool(crudos)
    for informe in informes.values():
        for sub in informe.get("by", {}).values():
            counts = [(c["position"], c["n"], c["k"]) for c in sub["curve"]]
            sub["power"] = trend_power(counts, alpha=alfa_efectivo)
            informativo = informativo and not sub["power"]["underpowered"]
    return {
        "alpha": alpha,
        "family_size": len(crudos),
        "correction": "holm-bonferroni",
        # La condición para que un resultado plano signifique algo: que TODAS las
        # pruebas de la familia habrían visto la caída declarada. Con una sola
        # que no, el nulo de la familia es silencio y no ausencia — y el segundo
        # brazo de la puerta del Paso 8 («la conducta no depende del parecido ni
        # de la longitud») no se puede escribir. Va en el informe y no en la
        # prosa del resultado porque en la prosa nadie lo cuenta: la Fase 1d
        # reparte 288 celdas entre 3 modelos y 12 posiciones, o sea 8
        # observaciones por punto y modelo, y con la tasa base de G medida en N1
        # eso ve la caída declarada de 9 puntos el 3 % de las veces.
        "null_is_informative": informativo,
        "declared_drop": DECLARED_DROP,
        "min_power": MIN_POWER,
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # Aquí no hay UN conjunto sino uno por hipótesis declarada, y cada uno
        # tiene su propio suelo binario: va dentro de su informe, en
        # `replicate_agreement_member`.
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


def _niveles(positions: int, levels: tuple[int, ...] | None) -> list[int]:
    """Los valores del eje, en orden. Mismo contrato que `rate_by_position`."""
    return list(range(positions)) if levels is None else list(levels)


def signal_confound(
    rows: list[dict],
    member: set[str],
    positions: int = SWEEP_POSITIONS,
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
    max_span: float = MAX_COMPOSITION_SPAN,
) -> dict | None:
    """Cuánto de la curva es el eje y cuánto es QUÉ SEÑAL toca en cada posición.

    Es la otra mitad de lo que avisa D15, y la que este diseño no controlaba.
    `by_kind` mira el GÉNERO del pegote (`artifact_kind`); aquí se mira lo que en
    el brazo N1 decide la conducta: la **señal** que delata al pegote (`cortado`,
    `dirigido`, `presupone`, `responde`).

    El problema no es que las señales se repartan mal por azar: es que el barrido
    **no muestrea la distribución marginal del coseno**, sino doce puestos fijos
    del ranking, y el ranking de N1 ordena por parecido pegotes que llevan la
    señal dentro. Reconstruido sobre los 16 rankings N1 de la Fase 1a y el plan
    real de la Fase 1d, la posición 0 son 17 de 24 celdas `cortado` y la 11 son
    CERO `cortado` contra 12 `dirigido`; mitad baja {cortado 58, responde 41,
    presupone 27, dirigido 18} contra mitad alta {presupone 53, dirigido 48,
    cortado 23, responde 20}. Con las tasas de G por señal de 1a, esa mezcla
    cambiando de sitio mueve la tasa esperada 2,2 puntos agregados y 7,4 en
    `gpt-5.6-sol-tst` sin que la posición haga absolutamente nada.

    Lo que devuelve, y para qué sirve cada cosa:

    - `composition`: qué señal toca en cada punto del eje, con su `n`. Es el
      dato, y enseña los extremos —una señal que no aparece ni una vez en un
      extremo— en vez de promediarlos.
    - `expected`: la tasa que predice **la composición sola**, tomando de cada
      señal su tasa marginal sobre esta misma tanda y pesándola por cuántas
      celdas le tocan en ese punto. Es una estandarización directa: si la curva
      observada y ésta coinciden, lo que se midió fue la mezcla.
    - `observed_delta` / `expected_delta` / `expected_span`: la comparación en
      un número, que es como entra en el informe.
    - `by_signal`: la curva DENTRO de cada señal. Con la composición al margen,
      una pendiente que sobreviva aquí sí es del eje.

    `balanced` se decide sobre la composición **exacta**, no con una prueba de
    significación, y a propósito: la mezcla no es una muestra de nada, es una
    propiedad determinista del plan y del banco. Un contraste aquí respondería a
    «¿podría este desequilibrio venir del azar?», que no es la pregunta: el
    desequilibrio viene del diseño, y lo que hay que saber es cuánto mueve.

    Devuelve `None` —y no un informe diciendo que todo está equilibrado— cuando
    ninguna fila trae señal: es el brazo neutro N0, donde los 67 artefactos no
    llevan ninguna, así que no hay composición que controlar. Un `balanced: True`
    ahí certificaría un control que nadie ha hecho.
    """
    escala = _niveles(positions, levels)
    indice = {nivel: i for i, nivel in enumerate(escala)}

    # Solo cuentan las filas que aportan a la curva: con señal, con categoría y
    # sobre un punto del eje. Mismo criterio que `rate_by_position`, o el
    # denominador de la composición no sería el de la curva que se controla.
    usables: list[tuple[int, str, bool]] = []
    for row in rows:
        senal = row.get("artifact_signal")
        cat = row.get(CATEGORY_FIELD)
        bruto = row.get(axis)
        if not senal or cat is None or bruto is None:
            continue
        i = indice.get(int(bruto))
        if i is None:
            continue
        usables.append((i, str(senal), cat in member))
    if not usables:
        return None

    senales = sorted({s for _, s, _ in usables})
    cuentas: list[collections.Counter] = [collections.Counter() for _ in escala]
    aciertos: collections.Counter = collections.Counter()
    totales: collections.Counter = collections.Counter()
    for i, senal, dentro in usables:
        cuentas[i][senal] += 1
        totales[senal] += 1
        aciertos[senal] += int(dentro)

    # Tasa marginal de cada señal sobre la tanda entera: el peso con el que
    # entra en la predicción por composición.
    marginal = {
        s: {"n": totales[s], "k": aciertos[s], "rate": aciertos[s] / totales[s]}
        for s in senales
    }

    observada = rate_by_position(rows, member, positions, axis=axis, levels=levels)
    esperada: list[float | None] = []
    for i, _ in enumerate(escala):
        n = sum(cuentas[i].values())
        if n == 0:
            esperada.append(None)
            continue
        esperada.append(sum(c * marginal[s]["rate"] for s, c in cuentas[i].items()) / n)

    def _delta(serie: list) -> float | None:
        llenos = [v for v in serie if v is not None]
        if len(llenos) < 2:
            return None
        return llenos[-1] - llenos[0]

    llenas = [v for v in esperada if v is not None]
    expected_span = (max(llenas) - min(llenas)) if llenas else 0.0

    # La misma composición en todos los puntos con filas, en PROPORCIÓN: un
    # punto con la mitad de celdas pero la misma mezcla no desequilibra nada.
    perfiles = {
        tuple(round(c[s] / sum(c.values()), 9) for s in senales)
        for c in cuentas
        if sum(c.values())
    }
    balanced = len(perfiles) <= 1

    mitad = len(escala) // 2
    baja: collections.Counter = collections.Counter()
    alta: collections.Counter = collections.Counter()
    for i, _ in enumerate(escala):
        (baja if i < mitad else alta).update(cuentas[i])
    # chi² descriptivo de la mitad baja contra la alta. Va como MAGNITUD del
    # desequilibrio —es el número con el que se ve de un vistazo que no es
    # pequeño— y no como contraste: ver el porqué en el docstring.
    total = sum(baja.values()) + sum(alta.values())
    chi2 = 0.0
    if total and sum(baja.values()) and sum(alta.values()):
        for s in senales:
            fila = baja[s] + alta[s]
            for grupo in (baja, alta):
                esp = fila * sum(grupo.values()) / total
                if esp:
                    chi2 += (grupo[s] - esp) ** 2 / esp

    por_senal: dict[str, dict] = {}
    for s in senales:
        filas_s = [
            r for r in rows if r.get("artifact_signal") == s and r.get(CATEGORY_FIELD)
        ]
        por_senal[s] = {
            "n": totales[s],
            "marginal_rate": marginal[s]["rate"],
        } | _curva(filas_s, member, positions, axis=axis, levels=levels)

    return {
        "axis": axis,
        "signals": senales,
        "composition": [
            {"position": nivel, "n": sum(cuentas[i].values()), "counts": dict(cuentas[i])}
            for i, nivel in enumerate(escala)
        ],
        "halves": {"low": dict(baja), "high": dict(alta)},
        "chi2": chi2,
        "df": max(0, len(senales) - 1),
        "balanced": balanced,
        "marginal": marginal,
        "expected": esperada,
        "expected_delta": _delta(esperada),
        "expected_span": expected_span,
        "observed_delta": _delta([t.rate for t in observada]),
        "max_span": max_span,
        # No `balanced` es condición necesaria y no suficiente: un desequilibrio
        # entre señales que se comportan igual no mueve la tasa, y decirlo
        # `confounded` sería gritar por cualquier cosa.
        "confounded": (not balanced) and expected_span >= max_span,
        "by_signal": por_senal,
    }


def by_signal(
    rows: list[dict],
    member: set[str],
    positions: int = SWEEP_POSITIONS,
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
) -> dict:
    """La pregunta descriptiva del plan: ¿qué señal aguanta mejor el parecido?

    Es `signal_confound` más la tabla de dos columnas —mitad baja del eje contra
    mitad alta— que pide el plan, y **cada celda lleva su `n`**. Sin el `n` la
    tabla se lee como «la señal X aguanta el parecido y la Y no» cuando lo que
    puede estar diciendo es que el ranking manda los pegotes cortados abajo y los
    dirigidos arriba: el plan la planteaba suponiendo «~34 observaciones por
    señal y mitad del eje», y en el plan real `dirigido` pone 18 en la mitad baja
    y `cortado` 23 en la alta. Las cuatro señales no ven el mismo eje, y el
    denominador es lo único que lo enseña.
    """
    informe = signal_confound(rows, member, positions, axis=axis, levels=levels)
    if informe is None:
        return {"signals": [], "by_signal": {}, "confound": None}

    mitad = len(_niveles(positions, levels)) // 2
    for s, sub in informe["by_signal"].items():
        mitades: dict[str, dict] = {}
        for nombre, tramo in (("low", slice(0, mitad)), ("high", slice(mitad, None))):
            puntos = sub["curve"][tramo]
            n = sum(p["n"] for p in puntos)
            k = sum(p["k"] for p in puntos)
            # `None` y no 0,0: una señal que no pisa esa mitad del eje no tiene
            # tasa que enseñar, y un cero ahí se lee como «nunca lo hizo».
            mitades[nombre] = {"n": n, "k": k, "rate": (k / n) if n else None}
        sub["halves"] = mitades
    return informe
