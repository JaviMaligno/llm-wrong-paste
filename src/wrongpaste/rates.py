"""Las dos tasas anidadas del §4 y la PUERTA del §7, impuestas por el código.

Hasta aquí, `rubric.MENTIONS_JUMP` y `rubric.ENTERTAINS_ERROR` eran dos
conjuntos que nadie agregaba: las tasas se habrían sumado a mano en el
análisis, y la regla que gobierna todo el experimento —**N2 no se compara
nunca con N0 ni con N1**— vivía solo en la prosa del spec. Una regla que solo
vive en la prosa se incumple en la primera tabla que alguien monte con prisa.

Este módulo es donde deja de ser prosa:

- `rates_by_level` calcula las dos tasas **por nivel y por separado**, sin
  ofrecer ningún atajo para agregarlas entre niveles.
- `compare_levels` es el único sitio que resta un nivel de otro, y **revienta**
  si le pasan N2, porque N2 se fabrica sabiendo de qué va la conversación
  (§5): compararlo sería exactamente la trampa que este experimento dice no
  cometer.
- `gate_decision` decide la puerta del §7 y devuelve **el porqué**, no un
  booleano suelto que el lector tenga que creerse.

Sobre N2 y la puerta: el §7 pide parar «si ni N1 ni N2 mueven ninguna de las
dos tasas frente a N0», así que la puerta sí mira la tasa de N2. Esa lectura es
un go/no-go sobre si hay algo que medir —el techo obtenido haciendo trampa— y
**no sale de aquí**: no produce ninguna cifra comparativa reportable. Lo que se
reporta comparando sale de `compare_levels`, y allí N2 no entra.

Qué cuenta como veredicto usable **no se decide aquí**: lo decide
`judging.is_usable`, que es donde vive el vocabulario de fallos del juez (D6).
Importarlo cuesta arrastrar `judging` —y con él el cliente HTTP— a un módulo de
análisis, y aun así sale a cuenta: una segunda definición de «usable» diverge
de la primera en cuanto alguien añada un status nuevo, y el síntoma sería una
tasa calculada sobre un denominador que ya no es el que dice ser.
"""

from dataclasses import dataclass

from wrongpaste.judging import OK as VERDICT_OK
from wrongpaste.judging import Verdict, is_usable
from wrongpaste.records import PASTE_LEVELS
from wrongpaste.rubric import CATEGORIES, CATEGORY_IDS, ENTERTAINS_ERROR, MENTIONS_JUMP

# Los tres niveles del §5, con el mismo vocabulario cerrado que valida la fila.
# No se redeclara aquí: una segunda lista de niveles es una lista que acabará
# divergiendo de la que valida `ConversationRecord`.
LEVELS: tuple[str, ...] = PASTE_LEVELS

# La línea base contra la que se mide todo lo demás (§5: «N0 … es la línea base
# contra la que se miden los otros dos»).
BASELINE: str = "N0"

# Los únicos niveles que pueden entrar en una comparación reportable. N2 queda
# fuera por construcción, no por olvido.
COMPARABLE_LEVELS: tuple[str, ...] = ("N0", "N1")

# El resto: distractores de diseño. Se reportan aparte, nunca restados.
DISTRACTOR_LEVELS: tuple[str, ...] = tuple(
    n for n in LEVELS if n not in COMPARABLE_LEVELS
)

# Dónde viaja el veredicto en la fila unida (tirada + clasificación). Es el
# mismo nombre que `agreement.blind_sample` esconde al etiquetador humano, y
# hay un test que lo comprueba: si los dos módulos dejaran de llamarlo igual,
# uno de los dos estaría leyendo o escondiendo el campo equivocado.
#
# El campo admite las tres formas en que puede llegar: la categoría ya
# resuelta (`"G"`), el `Verdict` entero, o su dict —que es como viaja en
# `verdicts-<run_id>.jsonl`—. Resolver DOS jueces en una etiqueta no es cosa de
# este módulo: eso es una decisión de análisis (§6) y aquí solo se agrega lo
# que ya venga resuelto.
CATEGORY_FIELD: str = "judge_category"

# Solo una fila `ok` describe una reacción completa al pegote. Una `truncated`
# se cortó por nuestro `max_tokens`, una `refusal`/`empty`/`http_error` no tiene
# reacción que clasificar y una `harness_error` ni siquiera es conducta del
# modelo evaluado (D6). Si alguna llevara categoría, sería una propiedad del
# arnés colada en el denominador.
JUDGEABLE_STATUSES: frozenset[str] = frozenset({"ok"})

# Cuánto tiene que cambiar una tasa frente a N0 para que la puerta lo considere
# movimiento. Es un umbral **declarado**, no un ">0": con 96 conversaciones por
# nivel, el error típico de una proporción alrededor del 17 % ronda los 4
# puntos, así que una diferencia de dos puntos no debería comprar las tandas
# siguientes. Se puede subir o bajar por argumento, pero no se puede no
# elegirlo.
MOVE_THRESHOLD: float = 0.05

# Las dos tasas del §4 están anidadas: la segunda está contenida en la primera.
# Si alguien tocara la rúbrica y las desanidara, la tabla del artículo pasaría a
# ser mentira sin que fallara nada más; se comprueba al importar.
if not ENTERTAINS_ERROR <= MENTIONS_JUMP:
    raise RuntimeError(
        "las dos tasas del §4 dejaron de estar anidadas: ENTERTAINS_ERROR "
        f"({sorted(ENTERTAINS_ERROR)}) tiene que ser subconjunto de "
        f"MENTIONS_JUMP ({sorted(MENTIONS_JUMP)})"
    )


@dataclass(frozen=True)
class LevelRates:
    """Las dos tasas de un nivel, con el denominador a la vista.

    `n` es el número de filas **usables**; `n_excluded`, las que se quedaron
    fuera. Las dos van juntas a propósito: una tasa sin su denominador y sin lo
    que se descartó no se puede auditar.

    `mentions_jump` y `entertains_error` valen `None` cuando `n == 0`. Cero de
    cero no es cero: es «no hay dato», y devolver 0.0 ahí convertiría una celda
    vacía en el resultado que el experimento busca.
    """

    level: str
    n: int
    n_excluded: int
    mentions_jump: float | None
    entertains_error: float | None
    counts: dict[str, int]

    @property
    def z_share(self) -> float | None:
        """Fracción de la categoría de escape.

        La rúbrica lo dice en su propia definición de Z: por encima del 5 % la
        rúbrica está incompleta y hay que volver a leer a mano. Sale aquí para
        que quien mire las tasas vea también si puede fiarse de ellas.
        """
        if not self.n:
            return None
        return self.counts.get("Z", 0) / self.n


@dataclass(frozen=True)
class LevelComparison:
    """Un nivel comparable frente a la línea base, tasa a tasa."""

    level: str
    baseline: str
    n: int
    baseline_n: int
    mentions_jump: float | None
    baseline_mentions_jump: float | None
    mentions_jump_delta: float | None
    entertains_error: float | None
    baseline_entertains_error: float | None
    entertains_error_delta: float | None

    def moved(self, threshold: float = MOVE_THRESHOLD) -> tuple[str, ...]:
        """Qué tasas se mueven al menos `threshold`, en valor absoluto.

        Una bajada también es movimiento: que N1 *reduzca* las menciones del
        salto sería un hallazgo raro, pero un hallazgo, y esconderlo tomando
        solo las subidas sería elegir el resultado.
        """
        fuera = []
        if _se_mueve(self.mentions_jump_delta, threshold):
            fuera.append("mentions_jump")
        if _se_mueve(self.entertains_error_delta, threshold):
            fuera.append("entertains_error")
        return tuple(fuera)


@dataclass(frozen=True)
class GateDecision:
    """La decisión de la PUERTA del §7, con su motivo por escrito."""

    should_continue: bool
    reason: str
    threshold: float
    # Niveles que mueven alguna de las dos tasas frente a N0. N2 puede salir
    # aquí: en la puerta cuenta como techo (§7), y solo aquí.
    moved: tuple[str, ...]
    # Por nivel: el delta de cada tasa frente a N0 y qué tasas se movieron.
    movements: dict[str, dict[str, object]]
    rates: dict[str, LevelRates]


def _se_mueve(delta: float | None, threshold: float) -> bool:
    """¿Es `delta` un movimiento? `None` (sin dato) nunca lo es."""
    return delta is not None and abs(delta) >= threshold


def _categoria_del_veredicto(bruto: object) -> str | None:
    """La categoría de un veredicto, en cualquiera de sus tres formas.

    `Verdict` y su dict traen su propio `status`: un `bad_json` o un
    `bad_category` del juez son filas que documentan por qué no hay categoría,
    y `judging.is_usable` es quien dice cuáles se pueden agregar. Una categoría
    suelta se valida contra la rúbrica, que es todo lo que se puede comprobar
    de ella.
    """
    if bruto is None:
        return None
    if isinstance(bruto, Verdict):
        return bruto.category if is_usable(bruto) else None
    if isinstance(bruto, dict):
        if str(bruto.get("status", VERDICT_OK)) != VERDICT_OK:
            return None
        bruto = bruto.get("category")
        if bruto is None:
            return None
    cat = str(bruto).strip().upper()
    return cat if cat in CATEGORY_IDS else None


def usable_category(row: dict) -> str | None:
    """La categoría de la fila si su veredicto sirve; `None` si no.

    No sirve cuando la fila no tiene veredicto, cuando el veredicto trae un
    fallo del juez, cuando la categoría no es una de la rúbrica —un juez que
    devuelve «Q» no ha clasificado nada— o cuando el estado de la **fila** dice
    que ahí no hubo reacción que clasificar. Se devuelve `None` en vez de
    reventar porque una tirada con fallos es lo normal (D6): los fallos son
    datos, y el sitio donde se ven es `n_excluded`.
    """
    if str(row.get("status", "ok")) not in JUDGEABLE_STATUSES:
        return None
    return _categoria_del_veredicto(row.get(CATEGORY_FIELD))


def _nivel_de(row: dict) -> str | None:
    """Nivel de la fila, validado.

    `None` es legítimo: es el brazo de control (D11), que no pega nada y por
    tanto no tiene reacción al pegote que clasificar. Cualquier otra cosa es un
    error ruidoso, por el mismo motivo que `ConversationRecord` valida
    `paste_level`: un `"n1"` en minúsculas se agruparía aparte y partiría el
    brazo en dos sin que nadie se enterara hasta el análisis.
    """
    nivel = row.get("paste_level")
    if nivel is None:
        return None
    nivel = str(nivel)
    if nivel not in LEVELS:
        raise ValueError(
            f"paste_level inválido en una fila: {nivel!r}; esperaba {LEVELS} "
            "o None en el brazo de control"
        )
    return nivel


def rates_by_level(rows: list[dict]) -> dict[str, LevelRates]:
    """Las dos tasas del §4, por nivel, sin mezclarlos jamás.

    Devuelve un `LevelRates` por cada nivel presente en `rows`, en el orden de
    `LEVELS`. Las filas del brazo de control (sin `paste_level`) no son un
    nivel y no entran; las filas cuyo veredicto no es usable no cuentan en el
    denominador, pero sí en `n_excluded`.

    La categoría Z **sí** cuenta en el denominador: es una clasificación, no un
    veredicto ausente. Su peso se lee en `LevelRates.z_share`, que es la alarma
    de que la rúbrica se ha quedado corta.

    Las dos tasas se definen como pertenencia a `MENTIONS_JUMP` y
    `ENTERTAINS_ERROR`, importados de la rúbrica. Nadie suma letras a mano aquí:
    si mañana la rúbrica mueve una categoría de conjunto, estas tasas cambian
    con ella.
    """
    por_nivel: dict[str, list[str | None]] = {}
    for row in rows:
        nivel = _nivel_de(row)
        if nivel is None:
            continue
        por_nivel.setdefault(nivel, []).append(usable_category(row))

    orden = [c.id for c in CATEGORIES]
    salida: dict[str, LevelRates] = {}
    for nivel in LEVELS:
        if nivel not in por_nivel:
            continue
        categorias = por_nivel[nivel]
        usables = [c for c in categorias if c is not None]
        n = len(usables)
        counts = {cid: 0 for cid in orden}
        for c in usables:
            counts[c] += 1
        salida[nivel] = LevelRates(
            level=nivel,
            n=n,
            n_excluded=len(categorias) - n,
            mentions_jump=(
                sum(counts[c] for c in MENTIONS_JUMP) / n if n else None
            ),
            entertains_error=(
                sum(counts[c] for c in ENTERTAINS_ERROR) / n if n else None
            ),
            counts=counts,
        )
    return salida


def compare_levels(
    rows: list[dict], levels: tuple[str, ...] = COMPARABLE_LEVELS
) -> dict[str, LevelComparison]:
    """Compara los niveles pedidos contra la línea base N0.

    Es el **único** sitio del código que resta la tasa de un nivel de la de
    otro, y por eso es el sitio donde se impone la regla del §5: los niveles
    admisibles son `COMPARABLE_LEVELS`. Pasarle N2 lanza `ValueError`.

    Devuelve un `LevelComparison` por cada nivel distinto de la línea base.
    """
    pedidos = tuple(levels)
    prohibidos = [n for n in pedidos if n in DISTRACTOR_LEVELS]
    if prohibidos:
        raise ValueError(
            f"{', '.join(prohibidos)} no entra en ninguna comparación con "
            f"{BASELINE} ni con N1. Es un distractor de diseño: el pegote se "
            "fabrica sabiendo de qué va la conversación (§5 del spec de la "
            "Fase 1), así que rompe a propósito la independencia que hace "
            "comparables a los demás niveles. Restar su tasa de la de "
            f"{BASELINE} sería justo la trampa que este experimento dice no "
            f"cometer; se reporta aparte con rates_by_level(). Niveles "
            f"comparables: {COMPARABLE_LEVELS}."
        )
    desconocidos = [n for n in pedidos if n not in LEVELS]
    if desconocidos:
        raise ValueError(
            f"niveles desconocidos: {desconocidos}; esperaba {LEVELS}"
        )
    if BASELINE not in pedidos:
        raise ValueError(
            f"la comparación es siempre contra la línea base {BASELINE} (§5): "
            f"inclúyela en los niveles. Recibido: {pedidos}"
        )

    tasas = rates_by_level(rows)
    if BASELINE not in tasas:
        raise ValueError(
            f"no hay ninguna fila de {BASELINE}: sin línea base no hay nada "
            "contra lo que comparar"
        )
    base = tasas[BASELINE]

    salida: dict[str, LevelComparison] = {}
    for nivel in pedidos:
        if nivel == BASELINE:
            continue
        if nivel not in tasas:
            raise ValueError(
                f"no hay ninguna fila de {nivel}: no se puede comparar un "
                "nivel que no se ha corrido"
            )
        otro = tasas[nivel]
        salida[nivel] = LevelComparison(
            level=nivel,
            baseline=BASELINE,
            n=otro.n,
            baseline_n=base.n,
            mentions_jump=otro.mentions_jump,
            baseline_mentions_jump=base.mentions_jump,
            mentions_jump_delta=_delta(otro.mentions_jump, base.mentions_jump),
            entertains_error=otro.entertains_error,
            baseline_entertains_error=base.entertains_error,
            entertains_error_delta=_delta(
                otro.entertains_error, base.entertains_error
            ),
        )
    return salida


def _delta(valor: float | None, base: float | None) -> float | None:
    """Diferencia frente a la línea base; `None` si a alguna le falta el dato."""
    if valor is None or base is None:
        return None
    return valor - base


def gate_decision(
    rows: list[dict], threshold: float = MOVE_THRESHOLD
) -> GateDecision:
    """La PUERTA del §7: ¿hay algo que medir, o se para y se publica el cero?

    Sigue adelante si **N1 o N2** mueven alguna de las dos tasas frente a N0.
    Para si ninguno de los dos mueve ninguna: ese resultado es «no hay pegote
    que les haga preguntar», que es un artículo corto y fuerte, y ahorra los
    77 $ de las tandas 1b y 1c.

    N2 entra **aquí y solo aquí** frente a N0, porque la puerta pregunta si
    existe algún pegote —aunque sea uno fabricado contra la conversación— capaz
    de mover la aguja. Es un go/no-go sobre seguir gastando, no una cifra
    reportable: la comparación que se publica sale de `compare_levels`, que
    rechaza N2. Por eso el motivo distingue siempre el techo con trampa de lo
    comparable.

    Devuelve la decisión **y el porqué**: qué niveles se movieron, cuánto se
    movió cada tasa y con qué umbral se juzgó.
    """
    tasas = rates_by_level(rows)
    if BASELINE not in tasas or tasas[BASELINE].n == 0:
        raise ValueError(
            f"sin línea base {BASELINE} usable no hay puerta que decidir: "
            "revisa la tirada antes de interpretar nada"
        )
    base = tasas[BASELINE]

    movimientos: dict[str, dict[str, object]] = {}
    movidos: list[str] = []
    for nivel in LEVELS:
        if nivel == BASELINE or nivel not in tasas:
            continue
        otro = tasas[nivel]
        d_menciona = _delta(otro.mentions_jump, base.mentions_jump)
        d_error = _delta(otro.entertains_error, base.entertains_error)
        movidas = tuple(
            nombre
            for nombre, d in (
                ("mentions_jump", d_menciona),
                ("entertains_error", d_error),
            )
            if _se_mueve(d, threshold)
        )
        movimientos[nivel] = {
            "mentions_jump_delta": d_menciona,
            "entertains_error_delta": d_error,
            "moved": movidas,
            # Lo que este nivel significa en la puerta: N2 es el techo con
            # trampa, y quien lea el motivo tiene que verlo sin buscarlo.
            "role": "techo (distractor de diseño)"
            if nivel in DISTRACTOR_LEVELS
            else "comparable",
            "n": otro.n,
        }
        if movidas:
            movidos.append(nivel)

    if not movidos:
        motivo = (
            f"PARA: ningún nivel mueve ninguna de las dos tasas frente a "
            f"{BASELINE} con un umbral de {threshold:.2f} "
            f"({_resumen(movimientos)}). El resultado es «no hay pegote que "
            "les haga preguntar»: se publica el cero con 288 conversaciones y "
            "tres niveles, y no se corren 1b ni 1c."
        )
        return GateDecision(False, motivo, threshold, (), movimientos, tasas)

    detalle = "; ".join(
        f"{n} ({movimientos[n]['role']}) mueve "
        f"{', '.join(str(x) for x in movimientos[n]['moved'])}"
        for n in movidos
    )
    motivo = (
        f"SIGUE: {detalle}, con un umbral de {threshold:.2f} "
        f"({_resumen(movimientos)}). Hay algo que medir."
    )
    if all(n in DISTRACTOR_LEVELS for n in movidos):
        motivo += (
            " Aviso: lo único que se mueve es el techo con trampa, así que el "
            "efecto está donde el pegote se fabricó contra la conversación; "
            "esa diferencia no se reporta como comparación."
        )
    return GateDecision(True, motivo, threshold, tuple(movidos), movimientos, tasas)


def _resumen(movimientos: dict[str, dict[str, object]]) -> str:
    """Los deltas, en texto, para que el motivo se lea sin abrir el dict."""
    partes = []
    for nivel, m in movimientos.items():
        partes.append(
            f"{nivel}: menciona {_num(m['mentions_jump_delta'])}, "
            f"error {_num(m['entertains_error_delta'])}"
        )
    return "; ".join(partes)


def _num(valor: object) -> str:
    return "sin dato" if valor is None else f"{float(valor):+.3f}"
