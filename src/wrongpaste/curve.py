"""El análisis del eje de similaridad: cómo cambia la mezcla de la taxonomía.

Sirve a dos tandas con muestreos distintos. La Fase 1b barrió doce puestos del
ranking sobre el brazo neutro (`sweep_position`); la Fase 1d muestrea bandas de
rango sobre el brazo con señal (`stratum`) y contrasta DOS —mitad baja contra
mitad alta—, que es donde llega la potencia. Cada tanda declara su familia de
hipótesis y su eje, y ninguna de las dos hereda los de la otra.

Se escribe **antes** de correr la tanda. Las hipótesis del plan quedan en
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
    el MISMO conjunto (G) y ejes distintos: H4 lo contrasta contra la BANDA del
    ranking y H5 contra la longitud de la conversación. Sin el eje dentro de
    la declaración, H5 no se puede correr, y una hipótesis que no se puede
    correr no entra en Holm: la familia se queda en la mitad de las pruebas que
    el plan declaró, el `p` más pequeño paga x3 en vez de x6, y un 0,012 crudo
    sale 0,036 —significativo— en lugar de 0,072 —no—.

    `levels` son los valores ordenados del eje, y hacen de puntuación de
    Cochran-Armitage. `None` significa `range(positions)`, que es el barrido.
    Con dos niveles —(2, 10) turnos— Cochran-Armitage es exactamente la prueba z
    de dos proporciones con varianza agrupada, así que no hace falta una prueba
    nueva: hace falta poder decirle sobre qué columna mirar.

    `fold_from` dice cuántas bandas trae la FILA cuando el contraste declara
    menos: las filas de la Fase 1d llevan una de las cuatro bandas del muestreo
    en `stratum` y H4 se contrasta sobre dos. Sin este campo, declarar el
    primario de dos bandas sobre un `levels` de (0, 1) no agruparía las bandas
    2 y 3: las tiraría, y la mitad de la tanda se caería del contraste que la
    paga sin que nada se quejara —exactamente el modo de fallo que este módulo
    lleva dos commits corrigiendo—. `None` es «el eje ya viene en los niveles
    que se contrastan», que es el caso de `sweep_position` y de `n_turns`.
    """

    member: frozenset[str]
    axis: str = "sweep_position"
    levels: tuple[int, ...] | None = None
    fold_from: int | None = None


def _como_hipotesis(valor: Hypothesis | frozenset[str] | set[str]) -> Hypothesis:
    """Un conjunto pelado sigue valiendo: es una hipótesis sobre el barrido.

    La familia de la Fase 1b está publicada como nombre -> conjunto y se queda
    así. Normalizar aquí es lo que permite que `primary_family_report` sin
    argumentos siga dando byte a byte lo mismo que cuando se publicó.
    """
    return valor if isinstance(valor, Hypothesis) else Hypothesis(frozenset(valor))


# El eje de la Fase 1d: BANDAS por rango del ranking, no puestos de un barrido.
#
# Viaja en la fila como `stratum`/`n_strata` porque una banda es literalmente un
# estrato por rango —la misma partición que `stratum_window` de la Fase 0—, y el
# runner deja `sweep_position` en `None`: escribir la banda ahí haría que un
# análisis conjunto leyera las dos tandas como si hubieran muestreado igual.
BAND_AXIS = "stratum"

# El campo en el que la FILA declara en cuántas bandas partió el ranking la
# tanda que la escribió. Es el mismo `n_strata` de la Fase 0 —una banda es un
# estrato por rango— y lo escribe `run_phase1d.run_cell`, también en las filas
# fallidas.
#
# Está aquí porque es lo único que tiene el análisis para saber si su pliegue le
# corresponde a estas filas. Sin mirarlo, plegar desde un número que no es el del
# muestreo no revienta: `fold_band` devuelve `None` para las bandas que sobran y
# los dos sitios que pliegan las descartan, así que sale una curva bien formada
# sobre media tanda.
BAND_COUNT_FIELD = "n_strata"

# Las bandas que MUESTREA el runner. Se escribe aquí y no se importa de
# `run_phase1d` para no atar el análisis al runner —`curve` no sabe de tiradas—,
# y `test_las_bandas_que_pliega_el_analisis_son_las_que_muestrea_este_runner`
# —en la suite del runner, que es la que puede importar los dos— comprueba que
# los dos números sigan siendo el mismo. Duplicar el 4 sin ese test sería la
# manera de que el día que alguien mueva las bandas el análisis siga leyendo
# cuatro; ese test cubre el código, y `_check_band_sampling` cubre las filas ya
# escritas, que es el caso que el test no puede ver.
BANDS = 4

# Las bandas que CONTRASTA el primario: dos, la mitad baja del ranking contra la
# alta. **El muestreo y el contraste no tienen por qué tener la misma
# resolución**, y aquí no la tienen a propósito.
#
# El runner muestrea cuatro porque es lo que hace que el banco N1 de 72 dé 18
# artefactos por banda —entero y múltiplo del plantel—. El contraste agrupa a dos
# porque la resolución del eje se paga en potencia y esta tanda no la puede
# pagar: con la tirada encogida a un solo modelo (Opus es el único que produce G
# en N1) las 1.152 celdas son 288 por punto con cuatro bandas y 576 con dos, y a
# la tasa base de Opus eso es la diferencia entre ver la caída declarada de 9
# puntos el 63 % de las veces y verla el **87 %**. La pregunta de la fase es si
# G baja con el parecido, no dibujar la forma de la bajada.
#
# Los dos números son al alfa suelto de 0,05. Al que de verdad paga el `p` más
# pequeño de la familia declarada —H4 y H5 sobre un modelo, o sea 0,025— son
# 0,52 y **0,797**, y ese 0,797 se queda tres milésimas por debajo de
# `MIN_POWER`: el pliegue casi arregla la potencia de esta tanda y no llega. Que
# el número esté ahí y no redondeado a «0,8» es lo único que impide que el nulo
# se acabe leyendo como ausencia.
#
# Se declara ANTES de mirar los datos, que es lo único que distingue esto de
# elegir el número de puntos por el `p` que sale. Las particiones de 3 y 4 se
# reportan, y se reportan como secundarias.
PRIMARY_BANDS = 2
PRIMARY_BAND_LEVELS: tuple[int, ...] = tuple(range(PRIMARY_BANDS))

# Las particiones que se reportan y NO deciden. Entran en `secondary_band_
# reports`, no en la familia de Holm: describir la forma del eje con más puntos
# es útil y no es un contraste, y meterlas en la familia le costaría potencia al
# primario por pruebas que no cierran ninguna puerta.
SECONDARY_BAND_SPLITS: tuple[int, ...] = (3, 4)


def fold_band(raw: int, n_bands: int, sampled: int = BANDS) -> int | None:
    """La banda de ANÁLISIS en la que cae una banda del MUESTREO.

    Se agrupa por el CENTRO de la banda muestreada: la banda `raw` cubre el
    tramo de rango [raw/sampled, (raw+1)/sampled) y se le asigna la banda de
    análisis donde cae su punto medio. Con 4 → 2 el reparto es exacto y
    cualquier regla razonable da lo mismo; con 4 → 3 no lo es, y ahí la regla se
    nota: por el centro sale la partición simétrica {0} {1,2} {3}, y con el
    `raw * n_bands // sampled` de toda la vida saldría {0,1} {2} {3}, que mueve
    la frontera baja del tercio a la mitad y deja el extremo menos parecido
    pesando el doble que el más parecido sin que nadie lo haya decidido.

    Devuelve `None` para una banda que el muestreo no declara, y lo dice en vez
    de dejarlo a la aritmética. Con ESTA fórmula un `stratum = -1` —el valor por
    defecto de una fila que reventó— cae fuera de la escala él solo y acabaría
    descartado igual; con una fórmula distinta, o con un `n_bands` distinto, no
    tiene por qué, y una celda fallida colándose en la banda baja es un sesgo en
    un extremo de la variable independiente que nadie vería. El contrato es
    «fuera del muestreo declarado no hay banda», no «la división redondea a mi
    favor».

    Pedir MÁS bandas de las que se muestrearon revienta: eso no es agrupar, es
    inventar puntos que nadie sembró, y saldría una curva con huecos que se
    leerían como tramos sin datos.

    **Quien cuente celdas de un PLAN por nivel tiene que plegar igual.** El
    análisis pliega las filas aquí, pero `run_phase1d.design_power` cuenta las
    celdas del plan mirando `celda["band"] == nivel` antes de que exista una sola
    fila: contra un `levels` de (0, 1) eso da 288 por nivel en vez de 576, y la
    potencia que sale —y con ella la puerta que decide si la tirada arranca— es
    la de media tanda. No es un fallo que se vea: los dos números son
    plausibles.
    """
    if n_bands > sampled:
        raise ValueError(
            f"no se puede analizar en {n_bands} bandas una tanda que muestreó "
            f"{sampled}: agrupar es juntar bandas contiguas, no partirlas, y la "
            "curva saldría con huecos que se leen como tramos sin filas"
        )
    if not 0 <= raw < sampled:
        return None
    # `(2*raw + 1) / (2*sampled)` es el centro de la banda en fracción de rango.
    return (2 * raw + 1) * n_bands // (2 * sampled)


def _check_band_sampling(
    rows: list[dict], axis: str, fold_from: int | None
) -> None:
    """Las filas tienen que venir del muestreo desde el que se pliega.

    `fold_band` devuelve `None` para una banda que el muestreo declarado no
    tiene, y los dos sitios que pliegan descartan esos `None` **en silencio**,
    porque ahí un `None` significa «fila fallida, fuera de la escala». Las dos
    cosas juntas hacen que plegar desde el número equivocado no se note: si el
    runner muestreó 6 bandas y el análisis pliega desde 4, las bandas 4 y 5 —384
    de las 1.152 conversaciones pagadas— salen del contraste primario sin
    excepción y sin bandera, y queda una curva de dos puntos perfectamente
    formada sobre dos tercios de la tanda. La guarda de `primary_family_report`
    tampoco salta: las bandas bajas sí miden.

    El descarte es invisible porque nadie mira el dato que la fila trae: cada
    fila de la Fase 1d declara su muestreo en `n_strata`. Aquí se mira, y un
    desacuerdo revienta en vez de encoger el denominador.

    **Revienta, no repliega desde lo que digan las filas.** Plegar 6 → 2 saldría
    bien —las bandas son tramos contiguos de rango y la mitad baja es la mitad
    baja se muestree en 4 o en 6—, pero entonces `fold_from` dejaría de ser algo
    declarado antes de ver los datos y pasaría a ser lo que traiga el fichero que
    se abra: dos tandas distintas se contrastarían bajo el mismo nombre sin que
    el informe lo dijera. El runner ya se niega por lo mismo a reanudar un JSONL
    con otro número de bandas en la cabecera.

    Solo mira las filas que de verdad contarían —con categoría y con el eje
    puesto—: una fila sin veredicto no entra en ninguna curva, y hacerla
    reventar aquí convertiría una celda fallida en una tanda que no se puede
    analizar.
    """
    if fold_from is None or axis != BAND_AXIS:
        return
    for row in rows:
        if row.get(CATEGORY_FIELD) is None or row.get(axis) is None:
            continue
        declaradas = row.get(BAND_COUNT_FIELD)
        if declaradas is None or int(declaradas) == fold_from:
            continue
        raise ValueError(
            f"estas filas declaran haber muestreado {int(declaradas)} bandas "
            f"(`{BAND_COUNT_FIELD}`) y el contraste pliega desde {fold_from}: "
            "las bandas que sobran no dan error al plegar, se descartan, y el "
            "contraste saldría bien formado sobre una fracción de las filas que "
            "se pagaron. Si el runner cambió de muestreo, `curve.BANDS` y el "
            "`fold_from` de la familia tienen que cambiar con él."
        )

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
        #
        # Y va sobre DOS bandas, no sobre las cuatro del muestreo: `fold_from`
        # agrupa las cuatro de la fila de dos en dos. Sin él, un `levels` de
        # (0, 1) no agruparía nada — se quedaría con las bandas 0 y 1 y tiraría
        # las otras 576 filas, que es el mismo modo de fallo de siempre con otra
        # cara: un contraste impecable sobre media tanda.
        "H4 contempla el error": Hypothesis(
            ENTERTAINS_ERROR,
            axis=BAND_AXIS,
            levels=PRIMARY_BAND_LEVELS,
            fold_from=BANDS,
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

# La tasa base con la que se calcula la potencia de la Fase 1d: **la de Opus**,
# 16/29 = 0,552, medida en el brazo N1 de la Fase 1a con el juez validado.
#
# No la agregada de los tres modelos (19/93 = 0,204), que es la tentación
# obvia —está publicada, es la que cita el plan para N1— y que aquí no vale por
# dos razones distintas, las dos suficientes:
#
# 1. **No es la tasa de ningún modelo.** Los tres dieron 16/29, 3/32 y 0/32, y
#    la tanda que se paga corre SOLO Opus precisamente porque los otros dos no
#    producen G. Una potencia calculada sobre una media que nadie va a producir
#    no describe ninguna tirada.
# 2. **No es conservadora, es al revés.** La varianza binomial `p(1-p)` es
#    máxima en 0,5, así que una tasa base más lejos del medio hace la prueba
#    MÁS sensible sobre el papel: con 0,204 el mismo diseño de dos bandas sale
#    con potencia 0,97 en vez de 0,87. Equivocarse aquí no infla un decimal,
#    autoriza a leer un nulo que no se podía leer.
PHASE1D_BASE_RATE = 16 / 29


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
    diseño habría visto la presencia. El reparto primario es por modelo
    (`trend_report(by="model_id")`); `pooled` no manda.

    Y es lo que decidió el eje de la Fase 1d, no un adorno del informe. Las
    1.152 celdas de la tanda —un solo modelo, porque Opus es el único que
    produce G en N1— dan 288 observaciones por punto repartidas en las cuatro
    bandas del muestreo y 576 plegadas a dos. A la tasa base de Opus (0,552) y
    con la caída declarada de 9 puntos, eso es potencia 0,629 contra **0,867**:
    la misma tanda, el mismo dinero y una diferencia de 24 puntos de potencia
    por elegir la resolución del contraste. Ver `PRIMARY_BANDS`.

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
# composición que mueve 2 ya es la cuarta parte del listón.
#
# **Y con el muestreo por bandas el desequilibrio no es la cuarta parte del
# listón: es más grande que el listón entero, y apunta hacia donde apunta H4.**
# Lo que había escrito aquí estaba reconstruido sobre el diseño descartado —doce
# puestos de `sweep_index(p, 44)`— y decía 2,2 puntos agregados hacia ARRIBA, o
# sea en contra de la hipótesis, que es el caso cómodo. Rehecha la cuenta sobre
# el eje que esta tanda sí muestrea:
#
# - Composición por banda: cada (prefijo, banda) agota su ventana entera (18 de
#   18), así que la mezcla de una banda es exactamente qué artefactos caen en su
#   tramo del ranking. Sobre los 16 rankings N1 de la Fase 1a, plegado al
#   contraste primario de dos bandas: mitad baja {`responde` 119, `cortado` 112,
#   `presupone` 66, `dirigido` 55} contra mitad alta {`dirigido` 121,
#   `presupone` 110, `cortado` 64, `responde` 57}. **chi² = 70,7 con 3 gl**
#   (99,9 con 9 gl sobre las cuatro bandas del muestreo). Las señales se ordenan
#   por coseno y las bandas lo heredan entero.
# - Magnitud propagada con las tasas de G **de Opus** en N1 (`responde` 8/8,
#   `cortado` 3/6, `dirigido` 2/4, `presupone` 3/11): la composición SOLA predice
#   0,626 en la banda baja y 0,510 en la alta, o sea **11,6 puntos de BAJADA**.
#   Con las tasas agregadas de los tres modelos serían 3,5 puntos, también hacia
#   abajo.
#
# Los 11,6 puntos son más que los 9 de `DECLARED_DROP` y van en la MISMA
# dirección que predice H4. En el diseño viejo la deriva enmascaraba el
# resultado; aquí lo fabrica. Por eso `signal_confound` cuelga de todo
# `trend_report` y por eso el eje que mira tiene que ser la banda: una H4
# significativa sin este control al lado no distingue «G baja con el parecido»
# de «al ranking le tocan arriba las señales con las que Opus duda menos».
#
# Dos avisos sobre esa cuenta, porque el número tiene que poder criticarse:
# (1) el ranking se reconstruye sobre el banco de **44** —el que llevaban las
# filas de la Fase 1a—, no sobre los 72 de hoy, porque rankear los 72 pide
# embeddings y el análisis no llama a la red; los 28 nuevos son 7 por señal, así
# que el banco sigue equilibrado en el margen, pero dónde cae cada uno está sin
# medir. (2) las tasas por señal de Opus salen de 29 filas repartidas 6/4/11/8,
# así que la MAGNITUD es aproximada; el SIGNO no depende de ella, porque viene de
# que `responde` y `cortado` —las señales con las que Opus más duda— son las de
# menor coseno medio, y eso se midió sobre las 96 filas N1.
#
# El umbral no es de validez: por debajo se mide y se reporta igual, solo deja de
# marcarse `confounded`.
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
    fold_from: int | None = None,
) -> list[PositionRate]:
    """Tasa de pertenencia a `member` en cada nivel del eje ordenado.

    Devuelve **una entrada por nivel**, incluidas las vacías: un hueco en la
    curva es un dato sobre la tanda, y borrarlo dibuja otra curva.

    `axis` y `levels` existen porque la Fase 1d contrasta el mismo conjunto
    sobre dos ejes: la banda del ranking (`stratum`, H4) y la longitud de la
    conversación (`n_turns`, con los dos niveles 2 y 10, H5). El eje cableado
    dejaba a H5 sin poder correrse, y una prueba que no se corre no entra en
    Holm.

    `position` lleva el VALOR del nivel, no su índice: la curva de H5 se lee
    «2 turnos / 10 turnos». Usarlo también como puntuación de Cochran-Armitage
    es inocuo porque la z es invariante a transformaciones afines de la
    puntuación, y con dos niveles la prueba es la z de dos proporciones se
    escriba (0, 1) o (2, 10).

    `fold_from` agrupa: la fila trae una de `fold_from` bandas del muestreo y se
    contrasta sobre las `len(escala)` del análisis (ver `fold_band`). Es lo que
    hace que el primario de dos bandas de la Fase 1d cuente las 1.152 filas en
    vez de quedarse con las dos bandas bajas y tirar la otra mitad.

    Solo cuentan las filas con categoría. Una fila sin veredicto no es una
    respuesta que no hizo lo que se mide: es una respuesta que no se clasificó.
    """
    _check_band_sampling(rows, axis, fold_from)
    escala = tuple(range(positions)) if levels is None else tuple(levels)
    indice = {nivel: i for i, nivel in enumerate(escala)}
    n = [0] * len(escala)
    k = [0] * len(escala)
    for row in rows:
        cat = row.get(CATEGORY_FIELD)
        bruto = row.get(axis)
        if cat is None or bruto is None:
            continue
        nivel = int(bruto)
        if fold_from is not None:
            plegado = fold_band(nivel, len(escala), fold_from)
            if plegado is None:
                continue
            nivel = plegado
        i = indice.get(nivel)
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
    fold_from: int | None = None,
) -> dict:
    tasas = rate_by_position(
        rows, member, positions, axis=axis, levels=levels, fold_from=fold_from
    )
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
    fold_from: int | None = None,
) -> dict:
    """Curva y prueba de tendencia, por grupo y agregada.

    `by="model_id"` es el reparto **primario** del plan. En la Fase 1b lo era
    por una razón que ya no aplica: dentro de un modelo las 96 celdas eran 96
    estímulos distintos y agregando los tres el mismo (prefijo, artefacto)
    aparecía repetido. El muestreo de la Fase 1d da un modelo por estímulo, así
    que ahí `pooled` no está degradado — y sigue sin mandar, porque los planes lo
    declaran secundario y cambiar el estadístico primario después de ver cuál da
    más potencia es elegir el análisis por su resultado.

    Los `p` que salen de aquí están **sin corregir**, y no pueden estar de otra
    manera: esto es una hipótesis de las tres, o sea un tercio de la familia, y
    Holm necesita verla entera. Quien contrasta las tres hipótesis del plan llama
    a `primary_family_report`. Lo que este informe sí lleva es `n_tests`, el
    número de pruebas que acaba de correr, para que nadie lea tres pruebas al
    5 % como si fueran una.
    """
    # Los niveles de VERDAD, que con `levels` no son `positions`. El informe
    # declaraba 12 puntos para la curva de dos de H5 —y habría declarado 12 para
    # las bandas de H4— porque copiaba el argumento en vez de la escala que se
    # acaba de usar. Un `positions` que no es el largo de `curve` es la misma
    # clase de mentira que este módulo está corrigiendo, en pequeño.
    niveles = _niveles(positions, levels)
    informe: dict = {
        "member": sorted(member),
        "positions": len(niveles),
        # El eje viaja en el informe porque dos hipótesis de la misma familia
        # pueden medir el MISMO conjunto sobre ejes distintos (H4 y H5 de la
        # Fase 1d): sin esto, las dos curvas se leen como la misma repetida.
        "axis": axis,
        "levels": niveles,
        # Cuántas bandas traía la fila cuando el contraste declara menos. Viaja
        # porque un informe de dos puntos sobre una tanda de cuatro bandas y uno
        # sobre una tanda de dos bandas se leen igual, y no son lo mismo.
        "folded_from": fold_from,
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # El que manda para juzgar ESTA curva: el binario de su conjunto.
        "replicate_agreement_member": replicate_agreement(member),
        "pooled": _curva(
            rows, member, positions, axis=axis, levels=levels, fold_from=fold_from
        ),
        # La curva NO se puede leer sola cuando el pegote lleva señal dentro:
        # ver `signal_confound`. Viaja dentro del informe y no en un guion
        # aparte porque esto es lo que lee quien contrasta la hipótesis, y un
        # control que hay que acordarse de correr es un control que no se corre.
        # En el brazo neutro sale `None` —N0 no tiene señales— y por eso el
        # informe de la Fase 1b no se mueve ni un decimal.
        "signal_confound": signal_confound(
            rows, member, positions, axis=axis, levels=levels, fold_from=fold_from
        ),
    }
    if by:
        grupos: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            grupos[str(row.get(by))].append(row)
        informe["by"] = {
            g: _curva(
                v, member, positions, axis=axis, levels=levels, fold_from=fold_from
            )
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
    constantes: con la familia de la Fase 1b y sus tres modelos son nueve
    pruebas, y con la de la 1d —dos hipótesis sobre el único modelo que la
    tanda corre— son dos. El multiplicador que paga el `p` más pequeño cambia
    con ello, y con él el alfa que la potencia tiene que batir.

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
            fold_from=h.fold_from,
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
    # Cuántas bandas contrasta esta familia, para quien lea el JSON sin el plan
    # al lado. Sale de las hipótesis declaradas y no de una constante repetida:
    # si alguien mueve `PRIMARY_BANDS` y se olvida de la familia, el informe
    # dice lo que se contrastó, no lo que la constante querría.
    bandas = sorted(
        {
            len(inf["levels"])
            for nombre, inf in informes.items()
            if inf["axis"] == BAND_AXIS
        }
    )
    if len(bandas) > 1:
        raise ValueError(
            f"la familia contrasta la banda con dos resoluciones a la vez "
            f"({bandas}): el informe no puede declarar cuál es la primaria, y "
            "un lector tomaría por primaria la que le convenga"
        )

    return {
        "alpha": alpha,
        "family_size": len(crudos),
        "correction": "holm-bonferroni",
        # `None` cuando la familia no mide bandas —la Fase 1b barrió puestos—:
        # inventar aquí un «4» de la constante sería declarar un eje que esa
        # tanda no tuvo.
        "bands": (
            None
            if not bandas
            else {
                "primary": bandas[0],
                "sampled": BANDS,
                "secondary": list(SECONDARY_BAND_SPLITS),
            }
        ),
        # La condición para que un resultado plano signifique algo: que TODAS las
        # pruebas de la familia habrían visto la caída declarada. Con una sola
        # que no, el nulo de la familia es silencio y no ausencia — y el segundo
        # brazo de la puerta del Paso 8 («la conducta no depende del parecido ni
        # de la longitud») no se puede escribir. Va en el informe y no en la
        # prosa del resultado porque en la prosa nadie lo cuenta, y en la Fase 1d
        # cuenta por tres milésimas: el contraste primario de dos bandas llega a
        # 0,867 al alfa suelto, pero el `p` más pequeño de una familia de dos
        # paga x2, y al 0,025 de Holm la misma tanda se queda en **0,797** contra
        # un `MIN_POWER` de 0,80. Un número que roza el umbral es justo el que se
        # acaba dando por bueno de memoria.
        "null_is_informative": informativo,
        "declared_drop": DECLARED_DROP,
        "min_power": MIN_POWER,
        "replicate_agreement_n0": REPLICATE_AGREEMENT_N0,
        # Aquí no hay UN conjunto sino uno por hipótesis declarada, y cada uno
        # tiene su propio suelo binario: va dentro de su informe, en
        # `replicate_agreement_member`.
        "hypotheses": informes,
    }


def secondary_band_reports(
    rows: list[dict],
    member: frozenset[str] | set[str] = ENTERTAINS_ERROR,
    by: str | None = "model_id",
    splits: tuple[int, ...] = SECONDARY_BAND_SPLITS,
    sampled: int = BANDS,
) -> dict[int, dict]:
    """Las particiones del eje que se REPORTAN y no deciden: 3 y 4 bandas.

    El primario contrasta dos bandas porque es lo que tiene potencia, y con eso
    la pregunta de la fase queda contestada o no contestada. Pero dos puntos no
    enseñan la forma del eje, y la forma es lo que hace falta para diseñar la
    Fase 2: si la caída está toda entre la banda 2 y la 3, no es lo mismo que si
    baja parejo.

    Van en una función aparte y con `primary: False` dentro porque la diferencia
    entre describir y contrastar no se sostiene sola: tres informes al lado del
    primario, con su `p` y su `z`, se leen como tres contrastes en cuanto uno
    salga bonito. Y si entraran en la familia, el `p` del primario pasaría de
    pagar x2 a pagar x8 por pruebas que no cierran ninguna puerta.

    La partición de 3 **no reparte igual** —4 bandas muestreadas no se parten en
    tres tercios— y por eso sus puntos salen 288/576/288 en vez de 384/384/384.
    No se corrige ni se disimula: el `n` viaja en cada punto de la curva, que es
    lo único que impide leerla como tres tramos comparables.
    """
    return {
        n_bandas: trend_report(
            rows,
            set(member),
            by=by,
            axis=BAND_AXIS,
            levels=tuple(range(n_bandas)),
            fold_from=sampled,
        )
        | {"primary": False, "bands": n_bandas}
        for n_bandas in splits
    }


def by_kind(
    rows: list[dict],
    member: set[str],
    min_span: float = MIN_KIND_SPAN,
    positions: int = SWEEP_POSITIONS,
    axis: str = "sweep_position",
    levels: tuple[int, ...] | None = None,
    fold_from: int | None = None,
) -> dict:
    """La mitad medible de D15: ¿se mueve la tasa DENTRO de un mismo género?

    Registro y similaridad son colineales por construcción (D15) y este diseño no
    los separa. Lo que sí puede hacer es mirar dentro de cada género: si en
    `job_ad`, que recorre 0,3 de coseno, la tasa sigue moviéndose, parte del
    efecto es de la similaridad y no solo del registro. Los géneros que no
    recorren eje se listan aparte **con su rango**, para que se vea por qué no
    dicen nada.

    `axis`/`levels`/`fold_from` son los mismos de `trend_report` y hay que
    pasarlos: con los valores por defecto esto mide `sweep_position`, y sobre
    una tanda de la Fase 1d —donde ese campo es `None` en las 1.152 filas— cada
    género saldría con la curva a ceros y un `p` de 1,0. Un informe por género
    bien formado y calculado sobre nada es justo el fallo que esta fase viene
    arrastrando; aquí no puede reventar como en la familia primaria, porque
    esto es exploratorio y no autoriza a concluir, pero tiene que poder mirar el
    eje bueno.
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
            anchos[kind] = base | _curva(
                filas, member, positions, axis=axis, levels=levels, fold_from=fold_from
            )
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
    fold_from: int | None = None,
) -> dict | None:
    """Cuánto de la curva es el eje y cuánto es QUÉ SEÑAL toca en cada posición.

    Es la otra mitad de lo que avisa D15, y la que este diseño no controlaba.
    `by_kind` mira el GÉNERO del pegote (`artifact_kind`); aquí se mira lo que en
    el brazo N1 decide la conducta: la **señal** que delata al pegote (`cortado`,
    `dirigido`, `presupone`, `responde`).

    El problema no es que las señales se repartan mal por azar. Con el muestreo
    por bandas es todavía más directo que con el barrido de puestos: cada
    (prefijo, banda) se lleva la ventana ENTERA de su tramo del ranking —18 de
    18—, así que la composición de una banda no es una muestra de nada, es
    exactamente qué artefactos caen en ese tramo. Y el ranking de N1 ordena por
    parecido pegotes que llevan la señal dentro: `responde` y `cortado` son las
    de menor coseno medio y se van abajo, `dirigido` y `presupone` arriba.

    Reconstruido sobre los 16 rankings N1 de la Fase 1a y plegado al contraste
    primario de dos bandas: mitad baja {responde 119, cortado 112, presupone 66,
    dirigido 55} contra mitad alta {dirigido 121, presupone 110, cortado 64,
    responde 57}, chi² = 70,7 con 3 gl. Con las tasas de G por señal de Opus en
    N1, esa mezcla cambiando de sitio mueve la tasa esperada **11,6 puntos hacia
    abajo** sin que la banda haga absolutamente nada — más que los 9 puntos que
    la puerta declara relevantes, y en la misma dirección que predice H4. Ver
    `MAX_COMPOSITION_SPAN` para la cuenta y sus dos avisos.

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
    _check_band_sampling(rows, axis, fold_from)
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
        nivel = int(bruto)
        if fold_from is not None:
            plegado = fold_band(nivel, len(escala), fold_from)
            if plegado is None:
                continue
            nivel = plegado
        i = indice.get(nivel)
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

    observada = rate_by_position(
        rows, member, positions, axis=axis, levels=levels, fold_from=fold_from
    )
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
        } | _curva(
            filas_s, member, positions, axis=axis, levels=levels, fold_from=fold_from
        )

    return {
        "axis": axis,
        "folded_from": fold_from,
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
    fold_from: int | None = None,
) -> dict:
    """La pregunta descriptiva del plan: ¿qué señal aguanta mejor el parecido?

    Es `signal_confound` más la tabla de dos columnas —mitad baja del eje contra
    mitad alta— que pide el plan, y **cada celda lleva su `n`**. Sin el `n` la
    tabla se lee como «la señal X aguanta el parecido y la Y no» cuando lo que
    puede estar diciendo es que el ranking manda abajo los pegotes que se delatan
    por estar cortados o por responder a otra cosa, y arriba los que se delatan
    por dirigirse a otro o por presuponer lo que no toca: el plan la planteaba
    suponiendo «~34 observaciones por señal y mitad del eje», y en el plan real
    por bandas `dirigido` pone 55 en la mitad baja contra 121 en la alta, y
    `responde` 119 contra 57. Las cuatro señales no ven el mismo eje, y el
    denominador es lo único que lo enseña.

    Sobre una tanda de la Fase 1d hay que pasarle el eje: `axis="stratum"`,
    `levels=PRIMARY_BAND_LEVELS` y `fold_from=BANDS`, que es lo que declara H4.
    Con los valores por defecto mira `sweep_position`, que en esas filas es
    `None`, y devuelve un informe vacío que se lee igual que «aquí no hay señal
    que controlar».
    """
    informe = signal_confound(
        rows, member, positions, axis=axis, levels=levels, fold_from=fold_from
    )
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
