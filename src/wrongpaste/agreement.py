"""Muestreo a ciegas y acuerdo entre etiquetadores (spec §6)."""

from dataclasses import fields

import numpy as np

from wrongpaste.judging import Verdict, cohen_kappa, raw_agreement

# Los campos REALES del veredicto, leídos del propio dataclass.
#
# Aquí había una lista escrita a mano —`judge_category`, `judge_quote`,
# `judge_confidence`, `verdicts`— y no cegaba nada: una fila unida con
# `asdict(verdict)` trae las claves `category`, `quote`, `confidence`,
# `rubric_version`, `judge_model` y `raw`, y ninguna de esas cuatro casaba con
# ninguna de ellas. La categoría del juez salía en la muestra "a ciegas".
#
# Por eso se leen del dataclass: una lista a mano se desincroniza en cuanto
# alguien renombra o añade un campo de `Verdict`, y el fallo es SILENCIOSO —no
# hay excepción ni columna de más, solo un etiquetador que ve la respuesta
# antes de dar la suya y un acuerdo del §6 que ya no mide nada—. Leerlos de
# `fields(Verdict)` hace que el renombrado viaje solo.
#
# No es hipotético: `Verdict` ganó `status` y `error` después de escribirse la
# lista a mano, y la lista ni se enteró. Con los campos reales entran solos.
#
# El precio de leerlos del dataclass es que `status` se llama igual en
# `ConversationRecord`, así que la fila unida pierde también el suyo. Se acepta
# a propósito: al etiquetador le hace falta la transcripción y la reacción, no
# el estado de la celda, y equivocarse por el lado de esconder de más cuesta
# una columna; por el otro cuesta el §6 entero.
CAMPOS_VERDICT: tuple[str, ...] = tuple(f.name for f in fields(Verdict))

# Nombre del campo de `Verdict` que guarda la categoría. Se comprueba contra
# los campos reales para que un renombrado reviente al importar, en vez de
# dejar el tercer eje del muestreo (§6.2) leyendo una clave que ya no existe.
CLAVE_CATEGORIA = "category"
if CLAVE_CATEGORIA not in CAMPOS_VERDICT:
    raise RuntimeError(
        f"Verdict ya no tiene el campo {CLAVE_CATEGORIA!r} (tiene "
        f"{CAMPOS_VERDICT}): actualiza CLAVE_CATEGORIA, o el muestreo del §6.2 "
        "deja de estratificar por categoría del juez sin avisar"
    )

# Además de los campos reales, los nombres con los que una fila unida puede
# arrastrar el veredicto sin llamarse como el dataclass: prefijados por el
# juez, anidados en singular o en plural. La regla de `_es_del_juez` los cubre
# por prefijo/subcadena; esta lista deja escritos los que ya sabemos que
# existen para que se lean de un vistazo.
OCULTO: tuple[str, ...] = CAMPOS_VERDICT + (
    "judge_category", "judge_quote", "judge_confidence", "verdict", "verdicts",
)


def _es_del_juez(clave: str) -> bool:
    """¿Puede esta clave delatar el veredicto del juez?

    Tres reglas, de más exacta a más amplia: los campos reales de `Verdict`
    (más los alias conocidos de `OCULTO`), cualquier clave que empiece por
    `judge` y cualquiera que contenga `verdict`. Las dos últimas no son celo:
    `Verdict.raw` guarda el JSON entero del juez —con su categoría dentro—,
    así que una fila unida como `judge_raw` o `verdict` filtra el veredicto
    completo sin que ninguna lista exacta se entere.

    Fallar cerrado cuesta, como mucho, esconder un campo de más a quien no lo
    necesita. Fallar abierto cuesta el §6 entero.
    """
    c = str(clave).lower()
    return c in OCULTO or c.startswith("judge") or "verdict" in c


def _categoria_del_juez(fila: dict) -> str | None:
    """Categoría que puso el juez, leída ANTES de cegar la fila.

    Es el tercer eje del muestreo (§6.2) y el que de verdad importa: G decide
    la puerta y se espera casi vacía, así que un muestreo ciego a la categoría
    puede no llevar ni una sola G y dejar sin comprobar justo la categoría de
    la que depende la decisión.

    La fila puede traerla prefijada (`judge_category`) o con el nombre del
    campo de `Verdict`, si se unió con `asdict(verdict)`. Si trae varios
    veredictos anidados vale el del primer juez: el estrato solo tiene que
    repartir la muestra, no arbitrar entre jueces.
    """
    for clave in ("judge_category", CLAVE_CATEGORIA):
        valor = fila.get(clave)
        if valor is not None:
            return str(valor)
    for clave in ("verdict", "verdicts"):
        valor = fila.get(clave)
        if isinstance(valor, dict):
            valor = [valor]
        if isinstance(valor, list):
            for v in valor:
                if isinstance(v, dict) and v.get(CLAVE_CATEGORIA) is not None:
                    return str(v[CLAVE_CATEGORIA])
    return None


# Los ejes de estratificación por defecto: los del §6.2 del spec de la Fase 1.
# El nivel del pegote era el factor de la Fase 1a; en tandas de un solo brazo
# —la 1b, que solo corre N0— ese eje vale lo mismo para todas las filas y no
# reparte nada, así que se sustituye por el que sí manda allí.
DEFAULT_STRATA_KEYS: tuple[str, ...] = ("paste_level", "model_id")


def blind_sample(
    rows: list[dict],
    n: int,
    seed: int,
    strata_keys: tuple[str, ...] = DEFAULT_STRATA_KEYS,
) -> list[dict]:
    """Muestra estratificada por `strata_keys` + categoría del juez, a ciegas.

    `strata_keys` son los ejes del diseño; la categoría del juez se añade
    siempre como último eje y **no es opcional**, porque es lo que garantiza que
    las categorías raras entren en la muestra en vez de quedarse fuera por
    sorteo. En la Fase 1a los ejes fueron `("paste_level", "model_id")`; en la
    1b, `("sweep_position", "model_id")`, porque allí el nivel es constante y
    la posición del barrido es la variable independiente.

    La categoría del juez se lee con
    `_categoria_del_juez` **antes** de cegar la fila y no vuelve a aparecer en
    la salida: se usa para repartir, no para enseñárselo a nadie.

    Los estratos se recorren en rueda empezando por los más pequeños. Cuando
    `n` no llega a cubrir un estrato por vuelta, el que se queda fuera con un
    orden alfabético es siempre el raro —G, precisamente—, y el raro es el que
    hay que mirar. Con el tamaño inicial como criterio, el estrato escaso entra
    en la primera vuelta. El orden se calcula una sola vez, con los tamaños de
    partida, para que la muestra siga siendo determinista dada la semilla.

    A ciegas significa a ciegas: si el etiquetador ve la categoría del juez, el
    acuerdo que salga no mide nada.
    """
    rng = np.random.default_rng(seed)
    estratos: dict[tuple, list[dict]] = {}
    for r in rows:
        eje = (*(r.get(k) for k in strata_keys), _categoria_del_juez(r))
        estratos.setdefault(eje, []).append(r)
    claves = sorted(
        estratos,
        key=lambda k: (len(estratos[k]), tuple(str(x) for x in k)),
    )
    elegidas: list[dict] = []
    i = 0
    while len(elegidas) < n and any(estratos[k] for k in claves):
        k = claves[i % len(claves)]
        i += 1
        if not estratos[k]:
            continue
        idx = int(rng.integers(0, len(estratos[k])))
        elegidas.append(estratos[k].pop(idx))
    return [{k: v for k, v in r.items() if not _es_del_juez(k)} for r in elegidas]


def agreement_report(human: dict[str, str], judges: dict[str, dict[str, str]]) -> dict:
    """Acuerdo bruto y kappa de cada juez contra las etiquetas humanas."""
    informe: dict = {}
    n_comun = None
    for nombre, etiquetas in judges.items():
        comunes = sorted(set(human) & set(etiquetas))
        if not comunes:
            raise ValueError(f"{nombre}: ningún id en común con las etiquetas humanas")
        a = [human[c] for c in comunes]
        b = [etiquetas[c] for c in comunes]
        informe[nombre] = {"raw": raw_agreement(a, b), "kappa": cohen_kappa(a, b),
                           "n": len(comunes)}
        n_comun = len(comunes) if n_comun is None else min(n_comun, len(comunes))
    informe["n"] = n_comun
    return informe
