"""Clasificación de las reacciones con la rúbrica v2, por dos jueces.

Dos jueces, no uno: una categoría mal definida se delata en el desacuerdo
entre ellos mucho antes que en una muestra humana pequeña. Los dos están
FUERA del plantel evaluado, para que nadie se puntúe a sí mismo.

**Los fallos del juez son datos, no caída (D6).** La Restricción Global del
plan —«ninguna celda se descarta en silencio; los fallos se escriben como
filas con `status`»— vale también aquí: son 576 llamadas de juez, y una
respuesta vacía o un JSON roto en la número 300 no puede matar la tanda ni,
peor, perder el caso. Por eso `judge_one` devuelve un `Verdict` con
`category=None` y el motivo en `status`, y solo lanza si el llamador pide
`strict=True`. Los agregados filtran con `is_usable`.
"""

import json
from collections import Counter
from dataclasses import dataclass

from wrongpaste.clients import chat
from wrongpaste.rubric import CATEGORY_IDS, RUBRIC_VERSION, rubric_prompt

# Spec §6 y §9: fuera del plantel, y de familias distintas para que el
# acuerdo entre ellos signifique algo.
JUDGES: tuple[str, ...] = ("gpt-5.5-tst", "gemini-2.5-flash")

# El mismo presupuesto que una respuesta evaluada, y por el mismo motivo: los
# dos jueces razonan, y el razonamiento **gasta de aquí**. Medido en vivo sobre
# una clasificación real de `gemini-2.5-flash`: 724 tokens de razonamiento para
# 69 de texto visible. Con 1.200, una cita algo larga dejaba el JSON cortado a
# media cadena en el 43 % de sus veredictos, y el corte se leía como «el juez no
# devuelve JSON» cuando el JSON era correcto y lo que faltaba era sitio.
JUDGE_MAX_TOKENS = 4000

# Vocabulario cerrado de `Verdict.status` (D6). `ok` es el único veredicto
# que se puede agregar; los otros son la fila que documenta por qué no hay
# categoría para esa celda.
#
# `truncated` existe separado de `bad_json` porque las dos se arreglan en
# sitios distintos: un JSON mal formado es cosa del prompt o del juez, y un
# JSON cortado es cosa de `JUDGE_MAX_TOKENS`. Juntarlas manda a arreglar lo
# que no está roto.
OK = "ok"
VERDICT_STATUSES: frozenset[str] = frozenset(
    {OK, "bad_json", "bad_category", "http_error", "empty", "truncated"}
)

# Lo que dice cada proveedor cuando se acabó el presupuesto de salida.
LENGTH_STOPS: frozenset[str] = frozenset({"length", "max_tokens", "MAX_TOKENS"})

# El detalle del error se recorta: es para leerlo en el CSV, no para guardar
# una traza entera por fila.
ERROR_CHARS = 500


@dataclass(frozen=True)
class Verdict:
    """Lo que dictamina un juez sobre UNA reacción.

    `category` es `None` exactamente cuando `status` no es `ok`: así ningún
    agregado puede confundir un fallo con una categoría, y `raw` conserva lo
    que llegó del juez aunque no se pudiera interpretar.
    """

    category: str | None
    quote: str
    confidence: float | None
    judge_model: str
    rubric_version: str = RUBRIC_VERSION
    raw: str = ""
    status: str = OK
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in VERDICT_STATUSES:
            raise ValueError(
                f"status inválido: {self.status!r}; esperaba {sorted(VERDICT_STATUSES)}"
            )
        if self.status == OK:
            if self.category not in CATEGORY_IDS:
                raise ValueError(f"categoría inválida del juez: {self.category!r}")
        elif self.category is not None:
            raise ValueError(
                f"un veredicto con status {self.status!r} no puede traer categoría "
                f"({self.category!r})"
            )


def is_usable(verdict: Verdict) -> bool:
    """¿Se puede meter este veredicto en un agregado?

    Solo los `ok`. Se expone como función —y no como comparación suelta en
    cada llamador— para que añadir un status nuevo no obligue a repasar todos
    los sitios donde se filtra.
    """
    return verdict.status == OK and verdict.category in CATEGORY_IDS


def _fallo(judge_model: str, status: str, raw: str, error: str) -> Verdict:
    return Verdict(
        category=None,
        quote="",
        confidence=None,
        judge_model=judge_model,
        rubric_version=RUBRIC_VERSION,
        raw=raw,
        status=status,
        error=error[:ERROR_CHARS],
    )


def judge_one(judge_model: str, reaction: str, paste_text: str,
              strict: bool = False) -> Verdict:
    """Clasifica una reacción con un juez. Un fallo vuelve como dato.

    Con `strict=True` los fallos se lanzan como `ValueError` (o como la
    excepción de transporte tal cual): es el modo que usan los tests que
    comprueban que una categoría inventada se rechaza. En la tirada real se
    llama SIEMPRE en modo no estricto, porque perder la celda es peor que
    tener una fila con `status`.
    """
    prompt = (
        f"{rubric_prompt()}\n\n"
        f"=== MENSAJE PEGADO ===\n{paste_text}\n\n"
        f"=== RESPUESTA A CLASIFICAR ===\n{reaction}\n"
    )
    try:
        reply = chat(judge_model, [{"role": "user", "content": prompt}],
                     max_tokens=JUDGE_MAX_TOKENS)
    except Exception as exc:  # transporte: cuota, timeout, 5xx del gateway
        if strict:
            raise
        return _fallo(judge_model, "http_error", "",
                      f"{exc.__class__.__name__}: {exc}")

    crudo = reply.text.strip().removeprefix("```json").removesuffix("```").strip()
    if not crudo:
        if strict:
            raise ValueError("el juez devolvió una respuesta vacía")
        return _fallo(judge_model, "empty", reply.text,
                      "el juez devolvió una respuesta vacía")

    try:
        datos = json.loads(crudo)
    except json.JSONDecodeError as exc:
        # El corte se distingue del JSON mal formado por el `stop_reason`, no
        # por la pinta del texto: los dos llegan aquí como JSONDecodeError.
        if reply.stop_reason in LENGTH_STOPS:
            if strict:
                raise ValueError(
                    f"el juez se quedó sin tokens: {crudo[-120:]!r}"
                ) from exc
            return _fallo(judge_model, "truncated", crudo,
                          "el juez agotó JUDGE_MAX_TOKENS antes de cerrar el JSON: "
                          f"{crudo[-120:]!r}")
        if strict:
            raise ValueError(f"el juez no devolvió JSON: {crudo[:200]!r}") from exc
        return _fallo(judge_model, "bad_json", crudo,
                      f"el juez no devolvió JSON: {crudo[:200]!r}")
    if not isinstance(datos, dict):
        if strict:
            raise ValueError(f"el juez no devolvió un objeto JSON: {crudo[:200]!r}")
        return _fallo(judge_model, "bad_json", crudo,
                      f"el juez no devolvió un objeto JSON: {crudo[:200]!r}")

    cat = str(datos.get("category", "")).strip().upper()[:1]
    if cat not in CATEGORY_IDS:
        if strict:
            raise ValueError(f"categoría inválida del juez: {cat!r}")
        return _fallo(judge_model, "bad_category", crudo,
                      f"categoría inválida del juez: {cat!r}")

    return Verdict(cat, str(datos.get("quote", "")),
                   datos.get("confidence"), judge_model, RUBRIC_VERSION, crudo)


def judge_all(reaction: str, paste_text: str) -> list[Verdict]:
    """Un veredicto por juez, pase lo que pase.

    Nunca lanza: si un juez falla, su fallo viaja como `Verdict` con `status`
    y el otro sigue contando. Una celda con un solo juez utilizable se puede
    clasificar; una celda perdida, no.
    """
    return [judge_one(j, reaction, paste_text) for j in JUDGES]


def raw_agreement(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("las dos series tienen que tener el mismo tamaño y no estar vacías")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Kappa de Cohen: acuerdo corregido por azar.

    Dos jueces que contestan siempre lo mismo coinciden al 100 % y no informan
    de nada; kappa vale 0 ahí, que es lo que queremos saber.
    """
    po = raw_agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 0.0
    return (po - pe) / (1 - pe)
