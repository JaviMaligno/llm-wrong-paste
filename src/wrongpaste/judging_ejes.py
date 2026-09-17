"""Juzga una muestra con la rúbrica de dos ejes y la compara con la v2.

Es una **medición sobre una muestra**, no un cambio de instrumento: sirve para
decidir si la Fase 2 cambia de rúbrica. Las tandas ya juzgadas con la v2 se
quedan como están, porque cambiar de rúbrica las haría incomparables.

Lo que hay que contestar con esto, y está todo en `comparar`:

1. **¿Cuántas respuestas ocupan una combinación que la v2 no podía expresar?**
   Esa es la pregunta. Si son pocas, la v2 valía y esto sobra.
2. **¿La casilla `duda` del eje 1 coincide con la G de la v2?** Si coincide, el
   resultado de las Fases 1a y 1d sigue en pie tal cual, porque las dos miden
   lo mismo con otro nombre.
3. **¿Qué se llevaba la v2 a `A` que en realidad reconocía el salto?** Es la
   sospecha concreta sobre la tasa «menciona el salto», que compite sin regla
   de precedencia y por eso puede estar contada por debajo.
"""

from __future__ import annotations

import collections
import json
from dataclasses import asdict, dataclass

from .clients import chat
from .judging import JUDGE_MAX_TOKENS, LENGTH_STOPS
from .rubric import MENTIONS_JUMP
from .rubric_ejes import (
    ACCION_IDS,
    RECONOCIMIENTO_IDS,
    RUBRIC_EJES_VERSION,
    rubric_ejes_prompt,
)

# Un solo juez: esto mide la rúbrica, no el acuerdo entre jueces. Se usa el
# validado en las fases anteriores para que la comparación con la v2 no mezcle
# «cambió la rúbrica» con «cambió el juez».
JUEZ = "gpt-5.5-tst"


@dataclass(frozen=True)
class VerdictoEjes:
    """Lo que el juez dice de una respuesta en los dos ejes.

    `reconocimiento` y `accion` valen `None` exactamente cuando `status` no es
    `ok`, por el mismo motivo que en la v2: que ningún agregado confunda un
    fallo con una casilla.
    """

    reconocimiento: str | None
    accion: str | None
    cita_reconocimiento: str = ""
    cita_accion: str = ""
    judge_model: str = JUEZ
    rubric_version: str = RUBRIC_EJES_VERSION
    raw: str = ""
    status: str = "ok"
    error: str = ""


def _fallo(status: str, raw: str, error: str) -> VerdictoEjes:
    return VerdictoEjes(None, None, raw=raw, status=status, error=error[:500])


def juzgar_ejes(reaction: str, paste_text: str) -> VerdictoEjes:
    """Un veredicto de dos ejes. Un fallo vuelve como dato, nunca lanza."""
    prompt = (
        f"{rubric_ejes_prompt()}\n\n"
        f"=== MENSAJE PEGADO ===\n{paste_text}\n\n"
        f"=== RESPUESTA A CLASIFICAR ===\n{reaction}\n"
    )
    try:
        reply = chat(JUEZ, [{"role": "user", "content": prompt}],
                     max_tokens=JUDGE_MAX_TOKENS)
    except Exception as exc:
        return _fallo("http_error", "", f"{exc.__class__.__name__}: {exc}")

    crudo = reply.text.strip().removeprefix("```json").removesuffix("```").strip()
    if not crudo:
        return _fallo("empty", reply.text, "el juez devolvió una respuesta vacía")
    try:
        datos = json.loads(crudo)
    except json.JSONDecodeError:
        if reply.stop_reason in LENGTH_STOPS:
            return _fallo("truncated", crudo,
                          "el juez agotó JUDGE_MAX_TOKENS antes de cerrar el JSON")
        return _fallo("bad_json", crudo, f"el juez no devolvió JSON: {crudo[:200]!r}")
    if not isinstance(datos, dict):
        return _fallo("bad_json", crudo, f"no devolvió un objeto: {crudo[:200]!r}")

    rec = str(datos.get("reconocimiento", "")).strip().lower()
    acc = str(datos.get("accion", "")).strip().lower()
    if rec not in RECONOCIMIENTO_IDS or acc not in ACCION_IDS:
        return _fallo("bad_category", crudo,
                      f"ejes inválidos: reconocimiento={rec!r} accion={acc!r}")
    return VerdictoEjes(rec, acc,
                        str(datos.get("cita_reconocimiento", "")),
                        str(datos.get("cita_accion", "")),
                        raw=crudo)


# --- la comparación, que es el objeto de todo esto -------------------------

# Lo que la v2 podía expresar: para cada categoría suya, la única combinación de
# ejes que le corresponde. Si una respuesta cae fuera de esta tabla, la v2 NO
# tenía casilla para ella y tuvo que elegir una de las dos verdades.
V2_EQUIVALE: dict[tuple[str, str], str] = {
    ("nada", "ejecuta"): "A",
    ("nada", "pregunta"): "B",
    ("nada", "rol"): "F",
    ("nada", "puente"): "E",
    ("salto", "nada"): "C",
    ("duda", "nada"): "G",
}


def comparar(filas: list[dict]) -> dict:
    """Qué dice la rúbrica de dos ejes sobre las etiquetas de la v2.

    `filas` son dicts con `v2` (la categoría de la v2) y `reconocimiento` /
    `accion` (los dos ejes). Devuelve las tres cifras que deciden si la v3 está
    justificada, más las tablas para mirarlas a ojo.
    """
    usables = [f for f in filas if f.get("reconocimiento") and f.get("accion")]
    n = len(usables)
    if not n:
        return {"n": 0}

    # 1. ¿Cuántas ocupan una combinación que la v2 no podía expresar?
    dobles = [f for f in usables
              if (f["reconocimiento"], f["accion"]) not in V2_EQUIVALE]

    # 2. ¿La casilla `duda` coincide con la G de la v2?
    duda = {f["conversation_id"] for f in usables if f["reconocimiento"] == "duda"}
    ges = {f["conversation_id"] for f in usables if f["v2"] == "G"}

    # 3. ¿Qué se llevaba la v2 a una categoría que NO menciona el salto, aunque
    #    el eje 1 diga que sí lo reconoce?
    reconocen = {"salto", "duda"}
    infracontadas = [f for f in usables
                     if f["reconocimiento"] in reconocen and f["v2"] not in MENTIONS_JUMP]

    return {
        "n": n,
        "combinaciones_que_v2_no_expresa": {
            "n": len(dobles),
            "porcentaje": round(100 * len(dobles) / n, 1),
            "reparto": dict(collections.Counter(
                f"{f['reconocimiento']}+{f['accion']}" for f in dobles).most_common()),
        },
        "duda_frente_a_G": {
            "duda": len(duda), "G": len(ges), "coinciden": len(duda & ges),
            "solo_duda": len(duda - ges), "solo_G": len(ges - duda),
        },
        "menciona_el_salto_infracontado": {
            "n": len(infracontadas),
            "porcentaje": round(100 * len(infracontadas) / n, 1),
            "a_que_categoria_fueron": dict(collections.Counter(
                f["v2"] for f in infracontadas).most_common()),
        },
        "tabla_ejes": dict(collections.Counter(
            f"{f['reconocimiento']}+{f['accion']}" for f in usables).most_common()),
        "v2_por_reconocimiento": {
            r: dict(collections.Counter(
                f["v2"] for f in usables if f["reconocimiento"] == r).most_common())
            for r in sorted({f["reconocimiento"] for f in usables})
        },
    }


def verdict_row(row: dict, v: VerdictoEjes) -> dict:
    return {
        "conversation_id": row.get("conversation_id"),
        "stratum": row.get("stratum"),
        "artifact_signal": row.get("artifact_signal"),
        "v2": row.get("v2"),
        **asdict(v),
    }
