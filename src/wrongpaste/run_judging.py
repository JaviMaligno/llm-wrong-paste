"""Clasifica una tirada con los dos jueces y persiste sus veredictos.

Escribe `verdicts-<run_id>.jsonl` al lado del JSONL de la tirada: **una línea
por (conversación, juez)**, no una por conversación. Los dos veredictos van
separados a propósito, porque el §6 del spec exige reportar el acuerdo entre
jueces: si se resolvieran aquí en una etiqueta única, el desacuerdo —que es el
dato que delata una categoría mal definida— se perdería antes de poder medirlo.
Resolver los dos jueces en una etiqueta es una decisión de análisis y vive en
`rates.py`, no aquí.

Tres cosas que este módulo NO hace, y que son deliberadas:

1. **No juzga filas que no sean `ok`** (`rates.JUDGEABLE_STATUSES`). Una
   `truncated` se cortó por nuestro `max_tokens` y una `empty` no tiene
   reacción: clasificarlas metería una propiedad del arnés en el numerador.
2. **No descarta los fallos del juez.** `judge_all` nunca lanza; un juez que
   falla vuelve como `Verdict` con `status`, y esa línea se escribe igual. Un
   fallo borrado es un denominador que nadie puede auditar.
3. **No vuelve a pagar lo ya pagado.** Al reanudar se saltan los pares
   (conversación, juez) que ya tengan un veredicto **usable**; los que fallaron
   se reintentan, que es justo lo que se quiere de una reanudación. El salto es
   **por juez, no por fila**: si un juez acertó y el otro falló, se llama solo
   al que falló. Pedir los dos y descartar el bueno costaría el doble — pasó en
   la primera reanudación de la Fase 1a, 242 llamadas para 121 veredictos.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .judging import JUDGES, Verdict, is_usable, judge_one
from .rates import JUDGEABLE_STATUSES


def verdicts_path(run_path: Path) -> Path:
    """`runs/phase1a/20260914T135814.jsonl` → `.../verdicts-20260914T135814.jsonl`."""
    return run_path.parent / f"verdicts-{run_path.stem}.jsonl"


def load_rows(run_path: Path) -> list[dict]:
    """Las filas de conversación de la tirada, sin la cabecera (D5)."""
    filas = []
    for linea in run_path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        fila = json.loads(linea)
        if fila.get("kind") == "run_header":
            continue
        filas.append(fila)
    return filas


def judgeable(rows: list[dict]) -> list[dict]:
    """Las que describen una reacción completa al pegote, y solo esas."""
    return [r for r in rows if r.get("status") in JUDGEABLE_STATUSES]


def done_pairs(path: Path) -> set[tuple[str, str]]:
    """Pares (conversación, juez) que ya tienen veredicto **usable**.

    Los fallidos no cuentan como hechos: al reanudar se reintentan. Es la
    diferencia entre reanudar y dar por buena una tirada a medias.
    """
    hechos: set[tuple[str, str]] = set()
    if not path.exists():
        return hechos
    for linea in path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            fila = json.loads(linea)
        except json.JSONDecodeError:
            continue
        if fila.get("status") == "ok":
            hechos.add((fila.get("conversation_id", ""), fila.get("judge_model", "")))
    return hechos


def verdict_row(row: dict, verdict: Verdict) -> dict:
    """El veredicto con lo justo de la fila para poder agrupar sin volver a unir.

    Se copian nivel y modelo —no se derivan luego del `conversation_id`— porque
    el nivel es el factor del experimento y un agregado no debería depender de
    que el formato del identificador no cambie nunca.
    """
    return {
        "conversation_id": row.get("conversation_id"),
        "paste_level": row.get("paste_level"),
        "model_id": row.get("model_id"),
        "topic_id": row.get("topic_id"),
        "n_turns": row.get("n_turns"),
        "artifact_signal": row.get("artifact_signal"),
        **dataclasses.asdict(verdict),
    }


def main(run_path: str | Path, out: str | Path | None = None) -> Path:
    """Juzga la tirada entera y devuelve la ruta de los veredictos."""
    run_path = Path(run_path)
    path = Path(out) if out is not None else verdicts_path(run_path)
    filas = judgeable(load_rows(run_path))
    hechos = done_pairs(path)

    pendientes = [
        f
        for f in filas
        if any((f.get("conversation_id"), j) not in hechos for j in JUDGES)
    ]
    llamadas = sum(
        1 for f in filas for j in JUDGES if (f.get("conversation_id"), j) not in hechos
    )
    print(
        f"--- {len(filas)} filas juzgables | {len(pendientes)} con algo pendiente "
        f"| {llamadas} llamadas (una por veredicto que falta, no por fila)"
    )

    escritas = 0
    with path.open("a", encoding="utf-8") as fh:
        for i, fila in enumerate(pendientes, 1):
            cid = fila.get("conversation_id")
            for juez in JUDGES:
                if (cid, juez) in hechos:
                    continue  # ya estaba, y era usable: no se vuelve a pagar
                veredicto = judge_one(
                    juez, fila.get("reaction", ""), fila.get("artifact_text", "")
                )
                fh.write(
                    json.dumps(verdict_row(fila, veredicto), ensure_ascii=False) + "\n"
                )
                fh.flush()
                escritas += 1
                if not is_usable(veredicto):
                    print(
                        f"    [{i}] {cid} {veredicto.judge_model}: "
                        f"{veredicto.status} {veredicto.error[:80]}"
                    )
            if i % 20 == 0 or i == len(pendientes):
                print(f"    [{i}/{len(pendientes)}] {escritas} veredictos escritos")

    print(f"--- {path}")
    return path


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1])
