"""Anotaciones manuales de la Fase 0: formato, persistencia y cuadre (D14).

Las 27 conversaciones de la Fase 0 se leen enteras a mano y de esa lectura sale
la rúbrica v1. **Es el único conjunto etiquetado a mano que va a existir** y el
que semilla el acuerdo juez-humano del spec §6: si se pierde, se desordena o se
desincroniza del JSONL de la tirada, no hay contra qué medir al juez-LLM de la
Fase 1. Por eso la anotación no vive en «un fichero de trabajo», sino en
`runs/phase0/annotations-<run_id>.jsonl`, con esquema y con un chequeo de
cuadre contra la tirada.

Lo que este módulo deliberadamente **no** hace: validar `category_guess`. La
rúbrica todavía no existe — derivarla es el producto de la Fase 0 —, así que la
categoría es texto libre. Un vocabulario cerrado aquí sería el error de fondo:
obligaría a encajar en las cinco categorías previas del spec §5 justo la
conducta nueva (por ejemplo, que un modelo **ejecute** un artefacto `prompt`)
que D4 se ha molestado en garantizar que aparezca en la muestra.

El módulo es hoja del grafo de importaciones, igual que `records.py`: no
importa clientes ni red, para que se pueda usar desde un análisis o desde un
cuaderno sin arrastrar nada.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

# Dónde viven las anotaciones: el mismo directorio que el JSONL de la tirada,
# para que la anotación viaje siempre pegada a los datos que describe. Se lee
# como global en cada función para que los tests lo redirijan a un `tmp_path`.
ANNOTATION_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase0"


@dataclass
class Annotation:
    """Lo que una persona anota tras leer una conversación entera.

    - `conversation_id`: la fila de la tirada que se está anotando; es la clave
      que permite cuadrar la anotación con el JSONL.
    - `annotator`: quién la escribió. Va en la fila porque el acuerdo entre dos
      lectores solo se puede calcular si se sabe quién dijo qué.
    - `category_guess`: **texto libre**, a propósito (ver el docstring del
      módulo).
    - `quote`: la cita literal de la transcripción que justifica la categoría.
      Es lo que después se copia como ejemplo en la rúbrica v1.
    - `notes`: lo demás — dudas, conductas que no encajan, desempates.
    """

    conversation_id: str
    annotator: str
    category_guess: str
    quote: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("conversation_id", "annotator", "category_guess"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} es obligatorio y no puede ir vacío: {value!r}"
                )

    def to_json(self) -> dict[str, Any]:
        """Diccionario listo para `json.dumps`, sin perder ningún campo."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Annotation":
        """Reconstruye una anotación de una línea del fichero.

        Es estricta con las claves desconocidas: estos ficheros se escriben a
        mano, y una errata en el nombre de un campo se tiene que ver al leer,
        no perderse en silencio.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"campos desconocidos en la anotación: {sorted(unknown)}"
            )
        return cls(**data)


def ANNOTATION_PATH(run_id: str) -> Path:  # noqa: N802 — es una ruta, no una clase
    """Ruta del fichero de anotaciones de una tirada, exista o no (D14)."""
    if not run_id or not str(run_id).strip():
        raise ValueError("run_id vacío: la anotación se guarda por tirada")
    return ANNOTATION_DIR / f"annotations-{run_id}.jsonl"


def write_annotations(run_id: str, anns: Iterable[Annotation]) -> Path:
    """Escribe las anotaciones en JSONL: **una línea por anotación**.

    Reescribe el fichero entero. Para añadir a una tanda anterior, el camino es
    `load_annotations` + `write_annotations` con la lista completa: así el
    fichero nunca queda a medias entre dos formatos y el orden es el que decide
    quien anota, no el del sistema de ficheros.

    `ensure_ascii=False` porque las citas son en español y en el fichero tienen
    que poder leerse a ojo, que es medio sentido de que esto sea JSONL.
    """
    path = ANNOTATION_PATH(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(a.to_json(), ensure_ascii=False, sort_keys=True)
        for a in anns
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def load_annotations(run_id: str) -> list[Annotation]:
    """Lee las anotaciones de una tirada, en el orden en que están escritas.

    Revienta si el fichero no está, igual que `prefixes.load_prefix`: que falte
    la anotación de una tirada es un error que hay que ver, no un caso normal
    que se resuelva devolviendo una lista vacía. Las líneas en blanco se
    ignoran (un editor de texto deja una al final).
    """
    raw = ANNOTATION_PATH(run_id).read_text(encoding="utf-8")
    out: list[Annotation] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"línea {number} de {ANNOTATION_PATH(run_id).name} no es JSON: {exc}"
            ) from exc
        out.append(Annotation.from_json(data))
    return out


def validate_against_run(
    anns: Iterable[Annotation], record_ids: Iterable[str]
) -> dict[str, list[str]]:
    """Cuadra las anotaciones contra los `conversation_id` de la tirada.

    Es lo que impide que la anotación se desincronice del JSONL — que es la
    forma silenciosa de perder el único conjunto etiquetado a mano. Devuelve
    tres listas ordenadas:

    - `unknown_ids`: anotaciones que apuntan a una conversación que **no está**
      en la tirada (típicamente una errata al copiar el id, o una anotación
      heredada de una tirada anterior).
    - `unannotated_ids`: conversaciones de la tirada **sin anotar**. Con 27
      conversaciones que hay que leer enteras, es el recuento que dice cuánto
      queda.
    - `duplicate_ids`: conversaciones que **el mismo anotador** ha anotado más
      de una vez. Que dos anotadores distintos anoten la misma conversación no
      es un duplicado: es exactamente el material con el que se mide el acuerdo
      entre lectores. Repetirse uno mismo, en cambio, es un despiste — y deja
      dos categorías para la misma fila sin decir cuál vale.

    Las tres vacías significa que la anotación cuadra con la tirada.
    """
    annotations = list(anns)
    ids = set(record_ids)

    annotated = {a.conversation_id for a in annotations}

    seen: set[tuple[str, str]] = set()
    duplicates: set[str] = set()
    for a in annotations:
        key = (a.conversation_id, a.annotator)
        if key in seen:
            duplicates.add(a.conversation_id)
        seen.add(key)

    return {
        "unknown_ids": sorted(annotated - ids),
        "unannotated_ids": sorted(ids - annotated),
        "duplicate_ids": sorted(duplicates),
    }
