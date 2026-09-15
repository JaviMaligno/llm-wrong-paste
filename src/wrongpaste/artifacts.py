import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data"

# Los dos bancos en disco. N2 NO está aquí a propósito: se fabrica contra cada
# conversación concreta, así que no es un banco (ver `contradictions.py`).
ARTIFACT_DIRS = {
    "N0": _DATA / "artifacts",
    "N1": _DATA / "artifacts-n1",
}

# Nombre antiguo del banco neutro, que era el único que había. Se mantiene
# porque hay código y notas que lo citan.
ARTIFACT_DIR = ARTIFACT_DIRS["N0"]


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    text: str
    entities: tuple[str, ...]
    # Nivel de pegote del spec §5 de la Fase 1: N0 neutro, N1 con señal
    # intrínseca. Por defecto N0, que es lo que era todo antes de la Fase 1.
    level: str = "N0"
    # Qué delata al pegote, y solo en N1: `cortado`, `dirigido`, `responde` o
    # `presupone`. La señal es parte del dato porque el análisis pregunta qué
    # TIPO de pista funciona, no solo si alguna funciona.
    signal: str | None = None
    # Los dos ejes de un pegote de género `prompt`, y solo de ese género. Se
    # declaran en el fichero porque la Fase 1b demostró que la categoría que
    # sale NO la decide el modelo sino el artefacto: cada uno de los cinco
    # `prompt` del banco original produjo siempre la misma letra, sin una sola
    # excepción, y el reparto A/B/F del género era en realidad el sorteo de qué
    # artefacto tocaba. Con los ejes fuera del fichero eso no se puede ni medir
    # ni equilibrar.
    #
    # `material` — qué trae el pegote sobre lo que trabajar:
    #   `autosuficiente`           la instrucción no necesita nada más
    #   `con_referente`            trae consigo el texto al que se refiere
    #   `sin_referente_sustituible` le falta el texto, pero la conversación
    #                              anterior sirve de material (y ahí es donde el
    #                              pegote secuestra la sesión)
    #   `sin_referente`            le falta el texto y nada puede sustituirlo
    # `papel` — `rol` si asigna un personaje, `tarea` si es un encargo suelto.
    material: str | None = None
    papel: str | None = None


PROMPT_MATERIALS: frozenset[str] = frozenset(
    {"autosuficiente", "con_referente", "sin_referente_sustituible", "sin_referente"}
)
PROMPT_PAPELES: frozenset[str] = frozenset({"rol", "tarea"})


def _parse(path: Path, level: str = "N0") -> Artifact:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name}: falta el frontmatter")
    _, front, body = raw.split("---\n", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    declarado = meta.get("level", level)
    if declarado != level:
        raise ValueError(
            f"{path.name}: declara level={declarado!r} pero vive en el banco "
            f"{level!r}. El directorio manda: mueve el fichero o arregla el "
            "frontmatter."
        )
    signal = meta.get("signal") or None
    if level == "N1" and signal is None:
        raise ValueError(
            f"{path.name}: un artefacto N1 sin `signal` no dice qué lo delata, "
            "y sin eso no se puede analizar qué tipo de señal funciona."
        )
    material = meta.get("material") or None
    papel = meta.get("papel") or None
    if meta["kind"] == "prompt":
        if material not in PROMPT_MATERIALS or papel not in PROMPT_PAPELES:
            raise ValueError(
                f"{path.name}: un pegote `prompt` tiene que declarar `material` "
                f"({sorted(PROMPT_MATERIALS)}) y `papel` ({sorted(PROMPT_PAPELES)}). "
                "Sin los dos ejes, el género vuelve a confundir 'qué hace el "
                "modelo' con 'qué artefacto tocó en el sorteo', que es lo que "
                "pasó en la Fase 1b."
            )
    elif material or papel:
        raise ValueError(
            f"{path.name}: `material`/`papel` son ejes del género `prompt` y "
            f"este es `{meta['kind']}`. Declararlos aquí sugiere un reparto que "
            "nadie mide."
        )
    return Artifact(
        id=meta["id"],
        kind=meta["kind"],
        text=body.strip(),
        entities=tuple(json.loads(meta["entities"])),
        level=level,
        signal=signal,
        material=material,
        papel=papel,
    )


def load_artifacts(level: str | None = None) -> list[Artifact]:
    """Carga el banco. Sin `level`, devuelve los dos niveles juntos.

    N2 NO vive aquí: se genera contra cada conversación (ver contradictions.py)
    y por eso no es un banco, es un distractor de diseño.

    Ojo al llamarla desde la Fase 0: su banco es el N0 y solo el N0. Pedirlo
    explícitamente —`load_artifacts(level="N0")`— es lo que impide que un
    artefacto de la Fase 1 se cuele en un recuento de la Fase 0.
    """
    if level is not None and level not in ARTIFACT_DIRS:
        raise ValueError(
            f"nivel desconocido: {level!r}; los bancos en disco son "
            f"{sorted(ARTIFACT_DIRS)}. N2 no es un banco: se fabrica contra "
            "cada conversación."
        )
    niveles = [level] if level is not None else list(ARTIFACT_DIRS)
    fuera = []
    for niv in niveles:
        for p in sorted(ARTIFACT_DIRS[niv].glob("*.md")):
            fuera.append(_parse(p, niv))
    return sorted(fuera, key=lambda a: a.id)


def normalise(text: str) -> str:
    """Normaliza un texto para el casado de entidades (D8).

    Pasa a minúsculas, descompone en NFKD y descarta las marcas combinantes
    (así «pimentón» y «Pimenton» acaban igual), convierte en separador todo
    carácter que no sea alfanumérico —puntuación, símbolos, guiones, el signo
    de euro— y colapsa los espacios. El resultado es una secuencia de palabras
    separadas por un único espacio, lo que permite comprobar fronteras de
    palabra con una simple búsqueda de subcadena.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    chars = [
        char if char.isalnum() else " "
        for char in decomposed
        if not unicodedata.combining(char)
    ]
    return " ".join("".join(chars).split())


def entity_hits(text: str, entities: Iterable[str]) -> list[str]:
    """Devuelve las `entities` que aparecen en `text` (D8).

    Ambos lados se normalizan con `normalise`: minúsculas, sin acentos,
    puntuación tratada como separador y espacios colapsados. Una entity cuenta
    como presente si aparece como subcadena **con fronteras de palabra** en el
    texto normalizado, de modo que «pan» no casa dentro de «pantalla».

    La lista se devuelve en el orden en que se pasaron las entities, con su
    grafía original y sin duplicados (dos entities que se normalizan igual
    cuentan como una sola).
    """
    haystack = f" {normalise(text)} "
    hits: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        needle = normalise(entity)
        if not needle or needle in seen:
            continue
        seen.add(needle)
        if f" {needle} " in haystack:
            hits.append(entity)
    return hits
