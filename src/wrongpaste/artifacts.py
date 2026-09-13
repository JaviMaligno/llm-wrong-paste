import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "artifacts"


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    text: str
    entities: tuple[str, ...]


def _parse(path: Path) -> Artifact:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name}: falta el frontmatter")
    _, front, body = raw.split("---\n", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return Artifact(
        id=meta["id"],
        kind=meta["kind"],
        text=body.strip(),
        entities=tuple(json.loads(meta["entities"])),
    )


def load_artifacts() -> list[Artifact]:
    return sorted(
        (_parse(p) for p in ARTIFACT_DIR.glob("*.md")),
        key=lambda a: a.id,
    )


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
