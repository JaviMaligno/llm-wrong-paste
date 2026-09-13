import json
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
