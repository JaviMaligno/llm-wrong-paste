from dataclasses import dataclass
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[2] / "data" / "topics"


@dataclass(frozen=True)
class Topic:
    id: str
    opening: str
    goals: tuple[str, ...]


def _parse(path: Path) -> Topic:
    raw = path.read_text(encoding="utf-8")
    _, front, body = raw.split("---\n", 2)
    topic_id = front.strip().split(":", 1)[1].strip()
    opening = ""
    goals = []
    for line in body.strip().splitlines():
        if line.startswith("opening:"):
            opening = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            goals.append(line[2:].strip())
    return Topic(id=topic_id, opening=opening, goals=tuple(goals))


def load_topics() -> list[Topic]:
    return sorted((_parse(p) for p in TOPIC_DIR.glob("*.md")), key=lambda t: t.id)
