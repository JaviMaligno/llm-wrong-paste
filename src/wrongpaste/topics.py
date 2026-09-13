from dataclasses import dataclass
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[2] / "data" / "topics"

# Campos opcionales del cuerpo que reserva D13 para la tarea verificable.
# En Fase 0 están comentados en los ocho ficheros y por tanto valen None.
_OPTIONAL_FIELDS = ("task", "expected", "verifier")


@dataclass(frozen=True)
class Topic:
    """Un tema de conversación doméstica del banco.

    `task`, `expected` y `verifier` son el hueco que exige D13: la Fase 2 necesita
    temas con una tarea verificable aguas abajo, y si el campo no existiera ya en
    Fase 0 la Fase 2 tendría que correr sobre temas nuevos, con lo que dejaría de
    estar pareada con la Fase 1. En Fase 0 los tres valen None a propósito: aquí
    no se inventa ninguna tarea.
    """

    id: str
    opening: str
    goals: tuple[str, ...]
    task: str | None = None
    expected: str | None = None
    verifier: str | None = None


def _parse(path: Path) -> Topic:
    """Lee un fichero de tema: frontmatter con `id` y cuerpo con el resto.

    Los campos opcionales se leen del cuerpo si están presentes como líneas
    `task:`, `expected:` o `verifier:`. Si la línea no está —o está comentada, o
    trae el valor vacío— el campo queda a None.
    """
    raw = path.read_text(encoding="utf-8")
    _, front, body = raw.split("---\n", 2)
    topic_id = front.strip().split(":", 1)[1].strip()
    opening = ""
    goals = []
    optional: dict[str, str | None] = {name: None for name in _OPTIONAL_FIELDS}
    for line in body.strip().splitlines():
        if line.startswith("opening:"):
            opening = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            goals.append(line[2:].strip())
        else:
            for name in _OPTIONAL_FIELDS:
                if line.startswith(f"{name}:"):
                    value = line.split(":", 1)[1].strip()
                    # Un valor vacío es un hueco declarado, no una tarea.
                    optional[name] = value or None
                    break
    return Topic(
        id=topic_id,
        opening=opening,
        goals=tuple(goals),
        task=optional["task"],
        expected=optional["expected"],
        verifier=optional["verifier"],
    )


def load_topics() -> list[Topic]:
    return sorted((_parse(p) for p in TOPIC_DIR.glob("*.md")), key=lambda t: t.id)
