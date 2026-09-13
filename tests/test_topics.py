from pathlib import Path

from wrongpaste.topics import Topic, _parse, load_topics


def test_there_are_eight_topics():
    assert len(load_topics()) == 8


def test_every_topic_has_an_opening_and_goals():
    for topic in load_topics():
        assert topic.opening.strip()
        assert len(topic.goals) >= 4, f"{topic.id} tiene pocos objetivos"


def test_topic_ids_are_unique():
    ids = [t.id for t in load_topics()]
    assert len(ids) == len(set(ids))


def test_the_verifiable_task_fields_are_none_in_phase_0():
    """D13: el hueco existe, pero en Fase 0 no hay tarea verificable."""
    for topic in load_topics():
        assert topic.task is None, f"{topic.id} trae task en Fase 0"
        assert topic.expected is None, f"{topic.id} trae expected en Fase 0"
        assert topic.verifier is None, f"{topic.id} trae verifier en Fase 0"


def test_the_eight_topic_files_declare_the_hole():
    """El hueco tiene que verse en el fichero, aunque el valor sea None."""
    from wrongpaste.topics import TOPIC_DIR

    paths = sorted(TOPIC_DIR.glob("*.md"))
    assert len(paths) == 8
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for field in ("task:", "expected:", "verifier:"):
            assert field in text, f"{path.name} no declara el hueco {field}"


def test_the_fields_default_to_none_when_the_topic_is_built_by_hand():
    topic = Topic("t", "quiero mudarme", ("a", "b", "c", "d"))
    assert (topic.task, topic.expected, topic.verifier) == (None, None, None)


def _write_topic(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tema-de-prueba.md"
    path.write_text(
        "---\nid: tema-de-prueba\n---\n"
        "opening: Una apertura cualquiera para el tema de prueba.\n"
        "goals:\n- Objetivo uno\n- Objetivo dos\n" + body,
        encoding="utf-8",
    )
    return path


def test_a_topic_with_the_three_fields_filled_in_parses(tmp_path):
    """Cuando la Fase 2 los rellene, el parser ya los lee."""
    path = _write_topic(
        tmp_path,
        "task: Calcula el ritmo medio para 10 km en 55 minutos\n"
        "expected: 5:30 min/km\n"
        "verifier: regex\n",
    )
    topic = _parse(path)
    assert topic.id == "tema-de-prueba"
    assert topic.goals == ("Objetivo uno", "Objetivo dos")
    assert topic.task == "Calcula el ritmo medio para 10 km en 55 minutos"
    assert topic.expected == "5:30 min/km"
    assert topic.verifier == "regex"


def test_a_value_with_colons_keeps_everything_after_the_first_one(tmp_path):
    path = _write_topic(tmp_path, "expected: ritmo 5:30 min/km\n")
    assert _parse(path).expected == "ritmo 5:30 min/km"


def test_an_empty_or_commented_field_is_none(tmp_path):
    path = _write_topic(tmp_path, "task:\n# expected: algo comentado\n")
    topic = _parse(path)
    assert topic.task is None
    assert topic.expected is None
    assert topic.verifier is None


def test_the_optional_fields_do_not_become_goals(tmp_path):
    path = _write_topic(tmp_path, "task: Una tarea\n")
    assert _parse(path).goals == ("Objetivo uno", "Objetivo dos")
