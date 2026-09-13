from wrongpaste.topics import load_topics


def test_there_are_eight_topics():
    assert len(load_topics()) == 8


def test_every_topic_has_an_opening_and_goals():
    for topic in load_topics():
        assert topic.opening.strip()
        assert len(topic.goals) >= 4, f"{topic.id} tiene pocos objetivos"


def test_topic_ids_are_unique():
    ids = [t.id for t in load_topics()]
    assert len(ids) == len(set(ids))
