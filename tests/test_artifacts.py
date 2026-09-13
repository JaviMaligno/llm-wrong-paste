from wrongpaste.artifacts import load_artifacts

MIN_ARTIFACTS = 60


def test_bank_is_big_enough():
    assert len(load_artifacts()) >= MIN_ARTIFACTS


def test_every_artifact_declares_entities():
    for art in load_artifacts():
        assert art.entities, f"{art.id} no declara entities"


def test_kinds_are_varied():
    kinds = {a.kind for a in load_artifacts()}
    assert len(kinds) >= 8


def test_ids_are_unique():
    ids = [a.id for a in load_artifacts()]
    assert len(ids) == len(set(ids))
