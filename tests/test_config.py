from wrongpaste.config import MODELS, Model


def test_roster_has_nine_models():
    assert len(MODELS) == 9


def test_every_model_has_a_known_provider():
    providers = {m.provider for m in MODELS.values()}
    assert providers == {"gateway", "vertex_anthropic", "vertex_openai"}


def test_claude_models_are_vertex_anthropic():
    assert MODELS["claude-opus-5"].provider == "vertex_anthropic"
    assert MODELS["claude-sonnet-5"].provider == "vertex_anthropic"


def test_size_ladder_tiers_are_distinct():
    ladder = [m for m in MODELS.values() if m.tier in {"large", "medium", "small"}]
    assert len(ladder) >= 3
