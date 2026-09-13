from wrongpaste.run_phase0 import PHASE0_MODELS, plan_phase0


def test_plan_uses_the_three_phase0_models():
    plan = plan_phase0(seed=1)
    assert {c["model_id"] for c in plan} == set(PHASE0_MODELS)


def test_plan_is_about_thirty_conversations():
    plan = plan_phase0(seed=1)
    assert 24 <= len(plan) <= 36


def test_plan_covers_both_lengths():
    plan = plan_phase0(seed=1)
    assert {c["n_turns"] for c in plan} == {2, 10}


def test_plan_is_deterministic_for_a_seed():
    a = [(c["model_id"], c["topic_id"], c["n_turns"]) for c in plan_phase0(seed=3)]
    b = [(c["model_id"], c["topic_id"], c["n_turns"]) for c in plan_phase0(seed=3)]
    assert a == b
