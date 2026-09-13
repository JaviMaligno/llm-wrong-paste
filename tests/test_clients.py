from wrongpaste.clients import _anthropic_body, _gateway_body, _vertex_openai_body


def test_anthropic_body_moves_system_out_of_messages():
    msgs = [
        {"role": "system", "content": "eres útil"},
        {"role": "user", "content": "hola"},
    ]
    body = _anthropic_body(msgs, max_tokens=16)
    assert body["system"] == "eres útil"
    assert body["messages"] == [{"role": "user", "content": "hola"}]
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert body["max_tokens"] == 16


def test_anthropic_body_never_ends_on_assistant_turn():
    # El prefill está eliminado en la familia 5: un último turno de
    # assistant daría 400. El cliente debe negarse antes de llamar.
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    try:
        _anthropic_body(msgs, max_tokens=16)
    except ValueError as exc:
        assert "prefill" in str(exc).lower()
    else:
        raise AssertionError("debería haber lanzado ValueError")


def test_gateway_body_uses_max_completion_tokens():
    body = _gateway_body("gpt-5.6-sol-tst", [{"role": "user", "content": "x"}], 32)
    assert body["max_completion_tokens"] == 32
    assert "max_tokens" not in body


def test_vertex_openai_body_prefixes_model_with_google():
    body = _vertex_openai_body("gemini-2.5-pro", [{"role": "user", "content": "x"}], 32)
    assert body["model"] == "google/gemini-2.5-pro"
