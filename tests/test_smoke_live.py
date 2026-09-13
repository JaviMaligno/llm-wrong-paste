import pytest

from wrongpaste.clients import chat, embed

LIVE = [
    "gpt-5.6-luna-tst",      # gateway, el más barato
    "claude-sonnet-5",       # vertex_anthropic
    "gemini-2.5-flash",      # vertex_openai
]


@pytest.mark.live
@pytest.mark.parametrize("model_id", LIVE)
def test_each_provider_answers(model_id):
    reply = chat(model_id, [{"role": "user", "content": "Responde solo: ok"}], max_tokens=16)
    assert reply.text.strip().lower().startswith("ok")


@pytest.mark.live
def test_embeddings_have_1536_dimensions():
    vecs = embed(["hola", "adiós"])
    assert vecs.shape == (2, 1536)
