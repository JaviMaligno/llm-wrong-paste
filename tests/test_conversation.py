import wrongpaste.conversation as conv
from wrongpaste.artifacts import Artifact
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic(id="t", opening="quiero mudarme", goals=("a", "b", "c", "d"))
ART = Artifact("art1", "recipe", "300 g de lentejas", ("lentejas",))


def _stub(monkeypatch, replies=None, seen=None):
    """Sustituye las dos únicas puertas de red del módulo.

    Si se pasa `seen`, se van apuntando los `messages` con los que se llamó a
    `chat`, para poder comprobar que salen sin el `tag` (que es metadato
    nuestro y los proveedores rechazan campos desconocidos).
    """
    seq = iter(replies or [Reply(f"r{i}", {}, {}) for i in range(20)])

    def fake_chat(model_id, messages, **kwargs):
        if seen is not None:
            seen.append(messages)
        return next(seq)

    monkeypatch.setattr(conv, "chat", fake_chat)
    monkeypatch.setattr(conv, "next_user_turn", lambda *a, **k: "turno de usuario")


def test_prefix_alternates_roles_and_ends_on_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, usages = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=3)

    assert transcript[0]["role"] == "user"
    assert transcript[-1]["role"] == "assistant"
    assert [m["role"] for m in transcript] == ["user", "assistant"] * 3
    assert len(usages) == 3


def test_prefix_never_ends_on_user_so_claude_accepts_it(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("claude-opus-5", TOPIC, n_turns=2)
    assert transcript[-1]["role"] == "assistant"


def test_prefix_tags_opening_user_sim_and_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=3)

    assert [m["tag"] for m in transcript] == [
        "opening",
        "assistant",
        "user_sim",
        "assistant",
        "user_sim",
        "assistant",
    ]


def test_only_the_first_message_is_tagged_opening(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=4)

    assert [m["tag"] for m in transcript].count("opening") == 1
    assert transcript[0]["content"] == TOPIC.opening


def test_chat_never_sees_the_tag_field(monkeypatch):
    seen: list[list[dict]] = []
    _stub(monkeypatch, seen=seen)

    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert seen, "el stub tiene que haber visto llamadas"
    for messages in seen:
        for m in messages:
            assert set(m) == {"role", "content"}


def test_inject_paste_appends_artifact_verbatim(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    before = len(transcript)

    reaction, usage, paste_index = conv.inject_paste(
        "gpt-5.6-luna-tst", transcript, ART
    )

    assert transcript[before]["role"] == "user"
    assert transcript[before]["content"] == ART.text
    assert transcript[-1]["role"] == "assistant"
    assert reaction == transcript[-1]["content"]
    assert paste_index == before


def test_inject_paste_tags_the_paste_and_the_reaction(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)

    _, _, paste_index = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    assert transcript[paste_index]["tag"] == "paste"
    assert transcript[paste_index]["content"] == ART.text
    assert transcript[-1]["tag"] == "assistant"
    assert [m["tag"] for m in transcript].count("paste") == 1


def test_inject_paste_adds_no_preamble(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    pasted = [m for m in transcript if m["content"] == ART.text]
    assert len(pasted) == 1, "el pegote debe ir tal cual, sin envoltorio"


def test_continue_after_paste_adds_four_messages(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    before = len(transcript)

    post_indices, usages = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC
    )

    assert len(transcript) - before == 4
    assert len(usages) == 2
    assert post_indices == [before, before + 1, before + 2, before + 3]


def test_continue_after_paste_tags_post_and_assistant(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    before = len(transcript)

    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert [m["tag"] for m in transcript[before:]] == [
        "post",
        "assistant",
        "post",
        "assistant",
    ]
    assert [m["role"] for m in transcript[before:]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_continue_after_paste_never_repairs(monkeypatch):
    """La variante (a) del spec §7: nadie menciona el pegote por el usuario."""
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)
    conv.continue_after_paste("gpt-5.6-luna-tst", transcript, TOPIC)

    assert all(m["tag"] != "repair" for m in transcript)


def test_post_indices_point_at_the_messages_added_after_the_reaction(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    _, _, paste_index = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    post_indices, _ = conv.continue_after_paste(
        "gpt-5.6-luna-tst", transcript, TOPIC, n_post=3
    )

    assert len(post_indices) == 6
    assert all(i > paste_index for i in post_indices)
    assert [transcript[i]["tag"] for i in post_indices[::2]] == ["post"] * 3
