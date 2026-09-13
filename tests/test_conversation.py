import wrongpaste.conversation as conv
from wrongpaste.artifacts import Artifact
from wrongpaste.clients import Reply
from wrongpaste.topics import Topic

TOPIC = Topic("t", "quiero mudarme", ("a", "b", "c", "d"))
ART = Artifact("art1", "recipe", "300 g de lentejas", ("lentejas",))


def _stub(monkeypatch, replies=None):
    seq = iter(replies or [Reply(f"r{i}", {}, {}) for i in range(20)])
    monkeypatch.setattr(conv, "chat", lambda *a, **k: next(seq))
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


def test_inject_paste_appends_artifact_verbatim(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    before = len(transcript)

    reaction, usage = conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    assert transcript[before]["role"] == "user"
    assert transcript[before]["content"] == ART.text
    assert transcript[-1]["role"] == "assistant"
    assert reaction == transcript[-1]["content"]


def test_inject_paste_adds_no_preamble(monkeypatch):
    _stub(monkeypatch)
    transcript, _ = conv.build_prefix("gpt-5.6-luna-tst", TOPIC, n_turns=2)
    conv.inject_paste("gpt-5.6-luna-tst", transcript, ART)

    pasted = [m for m in transcript if m["content"] == ART.text]
    assert len(pasted) == 1, "el pegote debe ir tal cual, sin envoltorio"
