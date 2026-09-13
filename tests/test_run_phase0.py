"""Tests del runner de la Fase 0. Ninguno hace una sola llamada a un modelo.

Todo lo que sale a la red —`ensure_prefix`, `rank_artifacts`, `inject_paste`,
`continue_after_paste`— se sustituye por un doble determinista. Lo que se prueba
aquí es el diseño: la rotación de ejes (D3), la cobertura de géneros (D4), la
identidad de las filas (D5), que un fallo escriba fila y que la tirada se pueda
reanudar (D6), los dos turnos posteriores (D7), las celdas de control (D11) y la
medición del ancho del eje (D12).
"""

import json
from collections import Counter

import numpy as np
import pytest

import wrongpaste.run_phase0 as rp
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.run_phase0 import (
    LENGTHS,
    PHASE0_MODELS,
    STRATA,
    STRATUM_STEP,
    pick_artifact,
    plan_phase0,
    stratum_window,
)
from wrongpaste.topics import Topic, load_topics

TOPICS = load_topics()
ARTS = load_artifacts()
ALL_KINDS = {a.kind for a in ARTS}


def _paste_cells(plan: list[dict]) -> list[dict]:
    return [c for c in plan if c["condition"] == "paste"]


def _control_cells(plan: list[dict]) -> list[dict]:
    return [c for c in plan if c["condition"] == "no_paste"]


# --- plan: lo que ya se exigía --------------------------------------------


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


# --- D3: rotación de ejes --------------------------------------------------


def test_stratum_travels_inside_the_plan():
    """El estrato es un dato de la celda, no el orden de iteración."""
    plan = plan_phase0(seed=1)
    assert all("stratum" in c for c in plan)
    strata = {c["stratum"] for c in _paste_cells(plan)}
    assert strata <= set(range(STRATA))


def test_stratum_follows_the_documented_formula():
    plan = _paste_cells(plan_phase0(seed=1))
    by_cell = {(c["topic_id"], c["model_id"]): c for c in plan}
    for t, topic in enumerate(TOPICS):
        for m, model_id in enumerate(PHASE0_MODELS):
            cell = by_cell[(topic.id, model_id)]
            assert cell["stratum"] == (t + STRATUM_STEP * m) % STRATA
            assert cell["n_turns"] == LENGTHS[(t + m) % 2]


def test_no_stratum_equals_the_topic_index_for_all_three_models():
    """Si estrato ≡ tema, el eje de similaridad y el de tema son el mismo eje."""
    plan = _paste_cells(plan_phase0(seed=1))
    for t, topic in enumerate(TOPICS):
        strata = [c["stratum"] for c in plan if c["topic_id"] == topic.id]
        assert len(strata) == len(PHASE0_MODELS)
        assert not all(s == t for s in strata)
        assert len(set(strata)) == len(PHASE0_MODELS), (
            "los tres modelos de un tema tienen que caer en estratos distintos"
        )


def test_every_stratum_lands_on_three_different_topics():
    plan = _paste_cells(plan_phase0(seed=1))
    per_stratum: dict[int, set[str]] = {}
    for cell in plan:
        per_stratum.setdefault(cell["stratum"], set()).add(cell["topic_id"])
    assert set(per_stratum) == set(range(STRATA))
    assert all(len(topics) == 3 for topics in per_stratum.values())


def test_every_topic_appears_in_both_lengths():
    plan = _paste_cells(plan_phase0(seed=1))
    for topic in TOPICS:
        lengths = {c["n_turns"] for c in plan if c["topic_id"] == topic.id}
        assert lengths == set(LENGTHS), f"{topic.id} no cubre las dos longitudes"


def test_execution_order_is_round_robin_of_models_inside_each_topic():
    """Modelo-mayor confundiría el modelo con la hora de reloj del gateway."""
    plan = _paste_cells(plan_phase0(seed=1))
    assert [c["model_id"] for c in plan[: len(PHASE0_MODELS)]] == PHASE0_MODELS
    topic_order = [c["topic_id"] for c in plan[:: len(PHASE0_MODELS)]]
    assert topic_order == [t.id for t in TOPICS]


# --- D11: brazo de control -------------------------------------------------


def test_plan_includes_three_control_cells_without_paste():
    controls = _control_cells(plan_phase0(seed=1))
    assert len(controls) == rp.N_CONTROL_CELLS == 3
    assert all(c["condition"] == "no_paste" for c in controls)
    assert all(c["stratum"] == -1 for c in controls)


def test_control_cells_spread_over_the_three_models():
    controls = _control_cells(plan_phase0(seed=1))
    assert {c["model_id"] for c in controls} == set(PHASE0_MODELS)


def test_control_cells_reuse_a_length_that_the_plan_already_builds():
    """El control no puede inventarse un prefijo nuevo: sería gastar por gusto."""
    plan = plan_phase0(seed=1)
    pairs = {(c["topic_id"], c["n_turns"]) for c in _paste_cells(plan)}
    for control in _control_cells(plan):
        assert (control["topic_id"], control["n_turns"]) in pairs


# --- D5: identidad ---------------------------------------------------------


def test_conversation_ids_are_unique():
    plan = plan_phase0(seed=1)
    ids = [c["conversation_id"] for c in plan]
    assert len(set(ids)) == len(ids), "hay dos celdas que escribirían el mismo id"


def test_conversation_ids_are_deterministic_across_runs_and_seeds():
    """El id es la clave de reanudación: no puede depender de la semilla."""
    one = [c["conversation_id"] for c in plan_phase0(seed=1)]
    two = [c["conversation_id"] for c in plan_phase0(seed=1)]
    other_seed = [c["conversation_id"] for c in plan_phase0(seed=99)]
    assert one == two == other_seed


def test_conversation_id_is_readable():
    cell = plan_phase0(seed=1)[0]
    assert cell["conversation_id"] == (
        f"p0-{cell['model_id']}-{cell['topic_id']}-{cell['n_turns']}-r0"
    )


# --- D4: cobertura de género ----------------------------------------------


def _fake_ranking(arts: list[Artifact], rng: np.random.Generator):
    """Un ranking plausible: el banco barajado, con cosenos crecientes."""
    order = list(rng.permutation(len(arts)))
    return [(arts[i], n / (len(arts) - 1)) for n, i in enumerate(order)]


def test_stratum_window_splits_the_ranking_without_gaps():
    ranked = _fake_ranking(ARTS, np.random.default_rng(0))
    seen: list[str] = []
    for s in range(STRATA):
        seen += [a.id for a, _ in stratum_window(ranked, s)]
    assert seen == [a.id for a, _ in ranked]


def test_stratum_window_rejects_a_stratum_out_of_range():
    ranked = _fake_ranking(ARTS, np.random.default_rng(0))
    with pytest.raises(ValueError):
        stratum_window(ranked, STRATA)


def test_pick_artifact_prefers_the_least_represented_kind():
    window = [
        (Artifact("a", "recipe", "t", ()), 0.1),
        (Artifact("b", "prompt", "t", ()), 0.2),
        (Artifact("c", "recipe", "t", ()), 0.3),
    ]
    counts = Counter({"recipe": 2, "prompt": 5})
    # Un único estrato que es la ventana entera, para aislar la elección.
    art, _ = pick_artifact(window, 0, counts, np.random.default_rng(0), n_strata=1)
    assert art.kind == "recipe"

    counts["recipe"] = 9
    art, _ = pick_artifact(window, 0, counts, np.random.default_rng(0), n_strata=1)
    assert art.kind == "prompt"


def test_pick_artifact_stays_inside_its_stratum():
    ranked = _fake_ranking(ARTS, np.random.default_rng(1))
    counts: Counter = Counter()
    for stratum in range(STRATA):
        art, sim = pick_artifact(ranked, stratum, counts, np.random.default_rng(2))
        assert (art, sim) in stratum_window(ranked, stratum)


def test_the_sampling_plan_covers_all_eleven_kinds():
    """D4: con 24 celdas y 11 géneros, dejarlo a suerte deja géneros sin ver."""
    counts: Counter = Counter()
    for cell in _paste_cells(plan_phase0(seed=rp.MASTER_SEED)):
        rng = np.random.default_rng(cell["seed"])
        ranked = _fake_ranking(ARTS, rng)
        art, _ = pick_artifact(ranked, cell["stratum"], counts, rng)
        counts[art.kind] += 1

    assert len(ALL_KINDS) == 11
    assert set(counts) == ALL_KINDS, f"géneros sin cubrir: {ALL_KINDS - set(counts)}"


def test_sampling_without_the_kind_rule_would_miss_kinds():
    """Contraste: el muestreo a suerte es el que D4 sustituye, y se le nota."""
    missed = 0
    for seed in range(20):
        kinds = set()
        for cell in _paste_cells(plan_phase0(seed=rp.MASTER_SEED)):
            rng = np.random.default_rng(cell["seed"] + seed)
            ranked = _fake_ranking(ARTS, rng)
            window = stratum_window(ranked, cell["stratum"])
            kinds.add(window[int(rng.integers(0, len(window)))][0].kind)
        missed += len(ALL_KINDS - kinds)
    assert missed > 0


# --- dobles para la tirada -------------------------------------------------


FAKE_TOPICS = [
    Topic(id=f"tema-{i}", opening=f"apertura {i}", goals=("a", "b")) for i in range(8)
]

FAKE_ARTS = [
    Artifact(f"art-{i:02d}", f"kind-{i % 4}", f"texto {i}", (f"e{i}",))
    for i in range(16)
]


def _prefix_for(topic, n_turns):
    return {
        "prefix_id": f"pfx-{topic.id}-{n_turns}",
        "topic_id": topic.id,
        "n_turns": n_turns,
        "prefix_model": "gpt-5.6-terra-tst",
        "transcript": [
            {"role": "user", "content": f"apertura de {topic.id}", "tag": "opening"},
            {"role": "assistant", "content": "prosa", "tag": "assistant"},
        ],
        "usages": [{"total_tokens": 10}],
    }


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Sustituye las cuatro puertas de red del runner y lo aísla en `tmp_path`."""
    calls: dict[str, list] = {"paste": [], "post": [], "rank": [], "prefix": []}

    def fake_ensure_prefix(topic, n_turns):
        calls["prefix"].append((topic.id, n_turns))
        return _prefix_for(topic, n_turns)

    def fake_rank_artifacts(text, arts, **kwargs):
        calls["rank"].append(text)
        n = len(arts)
        return [(a, i / (n - 1)) for i, a in enumerate(arts)], False

    def fake_inject_paste(model_id, transcript, artifact):
        calls["paste"].append((model_id, artifact.id))
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append(
            {"role": "assistant", "content": f"reacción de {model_id}", "tag": "assistant"}
        )
        return f"reacción de {model_id}", {"total_tokens": 5}, index

    def fake_continue(model_id, transcript, topic, n_post=2):
        calls["post"].append((model_id, n_post))
        indices = []
        for _ in range(n_post):
            indices.append(len(transcript))
            transcript.append({"role": "user", "content": "sigo", "tag": "post"})
            indices.append(len(transcript))
            transcript.append({"role": "assistant", "content": "ya", "tag": "assistant"})
        return indices, [{"total_tokens": 3}] * n_post

    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", lambda: list(FAKE_ARTS))
    monkeypatch.setattr(rp, "ensure_prefix", fake_ensure_prefix)
    monkeypatch.setattr(rp, "rank_artifacts", fake_rank_artifacts)
    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue)
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase0")
    return calls


def _rows(path):
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    return lines[0], lines[1:]


# --- D12: ancho del eje ----------------------------------------------------


def test_measure_axis_reports_min_median_and_max_per_topic(harness, capsys):
    axis = rp.measure_axis(topics=FAKE_TOPICS, arts=FAKE_ARTS, run_id="r1")

    assert axis["run_id"] == "r1"
    assert [e["topic_id"] for e in axis["topics"]] == [t.id for t in FAKE_TOPICS]
    for entry in axis["topics"]:
        assert entry["min"] <= entry["median"] <= entry["max"]
        assert entry["range"] == pytest.approx(entry["max"] - entry["min"])
        assert len(entry["top"]) == rp.AXIS_TOP_N
    out = capsys.readouterr().out
    assert "tema-0" in out and "med=" in out


def test_measure_axis_embeds_the_user_side_of_the_prefix(harness):
    rp.measure_axis(topics=FAKE_TOPICS[:2], arts=FAKE_ARTS, run_id="r1")
    # D2: se embebe el lado del usuario, no la prosa del asistente.
    assert harness["rank"] == ["apertura de tema-0", "apertura de tema-1"]


def test_measure_axis_flags_a_narrow_topic(harness, monkeypatch):
    def flat(text, arts, **kwargs):
        return [(a, 0.5 + i * 0.001) for i, a in enumerate(arts)], False

    monkeypatch.setattr(rp, "rank_artifacts", flat)
    axis = rp.measure_axis(topics=FAKE_TOPICS[:1], arts=FAKE_ARTS, run_id="r1")

    assert axis["topics"][0]["narrow"] is True
    assert axis["narrow_topics"] == ["tema-0"]


def test_main_writes_the_axis_report_before_the_run(harness, tmp_path):
    path = rp.main(seed=7)
    axis_file = rp.axis_path(path.stem)
    assert axis_file.exists()
    assert json.loads(axis_file.read_text(encoding="utf-8"))["topics"]


# --- D5/D6/D7/D11 sobre la tirada -----------------------------------------


def test_main_writes_the_run_header_first(harness):
    path = rp.main(seed=7)
    header, rows = _rows(path)

    assert header["kind"] == "run_header"
    assert header["planned_cells"] == len(rows) == len(plan_phase0(7))
    assert header["roster"] == PHASE0_MODELS
    assert header["master_seed"] == 7
    assert header["bank_sha"] and header["topics_sha"]
    assert header["prefix_model"] and header["user_model"]
    assert header["user_system_prompt"]


def test_main_writes_one_row_per_cell_with_identity(harness):
    path = rp.main(seed=7)
    _, rows = _rows(path)
    plan = plan_phase0(7)

    assert [r["conversation_id"] for r in rows] == [
        c["conversation_id"] for c in plan
    ]
    assert [r["cell_index"] for r in rows] == [c["cell_index"] for c in plan]
    assert all(r["run_id"] == path.stem for r in rows)
    assert all(r["model_label"] for r in rows)
    assert all(r["prefix_id"].startswith("pfx-") for r in rows)


def test_main_continues_two_turns_after_the_paste(harness):
    rp.main(seed=7)
    assert harness["post"], "nadie continuó después del pegote"
    assert {n for _, n in harness["post"]} == {2}


def test_rows_carry_the_post_turns_and_the_paste_index(harness):
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    for row in pasted:
        assert len(row["post_indices"]) == 4
        assert row["transcript"][row["paste_index"]]["tag"] == "paste"
        assert all(i > row["paste_index"] for i in row["post_indices"])
        assert [row["transcript"][i]["tag"] for i in row["post_indices"][::2]] == [
            "post",
            "post",
        ]


def test_control_rows_have_no_artifact_and_no_similarity(harness):
    path = rp.main(seed=7)
    _, rows = _rows(path)
    controls = [r for r in rows if r["condition"] == "no_paste"]

    assert len(controls) == rp.N_CONTROL_CELLS
    for row in controls:
        assert row["artifact_id"] is None
        assert row["artifact_kind"] is None
        assert row["similarity_user"] is None
        assert row["similarity_full"] is None
        assert row["paste_index"] is None
        assert row["reaction"] is None
        assert row["status"] == "ok"
        # El control sí continúa: su valor es la tasa base de entidades.
        assert len(row["post_indices"]) == 4


def test_paste_rows_carry_both_similarities_and_the_ranking(harness):
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    for row in pasted:
        assert row["similarity_user"] is not None
        assert row["similarity_full"] is not None
        assert len(row["ranking"]) == len(FAKE_ARTS)
        assert row["ranking"][0]["artifact_id"]
        assert 0 <= row["similarity_pct"] <= 1
        assert row["artifact_text"] and row["artifact_entities"]


def test_the_three_models_of_a_topic_share_the_prefix_ranking(harness):
    """D1/D2: el eje x no puede cambiar según quién conteste."""
    rp.main(seed=7, measure=False)

    prefixes = {
        (c["topic_id"], c["n_turns"]) for c in _paste_cells(plan_phase0(7))
    }
    # Dos llamadas por prefijo distinto —la similaridad de usuario y la de la
    # conversación entera—, no dos por celda: los tres modelos de un tema
    # comparten prefijo, así que tienen que compartir ranking.
    assert len(harness["rank"]) == 2 * len(prefixes)
    assert len(prefixes) < len(_paste_cells(plan_phase0(7)))


def test_main_is_idempotent_and_resumes(harness):
    path = rp.main(seed=7)
    _, rows = _rows(path)
    kept = rows[:5]

    partial = path.parent / "partial.jsonl"
    header, _ = _rows(path)
    partial.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False) for r in [header, *kept]
        )
        + "\n",
        encoding="utf-8",
    )

    again = rp.main(seed=7, out=partial)
    _, resumed = _rows(again)

    assert again == partial
    assert len(resumed) == len(rows)
    assert [r["conversation_id"] for r in resumed] == [
        r["conversation_id"] for r in rows
    ]
    # Las cinco que ya estaban no se han vuelto a correr: siguen tal cual.
    assert resumed[:5] == kept


def test_resume_keeps_the_kind_coverage_going(harness):
    """Reanudar sin recontar géneros rompería D4 en las celdas que faltan."""
    done, kinds = rp.resume_state(
        _write_rows(
            harness,
            [
                {"conversation_id": "p0-x", "artifact_kind": "recipe"},
                {"conversation_id": "p0-y", "artifact_kind": "sql"},
                {"conversation_id": "p0-z", "artifact_kind": "recipe"},
            ],
        )
    )
    assert done == {"p0-x", "p0-y", "p0-z"}
    assert kinds == Counter({"recipe": 2, "sql": 1})


def _write_rows(harness, rows):
    path = rp.OUT_DIR / "resume.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [json.dumps({"kind": "run_header", "run_id": "r0"})]
            + [json.dumps(r) for r in rows]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_resume_state_survives_a_truncated_last_line(harness):
    path = rp.OUT_DIR / "roto.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kind": "run_header", "run_id": "r0"})
        + "\n"
        + json.dumps({"conversation_id": "p0-a", "artifact_kind": "sql"})
        + "\n"
        + '{"conversation_id": "p0-b", "artifac',
        encoding="utf-8",
    )
    done, kinds = rp.resume_state(path)
    assert done == {"p0-a"}
    assert kinds == Counter({"sql": 1})


# --- D6: los fallos son datos ---------------------------------------------


def test_a_failing_cell_still_writes_a_row_and_the_run_goes_on(harness, monkeypatch):
    import httpx

    bad = {"call": 0}

    def flaky(model_id, transcript, artifact):
        bad["call"] += 1
        if bad["call"] == 2:
            request = httpx.Request("POST", "https://gateway.example/v1")
            response = httpx.Response(429, text="rate limited", request=request)
            error = httpx.HTTPStatusError("429", request=request, response=response)
            error.attempts = 3
            raise error
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "ok", "tag": "assistant"})
        return "ok", {}, index

    monkeypatch.setattr(rp, "inject_paste", flaky)
    path = rp.main(seed=7)
    _, rows = _rows(path)

    assert len(rows) == len(plan_phase0(7)), "una celda rota no puede acortar la tirada"
    failed = [r for r in rows if r["status"] != "ok"]
    assert len(failed) == 1
    assert failed[0]["status"] == "http_error"
    assert failed[0]["error_code"] == 429
    assert failed[0]["error_body"] == "rate limited"
    assert failed[0]["attempts"] == 3
    # Lo que se sabía antes de reventar se queda en la fila.
    assert failed[0]["prefix_id"].startswith("pfx-")
    assert failed[0]["artifact_id"]


def test_an_empty_reaction_is_not_an_ok(harness, monkeypatch):
    def silent(model_id, transcript, artifact):
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "  ", "tag": "assistant"})
        return "  ", {}, index

    monkeypatch.setattr(rp, "inject_paste", silent)
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["status"] == "empty" for r in pasted)


def test_a_timeout_is_recorded_as_a_timeout(harness, monkeypatch):
    import httpx

    def slow(model_id, transcript, artifact):
        raise httpx.ReadTimeout("demasiado lento")

    monkeypatch.setattr(rp, "inject_paste", slow)
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["status"] == "timeout" for r in pasted)


def test_main_writes_the_final_summary(harness, capsys):
    path = rp.main(seed=7)
    summary = json.loads(
        rp.summary_path(path.stem).read_text(encoding="utf-8")
    )

    assert summary["planned"] == len(plan_phase0(7))
    assert summary["completed"] == summary["planned"]
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    out = capsys.readouterr().out
    assert "planificadas" in out and "completadas" in out and "fallidas" in out
