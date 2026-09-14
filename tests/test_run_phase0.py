"""Tests del runner de la Fase 0. Ninguno hace una sola llamada a un modelo.

Todo lo que sale a la red —`ensure_prefix`, `rank_artifacts`, `inject_paste`,
`continue_after_paste`, `chat`— se sustituye por un doble determinista. Lo que se
prueba aquí es el diseño: la rotación de ejes (D3), la cobertura de géneros (D4),
la identidad y la trazabilidad de las filas (D5/D9), que un fallo escriba fila y
que la tirada se pueda reanudar de verdad (D6), los dos turnos posteriores (D7),
la sonda de caché (D10), las celdas de control (D11) y la medición del ancho del
eje en las dos longitudes (D12).
"""

import json
import warnings
from collections import Counter

import httpx
import numpy as np
import pytest

import wrongpaste.conversation as conv
import wrongpaste.prefixes as pfx
import wrongpaste.run_phase0 as rp
import wrongpaste.simulated_user as su
from wrongpaste import records
from wrongpaste.artifacts import Artifact, load_artifacts
from wrongpaste.clients import Reply
from wrongpaste.run_phase0 import (
    LENGTHS,
    PHASE0_MODELS,
    STRATA,
    STRATUM_SHIFTS,
    pick_artifact,
    plan_phase0,
    stratum_window,
)
from wrongpaste.topics import Topic, load_topics

TOPICS = load_topics()
ARTS = load_artifacts(level="N0")
ALL_KINDS = {a.kind for a in ARTS}


def _fake_load_artifacts(level: str | None = None) -> list[Artifact]:
    """Doble de `load_artifacts` con la firma de verdad, y exigente con ella.

    Desde la Fase 1 hay dos bancos en disco y `load_artifacts()` sin nivel
    devuelve los dos. La Fase 0 solo puede ver el N0, así que el doble se queja
    en voz alta si algún día deja de pedirlo: un doble más permisivo que la
    función real dejaría pasar justo el fallo que tiene que cazar.
    """
    assert level == "N0", f"la Fase 0 solo usa el banco N0, no {level!r}"
    return list(FAKE_ARTS)


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


def test_there_is_one_stratum_shift_per_model():
    """Un desplazamiento por modelo, y los tres distintos (tres temas/estrato)."""
    assert len(STRATUM_SHIFTS) == len(PHASE0_MODELS)
    assert len({s % STRATA for s in STRATUM_SHIFTS}) == len(PHASE0_MODELS)


def test_the_stratum_shifts_are_not_a_linear_step():
    """Ningún `(t + c·m) % 8` sirve, y los desplazamientos no son uno de ellos.

    Con `c` impar la paridad del estrato es el índice de longitud; con `c` par
    los tres temas de un estrato tienen todos la misma paridad. Los dos fallos
    se reconocen en la propia tupla: si los desplazamientos fueran `c·m`,
    existiría un `c` que los generase.
    """
    assert not any(
        all(shift % STRATA == (c * m) % STRATA for m, shift in enumerate(STRATUM_SHIFTS))
        for c in range(STRATA)
    ), "los desplazamientos son un salto lineal c·m: vuelve la confusión de ejes"


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
            assert cell["stratum"] == (t + STRATUM_SHIFTS[m]) % STRATA
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


def test_every_stratum_covers_topics_of_both_parities():
    """(a) Estrato × paridad del tema, sobre el plan de verdad.

    El agujero del arreglo anterior. Con `stratum = (t + 2·m) % 8` los tres
    temas de un estrato eran `t = s, s−2, s−4`: **todos de la paridad de `s`**.
    Los estratos pares solo veían temas pares y los impares solo impares, así
    que saber el estrato seguía siendo saber medio banco de temas, y cualquier
    efecto del tema (dominio, longitud del léxico, lo que sea) volvía a viajar
    pegado al eje de similaridad.

    Con desplazamientos (0, 3, 5) los temas de un estrato son `{s, s−3, s−5}` y
    sus paridades `{s, s+1, s+1}`: mixtas siempre.
    """
    plan = _paste_cells(plan_phase0(seed=1))
    index_of = {topic.id: t for t, topic in enumerate(TOPICS)}
    per_stratum: dict[int, set[int]] = {}
    for cell in plan:
        per_stratum.setdefault(cell["stratum"], set()).add(
            index_of[cell["topic_id"]] % 2
        )
    assert set(per_stratum) == set(range(STRATA))
    for stratum, parities in sorted(per_stratum.items()):
        assert parities == {0, 1}, (
            f"el estrato {stratum} solo cae en temas de paridad {parities}: "
            "estrato y tema siguen confundidos"
        )


def test_every_stratum_crosses_both_lengths():
    """(b) Estrato × longitud, sobre el plan de verdad.

    Era el único par de ejes sin comprobar, y por eso la suite en verde
    certificaba una rotación incompleta: con `stratum = (t + 3·m) % 8` los ocho
    estratos existían, cada uno caía en tres temas... y los pares salían
    **siempre** en conversaciones de 2 turnos y los impares **siempre** en las de
    10. Saber el estrato era saber la longitud, así que cualquier efecto de la
    similaridad y cualquier efecto de la longitud de la conversación quedaban
    pegados el uno al otro sin forma de separarlos después.

    Con desplazamientos (0, 3, 5) las longitudes de un estrato son `{s, s, s+1}`:
    las dos, siempre.
    """
    plan = _paste_cells(plan_phase0(seed=1))
    per_stratum: dict[int, set[int]] = {}
    for cell in plan:
        per_stratum.setdefault(cell["stratum"], set()).add(cell["n_turns"])
    assert set(per_stratum) == set(range(STRATA))
    for stratum, lengths in sorted(per_stratum.items()):
        assert lengths == set(LENGTHS), (
            f"el estrato {stratum} solo aparece con n_turns={sorted(lengths)}: "
            "estrato y longitud están confundidos"
        )


def test_the_parity_of_the_stratum_does_not_determine_the_length():
    """La forma concreta en que fallaba la fórmula impar, escrita como test."""
    plan = _paste_cells(plan_phase0(seed=1))
    for parity in (0, 1):
        lengths = {c["n_turns"] for c in plan if c["stratum"] % 2 == parity}
        assert lengths == set(LENGTHS), (
            f"los estratos de paridad {parity} solo salen con {sorted(lengths)}"
        )


def test_the_models_are_balanced_across_the_plan():
    """(c) Ningún modelo corre más celdas que otro: ni con pegote ni de control.

    Si un modelo llevara más celdas, llevaría también más estratos, más temas o
    más longitudes que los demás, y la comparación entre curvas —que es el eje
    declarado del spec §9— compararía además tamaños de muestra distintos.
    """
    plan = plan_phase0(seed=1)
    con_pegote = Counter(c["model_id"] for c in _paste_cells(plan))
    assert set(con_pegote) == set(PHASE0_MODELS)
    assert len(set(con_pegote.values())) == 1, f"plantel desequilibrado: {con_pegote}"
    assert set(con_pegote.values()) == {len(TOPICS)}
    control = Counter(c["model_id"] for c in _control_cells(plan))
    assert set(control) == set(PHASE0_MODELS)
    assert len(set(control.values())) == 1, f"control desequilibrado: {control}"


def test_every_stratum_lands_on_three_different_topics():
    plan = _paste_cells(plan_phase0(seed=1))
    per_stratum: dict[int, set[str]] = {}
    for cell in plan:
        per_stratum.setdefault(cell["stratum"], set()).add(cell["topic_id"])
    assert set(per_stratum) == set(range(STRATA))
    assert all(len(topics) == 3 for topics in per_stratum.values())


def test_every_model_walks_the_eight_strata():
    plan = _paste_cells(plan_phase0(seed=1))
    for model_id in PHASE0_MODELS:
        strata = {c["stratum"] for c in plan if c["model_id"] == model_id}
        assert strata == set(range(STRATA)), (
            f"{model_id} no recorre los ocho estratos: {sorted(strata)}"
        )


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


def test_phase0_has_no_replicates_and_says_what_phase1_must_do():
    """D1: las réplicas comparten prefijo Y artefacto. La Fase 0 no las usa.

    No se inventa el mecanismo aquí: se comprueba que no hay réplicas y que el
    docstring deja escrito el enganche que la Fase 1 tendrá que resolver —el
    artefacto tiene que pasar a ser función de `(prefix_id, stratum)` y no del
    seed de la celda, o dos réplicas elegirían artefactos distintos y dejarían
    de ser réplicas.
    """
    plan = plan_phase0(seed=1)
    assert {c["replicate_idx"] for c in plan} == {0}
    assert all(c["conversation_id"].endswith("-r0") for c in plan)

    doc = plan_phase0.__doc__ or ""
    assert "réplicas" in doc.lower()
    assert "prefix_id" in doc and "stratum" in doc
    assert "Fase 1" in doc


# --- D11: brazo de control -------------------------------------------------


def test_plan_includes_three_control_cells_without_paste():
    controls = _control_cells(plan_phase0(seed=1))
    assert len(controls) == rp.N_CONTROL_CELLS == 3
    assert all(c["condition"] == "no_paste" for c in controls)
    assert all(c["stratum"] == -1 for c in controls)


def test_control_cells_spread_over_the_three_models():
    controls = _control_cells(plan_phase0(seed=1))
    assert {c["model_id"] for c in controls} == set(PHASE0_MODELS)


def test_control_cells_cover_both_lengths():
    """Sin control corto, la tasa base no vale para la mitad de las celdas."""
    controls = _control_cells(plan_phase0(seed=1))
    assert {c["n_turns"] for c in controls} == set(LENGTHS)


def test_control_cells_use_distinct_topics():
    controls = _control_cells(plan_phase0(seed=1))
    assert len({c["topic_id"] for c in controls}) == len(controls)


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


# --- D13: la huella de los temas incluye la tarea verificable --------------


def test_topics_sha_covers_the_verifiable_task():
    """Dos tiradas con temas distintos no pueden declarar el mismo sha."""
    base = [Topic(id="t", opening="o", goals=("a",))]
    con_tarea = [
        Topic(id="t", opening="o", goals=("a",), task="calcula el total")
    ]
    con_esperado = [
        Topic(id="t", opening="o", goals=("a",), task="calcula el total", expected="42")
    ]
    con_verificador = [
        Topic(
            id="t",
            opening="o",
            goals=("a",),
            task="calcula el total",
            expected="42",
            verifier="igualdad",
        )
    ]
    shas = [
        rp.topics_sha(base),
        rp.topics_sha(con_tarea),
        rp.topics_sha(con_esperado),
        rp.topics_sha(con_verificador),
    ]
    assert len(set(shas)) == len(shas)


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


def _fill(request_params_out, model_id="claude-opus-5", **extra):
    """Emula lo que hace `chat()`: vacía el dict y escribe el cuerpo enviado.

    Construye el cuerpo con los MISMOS constructores que usa producción, en vez
    de fabricar siempre uno con forma de Claude. El doble anterior mentía: daba
    `max_tokens` + `thinking` también para los modelos del gateway, que en
    realidad mandan `max_completion_tokens` + `temperature` y ningún `thinking`.
    Un test escrito contra aquel doble certificaba una forma de cuerpo que 18 de
    las 27 filas de la tirada real no tienen.
    """
    if request_params_out is None:
        return
    from wrongpaste import clients
    from wrongpaste.config import MODELS

    mensajes = [{"role": "user", "content": "x"}]
    provider = MODELS[model_id].provider
    if provider == "gateway":
        cuerpo = clients._gateway_body(model_id, mensajes, rp.MAX_TOKENS)
    elif provider == "vertex_openai":
        cuerpo = clients._vertex_openai_body(model_id, mensajes, rp.MAX_TOKENS)
    else:
        cuerpo = clients._anthropic_body(mensajes, rp.MAX_TOKENS)
    cuerpo.pop("messages", None)
    request_params_out.clear()
    request_params_out.update({**cuerpo, **extra})


def _fake_reply(model_id, text, *, stop_reason="end_turn", attempts=1):
    return Reply(
        text=text,
        usage={"total_tokens": 5},
        raw={},
        stop_reason=stop_reason,
        response_model=f"{model_id}-20260101",
        attempts=attempts,
        latency_ms=12.5,
    )


def _fake_user_reply(text="sigo", *, stop_reason="stop"):
    """El `Reply` de un turno del usuario simulado, que no es el modelo evaluado."""
    return Reply(
        text=text,
        usage={"total_tokens": 2},
        raw={},
        stop_reason=stop_reason,
        response_model=f"{rp.USER_MODEL}-20260101",
        attempts=1,
        latency_ms=3.5,
    )


def _append_post_turns(
    model_id,
    transcript,
    n_post,
    *,
    stop_reason="end_turn",
    text="ya",
    user_replies_out=None,
    user_text="sigo",
    user_stop_reason="stop",
):
    """Doble de lo que hace `continue_after_paste` turno a turno (D7).

    Apila también el `Reply` del **usuario simulado** en `user_replies_out`,
    como el original: sin eso, el doble escondería justo la mitad de la
    conversación cuyos fallos eran invisibles.
    """
    indices, usages, replies = [], [], []
    for _ in range(n_post):
        if user_replies_out is not None:
            user_replies_out.append(
                _fake_user_reply(user_text, stop_reason=user_stop_reason)
            )
        indices.append(len(transcript))
        transcript.append({"role": "user", "content": user_text, "tag": "post"})
        indices.append(len(transcript))
        transcript.append({"role": "assistant", "content": text, "tag": "assistant"})
        usages.append({"total_tokens": 3})
        replies.append(_fake_reply(model_id, text, stop_reason=stop_reason))
    return indices, usages, replies


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Sustituye las puertas de red del runner y lo aísla en `tmp_path`."""
    calls: dict[str, list] = {"paste": [], "post": [], "rank": [], "prefix": []}

    def fake_ensure_prefix(topic, n_turns):
        calls["prefix"].append((topic.id, n_turns))
        return _prefix_for(topic, n_turns)

    def fake_rank_artifacts(text, arts, **kwargs):
        calls["rank"].append(text)
        n = len(arts)
        return [(a, i / (n - 1)) for i, a in enumerate(arts)], False

    def fake_inject_paste(model_id, transcript, artifact, request_params_out=None):
        calls["paste"].append((model_id, artifact.id))
        _fill(request_params_out, model_id)
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        text = f"reacción de {model_id}"
        transcript.append({"role": "assistant", "content": text, "tag": "assistant"})
        return text, {"total_tokens": 5}, index, _fake_reply(model_id, text)

    def fake_continue(
        model_id,
        transcript,
        topic,
        n_post=2,
        request_params_out=None,
        user_replies_out=None,
    ):
        calls["post"].append((model_id, n_post))
        _fill(request_params_out, model_id)
        return _append_post_turns(
            model_id, transcript, n_post, user_replies_out=user_replies_out
        )

    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", _fake_load_artifacts)
    monkeypatch.setattr(rp, "ensure_prefix", fake_ensure_prefix)
    # `ensure_prefix` va doblado, así que `find_prefix` —que `measure_axis` usa
    # solo para contar cuántos prefijos hubo que generar— no puede irse a mirar
    # el `runs/prefixes` del repo: el coste que reportaría sería el de la
    # máquina de quien corre la suite, no el de la tirada.
    monkeypatch.setattr(rp, "find_prefix", lambda *a, **k: None)
    monkeypatch.setattr(rp, "rank_artifacts", fake_rank_artifacts)
    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue)
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase0")
    return calls


def _rows(path):
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    return lines[0], lines[1:]


@pytest.fixture
def real_flow(tmp_path, monkeypatch):
    """Corta solo las puertas de red y deja correr el camino de verdad.

    El fixture `harness` dobla `ensure_prefix`, `inject_paste` y
    `continue_after_paste`. Va bien para probar el plan y el formato de las
    filas, pero es **ciego al gasto y al usuario simulado**: lo que gasta y quien
    escribe los turnos de usuario son exactamente las funciones que sustituye.

    Aquí se dobla solo lo que sale a la red —`clients.chat`, en los dos espacios
    de nombres donde se usa (`conversation` para el modelo evaluado y el
    asistente del prefijo, `simulated_user` para el usuario), y el embedder— y
    corre de verdad todo lo demás: `prefixes.ensure_prefix`,
    `conversation.build_prefix`, `inject_paste`, `continue_after_paste` y
    `simulated_user.next_user_turn`. Así cada llamada a un modelo queda contada,
    con el `model_id` que la pagó, y se puede afirmar quién gastó qué.

    `estado` permite romper el turno del usuario simulado a mitad de la prueba
    (texto vacío, `stop_reason` de corte) sin tocar el prefijo ya fabricado.
    """
    estado = {"user_text": "sigo con lo mío", "user_stop_reason": "stop"}
    llamadas: dict[str, list] = {"model": [], "user": [], "rank": []}

    def fake_chat(model_id, messages, max_tokens=1024, request_params_out=None):
        llamadas["model"].append(model_id)
        if request_params_out is not None:
            request_params_out.clear()
            request_params_out.update(
                {"max_tokens": max_tokens, "thinking": {"type": "adaptive"}}
            )
        return Reply(
            text=f"respuesta de {model_id}",
            usage={"total_tokens": 5},
            raw={},
            stop_reason="end_turn",
            response_model=f"{model_id}-20260101",
            attempts=1,
            latency_ms=9.0,
        )

    def fake_user_chat(model_id, messages, max_tokens=None, **kwargs):
        llamadas["user"].append(model_id)
        return Reply(
            text=estado["user_text"],
            usage={"total_tokens": 2},
            raw={},
            stop_reason=estado["user_stop_reason"],
            response_model=f"{model_id}-20260101",
            attempts=1,
            latency_ms=3.0,
        )

    def fake_rank(text, arts, **kwargs):
        llamadas["rank"].append(text)
        n = len(arts)
        return [(a, i / (n - 1)) for i, a in enumerate(arts)], False

    monkeypatch.setattr(conv, "chat", fake_chat)
    monkeypatch.setattr(su, "chat", fake_user_chat)
    monkeypatch.setattr(rp, "rank_artifacts", fake_rank)
    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", _fake_load_artifacts)
    monkeypatch.setattr(pfx, "PREFIX_DIR", tmp_path / "prefixes")
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase0")
    return {"estado": estado, "llamadas": llamadas, "tmp_path": tmp_path}


# --- D12: ancho del eje ----------------------------------------------------


def test_measure_axis_reports_min_median_and_max_per_topic_and_length(harness, capsys):
    axis = rp.measure_axis(topics=FAKE_TOPICS, arts=FAKE_ARTS, run_id="r1")

    assert axis["run_id"] == "r1"
    assert axis["lengths"] == LENGTHS
    assert [(e["topic_id"], e["n_turns"]) for e in axis["entries"]] == [
        (t.id, n) for t in FAKE_TOPICS for n in LENGTHS
    ]
    for entry in axis["entries"]:
        assert entry["min"] <= entry["median"] <= entry["max"]
        assert entry["range"] == pytest.approx(entry["max"] - entry["min"])
        assert len(entry["top"]) == rp.AXIS_TOP_N
    out = capsys.readouterr().out
    assert "tema-0" in out and "med=" in out and "n=10" in out


def test_measure_axis_measures_both_lengths(harness):
    """El criterio GO/NO-GO no puede medirse solo sobre los prefijos cortos."""
    rp.measure_axis(topics=FAKE_TOPICS[:2], arts=FAKE_ARTS, run_id="r1")

    assert harness["prefix"] == [
        ("tema-0", 2), ("tema-0", 10), ("tema-1", 2), ("tema-1", 10)
    ]
    assert len(harness["rank"]) == 4


def test_measure_axis_with_one_length_reports_only_that_one(harness):
    axis = rp.measure_axis(
        topics=FAKE_TOPICS[:1], arts=FAKE_ARTS, lengths=10, run_id="r1"
    )
    assert axis["lengths"] == [10]
    assert [e["n_turns"] for e in axis["entries"]] == [10]


def test_measure_axis_embeds_the_user_side_of_the_prefix(harness):
    rp.measure_axis(topics=FAKE_TOPICS[:1], arts=FAKE_ARTS, run_id="r1")
    # D2: se embebe el lado del usuario, no la prosa del asistente.
    assert harness["rank"] == ["apertura de tema-0", "apertura de tema-0"]


def test_measure_axis_flags_a_narrow_topic(harness, monkeypatch):
    def flat(text, arts, **kwargs):
        return [(a, 0.5 + i * 0.001) for i, a in enumerate(arts)], False

    monkeypatch.setattr(rp, "rank_artifacts", flat)
    axis = rp.measure_axis(topics=FAKE_TOPICS[:1], arts=FAKE_ARTS, run_id="r1")

    assert all(e["narrow"] is True for e in axis["entries"])
    assert axis["narrow_topics"] == ["tema-0"]
    assert axis["narrow_cells"] == [
        {"topic_id": "tema-0", "n_turns": 2},
        {"topic_id": "tema-0", "n_turns": 10},
    ]


def test_main_writes_the_axis_report_for_both_lengths(harness):
    path = rp.main(seed=7)
    axis_file = rp.axis_path(path.stem, out=path)
    assert axis_file.exists()
    axis = json.loads(axis_file.read_text(encoding="utf-8"))
    assert {e["n_turns"] for e in axis["entries"]} == set(LENGTHS)


def test_main_stops_before_spending_when_the_axis_is_narrow(real_flow, monkeypatch):
    """D12 dice GO/NO-GO, así que tiene que frenar, no solo imprimir «ESTRECHO».

    **Este test se escribió dos veces, y la primera no valía.** Doblaba
    `run_cell` y comprobaba que no se llamaba, dentro del fixture `harness`, que
    dobla `ensure_prefix`: era estructuralmente incapaz de ver el gasto, porque
    lo que gasta es justamente lo que el doble sustituía. El docstring de
    `main()` decía entonces que la puerta frena «antes de gastar un céntimo en
    modelos» y ningún test podía desmentirlo.

    Aquí corre el camino real (ver el fixture `real_flow`) y se cuentan las
    llamadas una a una. Lo que se afirma es la verdad, no el eslogan:

    - los 16 prefijos **sí** se pagan, porque para medir el eje hay que tenerlos;
    - los tres modelos evaluados **no** reciben ni una llamada, que es lo caro y
      es lo que la puerta ahorra.
    """
    monkeypatch.setattr(rp, "NARROW_RANGE", 2.0)
    out = rp.OUT_DIR / "estrecho.jsonl"

    with pytest.raises(rp.NarrowAxisError) as exc:
        rp.main(seed=7, out=out)

    llamadas = real_flow["llamadas"]
    # Lo que la puerta ahorra: las 27 celdas contra los modelos evaluados.
    assert not any(m in PHASE0_MODELS for m in llamadas["model"]), (
        "se gastó dinero en los modelos evaluados con el eje estrecho"
    )
    # Lo que la puerta NO ahorra, y ahora se ve: los 16 prefijos compartidos.
    prefijos = [(t, n) for t in FAKE_TOPICS for n in LENGTHS]
    assert llamadas["model"] == [pfx.PREFIX_MODEL] * sum(n for _, n in prefijos)
    assert llamadas["user"] == [su.USER_MODEL] * sum(n - 1 for _, n in prefijos)
    assert len(list((rp.OUT_DIR.parent / "prefixes").glob("*.json"))) == len(prefijos)

    assert exc.value.narrow_cells, "el error no dice qué celdas son estrechas"
    # El informe queda escrito: es la evidencia de por qué no se corrió, y dice
    # cuánto costó medir.
    axis = json.loads(
        rp.axis_path(out.stem, out=out).read_text(encoding="utf-8")
    )
    assert axis["narrow_cells"] == exc.value.narrow_cells
    assert axis["prefixes_generated"] == len(prefijos)
    assert axis["prefixes_reused"] == 0
    # Y no queda un JSONL a medias que luego parezca una tirada reanudable.
    assert not out.exists() or not out.read_text(encoding="utf-8").strip()


def test_a_wide_axis_does_reach_the_evaluated_models(real_flow):
    """El contraste que hace falsable al test de arriba.

    Si la puerta no frenara —o si el recuento de llamadas no viera nada—, este
    test y el anterior darían lo mismo. Aquí el eje es ancho, la tirada corre
    entera y cada modelo evaluado recibe exactamente las llamadas de sus celdas:
    tres por celda con pegote (pegote + los dos turnos de D7) y dos por celda de
    control (D11, sin pegote).
    """
    rp.main(seed=7)

    por_modelo = Counter(real_flow["llamadas"]["model"])
    plan = plan_phase0(7)
    for model_id in PHASE0_MODELS:
        con_pegote = len(
            [c for c in _paste_cells(plan) if c["model_id"] == model_id]
        )
        control = len([c for c in _control_cells(plan) if c["model_id"] == model_id])
        assert por_modelo[model_id] == con_pegote * (1 + rp.N_POST_TURNS) + (
            control * rp.N_POST_TURNS
        )


def test_a_narrow_axis_can_be_forced_on_purpose(harness, monkeypatch, capsys):
    monkeypatch.setattr(rp, "NARROW_RANGE", 2.0)
    path = rp.main(seed=7, out=rp.OUT_DIR / "forzado.jsonl", allow_narrow=True)
    _, rows = _rows(path)

    assert len(rows) == len(plan_phase0(7))
    assert "allow_narrow" in capsys.readouterr().out


def test_a_wide_axis_does_not_stop_anything(harness):
    """El camino normal: con el eje ancho, `main` no levanta nada."""
    path = rp.main(seed=7, out=rp.OUT_DIR / "ancho.jsonl")
    _, rows = _rows(path)
    assert len(rows) == len(plan_phase0(7))


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


def test_the_header_names_both_the_design_and_the_decisions(harness):
    """El diseño dice de qué va; las correcciones son las que gobiernan la tirada.

    `spec_path` apuntaba solo al documento de diseño, y el que fija la rotación
    de ejes, el vocabulario de `status` y el criterio GO/NO-GO es el de
    correcciones. Quien lea el JSONL dentro de seis meses necesita los dos.
    """
    header, _ = _rows(rp.main(seed=7))

    assert header["spec_path"].endswith("-design.md")
    assert header["corrections_path"].endswith("-correcciones.md")
    assert header["spec_path"] != header["corrections_path"]
    assert header["plan_path"].endswith("-fase-0.md")


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


# --- D5/D9: la fila dice lo que se envió y lo que contestó el proveedor ----


def test_rows_carry_the_request_params_that_were_really_sent(harness):
    """Antes salían 27 filas con `request_params: {}`."""
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    from wrongpaste.config import MODELS

    vistos = set()
    for row in rows:
        params = row["request_params"]
        assert params, "la fila no dice con qué cuerpo se corrió"
        assert "messages" not in params

        provider = MODELS[row["model_id"]].provider
        vistos.add(provider)
        if provider == "gateway":
            # D9: temperatura explícita, y el tope se llama distinto aquí.
            assert params["max_completion_tokens"] == rp.MAX_TOKENS
            assert params["temperature"] == 1.0
            assert "thinking" not in params
        elif provider == "vertex_openai":
            assert params["max_tokens"] == rp.MAX_TOKENS
            assert params["temperature"] == 1.0
            assert "thinking" not in params
        else:
            # Claude 5: thinking adaptativo y NADA de muestreo explícito (400).
            assert params["max_tokens"] == rp.MAX_TOKENS
            assert params["thinking"] == {"type": "adaptive"}
            assert "temperature" not in params

    # El test no vale de nada si todas las filas son del mismo proveedor.
    assert len(vistos) >= 2, f"solo se ejercitó {vistos}"


def test_rows_carry_stop_reasons_response_model_and_attempts(harness):
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    for row in rows:
        esperados = 3 if row["condition"] == "paste" else 2
        assert len(row["stop_reasons"]) == esperados, (
            "un stop_reason por turno del modelo evaluado, en orden"
        )
        assert all(reason == "end_turn" for reason in row["stop_reasons"])
        assert row["response_model"] == f"{row['model_id']}-20260101"
        assert row["attempts"] == 1


def test_a_retried_turn_is_visible_in_the_row(harness, monkeypatch):
    def con_reintentos(model_id, transcript, artifact, request_params_out=None):
        _fill(request_params_out, model_id)
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "ok", "tag": "assistant"})
        reply = _fake_reply(model_id, "ok", attempts=3)
        return "ok", {}, index, reply

    monkeypatch.setattr(rp, "inject_paste", con_reintentos)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["attempts"] == 3 for r in pasted)


def test_the_system_prompt_comes_from_what_was_really_sent():
    assert rp._system_prompt({}, []) is None
    assert rp._system_prompt({"system": "reglas"}, []) == "reglas"
    assert rp._system_prompt({}, [{"role": "system", "content": "en línea"}]) == (
        "en línea"
    )
    # El lado del usuario no es un system prompt.
    assert rp._system_prompt({}, [{"role": "user", "content": "hola"}]) is None


def test_a_system_prompt_reaches_the_row_when_the_body_carries_one(
    harness, monkeypatch
):
    def con_sistema(
        model_id,
        transcript,
        topic,
        n_post=2,
        request_params_out=None,
        user_replies_out=None,
    ):
        _fill(request_params_out, model_id, system="instrucciones del sistema")
        return _append_post_turns(
            model_id, transcript, n_post, user_replies_out=user_replies_out
        )

    monkeypatch.setattr(rp, "continue_after_paste", con_sistema)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert all(r["system_prompt"] == "instrucciones del sistema" for r in rows)


def test_phase0_rows_have_no_system_prompt_because_none_is_sent(harness):
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    assert all(r["system_prompt"] is None for r in rows)


def test_the_two_truncation_flags_are_recorded_separately(harness, monkeypatch):
    """D2: solo la conversación entera se pasa del tope, y eso no invalida el eje."""

    def rank(text, arts, **kwargs):
        n = len(arts)
        # `prosa` solo está en la conversación entera, no en el lado de usuario.
        return [(a, i / (n - 1)) for i, a in enumerate(arts)], "prosa" in text

    monkeypatch.setattr(rp, "rank_artifacts", rank)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert pasted
    for row in pasted:
        assert row["similarity_full_truncated"] is True
        assert row["similarity_user_truncated"] is False
        # La columna derivada sigue estando, para quien solo quiera el OR.
        assert row["similarity_text_truncated"] is True


def test_the_runner_does_not_use_the_deprecated_collapsed_flag(harness):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rp.main(seed=7, measure=False)


# --- D6: reanudación de verdad --------------------------------------------


def _write_rows(harness, rows, name="resume.jsonl"):
    path = rp.OUT_DIR / name
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
                {"conversation_id": "p0-x", "artifact_kind": "recipe", "status": "ok"},
                {"conversation_id": "p0-y", "artifact_kind": "sql", "status": "ok"},
                {"conversation_id": "p0-z", "artifact_kind": "recipe", "status": "ok"},
            ],
        )
    )
    assert done == {"p0-x", "p0-y", "p0-z"}
    assert kinds == Counter({"recipe": 2, "sql": 1})


def test_only_ok_and_refusal_count_as_done(harness):
    """Una celda muerta por un 429 tiene que volver a correrse."""
    done, kinds = rp.resume_state(
        _write_rows(
            harness,
            [
                {"conversation_id": "p0-ok", "artifact_kind": "recipe", "status": "ok"},
                {
                    "conversation_id": "p0-neg",
                    "artifact_kind": "prompt",
                    "status": "refusal",
                },
                {
                    "conversation_id": "p0-429",
                    "artifact_kind": "sql",
                    "status": "http_error",
                },
                {
                    "conversation_id": "p0-lento",
                    "artifact_kind": "config",
                    "status": "timeout",
                },
                {
                    "conversation_id": "p0-mudo",
                    "artifact_kind": "email",
                    "status": "empty",
                },
            ],
        )
    )
    assert done == {"p0-ok", "p0-neg"}
    # Los géneros de las celdas que se van a reintentar no se gastan dos veces.
    assert kinds == Counter({"recipe": 1, "prompt": 1})


def test_a_row_without_status_is_treated_as_done(harness):
    """Formato anterior a D6: el esquema tenía `ok` por defecto."""
    done, _ = rp.resume_state(
        _write_rows(harness, [{"conversation_id": "p0-viejo"}], name="viejo.jsonl")
    )
    assert done == {"p0-viejo"}


def _partial_with_a_dead_cell(harness, name="reintento.jsonl"):
    """Una tirada completa a la que se le rompe una celda y se trunca al 5.º."""
    path = rp.main(seed=7)
    header, rows = _rows(path)
    kept = [dict(r) for r in rows[:5]]
    kept[2]["status"] = "http_error"
    kept[2]["error_code"] = 429

    partial = path.parent / name
    partial.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [header, *kept]) + "\n",
        encoding="utf-8",
    )
    return partial, rows, kept[2]["conversation_id"]


def test_a_cell_that_died_on_a_transient_error_is_retried_on_resume(harness):
    partial, rows, muerta = _partial_with_a_dead_cell(harness)

    rp.main(seed=7, out=partial, measure=False)
    _, resumed = _rows(partial)

    por_id = {r["conversation_id"]: r for r in resumed}
    assert muerta in por_id
    assert por_id[muerta]["status"] == "ok", "la celda muerta no se reintentó"


def test_resuming_does_not_leave_two_rows_for_the_same_conversation(harness):
    """Reanudar dejaba 28 filas para 27 celdas, dos de ellas la misma.

    Con dos filas por conversación se rompe la única comprobación barata de
    integridad que da la cabecera (D5) —«¿hay tantas filas como
    `planned_cells`?»— y cualquier recuento cuenta dos veces justo las celdas
    que fallaron, que no son una muestra al azar: fallan más las conversaciones
    largas y los cosenos altos.
    """
    partial, rows, muerta = _partial_with_a_dead_cell(harness, name="sin-dobles.jsonl")

    rp.main(seed=7, out=partial, measure=False)
    header, resumed = _rows(partial)

    ids = [r["conversation_id"] for r in resumed]
    assert ids.count(muerta) == 1, "la fila vieja sigue dentro"
    assert len(ids) == len(set(ids)), "hay conversaciones repetidas"
    assert len(resumed) == header["planned_cells"] == len(plan_phase0(7))
    assert len(resumed) == len(rows)


def test_the_superseded_row_is_kept_out_of_the_way_not_deleted(harness):
    """La evidencia del 429 no se tira: se aparta del recuento."""
    partial, _, muerta = _partial_with_a_dead_cell(harness, name="apartadas.jsonl")

    rp.main(seed=7, out=partial, measure=False)

    header, _ = _rows(partial)
    trash = rp.superseded_path(header["run_id"], out=partial)
    retiradas = [
        json.loads(line)
        for line in trash.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["conversation_id"] for r in retiradas] == [muerta]
    assert retiradas[0]["status"] == "http_error"
    assert retiradas[0]["error_code"] == 429


def test_compaction_keeps_the_header_and_the_done_rows_in_order(harness):
    path = _write_rows(
        harness,
        [
            {"conversation_id": "p0-a", "status": "ok", "artifact_kind": "recipe"},
            {"conversation_id": "p0-b", "status": "http_error"},
            {"conversation_id": "p0-c", "status": "refusal"},
        ],
        name="compactar.jsonl",
    )
    retiradas = rp.compact_resume_file(path, out=path)

    header, filas = _rows(path)
    assert header["kind"] == "run_header"
    assert [r["conversation_id"] for r in filas] == ["p0-a", "p0-c"]
    assert [r["conversation_id"] for r in retiradas] == ["p0-b"]


def test_compaction_leaves_a_clean_file_untouched(harness):
    path = _write_rows(
        harness,
        [{"conversation_id": "p0-a", "status": "ok"}],
        name="limpio.jsonl",
    )
    antes = path.read_text(encoding="utf-8")
    assert rp.compact_resume_file(path, out=path) == []
    assert path.read_text(encoding="utf-8") == antes
    assert not rp.superseded_path("r0", out=path).exists()


def test_compaction_keeps_the_last_of_two_rows_for_the_same_cell(harness):
    """Ficheros de tiradas anteriores al arreglo: se desduplican al reanudar."""
    path = _write_rows(
        harness,
        [
            {"conversation_id": "p0-a", "status": "ok", "reaction": "vieja"},
            {"conversation_id": "p0-a", "status": "ok", "reaction": "nueva"},
        ],
        name="heredado.jsonl",
    )
    retiradas = rp.compact_resume_file(path, out=path)

    _, filas = _rows(path)
    assert [r["reaction"] for r in filas] == ["nueva"]
    assert [r["reaction"] for r in retiradas] == ["vieja"]


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


def test_an_empty_output_file_is_not_a_resume(harness, tmp_path):
    """Un fichero vacío dejaba la tirada entera sin `run_header`."""
    out = tmp_path / "vacio.jsonl"
    out.touch()
    assert rp.is_resumable(out) is False

    path = rp.main(seed=7, out=out, measure=False)
    header, rows = _rows(path)
    assert header["kind"] == "run_header"
    assert len(rows) == len(plan_phase0(7))


def test_a_file_with_rows_but_no_header_is_neither_resumed_nor_overwritten(
    harness, tmp_path
):
    out = tmp_path / "sin-cabecera.jsonl"
    original = json.dumps({"conversation_id": "p0-a", "status": "ok"}) + "\n"
    out.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="run_header"):
        rp.main(seed=7, out=out, measure=False)
    assert out.read_text(encoding="utf-8") == original


# --- D6: los fallos son datos ---------------------------------------------


def test_a_failing_cell_still_writes_a_row_and_the_run_goes_on(harness, monkeypatch):
    bad = {"call": 0}

    def flaky(model_id, transcript, artifact, request_params_out=None):
        bad["call"] += 1
        _fill(request_params_out, model_id)
        if bad["call"] == 2:
            request = httpx.Request("POST", "https://gateway.example/v1")
            response = httpx.Response(429, text="rate limited", request=request)
            error = httpx.HTTPStatusError("429", request=request, response=response)
            error.attempts = 3
            raise error
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "ok", "tag": "assistant"})
        return "ok", {}, index, _fake_reply(model_id, "ok")

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
    # Incluido el cuerpo que se llegó a enviar (D9).
    # El tope se llama distinto según el proveedor: el gateway usa
    # `max_completion_tokens` y los dos de Vertex `max_tokens`. Lo que importa
    # aquí es que una celda ROTA también dice con qué cuerpo se corrió.
    params_rotos = failed[0]["request_params"]
    tope = params_rotos.get("max_tokens", params_rotos.get("max_completion_tokens"))
    assert tope == rp.MAX_TOKENS


def test_an_empty_reaction_is_not_an_ok(harness, monkeypatch):
    def silent(model_id, transcript, artifact, request_params_out=None):
        _fill(request_params_out, model_id)
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "  ", "tag": "assistant"})
        return "  ", {}, index, _fake_reply(model_id, "  ", stop_reason="max_tokens")

    monkeypatch.setattr(rp, "inject_paste", silent)
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["status"] == "empty" for r in pasted)


def test_an_empty_answer_in_the_control_arm_is_not_an_ok_either(harness, monkeypatch):
    """La guarda de `empty` estaba condicionada a `condition == "paste"`.

    El brazo de control (D11) no tiene pegote: sus únicas respuestas son las de
    los turnos posteriores. Con la condición puesta, una conversación de control
    entera de respuestas vacías salía `ok` — y el control existe precisamente
    para dar la tasa base de fuga de entidades, que sobre respuestas que no
    existen es cero por construcción y nadie se entera al leer el fichero.
    """

    def mudo(
        model_id,
        transcript,
        topic,
        n_post=2,
        request_params_out=None,
        user_replies_out=None,
    ):
        _fill(request_params_out, model_id)
        return _append_post_turns(
            model_id,
            transcript,
            n_post,
            text="   ",
            user_replies_out=user_replies_out,
        )

    monkeypatch.setattr(rp, "continue_after_paste", mudo)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    control = [r for r in rows if r["condition"] == "no_paste"]

    assert control, "el plan no trae celdas de control que mirar"
    assert all(r["status"] == "empty" for r in control)


def test_a_timeout_is_recorded_as_a_timeout(harness, monkeypatch):
    def slow(model_id, transcript, artifact, request_params_out=None):
        raise httpx.ReadTimeout("demasiado lento")

    monkeypatch.setattr(rp, "inject_paste", slow)
    path = rp.main(seed=7)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["status"] == "timeout" for r in pasted)


def test_a_declared_refusal_is_a_result_not_a_failure(harness, monkeypatch):
    """D6: que un modelo se niegue es conducta, y cuenta como celda hecha."""

    def se_niega(model_id, transcript, artifact, request_params_out=None):
        _fill(request_params_out, model_id)
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        transcript.append({"role": "assistant", "content": "", "tag": "assistant"})
        return "", {}, index, _fake_reply(model_id, "", stop_reason="refusal")

    monkeypatch.setattr(rp, "inject_paste", se_niega)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert all(r["status"] == "refusal" for r in pasted)
    summary = json.loads(
        rp.summary_path(path.stem, out=path).read_text(encoding="utf-8")
    )
    assert summary["failed"] == 0
    assert summary["completed"] == summary["planned"]


# --- BLOQUEANTE: el origen del fallo, en el status -------------------------
#
# En una celda hablan tres modelos: el evaluado, el usuario simulado y el que
# fabricó el prefijo (los dos últimos, `gpt-5.6-terra-tst`). Solo el primero
# produce conducta. Si revienta cualquier otro y la fila lo llama `refusal`, el
# experimento se inventa una negativa que nadie dio.


def _content_filter_error() -> httpx.HTTPStatusError:
    """El 400 que devuelve un filtro de contenido, venga de quien venga."""
    return _http_error(400, json.dumps({"error": {"code": "content_filter"}}))


def test_a_failure_of_the_simulated_user_is_not_the_evaluated_models_conduct(
    harness, monkeypatch
):
    """El usuario simulado revienta antes de escribir su turno: arnés, no modelo.

    `continue_after_paste` añade el mensaje de usuario **después** de que
    `next_user_turn` haya devuelto, así que un transcript que sigue acabando en
    `assistant` dice que quien falló fue la llamada del usuario simulado.
    """

    def el_usuario_revienta(
        model_id,
        transcript,
        topic,
        n_post=2,
        request_params_out=None,
        user_replies_out=None,
    ):
        _fill(request_params_out, model_id)
        raise _content_filter_error()

    monkeypatch.setattr(rp, "continue_after_paste", el_usuario_revienta)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert {r["status"] for r in rows} == {"harness_error"}
    assert all(r["status"] != "refusal" for r in rows)
    # El qué pasó no se pierde: solo cambia de quién se dice que fue.
    assert all(r["error_code"] == 400 for r in rows)
    assert all("content_filter" in (r["error_body"] or "") for r in rows)


def test_a_refusal_of_the_evaluated_model_after_the_paste_is_still_conduct(
    harness, monkeypatch
):
    """Contraste: si el turno de usuario ya está escrito, el que falló es el modelo."""

    def el_modelo_revienta(
        model_id,
        transcript,
        topic,
        n_post=2,
        request_params_out=None,
        user_replies_out=None,
    ):
        _fill(request_params_out, model_id)
        transcript.append({"role": "user", "content": "sigo", "tag": "post"})
        raise _content_filter_error()

    monkeypatch.setattr(rp, "continue_after_paste", el_modelo_revienta)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert {r["status"] for r in rows} == {"refusal"}


def test_a_failure_building_the_prefix_is_a_harness_error(harness, monkeypatch):
    """El prefijo lo fabrican el usuario simulado y PREFIX_MODEL (D1)."""

    def sin_prefijo(topic, n_turns):
        raise _content_filter_error()

    monkeypatch.setattr(rp, "ensure_prefix", sin_prefijo)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert {r["status"] for r in rows} == {"harness_error"}


def test_a_failure_embedding_the_bank_is_a_harness_error(harness, monkeypatch):
    """El embedder tampoco es el modelo evaluado."""

    def sin_embeddings(text, arts, **kwargs):
        raise _content_filter_error()

    monkeypatch.setattr(rp, "rank_artifacts", sin_embeddings)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    pasted = [r for r in rows if r["condition"] == "paste"]
    assert pasted and {r["status"] for r in pasted} == {"harness_error"}


def test_the_two_extra_statuses_live_in_records_not_in_the_runner():
    """El vocabulario vive en `records`, sin depender de importar el runner.

    Antes, `run_phase0` ampliaba `records.STATUSES` como efecto secundario de
    ser importado. Un análisis que importara solo `records` —que es lo
    razonable, es la hoja del grafo— rechazaba como inválidas filas
    perfectamente buenas de su propio JSONL. Este test comprueba que el
    vocabulario está completo SIN pasar por el runner.
    """
    import subprocess
    import sys

    guion = (
        "from wrongpaste.records import ConversationRecord, STATUSES;"
        "assert 'harness_error' in STATUSES and 'truncated' in STATUSES;"
        "ConversationRecord(model_id='m', topic_id='t', status='harness_error');"
        "ConversationRecord(model_id='m', topic_id='t', status='truncated');"
        "print('ok')"
    )
    salida = subprocess.run(
        [sys.executable, "-c", guion], capture_output=True, text=True
    )
    assert salida.returncode == 0, salida.stderr
    assert "ok" in salida.stdout

    # Y el vocabulario sigue cerrado para lo que no es un status.
    from wrongpaste.records import ConversationRecord

    with pytest.raises(ValueError):
        ConversationRecord(model_id="m", topic_id="t", status="lo-que-sea")


def test_failure_origin_reads_the_stage_and_the_transcript():
    """La regla, aislada de la tirada."""
    asistente = [{"role": "assistant", "content": "x", "tag": "assistant"}]
    usuario = asistente + [{"role": "user", "content": "y", "tag": "post"}]

    assert rp.failure_origin(rp.STAGE_PREFIX, asistente) == rp.ORIGIN_HARNESS
    assert rp.failure_origin(rp.STAGE_RANK, asistente) == rp.ORIGIN_HARNESS
    assert rp.failure_origin(rp.STAGE_PASTE, usuario) == rp.ORIGIN_MODEL
    assert rp.failure_origin(rp.STAGE_POST, asistente) == rp.ORIGIN_HARNESS
    assert rp.failure_origin(rp.STAGE_POST, usuario) == rp.ORIGIN_MODEL
    # Sin etapa no hay conducta que atribuir todavía.
    assert rp.failure_origin("", None) == rp.ORIGIN_HARNESS


def test_a_harness_error_never_borrows_the_refusal_vocabulary():
    """Ni `refusal`, ni `timeout`, ni nada que se pueda leer como conducta."""
    status, code, _ = rp._status_for_error(
        _content_filter_error(), origin=rp.ORIGIN_HARNESS
    )
    assert status == "harness_error"
    assert code == 400

    lento, _, _ = rp._status_for_error(
        httpx.ReadTimeout("lento"), origin=rp.ORIGIN_HARNESS
    )
    assert lento == "harness_error"

    # Y con origen en el modelo evaluado, el vocabulario de siempre.
    assert rp._status_for_error(_content_filter_error())[0] == "refusal"
    assert rp._status_for_error(httpx.ReadTimeout("lento"))[0] == "timeout"


def test_a_harness_error_is_retried_on_resume(harness):
    """No es un resultado: es una avería, y la celda sigue sin correrse."""
    assert "harness_error" not in rp.DONE_STATUSES
    done, kinds = rp.resume_state(
        _write_rows(
            harness,
            [
                {"conversation_id": "p0-ok", "status": "ok", "artifact_kind": "sql"},
                {
                    "conversation_id": "p0-arnes",
                    "status": "harness_error",
                    "artifact_kind": "prompt",
                },
            ],
            name="arnes.jsonl",
        )
    )
    assert done == {"p0-ok"}
    assert kinds == Counter({"sql": 1})


# --- MEDIA: clasificar un fallo del arnés como negativa es inventar conducta


def _http_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gateway.example/v1")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


def test_a_malformed_request_mentioning_content_is_not_a_refusal():
    """La subcadena `content` sale en errores de petición corrientes."""
    body = json.dumps(
        {
            "error": {
                "message": "Invalid type for 'messages[0].content': expected string",
                "type": "invalid_request_error",
                "param": "messages[0].content",
                "code": "invalid_type",
            }
        }
    )
    status, code, _ = rp._status_for_error(_http_error(400, body))
    assert status == "http_error"
    assert code == 400


def test_a_non_json_body_with_the_word_content_is_not_a_refusal():
    status, _, _ = rp._status_for_error(
        _http_error(400, "bad request: missing content-length header")
    )
    assert status == "http_error"


def test_an_openai_content_filter_is_a_refusal():
    body = json.dumps(
        {"error": {"message": "filtrado", "code": "content_filter", "type": "error"}}
    )
    assert rp._status_for_error(_http_error(400, body))[0] == "refusal"


def test_a_nested_responsible_ai_violation_is_a_refusal():
    body = json.dumps(
        {
            "error": {
                "code": "BadRequest",
                "message": "la respuesta se ha filtrado",
                "innererror": {"code": "ResponsibleAIPolicyViolation"},
            }
        }
    )
    assert rp._status_for_error(_http_error(400, body))[0] == "refusal"


def test_a_vertex_block_reason_is_a_refusal():
    body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}})
    assert rp._status_for_error(_http_error(400, body))[0] == "refusal"


def test_a_server_error_is_never_a_refusal():
    body = json.dumps({"error": {"code": "internal", "type": "server_error"}})
    assert rp._status_for_error(_http_error(503, body))[0] == "http_error"


def test_the_body_is_analysed_whole_and_stored_truncated():
    """Recortar antes de mirar rompería el JSON de los errores largos."""
    relleno = "x" * (rp.ERROR_BODY_CHARS + 500)
    body = json.dumps({"error": {"message": relleno, "code": "content_filter"}})
    status, _, stored = rp._status_for_error(_http_error(400, body))
    assert status == "refusal"
    assert len(stored) == rp.ERROR_BODY_CHARS


def test_stop_reasons_decide_a_refusal_without_an_http_error():
    assert rp.status_for_stop_reasons(["end_turn", "refusal"]) == "refusal"
    assert rp.status_for_stop_reasons(["end_turn", "content_filter"]) == "refusal"
    assert rp.status_for_stop_reasons([None, None]) is None


def test_a_cut_by_max_tokens_is_not_an_ok():
    """El tope lo mandamos nosotros: una respuesta cortada no es conducta.

    Esta línea decía `is None`, o sea `ok`, para `max_tokens`. Una reacción
    cortada a media frase se lee luego como «el modelo lo ignoró y siguió a lo
    suyo», que es una de las cinco categorías del spec §5: el arnés se estaba
    inventando conducta.
    """
    assert rp.status_for_stop_reasons(["end_turn", "max_tokens"]) == "truncated"
    assert rp.status_for_stop_reasons(["length"]) == "truncated"
    assert rp.status_for_stop_reasons(["MAX_TOKENS"]) == "truncated"
    # Una negativa es conducta declarada y manda sobre el corte.
    assert rp.status_for_stop_reasons(["max_tokens", "refusal"]) == "refusal"


def test_a_truncated_row_says_so_instead_of_ok(harness, monkeypatch):
    """Sobre la tirada entera, no solo sobre la función."""

    def cortado(model_id, transcript, artifact, request_params_out=None):
        _fill(request_params_out, model_id)
        index = len(transcript)
        transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
        text = "empiezo a contestar y me qued"
        transcript.append({"role": "assistant", "content": text, "tag": "assistant"})
        return (
            text,
            {},
            index,
            _fake_reply(model_id, text, stop_reason="max_tokens"),
        )

    monkeypatch.setattr(rp, "inject_paste", cortado)
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)
    pasted = [r for r in rows if r["condition"] == "paste"]

    assert pasted, "no hay celdas con pegote que mirar"
    assert all(r["status"] == "truncated" for r in pasted)
    # Y no se reintenta al reanudar: el mismo `max_tokens` la volvería a cortar.
    assert "truncated" in rp.DONE_STATUSES
    done, _ = rp.resume_state(path)
    assert {r["conversation_id"] for r in pasted} <= done


# --- D10: sonda de caché ---------------------------------------------------


def _cache_usage(model_id: str, cached: int) -> dict:
    """Las dos formas reales de informar de caché, según el proveedor."""
    if model_id.startswith("claude"):
        return {
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0 if cached else 900,
        }
    return {"prompt_tokens_details": {"cached_tokens": cached}}


@pytest.fixture
def cache_harness(harness, monkeypatch):
    """`chat` falso que cachea a partir de la segunda llamada idéntica."""
    seen: list[tuple] = []

    def fake_chat(model_id, messages, max_tokens=1024, request_params_out=None):
        contenidos = tuple(m["content"] for m in messages)
        repetida = (model_id, contenidos) in seen
        seen.append((model_id, contenidos))
        _fill(request_params_out, model_id)
        return Reply(
            text="respuesta",
            usage=_cache_usage(model_id, 900 if repetida else 0),
            raw={},
            stop_reason="end_turn",
            response_model=model_id,
            attempts=1,
            latency_ms=10.0,
        )

    monkeypatch.setattr(rp, "chat", fake_chat)
    return seen


def test_probe_cache_repeats_the_very_same_call_twice_per_model(cache_harness):
    rp.probe_cache(
        models=["claude-opus-5", "gpt-5.6-sol-tst"],
        topic=FAKE_TOPICS[0],
        artifact=FAKE_ARTS[3],
        run_id="r1",
    )

    assert len(cache_harness) == 4, "cuatro llamadas de céntimos, ni una más"
    assert [m for m, _ in cache_harness] == [
        "claude-opus-5",
        "claude-opus-5",
        "gpt-5.6-sol-tst",
        "gpt-5.6-sol-tst",
    ]
    # Las dos llamadas de un modelo tienen que ser idénticas: si no, no se está
    # midiendo la caché sino otra cosa.
    assert cache_harness[0][1] == cache_harness[1][1]
    assert cache_harness[0][1][-1] == FAKE_ARTS[3].text


def test_probe_cache_reports_the_cache_read_of_each_call(cache_harness):
    report = rp.probe_cache(
        models=["claude-opus-5", "gpt-5.6-sol-tst"],
        topic=FAKE_TOPICS[0],
        artifact=FAKE_ARTS[0],
        run_id="r1",
    )

    for entry in report["models"]:
        assert entry["cache_reads"] == [0, 900]
        assert entry["criterion_met"] is True
    assert report["criterion_met"] is True
    # Desglosado por proveedor, como pide D10.
    assert report["by_provider"] == {"vertex_anthropic": True, "gateway": True}
    assert report["prefix_id"] == "pfx-tema-0-2"


def test_probe_cache_fails_the_criterion_when_nothing_is_cached(harness, monkeypatch):
    def sin_cache(model_id, messages, max_tokens=1024, request_params_out=None):
        _fill(request_params_out, model_id)
        return Reply(
            text="respuesta",
            usage=_cache_usage(model_id, 0),
            raw={},
            stop_reason="end_turn",
            response_model=model_id,
            attempts=1,
        )

    monkeypatch.setattr(rp, "chat", sin_cache)
    report = rp.probe_cache(
        models=["claude-opus-5"], topic=FAKE_TOPICS[0], artifact=FAKE_ARTS[0]
    )
    assert report["models"][0]["criterion_met"] is False
    assert report["criterion_met"] is False


def test_probe_cache_records_a_failed_call_instead_of_dying(harness, monkeypatch):
    def revienta(model_id, messages, max_tokens=1024, request_params_out=None):
        _fill(request_params_out, model_id)
        raise _http_error(429, "rate limited")

    monkeypatch.setattr(rp, "chat", revienta)
    report = rp.probe_cache(
        models=["claude-opus-5"], topic=FAKE_TOPICS[0], artifact=FAKE_ARTS[0]
    )
    calls = report["models"][0]["calls"]
    assert [c["status"] for c in calls] == ["http_error", "http_error"]
    assert report["criterion_met"] is False


def test_probe_cache_writes_its_report_next_to_the_run(cache_harness, tmp_path):
    out = tmp_path / "otro" / "tirada.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = rp.probe_cache(
        models=["claude-opus-5"],
        topic=FAKE_TOPICS[0],
        artifact=FAKE_ARTS[0],
        run_id="r9",
        out=out,
    )
    path = rp.cache_probe_path("r9", out=out)
    assert path == out.parent / "cache-probe-r9.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        k: v for k, v in report.items() if k != "path"
    }


def test_probe_cache_defaults_to_one_model_per_provider():
    proveedores = {
        rp.config.MODELS[m].provider for m in rp.CACHE_PROBE_MODELS
    }
    assert len(proveedores) == len(rp.CACHE_PROBE_MODELS) > 1
    assert rp.CACHE_PROBE_REPEATS == 2


def test_main_never_spends_money_on_the_cache_probe(harness, monkeypatch):
    """La sonda se lanza a mano: no se cuela en cada tirada."""
    llamadas = []
    monkeypatch.setattr(rp, "probe_cache", lambda *a, **k: llamadas.append(1))
    rp.main(seed=7, measure=False)
    assert not llamadas


# --- ficheros laterales ----------------------------------------------------


def test_main_writes_the_final_summary(harness, capsys):
    path = rp.main(seed=7)
    summary = json.loads(
        rp.summary_path(path.stem, out=path).read_text(encoding="utf-8")
    )

    assert summary["planned"] == len(plan_phase0(7))
    assert summary["completed"] == summary["planned"]
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    out = capsys.readouterr().out
    assert "planificadas" in out and "completadas" in out and "fallidas" in out


def test_the_sidecar_reports_follow_the_output_file(harness, tmp_path):
    """Un informe de ejes en otro directorio es un informe huérfano."""
    out = tmp_path / "otro" / "tirada.jsonl"
    rp.main(seed=7, out=out)

    assert (out.parent / "axis-tirada.json").exists()
    assert (out.parent / "summary-tirada.json").exists()
    assert not list((tmp_path / "phase0").glob("*.json"))


def test_the_sidecar_reports_default_to_out_dir(harness):
    path = rp.main(seed=7)
    assert rp.axis_path(path.stem).parent == rp.OUT_DIR
    assert rp.summary_path(path.stem).exists()


# --- BLOQUEANTE: los fallos del usuario simulado ya no son invisibles -------
#
# El usuario simulado no es el modelo bajo prueba. Si su turno sale vacío o
# cortado por el tope, la conversación que medimos está rota: la fila no puede
# decir `ok` (estaría certificando un dato que no existe) ni `refusal` (estaría
# atribuyéndole al modelo evaluado un silencio que no es suyo). Todos estos
# tests corren por el camino real —`real_flow` solo dobla la red—, así que lo
# que se ejerce es la ruta que recorre la tirada de verdad.


def _prefijos_ya_fabricados():
    """Fabrica los 16 prefijos con un usuario simulado que sí habla.

    Lo que se rompe después son los turnos **posteriores al pegote** (D7), que
    son los que paga cada celda. Así el prefijo compartido queda sano y el único
    fallo del experimento es el que el test quiere provocar.
    """
    rp.measure_axis(topics=FAKE_TOPICS, arts=FAKE_ARTS, run_id="previo")


def test_an_empty_simulated_user_turn_makes_the_cell_a_harness_error(real_flow):
    """Un turno vacío del usuario simulado: arnés, ni `ok` ni `refusal`."""
    _prefijos_ya_fabricados()
    real_flow["estado"]["user_text"] = "   "

    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert rows, "la tirada no escribió ninguna fila"
    assert {r["status"] for r in rows} == {"harness_error"}
    assert all(r["error_code"] == "SimulatedUserError" for r in rows)
    assert all("vacío" in (r["error_body"] or "") for r in rows)
    # El stop_reason del turno que falló llega a la fila igualmente.
    assert all(r["user_stop_reasons"] for r in rows)
    assert all("stop_reason" in (r["error_body"] or "") for r in rows)
    # Y no cuenta como celda hecha: al reanudar se vuelve a intentar.
    done, _ = rp.resume_state(path)
    assert done == set()


def test_a_truncated_simulated_user_turn_makes_the_cell_a_harness_error(real_flow):
    """Cortado por el tope: el tope es nuestro, así que es avería del arnés."""
    _prefijos_ya_fabricados()
    real_flow["estado"]["user_stop_reason"] = "max_tokens"

    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    assert {r["status"] for r in rows} == {"harness_error"}
    assert all("tope de tokens" in (r["error_body"] or "") for r in rows)
    for row in rows:
        # El stop_reason del usuario aparece en la fila, y en su columna: no se
        # mezcla con los del modelo evaluado.
        assert row["user_stop_reasons"][-1] == "max_tokens"
        assert "max_tokens" not in [s for s in row["stop_reasons"]]
        assert "max_tokens" in (row["error_body"] or "")


def test_a_broken_user_turn_inside_a_reused_prefix_is_a_harness_error(real_flow):
    """El prefijo compartido viene de disco: sus turnos también se auditan.

    `next_user_turn` se planta ahora ante un turno roto, así que un prefijo
    nuevo no puede nacer con uno dentro. Pero los prefijos se **reutilizan**
    (D1): uno fabricado por una versión anterior del código puede traer un turno
    de usuario cortado, y entonces el agujero está en el contexto que ven los
    tres modelos evaluados. Esta es la red de abajo: `user_side_problems` lo lee
    de `user_reply_traces` y manda la celda a `harness_error`.
    """
    _prefijos_ya_fabricados()
    roto = pfx.find_prefix("tema-0", 2)
    roto["user_reply_traces"][0]["stop_reason"] = "max_tokens"
    pfx.save_prefix(roto)

    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    afectadas = [r for r in rows if r["prefix_id"] == roto["prefix_id"]]
    assert len(afectadas) == 2, "tema-0 con n=2 lo corren dos modelos"
    for row in afectadas:
        assert row["status"] == "harness_error"
        assert "etapa prefix" in (row["error_body"] or "")
        assert "max_tokens" in (row["error_body"] or "")
        # Y dice cómo salir del bucle: `harness_error` se reintenta al reanudar,
        # y el reintento volvería a leer el mismo prefijo roto.
        assert f"runs/prefixes/{roto['prefix_id']}.json" in row["error_body"]
    # Y solo esas: el resto de la tirada sigue siendo buena.
    assert {r["status"] for r in rows if r not in afectadas} == {"ok"}


def test_an_empty_user_message_inside_a_reused_prefix_is_a_harness_error(real_flow):
    """Lo mismo pero leído del transcript: un prefijo viejo sin trazas.

    Un mensaje de usuario vacío no es solo un agujero en la conversación: en
    `vertex_anthropic` es un 400 seguro, y ese 400 estalla en la llamada del
    modelo evaluado, que es a quien se le acabaría atribuyendo.
    """
    _prefijos_ya_fabricados()
    roto = pfx.find_prefix("tema-0", 2)
    roto.pop("user_reply_traces")  # formato anterior: no hay trazas que mirar
    roto["transcript"].append({"role": "user", "content": "  ", "tag": "user_sim"})
    roto["transcript"].append({"role": "assistant", "content": "ya", "tag": "assistant"})
    pfx.save_prefix(roto)

    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    afectadas = [r for r in rows if r["prefix_id"] == roto["prefix_id"]]
    assert afectadas
    for row in afectadas:
        assert row["status"] == "harness_error"
        assert "está vacío" in (row["error_body"] or "")
        # El mensaje vacío cae dentro del tramo del prefijo, así que la fila lo
        # atribuye al prefijo compartido y no a esta celda.
        assert "etapa prefix" in row["error_body"]
        assert f"runs/prefixes/{roto['prefix_id']}.json" in row["error_body"]


def test_a_clean_run_carries_the_simulated_users_stop_reasons_in_every_row(real_flow):
    """La tirada buena: cada fila lleva el otro lado de la conversación (D5).

    Un `stop_reason` por llamada del usuario simulado, en orden y con la etapa:
    los `n_turns - 1` turnos del prefijo compartido más los dos de después del
    pegote (D7). Sin esta columna, un turno de usuario cortado no dejaba rastro
    en ningún fichero.
    """
    path = rp.main(seed=7)
    _, rows = _rows(path)

    assert rows
    for row in rows:
        del_prefijo = row["n_turns"] - 1
        assert row["status"] == "ok"
        assert row["user_stop_reasons"] == ["stop"] * (del_prefijo + rp.N_POST_TURNS)
        assert [t["stage"] for t in row["user_reply_traces"]] == (
            ["prefix"] * del_prefijo + ["post"] * rp.N_POST_TURNS
        )
        assert all(
            t["response_model"] == f"{su.USER_MODEL}-20260101"
            for t in row["user_reply_traces"]
        )
        # Y no se mezclan con las del modelo evaluado, que van en su columna.
        esperados = 3 if row["condition"] == "paste" else 2
        assert len(row["stop_reasons"]) == esperados


def test_the_three_models_of_a_prefix_carry_the_same_user_traces(real_flow):
    """D1: el prefijo es el mismo para los tres, y sus turnos de usuario también."""
    path = rp.main(seed=7, measure=False)
    _, rows = _rows(path)

    por_prefijo: dict[str, list] = {}
    for row in rows:
        del_prefijo = [
            t for t in row["user_reply_traces"] if t["stage"] == "prefix"
        ]
        por_prefijo.setdefault(row["prefix_id"], []).append(del_prefijo)
    compartidos = [v for v in por_prefijo.values() if len(v) > 1]
    assert compartidos, "ningún prefijo lo comparten dos celdas"
    for trazas in compartidos:
        assert all(t == trazas[0] for t in trazas)


# --- las piezas del lado del usuario, aisladas -----------------------------


def test_user_call_traces_marks_the_stage_of_each_turn():
    prefijo = [{"stop_reason": "stop", "attempts": 1}]
    replies = [_fake_user_reply("sigo")]

    trazas = rp.user_call_traces(prefijo, replies)

    assert [t["stage"] for t in trazas] == [rp.STAGE_PREFIX, rp.STAGE_POST]
    assert trazas[0]["stop_reason"] == "stop"
    assert trazas[1]["response_model"] == f"{rp.USER_MODEL}-20260101"


def test_user_call_traces_survives_a_prefix_without_traces():
    """Un prefijo de disco anterior a esta columna no puede tumbar la celda."""
    assert rp.user_call_traces([], []) == []
    assert rp.user_call_traces(None, []) == []


def test_user_side_problems_says_nothing_about_a_healthy_conversation():
    trazas = rp.user_call_traces([{"stop_reason": "stop"}], [_fake_user_reply()])
    transcript = [
        {"role": "user", "content": "hola", "tag": "opening"},
        {"role": "assistant", "content": "dime", "tag": "assistant"},
        {"role": "user", "content": "sigo", "tag": "user_sim"},
    ]
    assert rp.user_side_problems(trazas, transcript) == []


def test_user_side_problems_catches_a_cut_and_an_empty_turn():
    trazas = rp.user_call_traces(
        [{"stop_reason": "max_tokens"}], [_fake_user_reply(stop_reason="length")]
    )
    transcript = [
        {"role": "user", "content": "  ", "tag": "post"},
        # El pegote es texto nuestro, no del usuario simulado: no cuenta.
        {"role": "user", "content": "", "tag": "paste"},
    ]
    problemas = rp.user_side_problems(trazas, transcript)

    assert len(problemas) == 3
    assert sum("tope de tokens" in p for p in problemas) == 2
    assert sum("está vacío" in p for p in problemas) == 1
    assert "etapa prefix" in problemas[0] and "etapa post" in problemas[1]


def test_user_side_problems_says_whether_the_empty_turn_is_in_the_prefix():
    """El prefijo se reutiliza: arreglarlo y arreglar la celda no es lo mismo."""
    transcript = [
        {"role": "user", "content": "hola", "tag": "opening"},
        {"role": "user", "content": " ", "tag": "user_sim"},
        {"role": "assistant", "content": "ya", "tag": "assistant"},
        {"role": "user", "content": "", "tag": "post"},
    ]
    problemas = rp.user_side_problems([], transcript, prefix_len=3)

    assert len(problemas) == 2
    assert "etapa prefix" in problemas[0]
    assert "etapa post" in problemas[1]
    # Sin `prefix_len` no hay tramo que distinguir: todo es de esta celda.
    assert all(
        "etapa post" in p for p in rp.user_side_problems([], transcript)
    )


def test_a_simulated_user_error_is_always_a_harness_error():
    """Venga de la etapa que venga: quien se calló no es el modelo evaluado."""
    exc = su.SimulatedUserError(["salió vacío"], stop_reason="max_tokens")
    for origen in (rp.ORIGIN_MODEL, rp.ORIGIN_HARNESS):
        status, code, body = rp._status_for_error(exc, origin=origen)
        assert status == "harness_error"
        assert code == "SimulatedUserError"
        assert "max_tokens" in body
