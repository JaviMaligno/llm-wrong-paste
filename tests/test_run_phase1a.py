"""Tests del runner de la Fase 1a. Ninguno hace una sola llamada a un modelo.

Todo lo que sale a la red —`ensure_prefix`, `ensure_contradiction`,
`rank_artifacts`, `inject_paste`, `continue_after_paste`— se sustituye por un
doble determinista, y la red de seguridad del final del fichero (`_prohibido`)
revienta si alguna llamada se escapa por un camino que no se dobló.

Lo que se prueba aquí es lo que la Fase 1a **añade** sobre la Fase 0, que es
poco y está muy localizado:

- el plan de 288 celdas con el nivel de pegote como factor propio (§5 del spec);
- que **las dos réplicas de una celda comparten prefijo y artefacto** (D1), y
  que por tanto el artefacto NO puede depender del `seed` de la celda —que es
  justo lo que las distingue— ni del orden en que se reanude la tirada;
- que N2 se fabrica contra el prefijo ya construido y se reutiliza, y que sus
  filas no traen estrato ni similaridad, porque no sale de ningún banco;
- que un fallo del generador de N2 es del **arnés** y nunca conducta del modelo
  evaluado.

Los dobles declaran la firma exacta de la función que sustituyen, y hay un test
que lo comprueba con `inspect.signature`: un doble más permisivo que el original
ejercita un camino que producción no recorre, que es el patrón que ya ha roto
este repositorio cuatro veces.
"""

import collections
import inspect
import json

import httpx
import numpy as np
import pytest

import wrongpaste.artifacts as arts_mod
import wrongpaste.contradictions as cd
import wrongpaste.conversation as conv
import wrongpaste.prefixes as pfx
import wrongpaste.run_phase0 as rp0
import wrongpaste.run_phase1a as rp
import wrongpaste.similarity as sim
import wrongpaste.simulated_user as su
from wrongpaste.artifacts import Artifact
from wrongpaste.clients import Reply
from wrongpaste.records import PASTE_LEVELS
from wrongpaste.run_phase1a import LEVELS, PHASE1A_MODELS, plan_phase1a
from wrongpaste.topics import Topic

# --- los seis tests del plan ----------------------------------------------


def test_el_plan_tiene_288_celdas():
    # 3 niveles x 8 temas x 2 longitudes x 2 réplicas x 3 modelos
    assert len(plan_phase1a(1)) == 3 * 8 * 2 * 2 * 3


def test_los_tres_niveles_estan_equilibrados():
    c = collections.Counter(x["paste_level"] for x in plan_phase1a(1))
    assert set(c) == set(LEVELS)
    assert len(set(c.values())) == 1, "los tres niveles con las mismas celdas"


def test_cada_modelo_ve_los_tres_niveles_en_ambas_longitudes():
    plan = plan_phase1a(1)
    for m in PHASE1A_MODELS:
        for niv in LEVELS:
            longs = {x["n_turns"] for x in plan
                     if x["model_id"] == m and x["paste_level"] == niv}
            assert longs == {2, 10}, f"{m}/{niv} solo en {longs}"


def test_las_replicas_comparten_prefijo_y_se_distinguen():
    plan = plan_phase1a(1)
    por_celda = collections.defaultdict(list)
    for x in plan:
        por_celda[(x["model_id"], x["topic_id"], x["n_turns"], x["paste_level"])].append(x)
    for celda, xs in por_celda.items():
        assert len(xs) == 2, f"{celda} no tiene 2 réplicas"
        assert {x["replicate_idx"] for x in xs} == {0, 1}


def test_conversation_id_unico():
    ids = [x["conversation_id"] for x in plan_phase1a(1)]
    assert len(ids) == len(set(ids))


def test_el_plan_es_determinista():
    assert plan_phase1a(9) == plan_phase1a(9)


# --- el plan, lo que la fila necesita de él -------------------------------


def test_el_plan_trae_todo_lo_que_la_fila_pide():
    """Sin `cell_index` ni `condition` ni `seed`, `run_cell` revienta con KeyError.

    No es una comprobación de estilo: la fila de la Fase 0 (D5) los lee del
    dict del plan, y una celda que no los traiga no llega ni a escribirse como
    fallo.
    """
    obligatorias = {
        "cell_index", "model_id", "topic_id", "n_turns", "paste_level",
        "replicate_idx", "stratum", "condition", "seed", "conversation_id",
    }
    plan = plan_phase1a(1)
    for celda in plan:
        assert obligatorias <= set(celda), f"faltan {obligatorias - set(celda)}"
    assert [c["cell_index"] for c in plan] == list(range(len(plan)))


def test_los_niveles_son_los_del_esquema_de_fila():
    """`paste_level` tiene vocabulario cerrado en `records`: no inventar uno."""
    assert set(LEVELS) == set(PASTE_LEVELS)


def test_el_plantel_es_el_evaluado_y_ningun_juez_esta_dentro():
    from wrongpaste.config import EVALUATED
    from wrongpaste.judging import JUDGES

    assert set(PHASE1A_MODELS) <= set(EVALUATED)
    assert not set(PHASE1A_MODELS) & set(JUDGES), "nadie se puntúa a sí mismo"


def test_el_estrato_sigue_la_formula_de_la_fase_0():
    """El nivel es un factor nuevo, pero la rotación de ejes de D3 no cambia."""
    plan = plan_phase1a(1)
    por_celda = {(c["topic_id"], c["model_id"]): c["stratum"] for c in plan}
    for t, topic in enumerate(rp.load_topics()):
        for m, model_id in enumerate(PHASE1A_MODELS):
            esperado = (t + rp.STRATUM_SHIFTS[m]) % rp.STRATA
            assert por_celda[(topic.id, model_id)] == esperado


# --- la semilla del artefacto (D1) ----------------------------------------


def test_la_semilla_del_artefacto_no_ve_el_seed_de_la_celda():
    """D1: lo que eligen las réplicas en común no puede salir de lo que las separa."""
    parametros = list(inspect.signature(rp.artifact_seed).parameters)
    assert parametros == ["prefix_id", "paste_level", "stratum"]


def test_la_semilla_del_artefacto_separa_niveles_y_estratos():
    base = rp.artifact_seed("pfx-1", "N0", 3)
    assert base == rp.artifact_seed("pfx-1", "N0", 3)
    assert base != rp.artifact_seed("pfx-1", "N1", 3)
    assert base != rp.artifact_seed("pfx-1", "N0", 4)
    assert base != rp.artifact_seed("pfx-2", "N0", 3)


def test_el_fallo_del_generador_de_n2_es_del_arnes():
    """Un 400 del generador de N2 no es una negativa del modelo evaluado.

    `failure_origin` decide el origen por el nombre de la etapa, con una lista
    cerrada. Si la etapa del generador quedara fuera de esa lista, la fila diría
    `refusal` de un modelo que ni siquiera había hablado todavía.
    """
    assert rp0.failure_origin(rp.STAGE_CONTRADICTION, []) == rp0.ORIGIN_HARNESS


# --- dobles con la firma de verdad ----------------------------------------


FAKE_TOPICS = [
    Topic(id=f"tema-{i}", opening=f"apertura {i}", goals=("a", "b")) for i in range(8)
]

# Los bancos falsos tienen el TAMAÑO y la MEZCLA DE GÉNEROS de los de verdad
# (64 artefactos en N0, 32 en N1, con los géneros repartidos de forma desigual y
# sin agruparse por posición en el ranking). No es decoración: con un banco
# pequeño y uniforme, la ventana de un estrato tiene dos artefactos de dos
# géneros distintos y el recuento de D4 casi nunca decide nada, así que un test
# escrito contra ese banco da por bueno un runner que elige el pegote de otra
# manera. Comprobado a mano: con el banco uniforme de 16, quitar la recuperación
# de elecciones al reanudar dejaba la suite en verde.
KINDS_N0 = (
    "stacktrace", "sql", "config", "changelog", "prompt", "email",
    "receta", "nota", "mensaje", "albaran", "aviso",
)
KINDS_N1 = ("email", "nota", "mensaje", "albaran", "aviso", "sms", "carta", "acta")

FAKE_BANKS = {
    "N0": [
        Artifact(
            f"n0-{i:02d}", KINDS_N0[(i * 5) % len(KINDS_N0)],
            f"texto n0 {i}", (f"e{i}",), level="N0",
        )
        for i in range(64)
    ],
    "N1": [
        Artifact(
            f"n1-{i:02d}", KINDS_N1[(i * 3) % len(KINDS_N1)],
            f"texto n1 {i}", (f"e{i}",), level="N1", signal="cortado",
        )
        for i in range(32)
    ],
}

CALLS: dict[str, list] = {}
N2_MEMO: dict[tuple[str, int], Artifact] = {}


def _reset_calls() -> dict[str, list]:
    CALLS.clear()
    CALLS.update(
        {"prefix": [], "rank": [], "paste": [], "post": [], "n2": [], "n2_generated": []}
    )
    N2_MEMO.clear()
    return CALLS


def fake_load_artifacts(level: str | None = None) -> list[Artifact]:
    """Doble de `load_artifacts`, exigente con el nivel.

    La Fase 1a nunca puede pedir el banco entero: N0 y N1 se estratifican por
    separado y mezclarlos cambiaría el ranking de los dos a la vez.
    """
    assert level in FAKE_BANKS, f"la Fase 1a pide el banco por nivel, no {level!r}"
    return list(FAKE_BANKS[level])


def fake_ensure_prefix(topic, n_turns):
    CALLS["prefix"].append((topic.id, n_turns))
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
        "user_reply_traces": [],
    }


def fake_ensure_contradiction(prefix_id, topic, transcript, seed):
    """Doble de la puerta de N2, memoizada **como la de verdad**.

    La real reutiliza `runs/contradictions/<prefijo>-<semilla>.json`, que es lo
    que hace que las dos réplicas de una celda N2 compartan artefacto (D1) y que
    reanudar no vuelva a pagar el generador. Un doble que fabricara un texto
    nuevo en cada llamada escondería exactamente el fallo que hay que cazar.
    """
    CALLS["n2"].append((prefix_id, topic.id, [dict(m) for m in transcript], seed))
    clave = (prefix_id, int(seed))
    if clave not in N2_MEMO:
        CALLS["n2_generated"].append(clave)
        N2_MEMO[clave] = Artifact(
            id=f"n2-{prefix_id}-{seed}",
            kind="contradiction",
            text=f"contradicción de {prefix_id}",
            entities=("otra ciudad", "otra fecha"),
            level="N2",
            signal="contradiccion",
        )
    return N2_MEMO[clave]


def fake_rank_artifacts(text, arts, max_chars=sim.MAX_EMBED_CHARS):
    CALLS["rank"].append((text, len(arts)))
    n = len(arts)
    return [(a, i / (n - 1)) for i, a in enumerate(arts)], False


def _fake_reply(model_id, text, *, stop_reason="end_turn"):
    return Reply(
        text=text,
        usage={"total_tokens": 5},
        raw={},
        stop_reason=stop_reason,
        response_model=f"{model_id}-20260101",
        attempts=1,
        latency_ms=12.5,
    )


def _fill(request_params_out, model_id):
    """Emula lo que hace `chat()`: vacía el dict y escribe el cuerpo enviado."""
    if request_params_out is None:
        return
    from wrongpaste import clients
    from wrongpaste.config import MODELS

    mensajes = [{"role": "user", "content": "x"}]
    provider = MODELS[model_id].provider
    if provider == "gateway":
        cuerpo = clients._gateway_body(model_id, mensajes, conv.MAX_TOKENS)
    elif provider == "vertex_openai":
        cuerpo = clients._vertex_openai_body(model_id, mensajes, conv.MAX_TOKENS)
    else:
        cuerpo = clients._anthropic_body(mensajes, conv.MAX_TOKENS)
    cuerpo.pop("messages", None)
    request_params_out.clear()
    request_params_out.update(cuerpo)


def fake_inject_paste(
    model_id, transcript, artifact, request_params_out=None, max_tokens=conv.MAX_TOKENS
):
    CALLS["paste"].append((model_id, artifact.id))
    _fill(request_params_out, model_id)
    index = len(transcript)
    transcript.append({"role": "user", "content": artifact.text, "tag": "paste"})
    text = f"reacción de {model_id}"
    transcript.append({"role": "assistant", "content": text, "tag": "assistant"})
    return text, {"total_tokens": 5}, index, _fake_reply(model_id, text)


def fake_continue_after_paste(
    model_id,
    transcript,
    topic,
    n_post=conv.N_POST_TURNS,
    request_params_out=None,
    max_tokens=conv.MAX_TOKENS,
    user_max_tokens=su.USER_MAX_TOKENS,
    user_replies_out=None,
):
    CALLS["post"].append((model_id, n_post))
    _fill(request_params_out, model_id)
    indices, usages, replies = [], [], []
    for _ in range(n_post):
        if user_replies_out is not None:
            user_replies_out.append(
                Reply("sigo", {"total_tokens": 2}, {}, stop_reason="stop")
            )
        indices.append(len(transcript))
        transcript.append({"role": "user", "content": "sigo", "tag": "post"})
        indices.append(len(transcript))
        transcript.append({"role": "assistant", "content": "ya", "tag": "assistant"})
        usages.append({"total_tokens": 3})
        replies.append(_fake_reply(model_id, "ya"))
    return indices, usages, replies


def test_los_dobles_declaran_la_firma_de_la_funcion_real():
    """Un doble más permisivo que el original prueba un camino que no existe.

    Es el patrón que ya ha roto este repositorio cuatro veces, así que se
    comprueba en vez de confiarlo a la vista: mismos parámetros y en el mismo
    orden que la función de producción que sustituyen.
    """
    parejas = [
        (fake_load_artifacts, arts_mod.load_artifacts),
        (fake_ensure_prefix, pfx.ensure_prefix),
        (fake_ensure_contradiction, cd.ensure_contradiction),
        (fake_rank_artifacts, sim.rank_artifacts),
        (fake_inject_paste, conv.inject_paste),
        (fake_continue_after_paste, conv.continue_after_paste),
    ]
    for doble, real in parejas:
        assert list(inspect.signature(doble).parameters) == list(
            inspect.signature(real).parameters
        ), f"{doble.__name__} no casa con {real.__module__}.{real.__name__}"


def _prohibido(*args, **kwargs):
    raise AssertionError(
        "se ha escapado una llamada real a un modelo por un camino sin doblar"
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Sustituye las puertas de red del runner y lo aísla en `tmp_path`."""
    calls = _reset_calls()
    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(rp, "ensure_prefix", fake_ensure_prefix)
    monkeypatch.setattr(rp, "ensure_contradiction", fake_ensure_contradiction)
    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue_after_paste)
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase1a")
    # `rank_for_prefix` se reutiliza tal cual de la Fase 0, así que el embedder
    # se dobla **en su espacio de nombres**, que es el que producción recorre.
    monkeypatch.setattr(rp0, "rank_artifacts", fake_rank_artifacts)
    # Red de seguridad: cualquier camino que no se haya doblado revienta aquí en
    # vez de salir a la red.
    monkeypatch.setattr(conv, "chat", _prohibido)
    monkeypatch.setattr(su, "chat", _prohibido)
    monkeypatch.setattr(cd, "chat", _prohibido)
    monkeypatch.setattr(sim, "embed", _prohibido)
    return calls


def _rows(path):
    lineas = [
        json.loads(linea)
        for linea in path.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    return lineas[0], lineas[1:]


def _por_celda(rows):
    fuera = collections.defaultdict(list)
    for r in rows:
        fuera[(r["model_id"], r["topic_id"], r["n_turns"], r["paste_level"])].append(r)
    return fuera


# --- la tirada -------------------------------------------------------------


def test_la_tirada_escribe_una_fila_por_celda_con_su_nivel(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    header, rows = _rows(path)
    assert header["kind"] == "run_header"
    assert header["planned_cells"] == 288
    assert len(rows) == 288
    assert len({r["conversation_id"] for r in rows}) == 288
    niveles = collections.Counter(r["paste_level"] for r in rows)
    assert niveles == {"N0": 96, "N1": 96, "N2": 96}
    assert all(r["status"] == "ok" for r in rows)


def test_la_cabecera_declara_la_fase_y_los_dos_bancos(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    header, _ = _rows(path)
    assert header["phase"] == "1a"
    assert header["roster"] == list(PHASE1A_MODELS)
    assert header["levels"] == list(LEVELS)
    # Los dos bancos entran en la identidad: si cambia cualquiera de ellos, las
    # tiradas dejan de ser comparables y el sha lo dice.
    assert set(header["bank_sha_by_level"]) == {"N0", "N1"}
    assert header["bank_sha_by_level"]["N0"] != header["bank_sha_by_level"]["N1"]
    assert header["rubric_version"] == "v2"
    assert header["judges"] == ["gpt-5.5-tst", "gemini-2.5-flash"]
    assert header["contradiction_model"]


def test_cada_nivel_pega_artefactos_de_su_banco(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    for r in rows:
        if r["paste_level"] == "N0":
            assert r["artifact_id"].startswith("n0-")
        elif r["paste_level"] == "N1":
            assert r["artifact_id"].startswith("n1-")
        else:
            assert r["artifact_id"].startswith("n2-")
            assert r["artifact_kind"] == "contradiction"


def test_las_dos_replicas_comparten_prefijo_y_artefacto(harness, tmp_path):
    """D1, que es el punto de toda la Fase 1a: las réplicas miden el muestreo.

    Si cada réplica pega un artefacto distinto, la variación entre ellas mezcla
    «el modelo contesta distinto» con «le pegamos otra cosa», y deja de ser una
    réplica.
    """
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    por_celda = _por_celda(rows)
    assert len(por_celda) == 144
    for celda, rs in por_celda.items():
        assert len(rs) == 2, celda
        assert len({r["artifact_id"] for r in rs}) == 1, celda
        assert len({r["artifact_text"] for r in rs}) == 1, celda
        assert len({r["prefix_id"] for r in rs}) == 1, celda
        assert {r["replicate_idx"] for r in rs} == {0, 1}


def test_el_artefacto_no_lo_elige_la_semilla_maestra(harness, tmp_path):
    """La semilla de la celda cambia y el artefacto no: sale de (prefijo, nivel, estrato).

    Es la comprobación que distingue el arreglo de D1 de un arreglo aparente:
    con el artefacto elegido por el `seed` de la celda, dos tiradas con semillas
    distintas pegarían cosas distintas y las réplicas serían un accidente del
    orden de iteración.
    """
    a = rp.main(seed=1, out=tmp_path / "a.jsonl", measure=False)
    b = rp.main(seed=99, out=tmp_path / "b.jsonl", measure=False)
    _, filas_a = _rows(a)
    _, filas_b = _rows(b)
    arte_a = {r["conversation_id"]: r["artifact_id"] for r in filas_a}
    arte_b = {r["conversation_id"]: r["artifact_id"] for r in filas_b}
    assert arte_a == arte_b
    semillas_a = [r["seed"] for r in sorted(filas_a, key=lambda r: r["cell_index"])]
    semillas_b = [r["seed"] for r in sorted(filas_b, key=lambda r: r["cell_index"])]
    assert semillas_a != semillas_b, "el plan sí tiene que depender de la semilla"


def test_los_niveles_de_banco_cubren_los_generos(harness, tmp_path):
    """D4 sigue vigente, y se cuenta **por banco**: cada uno cubre los suyos."""
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    for nivel in ("N0", "N1"):
        kinds = {r["artifact_kind"] for r in rows if r["paste_level"] == nivel}
        assert kinds == {a.kind for a in FAKE_BANKS[nivel]}, nivel


# --- N2 --------------------------------------------------------------------


def test_la_contradiccion_se_fabrica_contra_el_prefijo_ya_construido(harness, tmp_path):
    rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    assert harness["n2"], "ninguna celda N2 llamó al generador"
    for prefix_id, topic_id, transcript, _seed in harness["n2"]:
        assert topic_id in prefix_id
        # El generador ve el prefijo y nada más: ni el pegote ni la reacción.
        assert [m["tag"] for m in transcript] == ["opening", "assistant"]


def test_la_contradiccion_se_paga_una_vez_por_celda(harness, tmp_path):
    """Las dos réplicas comparten artefacto también cuando lo escribe un modelo.

    Aquí D1 no es solo rigor: llamar dos veces al generador daría dos textos
    distintos, y además pagaría el doble.
    """
    rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    llamadas = [(p, s) for p, _t, _tr, s in harness["n2"]]
    assert len(llamadas) == 96, "una llamada a la puerta por celda N2"
    assert len(set(llamadas)) == 48, "una clave por (prefijo, estrato)"
    assert len(harness["n2_generated"]) == 48, "y una sola fabricación por clave"


def test_las_filas_n2_no_traen_estrato_ni_similaridad(harness, tmp_path):
    """Un pegote N2 no sale del banco: no tiene ranking, ni estrato, ni coseno.

    Escribir un estrato en una fila N2 sería inventar un dato: nada se muestreó
    de ninguna ventana. La convención es la misma que la del brazo de control de
    la Fase 0 (D11), que tampoco tiene estrato porque no pega nada del banco.
    """
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    n2 = [r for r in rows if r["paste_level"] == "N2"]
    assert n2
    for r in n2:
        assert r["stratum"] == -1
        assert r["similarity_user"] is None
        assert r["similarity_full"] is None
        assert r["ranking"] is None
        assert r["artifact_entities"] == ["otra ciudad", "otra fecha"]
    for r in rows:
        if r["paste_level"] != "N2":
            assert r["stratum"] >= 0
            assert r["similarity_user"] is not None
            assert r["ranking"] is not None


# --- fallos y reanudación (D6) --------------------------------------------


def test_una_celda_rota_escribe_fila_con_su_nivel(harness, tmp_path, monkeypatch):
    def revienta(model_id, transcript, artifact, request_params_out=None,
                 max_tokens=conv.MAX_TOKENS):
        if model_id == PHASE1A_MODELS[0]:
            raise httpx.HTTPStatusError(
                "500",
                request=httpx.Request("POST", "https://x.invalid"),
                response=httpx.Response(500, text="{}"),
            )
        return fake_inject_paste(
            model_id, transcript, artifact,
            request_params_out=request_params_out, max_tokens=max_tokens,
        )

    monkeypatch.setattr(rp, "inject_paste", revienta)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    assert len(rows) == 288, "una celda rota no se descarta en silencio"
    rotas = [r for r in rows if r["status"] != "ok"]
    assert rotas
    for r in rotas:
        assert r["model_id"] == PHASE1A_MODELS[0]
        assert r["status"] == "http_error"
        assert r["paste_level"] in set(LEVELS)
        assert r["error_code"] == 500


def test_un_fallo_del_generador_de_n2_no_se_cuenta_como_negativa(
    harness, tmp_path, monkeypatch
):
    """El generador lo paga otro modelo: su 400 no es conducta del evaluado."""

    def revienta(prefix_id, topic, transcript, seed):
        raise httpx.HTTPStatusError(
            "400",
            request=httpx.Request("POST", "https://x.invalid"),
            response=httpx.Response(400, text='{"error": {"code": "content_filter"}}'),
        )

    monkeypatch.setattr(rp, "ensure_contradiction", revienta)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    n2 = [r for r in rows if r["paste_level"] == "N2"]
    assert len(n2) == 96
    for r in n2:
        assert r["status"] == "harness_error", (
            "una negativa del generador no la dio el modelo evaluado"
        )


def test_la_tirada_se_reanuda_sin_duplicar_filas(harness, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"

    def a_medias(model_id, transcript, artifact, request_params_out=None,
                 max_tokens=conv.MAX_TOKENS):
        if artifact.level == "N1":
            raise httpx.TimeoutException("se acabó el tiempo")
        return fake_inject_paste(
            model_id, transcript, artifact,
            request_params_out=request_params_out, max_tokens=max_tokens,
        )

    monkeypatch.setattr(rp, "inject_paste", a_medias)
    rp.main(seed=1, out=path, measure=False)
    _, rows = _rows(path)
    assert sum(1 for r in rows if r["status"] == "timeout") == 96

    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    rp.main(seed=1, out=path, measure=False)
    _, rows = _rows(path)
    assert len(rows) == 288, "reanudar no puede dejar dos filas por conversación"
    assert len({r["conversation_id"] for r in rows}) == 288
    assert all(r["status"] == "ok" for r in rows)


def test_al_reanudar_la_replica_que_falta_pega_el_mismo_artefacto(harness, tmp_path):
    """D1 tiene que sobrevivir a una reanudación, no solo a un proceso.

    Si la réplica 0 ya está escrita y la 1 se vuelve a correr, la elección no
    puede depender del recuento de géneros que lleve **esa** ejecución: el
    artefacto de la réplica que falta sale de lo que ya está en el fichero.

    **Las réplicas que faltan van salteadas, y eso es lo que hace que el test
    sirva.** Si se quitan TODAS las réplicas 1, la reanudación vuelve a recorrer
    las mismas claves en el mismo orden, el recuento de géneros de D4 evoluciona
    igual que en la primera tirada y las elecciones coinciden por casualidad:
    comprobado quitando `resume_choices` del runner, la suite seguía en verde.
    Una tirada real se corta por donde se corta, así que el caso que hay que
    probar es el desordenado, donde el recuento de la ejecución nueva no se
    parece en nada al de la vieja.
    """
    path = tmp_path / "t.jsonl"
    rp.main(seed=1, out=path, measure=False)
    header, rows = _rows(path)
    faltan = {
        r["conversation_id"]
        for r in rows
        if r["replicate_idx"] == 1 and r["cell_index"] % 13 == 0
    }
    assert len(faltan) >= 8, "hacen falta bastantes huecos para que el test muerda"
    quedan = [r for r in rows if r["conversation_id"] not in faltan]
    path.write_text(
        "\n".join(json.dumps(fila, ensure_ascii=False) for fila in [header, *quedan])
        + "\n",
        encoding="utf-8",
    )

    rp.main(seed=1, out=path, measure=False)
    _, rows2 = _rows(path)
    assert len(rows2) == 288
    assert len({r["conversation_id"] for r in rows2}) == 288
    for celda, rs in _por_celda(rows2).items():
        assert len(rs) == 2, celda
        assert len({r["artifact_id"] for r in rs}) == 1, celda


# --- cobertura de señales en N1 -------------------------------------------


def _ranking_simulado(bank, rng):
    """Un orden de similaridad cualquiera, con los pares (artefacto, coseno).

    En producción el orden lo pone el embedder y depende del prefijo. Offline no
    se puede saber cuál sale, así que el sorteo que se prueba aquí ES el orden:
    lo que tiene que aguantar el muestreador es *cualquier* ranking, no uno.
    """
    orden = rng.permutation(len(bank))
    return [(bank[i], j / (len(bank) - 1)) for j, i in enumerate(orden)]


def _tirada_n1(bank, plan, rng):
    """Muestrea las 96 celdas N1 y devuelve cuántas veces sale cada señal."""
    kind_counts, signal_counts, chosen = (
        collections.Counter(),
        collections.Counter(),
        {},
    )
    rankings = {}
    for cell in plan:
        prefix_id = f"{cell['topic_id']}-{cell['n_turns']}"
        if prefix_id not in rankings:
            rankings[prefix_id] = _ranking_simulado(bank, rng)
        rp.choose_bank_artifact(
            prefix_id, cell, rankings[prefix_id], kind_counts, chosen, signal_counts
        )
    # Se recuenta desde `chosen` y no desde `signal_counts`: así el test no se
    # cree el contador que está juzgando. Las dos réplicas comparten artefacto
    # (D1), así que cada clave de `chosen` vale por las dos celdas.
    por_id = {a.id: a for a in bank}
    cuenta = collections.Counter()
    for cell in plan:
        clave = (f"{cell['topic_id']}-{cell['n_turns']}", "N1", int(cell["stratum"]))
        cuenta[por_id[chosen[clave]].signal] += 1
    return cuenta


def test_el_muestreo_de_n1_reparte_las_senales_con_el_banco_real():
    """Cubrir géneros no cubre señales, y la señal es la pregunta de N1.

    **Va contra el banco real a propósito.** Con el banco de juguete del resto
    del fichero esto no se puede medir: el desequilibrio nace de cómo se cruzan
    los 11 géneros con las 4 señales en los 44 artefactos escritos, así que un
    banco inventado para el test comprobaría una distribución que producción no
    tiene —el patrón que ya ha roto este repositorio varias veces—.

    Medido antes de arreglarlo, sobre 300 rankings: entre la señal más y la
    menos muestreada había 14 celdas de 96 (mediana), y el 46 % de las tiradas
    dejaba alguna señal por debajo de 18. Eso no sesga la comparación N0/N1
    —la que decide la puerta— pero sí deja sin potencia la pregunta de qué
    señal hace preguntar.
    """
    bank = arts_mod.load_artifacts(level="N1")
    señales = {a.signal for a in bank}
    assert len(señales) == 4, f"el banco N1 trae {len(señales)} señales, no 4"
    plan = [c for c in plan_phase1a(1) if c["paste_level"] == "N1"]
    assert len(plan) == 96

    for semilla in range(40):
        cuenta = _tirada_n1(bank, plan, np.random.default_rng(semilla))
        assert set(cuenta) == señales, f"semilla {semilla}: falta alguna señal"
        vals = sorted(cuenta.values())
        assert vals[-1] - vals[0] <= 8, f"semilla {semilla}: reparto {dict(cuenta)}"
        assert vals[0] >= 18, f"semilla {semilla}: reparto {dict(cuenta)}"


def test_la_senal_del_pegote_viaja_en_la_fila(harness, tmp_path):
    """Sin esto, «qué señal funciona» habría que derivarlo del banco después."""
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    por_nivel = collections.defaultdict(set)
    for r in rows:
        por_nivel[r["paste_level"]].add(r.get("artifact_signal"))
    assert por_nivel["N0"] == {None}, "N0 no tiene señal: es el banco neutro"
    assert None not in por_nivel["N1"], "toda fila N1 dice qué la delata"


def test_al_reanudar_las_senales_ya_gastadas_vuelven(harness, tmp_path):
    """Si la reanudación no las trajera, el equilibrio valdría por trozos.

    La segunda ejecución repartiría señales desde cero y las celdas que le
    tocasen se sumarían a las de la primera sin verlas.
    """
    path = tmp_path / "t.jsonl"
    rp.main(seed=1, out=path, measure=False)
    _, _, signal_counts = rp.resume_choices(path)
    assert sum(signal_counts["N1"].values()) == 96
    assert not signal_counts["N0"], "N0 no aporta señales al recuento"


# --- ancho del eje y resumen ----------------------------------------------


def test_el_ancho_del_eje_se_mide_para_los_dos_bancos(harness, tmp_path, monkeypatch):
    vistos = []

    def fake_measure_axis(topics=None, arts=None, lengths=(2, 10), run_id=""):
        vistos.append({a.level for a in arts})
        return {
            "run_id": run_id,
            "narrow_cells": [],
            "narrow_topics": [],
            "entries": [],
        }

    monkeypatch.setattr(rp, "measure_axis", fake_measure_axis)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=True)
    assert vistos == [{"N0"}, {"N1"}]
    informe = json.loads(
        (tmp_path / f"axis-{path.stem}.json").read_text(encoding="utf-8")
    )
    assert set(informe["levels"]) == {"N0", "N1"}


def test_un_eje_estrecho_avisa_pero_no_frena_la_fase_1a(
    harness, tmp_path, monkeypatch, capsys
):
    """En 1a la variable independiente es el NIVEL, no la similaridad.

    La puerta de D12 existe para 1b, donde la similaridad es el eje: frenar 1a
    porque un banco tenga el rango corto sería aplicar un criterio que no
    gobierna esta pregunta. Se mide, se escribe y se avisa.
    """

    def estrecho(topics=None, arts=None, lengths=(2, 10), run_id=""):
        return {
            "run_id": run_id,
            "narrow_cells": [{"topic_id": "tema-0", "n_turns": 2}],
            "narrow_topics": ["tema-0"],
            "entries": [],
        }

    monkeypatch.setattr(rp, "measure_axis", estrecho)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=True)
    _, rows = _rows(path)
    assert len(rows) == 288
    assert "estrecho" in capsys.readouterr().out.lower()


def test_el_resumen_cuenta_por_nivel(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    resumen = json.loads(
        (tmp_path / f"summary-{path.stem}.json").read_text(encoding="utf-8")
    )
    assert resumen["planned"] == 288
    assert resumen["completed"] == 288
    assert resumen["failed"] == 0
    assert resumen["by_level"]["N0"]["completed"] == 96
    assert resumen["by_level"]["N2"]["completed"] == 96


def test_el_presupuesto_de_llamadas_se_dice_antes_de_gastar(harness, tmp_path, capsys):
    presupuesto = rp.call_budget(plan_phase1a(1))
    # Por celda: el pegote más los dos turnos de D7.
    assert presupuesto["evaluated_model_calls"] == 288 * (1 + conv.N_POST_TURNS)
    assert presupuesto["simulated_user_calls"] == 288 * conv.N_POST_TURNS
    assert presupuesto["contradiction_calls"] == 48
    assert presupuesto["prefixes"] == 16
    rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    assert "llamadas" in capsys.readouterr().out.lower()
