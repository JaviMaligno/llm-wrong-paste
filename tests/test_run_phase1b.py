"""Tests de la Fase 1b. Ninguno llama a un modelo.

Lo que se prueba es lo único que 1b añade sobre 1a: que la similaridad pasa de
ser un estrato a ser la variable independiente, y que el barrido cubre el
ranking entero sin repetir ni saltarse los extremos.
"""

import collections
import inspect
import json

import pytest

import wrongpaste.conversation as conv
import wrongpaste.prefixes as pfx
import wrongpaste.run_phase0 as rp0
import wrongpaste.run_phase1b as rp
import wrongpaste.similarity as sim
import wrongpaste.simulated_user as su
from wrongpaste.artifacts import Artifact
from wrongpaste.run_phase1b import (
    LENGTHS,
    PHASE1B_MODELS,
    SWEEP_POSITIONS,
    choose_sweep_artifact,
    plan_phase1b,
    sweep_index,
)


def test_el_barrido_toca_los_dos_extremos_del_ranking():
    """Sin los extremos, el eje medido es más corto que el eje que existe."""
    assert sweep_index(0, 64) == 0
    assert sweep_index(SWEEP_POSITIONS - 1, 64) == 63


def test_el_barrido_no_repite_indice_en_un_banco_de_64():
    idx = [sweep_index(p, 64) for p in range(SWEEP_POSITIONS)]
    assert len(set(idx)) == SWEEP_POSITIONS
    assert idx == sorted(idx), "el barrido tiene que ir de menos a más parecido"


def test_el_barrido_aguanta_un_banco_mas_pequeno_que_el_numero_de_posiciones():
    """Con 5 artefactos y 12 posiciones hay repetición, y es preferible a reventar."""
    idx = [sweep_index(p, 5) for p in range(SWEEP_POSITIONS)]
    assert min(idx) == 0 and max(idx) == 4
    assert idx == sorted(idx)


def test_una_posicion_fuera_de_rango_es_un_error():
    with pytest.raises(ValueError):
        sweep_index(SWEEP_POSITIONS, 64)
    with pytest.raises(ValueError):
        sweep_index(-1, 64)


def test_el_plan_tiene_288_celdas():
    # 12 posiciones x 8 temas x 3 modelos, sin réplicas
    assert len(plan_phase1b(1)) == SWEEP_POSITIONS * 8 * len(PHASE1B_MODELS)


def test_cada_tema_y_modelo_recorre_el_barrido_entero():
    """Si un (tema, modelo) no ve todo el eje, su curva es de otro tramo."""
    plan = plan_phase1b(1)
    por_celda = collections.defaultdict(set)
    for x in plan:
        por_celda[(x["topic_id"], x["model_id"])].add(x["sweep_position"])
    assert len(por_celda) == 8 * len(PHASE1B_MODELS)
    for clave, posiciones in por_celda.items():
        assert posiciones == set(range(SWEEP_POSITIONS)), clave


def test_la_longitud_esta_equilibrada_dentro_de_cada_posicion():
    """Si una posición cayera entera en conversaciones cortas, la curva mediría
    la longitud disfrazada de parecido: es la confusión que D3 deshacía en 1a."""
    plan = plan_phase1b(1)
    for pos in range(SWEEP_POSITIONS):
        largos = collections.Counter(
            x["n_turns"] for x in plan if x["sweep_position"] == pos
        )
        assert set(largos) == set(LENGTHS), pos
        assert len(set(largos.values())) == 1, f"posición {pos}: {dict(largos)}"


def test_la_longitud_esta_equilibrada_dentro_de_cada_tema_y_modelo():
    plan = plan_phase1b(1)
    por_celda = collections.defaultdict(collections.Counter)
    for x in plan:
        por_celda[(x["topic_id"], x["model_id"])][x["n_turns"]] += 1
    for clave, largos in por_celda.items():
        assert set(largos) == set(LENGTHS), clave
        assert len(set(largos.values())) == 1, f"{clave}: {dict(largos)}"


def test_conversation_id_unico_y_determinista():
    ids = [x["conversation_id"] for x in plan_phase1b(1)]
    assert len(ids) == len(set(ids))
    assert plan_phase1b(9) == plan_phase1b(9)


def test_el_plan_no_lleva_estrato():
    """El estrato era el instrumento de 1a. Aquí la posición ES la variable, y
    dejar los dos invitaría a analizar por el que no toca."""
    assert "stratum" not in plan_phase1b(1)[0]


# --- el muestreo por posición ---------------------------------------------


def _ranking(n=64):
    """Ranking ascendente por coseno, como lo devuelve `rank_artifacts` (D2)."""
    return [
        (Artifact(id=f"a{i}", kind="email", text="x", entities=(), level="N0"),
         i / (n - 1))
        for i in range(n)
    ]


def test_la_posicion_cero_es_el_artefacto_menos_parecido():
    art, sim = choose_sweep_artifact(_ranking(), 0)
    assert art.id == "a0" and sim == 0.0


def test_la_ultima_posicion_es_el_mas_parecido():
    art, sim = choose_sweep_artifact(_ranking(), SWEEP_POSITIONS - 1)
    assert art.id == "a63" and sim == 1.0


def test_el_muestreo_no_depende_de_lo_ya_elegido():
    """A diferencia de 1a, aquí no hay recuento que evolucione: la misma posición
    sobre el mismo ranking da siempre el mismo artefacto. Sin eso, la tirada no
    sería reanudable sin arrastrar estado."""
    r = _ranking()
    assert choose_sweep_artifact(r, 5) == choose_sweep_artifact(r, 5)


# --- el runner -------------------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Las mismas puertas de red que dobla la Fase 1a, reapuntadas a 1b."""
    from tests.test_run_phase1a import (
        FAKE_TOPICS,
        _reset_calls,
        fake_continue_after_paste,
        fake_ensure_prefix,
        fake_inject_paste,
        fake_load_artifacts,
        fake_rank_artifacts,
    )

    # Los dobles de 1a llevan la cuenta de las llamadas en un dict de módulo que
    # hay que inicializar antes de usarlos; sin esto, el primer prefijo revienta
    # con KeyError en cuanto esta suite se corre sola.
    _reset_calls()
    monkeypatch.setattr(rp, "load_topics", lambda: list(FAKE_TOPICS))
    monkeypatch.setattr(rp, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(rp, "ensure_prefix", fake_ensure_prefix)
    monkeypatch.setattr(rp, "inject_paste", fake_inject_paste)
    monkeypatch.setattr(rp, "continue_after_paste", fake_continue_after_paste)
    monkeypatch.setattr(rp, "OUT_DIR", tmp_path / "phase1b")
    monkeypatch.setattr(rp0, "rank_artifacts", fake_rank_artifacts)
    for modulo in (conv, su, sim):
        monkeypatch.setattr(
            modulo, "chat" if modulo is not sim else "embed",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("llamada real a un modelo por un camino sin doblar")
            ),
        )
    return tmp_path


def _rows(path):
    lineas = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lineas[0], lineas[1:]


def test_la_tirada_escribe_una_fila_por_celda_con_su_posicion(harness, tmp_path):
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    header, rows = _rows(path)
    assert header["phase"] == "1b"
    assert header["planned_cells"] == 288
    assert len(rows) == 288
    assert all(r["paste_level"] == "N0" for r in rows)
    assert {r["sweep_position"] for r in rows} == set(range(rp.SWEEP_POSITIONS))


def test_la_similaridad_crece_con_la_posicion(harness, tmp_path):
    """Es la comprobación que dice que el eje existe. Si la similaridad media no
    creciera con la posición, el barrido estaría midiendo otra cosa."""
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    medias = []
    for pos in range(rp.SWEEP_POSITIONS):
        sims = [r["similarity_user"] for r in rows if r["sweep_position"] == pos]
        medias.append(sum(sims) / len(sims))
    assert medias == sorted(medias), medias
    assert medias[-1] > medias[0]


def test_el_eje_guarda_el_coseno_del_usuario_y_no_el_de_la_conversacion(
    harness, tmp_path, monkeypatch
):
    """La similaridad primaria es la del lado del usuario (D2), y es el eje x.

    `fake_rank_artifacts` ignora el `text` y ordena por la lista del banco, así
    que bajo él los dos rankings que saca `rank_for_prefix` —uno del texto del
    usuario, otro de la conversación entera— salen idénticos en orden y en
    valor, y guardar el secundario bajo el nombre del primario no rompe nada.
    El test de arriba, que mira si la media crece, tampoco lo vería.

    Aquí se dobla `rank_artifacts` de forma que los dos lados den valores
    distintos —el del usuario ascendente, el de la conversación descendente—
    para que la fila diga cuál de los dos cosenos se guardó. Importa porque
    `curve.by_kind` mide el rango por género leyendo `similarity_user`: con el
    coseno equivocado ahí, el desglose de D15 mediría otra variable sin avisar.
    """
    from tests.test_run_phase1a import FAKE_BANKS

    n = len(FAKE_BANKS["N0"])

    def rank_por_lado(text, arts, max_chars=sim.MAX_EMBED_CHARS):
        # El transcripto del prefijo doblado es apertura (usuario) + "prosa"
        # (asistente): `conversation_text` lleva la prosa y `user_text` no.
        if "prosa" in text:
            return [(a, 1.0 - i / (len(arts) - 1)) for i, a in enumerate(arts)], False
        return [(a, i / (len(arts) - 1)) for i, a in enumerate(arts)], False

    monkeypatch.setattr(rp0, "rank_artifacts", rank_por_lado)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)

    esperado = {pos: rp.sweep_index(pos, n) / (n - 1) for pos in range(rp.SWEEP_POSITIONS)}
    assert len(set(esperado.values())) == rp.SWEEP_POSITIONS, "el doble no separa"
    for fila in rows:
        primaria = esperado[fila["sweep_position"]]
        assert fila["similarity_user"] == pytest.approx(primaria), fila["conversation_id"]
        assert fila["similarity_full"] == pytest.approx(1.0 - primaria), (
            fila["conversation_id"]
        )


def test_una_celda_rota_no_tumba_la_tirada(harness, tmp_path, monkeypatch):
    from tests.test_run_phase1a import fake_inject_paste

    def revienta(model_id, transcript, artifact, request_params_out=None,
                 max_tokens=conv.MAX_TOKENS):
        if artifact.id.endswith("3"):
            raise RuntimeError("se cayó el gateway")
        return fake_inject_paste(model_id, transcript, artifact,
                                 request_params_out=request_params_out,
                                 max_tokens=max_tokens)

    monkeypatch.setattr(rp, "inject_paste", revienta)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=False)
    _, rows = _rows(path)
    assert len(rows) == 288, "una celda rota no se descarta en silencio"
    rotas = [r for r in rows if r["status"] != "ok"]
    assert rotas
    for r in rotas:
        # `http_error`, y NO `harness_error`: la llamada que revienta es la del
        # pegote, que la hace el modelo evaluado. D6 decide el origen por la
        # etapa (`failure_origin`), y marcar esto como fallo del arnés sería
        # decir que la celda no llegó a preguntarle nada al modelo. Es el mismo
        # status que comprueba el test equivalente de la Fase 1a.
        assert r["status"] == "http_error"
        # La posición viaja aunque la celda falle: es un dato del diseño, no un
        # resultado, y sin ella no se puede saber si el hueco que deja la
        # pérdida cae en un extremo del barrido o repartido.
        assert r["sweep_position"] is not None
        # Y el nivel también. La fila de la Fase 0 no sabe nada de niveles, así
        # que `failed_record` se lo pone a mano; si esa línea se pierde o se
        # equivoca, las celdas rotas salen del recuento de pérdidas por nivel
        # —o peor, se cuentan como de un brazo que esta tanda no corrió, porque
        # 1b es de un solo brazo (N0)—.
        assert r["paste_level"] == "N0"


def test_la_tirada_se_reanuda_sin_duplicar_filas(harness, tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    rp.main(seed=1, out=path, measure=False)
    _, rows = _rows(path)
    header, quedan = _rows(path)
    path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [header, *quedan[:100]]) + "\n",
        encoding="utf-8",
    )
    rp.main(seed=1, out=path, measure=False)
    _, rows2 = _rows(path)
    assert len(rows2) == 288
    assert len({r["conversation_id"] for r in rows2}) == 288


def test_los_dobles_declaran_la_firma_de_la_funcion_real():
    from tests.test_run_phase1a import (
        fake_continue_after_paste,
        fake_ensure_prefix,
        fake_inject_paste,
        fake_load_artifacts,
        fake_rank_artifacts,
    )
    from wrongpaste import artifacts as arts_mod

    for doble, real in [
        (fake_load_artifacts, arts_mod.load_artifacts),
        (fake_ensure_prefix, pfx.ensure_prefix),
        (fake_rank_artifacts, sim.rank_artifacts),
        (fake_inject_paste, conv.inject_paste),
        (fake_continue_after_paste, conv.continue_after_paste),
        # Los dos dobles del eje son de esta suite, y entran aquí por lo mismo:
        # la puerta de D12 llama a `measure_axis` por palabra clave, y un doble
        # con otros nombres de parámetro dejaría pasar una llamada que en
        # producción reventaría.
        (fake_measure_axis_estrecho, rp0.measure_axis),
        (fake_measure_axis_ancho, rp0.measure_axis),
    ]:
        assert list(inspect.signature(doble).parameters) == list(
            inspect.signature(real).parameters
        ), doble.__name__


# --- la puerta del eje (D12) -----------------------------------------------


def fake_measure_axis_estrecho(topics=None, arts=None, lengths=(2, 10), run_id=""):
    """Informe de eje con TODAS las celdas estrechas.

    Las celdas llevan `n_turns` porque las de `measure_axis` lo llevan siempre:
    un doble más permisivo que la función real ejercita un camino que producción
    no recorre.
    """
    return {
        "run_id": run_id,
        "narrow_cells": [
            {"topic_id": t.id, "n_turns": n} for t in (topics or []) for n in lengths
        ],
        "narrow_topics": [t.id for t in (topics or [])],
        "entries": [],
    }


def fake_measure_axis_ancho(topics=None, arts=None, lengths=(2, 10), run_id=""):
    """Informe de eje sin ninguna celda estrecha: el barrido tiene dónde barrer."""
    return {"run_id": run_id, "narrow_cells": [], "narrow_topics": [], "entries": []}


def test_la_puerta_del_eje_para_la_tirada_si_el_eje_es_estrecho(
    harness, tmp_path, monkeypatch
):
    """Un eje estrecho hace decorativo el barrido entero: mejor no pagarlo."""
    monkeypatch.setattr(rp, "measure_axis", fake_measure_axis_estrecho)
    with pytest.raises(rp.NarrowAxisError):
        rp.main(seed=1, out=tmp_path / "t.jsonl", measure=True)
    assert not (tmp_path / "t.jsonl").exists(), "no se escribe nada si no se corre"


def test_el_informe_del_eje_queda_escrito_aunque_la_puerta_salte(
    harness, tmp_path, monkeypatch
):
    """La evidencia de por qué no se corrió es justo lo que hay que mirar después.

    Es la misma convención que la puerta de la Fase 0: el informe se escribe
    antes de levantar el error, no después de decidir que hay tirada.
    """
    monkeypatch.setattr(rp, "measure_axis", fake_measure_axis_estrecho)
    with pytest.raises(rp.NarrowAxisError):
        rp.main(seed=1, out=tmp_path / "t.jsonl", measure=True)
    assert rp0.axis_path("t", out=tmp_path / "t.jsonl").exists()


def test_con_el_eje_ancho_la_tirada_sigue(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "measure_axis", fake_measure_axis_ancho)
    path = rp.main(seed=1, out=tmp_path / "t.jsonl", measure=True)
    _, rows = _rows(path)
    assert len(rows) == 288
