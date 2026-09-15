"""Tests del runner de clasificación. Ninguno llama a un juez de verdad.

Lo que se prueba es lo que separa este módulo de un bucle cualquiera: que no
juzga lo que no es `ok`, que un fallo del juez se escribe en vez de perderse, y
que reanudar no vuelve a pagar lo ya pagado **pero sí reintenta lo fallido**.
"""

import json

import pytest

import wrongpaste.run_judging as rj
from wrongpaste.judging import JUDGES, Verdict


def _fila(cid, status="ok", level="N0"):
    return {
        "conversation_id": cid,
        "status": status,
        "paste_level": level,
        "model_id": "gpt-5.6-sol-tst",
        "topic_id": "carrera-10k",
        "n_turns": 2,
        "artifact_signal": None,
        "reaction": f"reacción de {cid}",
        "artifact_text": "pegote",
    }


def _escribir(path, filas, header=True):
    lineas = [{"kind": "run_header", "phase": "1a"}] if header else []
    lineas += filas
    path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lineas) + "\n",
        encoding="utf-8",
    )


def _leer(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture
def jueces_falsos(monkeypatch):
    """Un veredicto por juez, y la lista de (juez, reacción) que se pidieron."""
    llamadas = []

    def fake_judge_one(judge_model, reaction, paste_text):
        llamadas.append((judge_model, reaction))
        return Verdict("A", "cita", 0.9, judge_model)

    monkeypatch.setattr(rj, "judge_one", fake_judge_one)
    return llamadas


def test_solo_se_juzgan_las_filas_ok(tmp_path, jueces_falsos):
    """Una `truncated` se cortó por nuestro max_tokens: no es conducta completa."""
    run = tmp_path / "20260101T000000.jsonl"
    _escribir(run, [_fila("a"), _fila("b", status="truncated"), _fila("c", status="empty")])
    out = rj.main(run)
    filas = _leer(out)
    assert {f["conversation_id"] for f in filas} == {"a"}
    assert len(jueces_falsos) == len(JUDGES), "las no-ok ni se mandan al juez"


def test_cada_conversacion_deja_una_linea_por_juez(tmp_path, jueces_falsos):
    """Separados a propósito: resolverlos aquí borraría el desacuerdo (§6)."""
    run = tmp_path / "20260101T000000.jsonl"
    _escribir(run, [_fila("a"), _fila("b")])
    filas = _leer(rj.main(run))
    assert len(filas) == 2 * len(JUDGES)
    for cid in ("a", "b"):
        jueces = {f["judge_model"] for f in filas if f["conversation_id"] == cid}
        assert jueces == set(JUDGES)


def test_el_nivel_y_el_modelo_viajan_con_el_veredicto(tmp_path, jueces_falsos):
    run = tmp_path / "20260101T000000.jsonl"
    _escribir(run, [_fila("a", level="N1")])
    fila = _leer(rj.main(run))[0]
    assert fila["paste_level"] == "N1"
    assert fila["model_id"] == "gpt-5.6-sol-tst"
    assert fila["category"] == "A"


def test_un_fallo_del_juez_se_escribe_en_vez_de_perderse(tmp_path, monkeypatch):
    """Un fallo borrado es un denominador que nadie puede auditar."""
    def judge_mitad(judge_model, reaction, paste_text):
        if judge_model == JUDGES[0]:
            return Verdict("A", "cita", 0.9, judge_model)
        return Verdict(None, "", None, judge_model, status="http_error", error="429")

    monkeypatch.setattr(rj, "judge_one", judge_mitad)
    run = tmp_path / "20260101T000000.jsonl"
    _escribir(run, [_fila("a")])
    filas = _leer(rj.main(run))
    assert len(filas) == 2
    malo = [f for f in filas if f["status"] != "ok"][0]
    assert malo["category"] is None and malo["error"] == "429"


def test_reanudar_no_repaga_lo_usable_pero_reintenta_lo_fallido(tmp_path, monkeypatch):
    """La diferencia entre reanudar y dar por buena una tirada a medias."""
    fallar = {"si": True}
    llamadas = []

    def judge_inestable(judge_model, reaction, paste_text):
        llamadas.append(judge_model)
        if fallar["si"] and judge_model == JUDGES[1]:
            return Verdict(None, "", None, judge_model, status="http_error", error="429")
        return Verdict("A" if fallar["si"] else "B", "cita", 0.9, judge_model)

    monkeypatch.setattr(rj, "judge_one", judge_inestable)
    run = tmp_path / "20260101T000000.jsonl"
    _escribir(run, [_fila("a")])
    out = rj.main(run)

    fallar["si"] = False
    llamadas.clear()
    rj.main(run)
    # Y sobre todo: se llama SOLO al juez que faltaba. Pedir los dos y tirar el
    # bueno duplica la factura de cada reanudación.
    assert llamadas == [JUDGES[1]], f"repagó a {llamadas}"
    filas = _leer(out)
    # El juez que ya había respondido no se vuelve a pagar: sigue con su 'A'.
    buenos = [f for f in filas if f["status"] == "ok"]
    assert [f["judge_model"] for f in buenos if f["category"] == "A"] == [JUDGES[0]]
    # El que falló se reintentó y ahora tiene veredicto usable.
    assert any(
        f["judge_model"] == JUDGES[1] and f["status"] == "ok" for f in filas
    ), "el veredicto fallido no se reintentó al reanudar"


def test_la_ruta_de_veredictos_sale_del_nombre_de_la_tirada(tmp_path):
    run = tmp_path / "20260914T135814.jsonl"
    assert rj.verdicts_path(run).name == "verdicts-20260914T135814.jsonl"
