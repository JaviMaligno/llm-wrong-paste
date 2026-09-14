from dataclasses import asdict, fields

import pytest

from wrongpaste.agreement import agreement_report, blind_sample
from wrongpaste.judging import Verdict
from wrongpaste.rubric import RUBRIC_VERSION


def _veredicto(categoria: str, i: int = 0) -> Verdict:
    """Un `Verdict` real, con todos sus campos llenos.

    Nada de diccionarios inventados: la fila que verá `blind_sample` en
    producción sale de unir el registro con `asdict(verdict)`, así que los
    tests parten de ahí. Construirla a mano con los nombres que le convienen al
    test es justo lo que dejó pasar que la categoría del juez saliera en la
    muestra "a ciegas".
    """
    return Verdict(
        category=categoria,
        quote=f"la cita que justificó {categoria} ({i})",
        confidence=0.87,
        judge_model="gpt-5.5-tst",
        rubric_version=RUBRIC_VERSION,
        raw=f'{{"category": "{categoria}", "quote": "…"}}',
    )


def _fila(i: int, categoria: str | None = None) -> dict:
    """Fila unida como la unirá producción: registro + `asdict(verdict)`."""
    cat = categoria if categoria is not None else "ABCDEFG"[i % 7]
    return {
        "conversation_id": f"c{i}",
        "paste_level": ["N0", "N1", "N2"][i % 3],
        "model_id": ["m1", "m2", "m3"][i % 3],
        "transcript": [{"role": "user", "content": "…"}],
        "reaction": "…",
        **asdict(_veredicto(cat, i)),
    }


def _filas(n):
    return [_fila(i) for i in range(n)]


def _no_del_juez(fila: dict) -> set[str]:
    """Claves de la fila que NO vienen del veredicto."""
    del_juez = {f.name for f in fields(Verdict)}
    return {k for k in fila if k not in del_juez}


def test_la_muestra_es_estratificada_y_determinista():
    a = blind_sample(_filas(300), n=120, seed=3)
    b = blind_sample(_filas(300), n=120, seed=3)
    assert [r["conversation_id"] for r in a] == [r["conversation_id"] for r in b]
    assert len(a) == 120
    assert len({r["paste_level"] for r in a}) == 3
    assert len({r["model_id"] for r in a}) == 3


def test_la_muestra_no_lleva_el_veredicto_del_juez():
    """La fila entra con `asdict(verdict)` dentro y sale sin nada de él."""
    fila = _fila(6)  # i % 7 == 6 → categoría G
    assert fila["category"] == "G", "la fila de partida sí trae el veredicto"

    (salida,) = blind_sample([fila], n=1, seed=1)

    assert set(salida) == _no_del_juez(fila)
    assert set(salida) == {"conversation_id", "paste_level", "model_id",
                           "transcript", "reaction"}


def test_ningun_campo_de_verdict_sobrevive_al_cegado():
    """Guarda: si `Verdict` gana o renombra un campo, este test lo ve.

    Recorre los campos REALES del dataclass, no una lista escrita aquí, que es
    exactamente la clase de lista que se desincroniza en silencio.
    """
    muestra = blind_sample(_filas(300), n=30, seed=1)
    assert muestra
    for fila in muestra:
        for f in fields(Verdict):
            assert f.name not in fila, (
                f"el campo {f.name!r} de Verdict se coló en la muestra a ciegas"
            )


def test_el_cegado_falla_cerrado_cuando_un_nombre_colisiona():
    """`status` es campo de `Verdict` y del registro: se esconde igual.

    Leer los campos del dataclass se lleva por delante los nombres que el
    registro comparte con el veredicto. Es la dirección buena del error: al
    etiquetador le sobra el estado de la celda y le falta —jamás— la categoría
    del juez. Queda fijado aquí para que sea una decisión y no un susto.
    """
    assert "status" in {f.name for f in fields(Verdict)}
    fila = {**_fila(0), "status": "ok"}

    (salida,) = blind_sample([fila], n=1, seed=0)

    assert "status" not in salida
    assert {"transcript", "reaction"} <= set(salida), "lo que sí hace falta sigue"


def test_ningun_campo_del_veredicto_sobrevive_aunque_cambie_de_nombre():
    """La lista exacta no basta: `verdict` en singular filtraba igual."""
    fila = {"conversation_id": "c0", "paste_level": "N0", "model_id": "m1",
            "transcript": "…", "judge_category": "G", "judge_model": "gpt-5.5-tst",
            "judge_raw": '{"category": "G"}', "verdict": {"category": "G"},
            "verdicts": [{"category": "G"}]}
    (salida,) = blind_sample([fila], n=1, seed=0)
    assert set(salida) == {"conversation_id", "paste_level", "model_id", "transcript"}


def test_la_muestra_estratifica_tambien_por_categoria_del_juez():
    """Si el corpus tiene Gs, la muestra lleva alguna.

    G es la categoría que decide la puerta y se espera casi vacía: un muestreo
    ciego a la categoría podría no incluir ni una y dejar sin comprobar justo
    el número del que depende la decisión. Aquí se vuelve escasa a propósito
    —dos filas de 200— y la muestra es pequeña.
    """
    filas = [_fila(i, "G" if i in (7, 101) else "A") for i in range(200)]
    ids_g = {f["conversation_id"] for f in filas if f["category"] == "G"}
    assert len(ids_g) == 2

    muestra = blind_sample(filas, n=4, seed=5)

    # La categoría ya está cegada en la salida: las Gs se reconocen por su id.
    assert ids_g & {r["conversation_id"] for r in muestra}, (
        "la muestra no incluyó ninguna G: el tercer eje del §6.2 no estratifica"
    )


def test_la_categoria_se_lee_del_campo_prefijado_si_es_el_que_viene():
    """Mismo eje aunque la fila traiga `judge_category` en vez de `category`."""
    filas = [{"conversation_id": f"c{i}", "paste_level": "N0", "model_id": "m1",
              "judge_category": "G" if i in (7, 101) else "A"} for i in range(200)]
    ids_g = {f["conversation_id"] for f in filas if f["judge_category"] == "G"}

    muestra = blind_sample(filas, n=4, seed=5)

    assert ids_g & {r["conversation_id"] for r in muestra}


def test_el_informe_da_acuerdo_y_kappa_por_juez():
    humano = {"c1": "A", "c2": "B", "c3": "C"}
    jueces = {"j1": {"c1": "A", "c2": "B", "c3": "C"},
              "j2": {"c1": "A", "c2": "B", "c3": "G"}}
    inf = agreement_report(humano, jueces)
    assert inf["j1"]["raw"] == 1.0
    assert inf["j2"]["raw"] == pytest.approx(2 / 3)
    assert "kappa" in inf["j1"]
    assert inf["n"] == 3


def test_el_informe_falla_si_no_hay_solape():
    with pytest.raises(ValueError):
        agreement_report({"c1": "A"}, {"j1": {"c9": "A"}})
