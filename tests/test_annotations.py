"""Tests de las anotaciones manuales de la Fase 0 (D14).

Lo que se protege aquí es el único conjunto etiquetado a mano del proyecto: que
se escriba en JSONL de verdad, que vuelva igual al leerlo y que no se pueda
desincronizar en silencio del JSONL de la tirada.
"""

import json

import pytest

import wrongpaste.annotations as annotations
from wrongpaste.annotations import (
    Annotation,
    load_annotations,
    validate_against_run,
    write_annotations,
)

RUN_ID = "20260913T101500"


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Ningún test escribe en `runs/phase0` de verdad."""
    monkeypatch.setattr(annotations, "ANNOTATION_DIR", tmp_path / "phase0")


def _ann(cid: str = "p0-claude-opus-5-hacer-pan-10-r0", **kw) -> Annotation:
    base = dict(
        conversation_id=cid,
        annotator="javier",
        category_guess="puente_confabulado",
        quote="Entiendo que el log tiene que ver con la fermentación…",
        notes="no pregunta; enlaza el stacktrace con el horno en la primera frase",
    )
    base.update(kw)
    return Annotation(**base)


# --- persistencia -----------------------------------------------------------


def test_ruta_sigue_el_formato_de_d14():
    path = annotations.ANNOTATION_PATH(RUN_ID)

    assert path.name == f"annotations-{RUN_ID}.jsonl"
    assert path.parent == annotations.ANNOTATION_DIR


def test_ida_y_vuelta_de_escritura_y_lectura():
    anns = [_ann(), _ann("p0-gpt-5.6-luna-tst-mudanza-2-r0", category_guess="pivota")]

    path = write_annotations(RUN_ID, anns)

    assert path.exists()
    assert load_annotations(RUN_ID) == anns


def test_el_fichero_es_jsonl_de_verdad_una_linea_por_anotacion():
    """Cada línea parsea sola: es lo que permite leerlo y añadirle con `>>`."""
    anns = [
        _ann(),
        _ann("p0-gpt-5.6-luna-tst-mudanza-2-r0"),
        _ann("p0-claude-sonnet-5-elegir-camara-2-r1"),
    ]

    path = write_annotations(RUN_ID, anns)
    raw = path.read_text(encoding="utf-8")
    lineas = raw.splitlines()

    assert len(lineas) == 3
    assert raw.endswith("\n")
    parsed = [json.loads(linea) for linea in lineas]
    assert [p["conversation_id"] for p in parsed] == [
        a.conversation_id for a in anns
    ]
    assert set(parsed[0]) == {
        "conversation_id",
        "annotator",
        "category_guess",
        "quote",
        "notes",
    }


def test_las_citas_se_guardan_legibles_con_acentos():
    """`ensure_ascii=False`: media gracia del JSONL es poder leerlo a ojo."""
    path = write_annotations(RUN_ID, [_ann(quote="la fermentación en frío")])

    assert "la fermentación en frío" in path.read_text(encoding="utf-8")


def test_las_lineas_en_blanco_se_ignoran_al_leer():
    path = write_annotations(RUN_ID, [_ann()])
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert len(load_annotations(RUN_ID)) == 1


def test_escribir_reemplaza_el_fichero_entero():
    write_annotations(RUN_ID, [_ann(), _ann("p0-gpt-5.6-luna-tst-mudanza-2-r0")])

    write_annotations(RUN_ID, [_ann()])

    assert len(load_annotations(RUN_ID)) == 1


def test_leer_una_tirada_sin_anotar_revienta():
    """Que falte la anotación es un error visible, no una lista vacía."""
    with pytest.raises(FileNotFoundError):
        load_annotations(RUN_ID)


def test_una_linea_que_no_es_json_dice_que_linea_es():
    path = annotations.ANNOTATION_PATH(RUN_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"conversation_id": "a"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="línea 1"):
        load_annotations(RUN_ID)


# --- esquema ----------------------------------------------------------------


def test_category_guess_es_texto_libre():
    """La rúbrica todavía no existe: ese es justo el punto de la Fase 0."""
    for guess in ("ejecuta el prompt pegado", "???", "no encaja en ninguna"):
        write_annotations(RUN_ID, [_ann(category_guess=guess)])
        assert load_annotations(RUN_ID)[0].category_guess == guess


@pytest.mark.parametrize(
    "campo", ["conversation_id", "annotator", "category_guess"]
)
def test_los_campos_que_sostienen_el_cuadre_no_pueden_ir_vacios(campo):
    with pytest.raises(ValueError, match=campo):
        _ann(**{campo: "   "})


def test_quote_y_notes_pueden_faltar():
    ann = Annotation(
        conversation_id="p0-gpt-5.6-luna-tst-mudanza-2-r0",
        annotator="javier",
        category_guess="pivota",
    )

    assert ann.quote == ""
    assert ann.notes == ""


def test_un_campo_desconocido_al_leer_es_un_error():
    """Estos ficheros se escriben a mano; una errata se tiene que ver."""
    with pytest.raises(ValueError, match="categoria_guess"):
        Annotation.from_json(
            {
                "conversation_id": "p0-x-y-2-r0",
                "annotator": "javier",
                "categoria_guess": "pivota",
            }
        )


# --- cuadre contra la tirada (lo que impide la desincronización) ------------


def test_validate_en_verde_cuando_todo_cuadra():
    ids = ["p0-a-t-2-r0", "p0-b-t-2-r0"]
    anns = [_ann(cid) for cid in ids]

    assert validate_against_run(anns, ids) == {
        "unknown_ids": [],
        "unannotated_ids": [],
        "duplicate_ids": [],
    }


def test_validate_detecta_un_id_que_no_existe_en_la_tirada():
    report = validate_against_run(
        [_ann("p0-a-t-2-r0"), _ann("p0-inventado-t-2-r0")],
        ["p0-a-t-2-r0"],
    )

    assert report["unknown_ids"] == ["p0-inventado-t-2-r0"]
    assert report["unannotated_ids"] == []
    assert report["duplicate_ids"] == []


def test_validate_detecta_una_conversacion_sin_anotar():
    report = validate_against_run(
        [_ann("p0-a-t-2-r0")],
        ["p0-a-t-2-r0", "p0-b-t-2-r0", "p0-c-t-10-r0"],
    )

    assert report["unannotated_ids"] == ["p0-b-t-2-r0", "p0-c-t-10-r0"]
    assert report["unknown_ids"] == []


def test_validate_detecta_un_duplicado_del_mismo_anotador():
    report = validate_against_run(
        [_ann("p0-a-t-2-r0", category_guess="pivota"),
         _ann("p0-a-t-2-r0", category_guess="puente")],
        ["p0-a-t-2-r0"],
    )

    assert report["duplicate_ids"] == ["p0-a-t-2-r0"]


def test_dos_anotadores_sobre_la_misma_conversacion_no_son_un_duplicado():
    """Es el material con el que se mide el acuerdo entre lectores."""
    report = validate_against_run(
        [_ann("p0-a-t-2-r0", annotator="javier"),
         _ann("p0-a-t-2-r0", annotator="segundo-lector")],
        ["p0-a-t-2-r0"],
    )

    assert report == {
        "unknown_ids": [],
        "unannotated_ids": [],
        "duplicate_ids": [],
    }


def test_validate_junta_los_tres_problemas_a_la_vez():
    report = validate_against_run(
        [_ann("p0-a-t-2-r0"), _ann("p0-a-t-2-r0"), _ann("p0-fantasma-t-2-r0")],
        ["p0-a-t-2-r0", "p0-b-t-2-r0"],
    )

    assert report == {
        "unknown_ids": ["p0-fantasma-t-2-r0"],
        "unannotated_ids": ["p0-b-t-2-r0"],
        "duplicate_ids": ["p0-a-t-2-r0"],
    }


def test_validate_cuadra_con_lo_que_escribe_el_runner():
    """El ciclo real: se anota, se guarda, se relee y se cuadra."""
    ids = ["p0-a-t-2-r0", "p0-b-t-10-r1"]
    write_annotations(RUN_ID, [_ann(cid) for cid in ids])

    report = validate_against_run(load_annotations(RUN_ID), ids)

    assert report["unknown_ids"] == []
    assert report["unannotated_ids"] == []
    assert report["duplicate_ids"] == []


# --- la carpeta de la fase (Fase 1a escribe en `runs/phase1a`) --------------


def test_sin_carpeta_explicita_se_guarda_donde_siempre():
    """El comportamiento por defecto no cambia: la Fase 0 no se entera."""
    path = write_annotations(RUN_ID, [_ann()])

    assert path.parent == annotations.ANNOTATION_DIR


def test_una_tirada_de_otra_fase_se_guarda_en_su_carpeta(tmp_path):
    destino = tmp_path / "phase1a"

    path = write_annotations(RUN_ID, [_ann()], directory=destino)

    assert path == destino / f"annotations-{RUN_ID}.jsonl"
    assert path.exists()


def test_ida_y_vuelta_con_carpeta_explicita(tmp_path):
    destino = tmp_path / "phase1a"
    anns = [_ann(), _ann("p1a-gpt-5.6-luna-tst-mudanza-2-N1-r0")]

    write_annotations(RUN_ID, anns, directory=destino)

    assert load_annotations(RUN_ID, directory=destino) == anns


def test_dos_fases_con_el_mismo_run_id_no_se_pisan(tmp_path):
    """La carpeta es lo único que separa dos tiradas homónimas."""
    destino = tmp_path / "phase1a"
    write_annotations(RUN_ID, [_ann("p0-a-t-2-r0")])
    write_annotations(
        RUN_ID,
        [_ann("p1a-a-t-2-N0-r0"), _ann("p1a-b-t-10-N2-r1")],
        directory=destino,
    )

    fase0 = load_annotations(RUN_ID)
    fase1a = load_annotations(RUN_ID, directory=destino)

    assert [a.conversation_id for a in fase0] == ["p0-a-t-2-r0"]
    assert [a.conversation_id for a in fase1a] == [
        "p1a-a-t-2-N0-r0",
        "p1a-b-t-10-N2-r1",
    ]
