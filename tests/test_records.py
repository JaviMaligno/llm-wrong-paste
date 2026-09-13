"""Tests del esquema de fila: lo que se escribe al JSONL y no se puede recuperar
después si sale mal (D5, D6, D7, D11, D15/D17)."""

import dataclasses
import json

import pytest

from wrongpaste.records import (
    MESSAGE_TAGS,
    SCHEMA_VERSION,
    ConversationRecord,
    RunHeader,
    make_conversation_id,
    run_header_line,
)


def _control_record() -> ConversationRecord:
    """Celda de control de D11: misma conversación, sin pegote."""
    return ConversationRecord(
        run_id="p0-20260913T101500",
        cell_index=24,
        model_id="gpt-5.6-luna-tst",
        model_label="GPT-5.6 luna",
        topic_id="pan-masa-madre",
        n_turns=2,
        condition="no_paste",
        transcript=[
            {"role": "user", "content": "no me sube la masa", "tag": "opening"},
            {"role": "assistant", "content": "¿qué harina usas?", "tag": "assistant"},
        ],
    )


def test_control_sin_pegote_se_construye_y_serializa():
    """(a) El brazo sin pegote no necesita artefacto ni similaridad."""
    rec = _control_record()

    assert rec.condition == "no_paste"
    assert rec.artifact_id is None
    assert rec.artifact_kind is None
    assert rec.artifact_text is None
    assert rec.artifact_entities is None
    assert rec.similarity_user is None
    assert rec.similarity_full is None
    assert rec.paste_index is None
    assert rec.post_indices == []

    linea = json.dumps(rec.to_json(), ensure_ascii=False)
    assert json.loads(linea)["condition"] == "no_paste"


def test_conversation_id_sigue_el_formato_de_d5():
    """(b) `p0-{model}-{topic}-{n_turns}-r{replicate}`, determinista."""
    assert (
        make_conversation_id("claude-opus-5", "mudanza", 10, 2)
        == "p0-claude-opus-5-mudanza-10-r2"
    )

    rec = ConversationRecord(
        model_id="claude-opus-5", topic_id="mudanza", n_turns=10, replicate_idx=2
    )
    assert rec.conversation_id == "p0-claude-opus-5-mudanza-10-r2"

    # Dos construcciones del mismo plan dan el mismo id: es la clave con la que
    # `main()` se salta celdas ya escritas al reanudar (D6).
    otro = ConversationRecord(
        model_id="claude-opus-5", topic_id="mudanza", n_turns=10, replicate_idx=2
    )
    assert otro.conversation_id == rec.conversation_id


def test_conversation_id_explicito_no_se_pisa():
    rec = ConversationRecord(
        conversation_id="p0-fijado-a-mano",
        model_id="claude-opus-5",
        topic_id="mudanza",
        n_turns=10,
    )
    assert rec.conversation_id == "p0-fijado-a-mano"


def test_to_json_no_pierde_ningun_campo():
    """(c) Todo campo del dataclass acaba en la fila, y la fila es JSON."""
    esperados = {f.name for f in dataclasses.fields(ConversationRecord)}
    fila = _control_record().to_json()

    assert set(fila) == esperados
    # Y sobrevive al viaje de ida y vuelta por JSON sin perder claves.
    assert set(json.loads(json.dumps(fila, ensure_ascii=False))) == esperados


def test_la_fila_lleva_los_campos_que_exige_el_documento_de_correcciones():
    obligatorios = {
        # D5: identidad y trazabilidad
        "schema_version",
        "run_id",
        "conversation_id",
        "cell_index",
        "replicate_idx",
        "model_id",
        "model_label",
        "response_model",
        "topic_id",
        "n_turns",
        "stratum",
        "n_strata",
        "prefix_id",
        "condition",
        "arm",
        "parent_id",
        # D15/D17: género del artefacto y etiqueta del modelo
        "artifact_id",
        "artifact_kind",
        "artifact_text",
        "artifact_entities",
        # D2: las dos similaridades
        "similarity_user",
        "similarity_full",
        "similarity_rank",
        "similarity_pct",
        "ranking",
        "similarity_text_truncated",
        # D7: el pegote y los dos turnos posteriores
        "paste_index",
        "post_indices",
        "transcript",
        "reaction",
        # spec §4.2 y D9: lo que se envió
        "request_params",
        "system_prompt",
        "user_model",
        "prefix_model",
        "max_tokens",
        "stop_reasons",
        "usages",
        "seed",
        "started_at",
        "ended_at",
        "latency_ms",
        # D6: fallos como datos
        "status",
        "error_code",
        "error_body",
        "attempts",
    }
    assert obligatorios <= set(ConversationRecord.field_names())


def test_fila_con_pegote_y_turnos_posteriores():
    """D7: los dos turnos `post` se localizan por índice, no por adivinanza."""
    rec = ConversationRecord(
        model_id="gpt-5.6-sol-tst",
        topic_id="pan-masa-madre",
        n_turns=2,
        artifact_id="stack-03",
        artifact_kind="stacktrace",
        artifact_text="Traceback (most recent call last):",
        artifact_entities=["ConnectionResetError"],
        similarity_user=0.31,
        similarity_full=0.28,
        similarity_rank=57,
        similarity_pct=0.11,
        paste_index=4,
        post_indices=[6, 8],
        transcript=[{"role": "user", "content": "x", "tag": tag} for tag in MESSAGE_TAGS],
    )
    fila = rec.to_json()

    assert fila["condition"] == "paste"
    assert fila["post_indices"] == [6, 8]
    assert fila["artifact_kind"] == "stacktrace"
    assert all(m["tag"] in MESSAGE_TAGS for m in fila["transcript"])


def test_celda_fallida_se_registra_como_dato():
    """D6: un 429 o un rechazo escriben fila, no tumban la tirada."""
    rec = ConversationRecord(
        model_id="claude-opus-5",
        topic_id="mudanza",
        n_turns=10,
        status="http_error",
        error_code=429,
        error_body="rate limit",
        attempts=3,
        reaction=None,
    )
    fila = rec.to_json()

    assert fila["status"] == "http_error"
    assert fila["error_code"] == 429
    assert fila["attempts"] == 3
    assert fila["reaction"] is None
    assert json.loads(json.dumps(fila))["error_body"] == "rate limit"


@pytest.mark.parametrize("valor", ["error", "OK", "", "paste"])
def test_status_fuera_del_vocabulario_de_d6_revienta(valor):
    with pytest.raises(ValueError):
        ConversationRecord(model_id="m", topic_id="t", status=valor)


@pytest.mark.parametrize("valor", ["control", "sin_pegote", ""])
def test_condition_fuera_del_vocabulario_de_d11_revienta(valor):
    with pytest.raises(ValueError):
        ConversationRecord(model_id="m", topic_id="t", condition=valor)


def test_run_header_serializa_con_kind_run_header():
    """(d) La primera línea del JSONL se identifica sola."""
    header = RunHeader(
        run_id="p0-20260913T101500",
        phase="0",
        plan_path="docs/superpowers/plans/2026-09-13-pegado-accidental-fase-0.md",
        spec_path="docs/superpowers/specs/2026-09-13-pegado-accidental-design.md",
        code_sha="abc1234",
        bank_sha="def5678",
        topics_sha="9012ghi",
        master_seed=20260913,
        roster=["gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5"],
        planned_cells=24,
        embedding_model="text-embedding-3-small-tst",
        prefix_model="gpt-5.6-terra-tst",
        user_model="gpt-5.6-terra-tst",
        user_system_prompt="Eres una persona escribiendo a un asistente.",
        started_at=1789000000.0,
    )
    linea = run_header_line(header)

    assert linea["kind"] == "run_header"
    assert linea["run_id"] == "p0-20260913T101500"
    assert linea["planned_cells"] == 24
    assert linea["schema_version"] == SCHEMA_VERSION
    # Todos los campos de D5 viajan en la cabecera.
    assert set(linea) == {"kind"} | {
        f.name for f in dataclasses.fields(RunHeader)
    }
    assert json.loads(json.dumps(linea, ensure_ascii=False))["phase"] == "0"


def test_run_header_line_acepta_campos_sueltos():
    linea = run_header_line(run_id="p0-x", planned_cells=3)
    assert linea["kind"] == "run_header"
    assert linea["run_id"] == "p0-x"
    assert linea["planned_cells"] == 3


def test_run_header_line_rechaza_campos_inventados():
    with pytest.raises(TypeError):
        run_header_line(run_id="p0-x", numero_de_la_suerte=7)
