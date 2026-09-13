import json
import time
from pathlib import Path

import numpy as np

from wrongpaste.artifacts import load_artifacts
from wrongpaste.conversation import (
    MAX_TOKENS,
    build_prefix,
    conversation_text,
    inject_paste,
)
from wrongpaste.records import ConversationRecord
from wrongpaste.similarity import rank_artifacts
from wrongpaste.simulated_user import USER_MODEL
from wrongpaste.topics import load_topics

PHASE0_MODELS = ["gpt-5.6-sol-tst", "gpt-5.6-luna-tst", "claude-opus-5"]
LENGTHS = [2, 10]
STRATA = 8
OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "phase0"


def plan_phase0(seed: int) -> list[dict]:
    """Cada modelo recorre los 8 temas; la longitud alterna por tema.

    No es un factorial completo a propósito: la Fase 0 es para leer, no para
    contar, y 30 transcripciones es lo que se puede leer de una sentada.
    """
    topics = load_topics()
    rng = np.random.default_rng(seed)
    plan = []
    for model_id in PHASE0_MODELS:
        for i, topic in enumerate(topics):
            plan.append(
                {
                    "model_id": model_id,
                    "topic_id": topic.id,
                    "n_turns": LENGTHS[i % 2],
                    "seed": int(rng.integers(0, 2**31)),
                }
            )
    return plan


def main(seed: int = 20260913) -> Path:
    topics = {t.id: t for t in load_topics()}
    arts = load_artifacts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"

    with out.open("w", encoding="utf-8") as fh:
        for i, cell in enumerate(plan_phase0(seed)):
            topic = topics[cell["topic_id"]]
            rng = np.random.default_rng(cell["seed"])

            # 1. Prefijo primero: la similaridad se mide contra la conversación
            #    hasta el turno del pegote (spec §4.1), no contra el tema.
            transcript, usages = build_prefix(
                cell["model_id"], topic, cell["n_turns"]
            )

            # 2. Medir el banco entero contra ese prefijo y muestrear.
            #    En Fase 0 se recorre el estrato i para que las 24 celdas
            #    cubran el rango de similaridad entre todas.
            ranked = rank_artifacts(conversation_text(transcript), arts)
            stratum = i % STRATA
            lo = len(ranked) * stratum // STRATA
            hi = len(ranked) * (stratum + 1) // STRATA
            artifact, sim = ranked[int(rng.integers(lo, max(hi, lo + 1)))]

            # 3. Inyectar y registrar.
            reaction, paste_usage = inject_paste(
                cell["model_id"], transcript, artifact
            )
            usages.append(paste_usage)

            rec = ConversationRecord(
                model_id=cell["model_id"],
                topic_id=topic.id,
                artifact_id=artifact.id,
                artifact_kind=artifact.kind,
                artifact_entities=list(artifact.entities),
                similarity=sim,
                n_turns=cell["n_turns"],
                seed=cell["seed"],
                user_model=USER_MODEL,
                max_tokens=MAX_TOKENS,
                transcript=transcript,
                reaction=reaction,
                usages=usages,
            )
            fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{rec.model_id} / {rec.topic_id} / sim={sim:.3f}")

    return out


if __name__ == "__main__":
    print(main())
