"""N2: pegotes que contradicen la conversación. Distractor de diseño.

A diferencia de N0 y N1, aquí el artefacto SÍ se fabrica sabiendo de qué va la
conversación. Eso rompe a propósito la independencia del §4.1 del spec
original, así que N2 no se compara nunca con N0 ni con N1: se reporta aparte,
como el brazo donde buscamos el techo haciendo trampa.

La generación va partida en dos, igual que en `prefixes.py`:
`make_contradiction` fabrica y no toca el disco, y `ensure_contradiction`
—la puerta que usa el runner— reutiliza lo que ya haya guardado. Esa segunda
mitad no es comodidad: D1 exige que las dos réplicas de una celda compartan
prefijo **y** artefacto, y en N2 el artefacto lo escribe un modelo, de modo que
llamarlo dos veces daría dos textos distintos y las réplicas medirían otra cosa.
"""

import hashlib
import json
from pathlib import Path

from wrongpaste.artifacts import Artifact
from wrongpaste.clients import chat
from wrongpaste.topics import Topic

CONTRADICTION_MODEL = "gpt-5.6-terra-tst"  # el mismo que escribe los prefijos
# Se lee como global dentro de cada función para que los tests lo redirijan a
# un `tmp_path`, como en `prefixes.PREFIX_DIR`.
CONTRADICTION_DIR = Path(__file__).resolve().parents[2] / "runs" / "contradictions"

_SYSTEM = """Escribes fragmentos de texto que alguien podría tener en el
portapapeles: un trozo de correo, una nota, un mensaje suelto.

Te doy una conversación. Escribe un fragmento que CONTRADIGA algún dato
concreto de ella: otra ciudad, otra fecha, otra persona, otra cantidad. El
fragmento NO debe mencionar la conversación ni sonar a respuesta: es un texto
independiente que casualmente choca con lo que se ha dicho.

Entre 2 y 6 líneas. Todo inventado: nada de empresas, personas ni lugares
reales.

Responde SOLO con JSON:
{"text": "<el fragmento>", "entities": ["<3 a 5 términos distintivos que
aparezcan literalmente en el fragmento>"]}"""


def contradiction_path(prefix_id: str, seed: int) -> Path:
    return CONTRADICTION_DIR / f"{prefix_id}-{seed}.json"


def make_contradiction(topic: Topic, transcript: list[dict], seed: int) -> Artifact:
    """Fabrica una contradicción contra `transcript`. No toca el disco.

    Guardar es cosa de `save_contradiction`, para que los tests y cualquier
    exploración manual puedan generar sin ensuciar `runs/`.
    """
    conversacion = "\n".join(
        f"{m['role']}: {m['content']}" for m in transcript
    )
    prompt = f"Conversación:\n{conversacion}\n\nEscribe el fragmento."
    reply = chat(
        CONTRADICTION_MODEL,
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    crudo = reply.text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        datos = json.loads(crudo)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"el generador de N2 no devolvió JSON: {reply.text[:200]!r}"
        ) from exc

    # Un JSON válido pero incompleto no puede pasar en silencio: un artefacto
    # sin `entities` produce un `Artifact` perfectamente válido cuyo recuento
    # de fuga (D8) vale cero para siempre, y eso no se distingue de un modelo
    # que no filtró nada. Vale más una fila con `status` (D6) que un dato mudo.
    texto = str(datos.get("text", "")).strip()
    if not texto:
        raise ValueError(
            f"el generador de N2 devolvió JSON sin `text`: {crudo[:200]!r}"
        )
    entities = datos.get("entities") or []
    if not isinstance(entities, list) or not entities:
        raise ValueError(
            f"el generador de N2 devolvió JSON sin `entities`: {crudo[:200]!r}"
        )

    huella = hashlib.sha256(
        f"{topic.id}|{seed}|{texto}".encode()
    ).hexdigest()[:12]
    return Artifact(
        id=f"n2-{topic.id}-{seed}-{huella}",
        kind="contradiction",
        text=texto,
        entities=tuple(str(e) for e in entities),
        level="N2",
        signal="contradiccion",
    )


def save_contradiction(prefix_id: str, seed: int, artifact: Artifact) -> Path:
    """Persiste la contradicción en `runs/contradictions/<prefijo>-<semilla>.json`.

    Se guardan también `prefix_id` y `seed` dentro del fichero, y no solo en el
    nombre: el nombre se puede renombrar y el contenido no.
    """
    path = contradiction_path(prefix_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prefix_id": prefix_id,
        "seed": int(seed),
        "generator_model": CONTRADICTION_MODEL,
        "artifact": {
            "id": artifact.id,
            "kind": artifact.kind,
            "text": artifact.text,
            "entities": list(artifact.entities),
            "level": artifact.level,
            "signal": artifact.signal,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_contradiction(prefix_id: str, seed: int) -> Artifact | None:
    """Lee la contradicción de (prefijo, semilla), o `None` si no está."""
    path = contradiction_path(prefix_id, seed)
    if not path.is_file():
        return None
    datos = json.loads(path.read_text(encoding="utf-8"))["artifact"]
    return Artifact(
        id=datos["id"],
        kind=datos["kind"],
        text=datos["text"],
        entities=tuple(datos["entities"]),
        level=datos["level"],
        signal=datos["signal"],
    )


def ensure_contradiction(
    prefix_id: str, topic: Topic, transcript: list[dict], seed: int
) -> Artifact:
    """Devuelve la contradicción de (prefijo, semilla), generándola si falta.

    Es la puerta que usa el runner, y es lo que hace que las dos réplicas de
    una celda N2 compartan artefacto (D1) y que reanudar una tirada no vuelva a
    pagar el generador.
    """
    existente = load_contradiction(prefix_id, seed)
    if existente is not None:
        return existente
    artifact = make_contradiction(topic, transcript, seed)
    save_contradiction(prefix_id, seed, artifact)
    return artifact
