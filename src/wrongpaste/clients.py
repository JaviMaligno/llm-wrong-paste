import subprocess
import time
from dataclasses import dataclass

import httpx
import numpy as np

from wrongpaste import config

_TIMEOUT = httpx.Timeout(300.0)

# D9: muestreo explícito. No se delega en el default de cada proveedor, que
# puede cambiar sin avisar y que no es el mismo en los tres.
TEMPERATURE = 1.0

# D6: los fallos son datos, no caída. Tres intentos con backoff exponencial
# (1 s / 2 s / 4 s) para 429 y 5xx; el resto de 4xx son errores nuestros y no
# mejoran esperando, así que se propagan al primer intento.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
_RETRIABLE_STATUS = {429}


@dataclass
class Reply:
    text: str
    usage: dict
    raw: dict
    stop_reason: str | None = None
    response_model: str | None = None
    attempts: int = 1
    latency_ms: float | None = None


def _gcp_token() -> str:
    out = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _api_message(message: dict) -> dict:
    """Deja el mensaje como lo espera el proveedor: solo `role` y `content`.

    El transcript lleva además `tag` (D5), que es metadato nuestro: los
    proveedores rechazan claves desconocidas dentro de `messages`. Sanear aquí
    permite pasarle a `chat()` la transcripción etiquetada, que es la única
    forma de que el breakpoint de caché sepa dónde acaba el prefijo.
    """
    return {"role": message["role"], "content": message["content"]}


def _gateway_body(model_id: str, messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": model_id,
        "messages": [_api_message(m) for m in messages],
        "max_completion_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }


def _vertex_openai_body(model_id: str, messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": f"google/{model_id}",
        "messages": [_api_message(m) for m in messages],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }


def _as_blocks(content) -> list[dict]:
    """Normaliza el contenido de un mensaje a lista de bloques.

    Claude acepta `content` como cadena o como lista de bloques, pero
    `cache_control` solo se puede colgar de un bloque.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [dict(block) for block in content]


# Etiquetas (D5) que marcan que el prefijo compartido ya se acabó: el primer
# mensaje con una de ellas es el pegote, la reparación o un turno posterior,
# y ninguno de los tres pertenece al prefijo que se sirve de caché (D10).
POST_PREFIX_TAGS: frozenset[str] = frozenset({"paste", "repair", "post"})


def _prefix_breakpoint_index(convo: list[dict]) -> int | None:
    """Índice del mensaje donde acaba el prefijo compartido, o `None`.

    Con etiquetas (el caso normal: todo lo que fabrica `conversation.py`), el
    prefijo acaba en el **último mensaje con `tag == "assistant"` anterior al
    primer mensaje etiquetado con `POST_PREFIX_TAGS`**. No se supone ninguna
    posición: en la llamada del pegote ese mensaje es el penúltimo, pero en
    los dos turnos `post` de D7 y en las celdas de control de D11 el último
    mensaje es un turno de usuario posterior, y contar posiciones dejaría el
    breakpoint sobre la reacción al pegote o sobre un asistente de después —
    es decir, sobre texto que cambia de celda en celda y que por tanto no se
    puede cachear entre celdas que comparten `prefix_id`.

    Sin etiquetas (llamada suelta, transcripciones crudas) no hay forma de
    saber dónde acaba el prefijo: se **degrada** a marcar el penúltimo
    mensaje, que es lo correcto solo si el último es el pegote. Quien llame
    con transcripciones sin etiquetar asume esa degradación.
    """
    if not any(message.get("tag") for message in convo):
        return len(convo) - 2 if len(convo) >= 2 else None

    boundary = len(convo)
    for index, message in enumerate(convo):
        if message.get("tag") in POST_PREFIX_TAGS:
            boundary = index
            break

    for index in range(boundary - 1, -1, -1):
        if convo[index].get("tag") == "assistant":
            return index
    # Prefijo sin ninguna respuesta del asistente: no hay nada estable que
    # cachear delante del pegote.
    return None


def _mark_cacheable_prefix(convo: list[dict]) -> list[dict]:
    """Pone el breakpoint de caché al final del prefijo (D10).

    El prefijo —idéntico entre todas las celdas que comparten `prefix_id`— es
    lo que interesa servir de caché; lo que venga después (pegote, reacción,
    turnos `post`) cambia en cada celda. Dónde acaba el prefijo lo decide
    `_prefix_breakpoint_index` a partir de las etiquetas, no de la posición.

    Devuelve una lista nueva, ya en forma de API (sin `tag`): no muta la
    transcripción del llamante, que se guarda tal cual en el JSONL.
    """
    marked = [_api_message(m) for m in convo]
    target_index = _prefix_breakpoint_index(convo)
    if target_index is None:
        return marked
    target = marked[target_index]
    blocks = _as_blocks(target["content"])
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    target["content"] = blocks
    return marked


def _anthropic_body(messages: list[dict], max_tokens: int) -> dict:
    if messages and messages[-1]["role"] == "assistant":
        raise ValueError(
            "prefill de turno de asistente eliminado en la familia Claude 5"
        )
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    body = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": _mark_cacheable_prefix(convo),
        "max_tokens": max_tokens,
        # OJO (D9): `budget_tokens` devuelve 400 en Opus 5 y Sonnet 5. El modo
        # adaptativo es la única forma de razonamiento en la familia 5, y hay
        # que mandarlo siempre para que Sonnet 5 y Opus 5 sean comparables.
        "thinking": {"type": "adaptive"},
        # NO mandar `temperature` / `top_p` / `top_k`: en la familia Claude 5
        # los parámetros de muestreo están ELIMINADOS y devuelven 400, igual
        # que `budget_tokens`. La versión original de D9 pedía temperatura
        # explícita en los tres cuerpos; era un error y habría tumbado todas
        # las celdas de Claude. Aquí el muestreo lo fija el proveedor, y eso
        # se registra como tal en `request_params`.
    }
    if system:
        body["system"] = system
    return body


def _backoff_seconds(attempt: int) -> float:
    """Espera antes del intento `attempt + 1`: 1 s, 2 s, 4 s."""
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


def _is_retriable(status_code: int) -> bool:
    return status_code in _RETRIABLE_STATUS or status_code >= 500


def _post_with_retries(url: str, headers: dict, body: dict) -> tuple[httpx.Response, int, float]:
    """POST con reintentos de 429 y 5xx (D6).

    Devuelve la respuesta, el número de intentos consumidos y la latencia
    total en milisegundos (sueltas de backoff incluidas). Si se agotan los
    intentos, propaga el `HTTPStatusError` con el recuento de intentos
    colgado en el atributo `attempts`, para que el runner pueda registrarlo.
    """
    started = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = httpx.post(url, headers=headers, json=body, timeout=_TIMEOUT)
        if resp.status_code < 400:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return resp, attempt, latency_ms
        if attempt < MAX_ATTEMPTS and _is_retriable(resp.status_code):
            time.sleep(_backoff_seconds(attempt))
            continue
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            exc.attempts = attempt
            raise
    raise AssertionError("inalcanzable")  # pragma: no cover


def _openai_reply(data: dict, attempts: int, latency_ms: float) -> Reply:
    choice = (data.get("choices") or [{}])[0]
    return Reply(
        text=(choice.get("message") or {}).get("content") or "",
        usage=data.get("usage", {}),
        raw=data,
        stop_reason=choice.get("finish_reason"),
        response_model=data.get("model"),
        attempts=attempts,
        latency_ms=latency_ms,
    )


def chat(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 1024,
    request_params_out: dict | None = None,
) -> Reply:
    """Una llamada al modelo, con reintentos y trazabilidad.

    **`messages` se pasa ETIQUETADO** (con el `tag` de D5) siempre que venga de
    `conversation.py`. Quitar el `tag` es trabajo de este módulo y se hace en el
    último momento, dentro de cada `_*_body`, **después** de decidir dónde va el
    breakpoint de caché: las etiquetas son lo único que dice dónde acaba el
    prefijo compartido (D10). Si llegan mensajes sin etiquetar —una llamada
    suelta, el usuario simulado—, `_prefix_breakpoint_index` cae en su rama
    degradada, que solo acierta si el último mensaje es el pegote.

    `request_params_out`, si se pasa, se rellena con el cuerpo enviado sin
    `messages`, que es lo que cada fila del JSONL guarda como `request_params`
    (D9). Se rellena antes de llamar, así que también queda disponible cuando
    la llamada acaba en error.
    """
    model = config.MODELS[model_id]

    if model.provider == "gateway":
        url = f"{config.GATEWAY_URL}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.gateway_key()}"}
        body = _gateway_body(model_id, messages, max_tokens)
    elif model.provider == "vertex_openai":
        url = config.VERTEX_OPENAI_URL.format(project=config.GCP_PROJECT)
        headers = {"Authorization": f"Bearer {_gcp_token()}"}
        body = _vertex_openai_body(model_id, messages, max_tokens)
    elif model.provider == "vertex_anthropic":
        url = config.VERTEX_ANTHROPIC_URL.format(
            project=config.GCP_PROJECT, model=model_id
        )
        headers = {
            "Authorization": f"Bearer {_gcp_token()}",
            "x-goog-user-project": config.GCP_PROJECT,
        }
        body = _anthropic_body(messages, max_tokens)
    else:
        raise ValueError(f"proveedor desconocido: {model.provider}")

    if request_params_out is not None:
        request_params_out.clear()
        request_params_out.update({k: v for k, v in body.items() if k != "messages"})

    resp, attempts, latency_ms = _post_with_retries(url, headers, body)
    data = resp.json()

    if model.provider in ("gateway", "vertex_openai"):
        return _openai_reply(data, attempts, latency_ms)

    text = "".join(b["text"] for b in data.get("content", []) if b["type"] == "text")
    return Reply(
        text=text,
        usage=data.get("usage", {}),
        raw=data,
        stop_reason=data.get("stop_reason"),
        response_model=data.get("model"),
        attempts=attempts,
        latency_ms=latency_ms,
    )


def embed(texts: list[str]) -> np.ndarray:
    resp = httpx.post(
        f"{config.GATEWAY_URL}/v1/embeddings",
        headers={"Authorization": f"Bearer {config.gateway_key()}"},
        json={"model": config.EMBEDDING_MODEL, "input": texts},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    rows = sorted(resp.json()["data"], key=lambda d: d["index"])
    return np.array([r["embedding"] for r in rows], dtype=np.float32)
