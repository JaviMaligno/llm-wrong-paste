import subprocess
from dataclasses import dataclass

import httpx
import numpy as np

from wrongpaste import config

_TIMEOUT = httpx.Timeout(300.0)


@dataclass
class Reply:
    text: str
    usage: dict
    raw: dict


def _gcp_token() -> str:
    out = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _gateway_body(model_id: str, messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }


def _vertex_openai_body(model_id: str, messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": f"google/{model_id}",
        "messages": messages,
        "max_tokens": max_tokens,
    }


def _anthropic_body(messages: list[dict], max_tokens: int) -> dict:
    if messages and messages[-1]["role"] == "assistant":
        raise ValueError(
            "prefill de turno de asistente eliminado en la familia Claude 5"
        )
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    body = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": convo,
        "max_tokens": max_tokens,
    }
    if system:
        body["system"] = system
    return body


def chat(model_id: str, messages: list[dict], max_tokens: int = 1024) -> Reply:
    model = config.MODELS[model_id]

    if model.provider == "gateway":
        resp = httpx.post(
            f"{config.GATEWAY_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.gateway_key()}"},
            json=_gateway_body(model_id, messages, max_tokens),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return Reply(data["choices"][0]["message"]["content"] or "", data.get("usage", {}), data)

    if model.provider == "vertex_openai":
        resp = httpx.post(
            config.VERTEX_OPENAI_URL.format(project=config.GCP_PROJECT),
            headers={"Authorization": f"Bearer {_gcp_token()}"},
            json=_vertex_openai_body(model_id, messages, max_tokens),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return Reply(data["choices"][0]["message"]["content"] or "", data.get("usage", {}), data)

    if model.provider == "vertex_anthropic":
        url = config.VERTEX_ANTHROPIC_URL.format(
            project=config.GCP_PROJECT, model=model_id
        )
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {_gcp_token()}",
                "x-goog-user-project": config.GCP_PROJECT,
            },
            json=_anthropic_body(messages, max_tokens),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        return Reply(text, data.get("usage", {}), data)

    raise ValueError(f"proveedor desconocido: {model.provider}")


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
