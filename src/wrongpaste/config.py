"""Configuración del arnés: plantel de modelos y puntos de acceso.

El endpoint del gateway y el identificador del proyecto de GCP **no se
versionan** (D17). No son secretos, pero sí material de trabajo interno, y este
repositorio es público porque los datos crudos de `runs/` son el producto. Se
leen de las variables de entorno `WRONGPASTE_GATEWAY_URL` y
`WRONGPASTE_GCP_PROJECT`, sin valor por defecto: cualquier default plausible
publicaría justo lo que se quiere dejar fuera.

La lectura es **perezosa** —dentro de la función que necesita el valor, no al
importar— para que importar este módulo siga funcionando sin entorno. Eso es lo
que permite que la suite offline entera corra en una máquina recién clonada.
"""

import os
from dataclasses import dataclass
from pathlib import Path

GATEWAY_URL_ENV = "WRONGPASTE_GATEWAY_URL"
GCP_PROJECT_ENV = "WRONGPASTE_GCP_PROJECT"

GATEWAY_KEY_PATH = Path.home() / ".acp-blog-paste-key"

VERTEX_ANTHROPIC_URL = (
    "https://aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/global/publishers/anthropic/models/{model}:rawPredict"
)
VERTEX_OPENAI_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project}"
    "/locations/us-central1/endpoints/openapi/chat/completions"
)

EMBEDDING_MODEL = "text-embedding-3-small-tst"


class MissingConfig(RuntimeError):
    """Falta un ajuste de entorno obligatorio para hablar con un proveedor."""


def _required_env(env_var: str, que_es: str) -> str:
    """Devuelve la variable de entorno, o explica cómo ponerla.

    El mensaje tiene que bastar para desatascarse sin leer el código: qué falta,
    para qué sirve y por qué no viene puesta de fábrica.
    """
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise MissingConfig(
            f"Falta la variable de entorno {env_var} ({que_es}). "
            "No se versiona en este repositorio, que es público, así que hay "
            "que exportarla antes de cualquier llamada real:\n"
            f"    export {env_var}=...\n"
            "Los tests offline (`pytest -m \"not live\"`) no la necesitan."
        )
    return value


def gateway_url() -> str:
    """URL base del gateway LiteLLM, sin barra final."""
    return _required_env(GATEWAY_URL_ENV, "URL base del gateway LiteLLM").rstrip("/")


def gcp_project() -> str:
    """Identificador del proyecto de GCP donde vive Vertex AI."""
    return _required_env(GCP_PROJECT_ENV, "proyecto de GCP con Vertex AI habilitado")


def __getattr__(name: str) -> str:
    """Resuelve `config.GATEWAY_URL` y `config.GCP_PROJECT` de forma perezosa.

    PEP 562: Python solo llama aquí cuando el nombre **no** existe como global
    del módulo, así que estos dos nombres se leen del entorno en el momento de
    usarlos —y fallan con el mensaje de `_required_env`— en vez de quedar
    congelados al importar. Es compatibilidad para los usos que todavía los
    tratan como constantes; lo que se debe llamar es `gateway_url()` y
    `gcp_project()`.
    """
    if name == "GATEWAY_URL":
        return gateway_url()
    if name == "GCP_PROJECT":
        return gcp_project()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    label: str
    tier: str


_ROSTER = [
    Model("gpt-5.6-sol-tst", "gateway", "GPT-5.6 sol", "large"),
    Model("gpt-5.6-terra-tst", "gateway", "GPT-5.6 terra", "medium"),
    Model("gpt-5.6-luna-tst", "gateway", "GPT-5.6 luna", "small"),
    Model("gpt-5.4-tst", "gateway", "GPT-5.4", "prev_large"),
    Model("gpt-5.4-mini-tst", "gateway", "GPT-5.4 mini", "prev_small"),
    Model("claude-opus-5", "vertex_anthropic", "Claude Opus 5", "large"),
    Model("claude-sonnet-5", "vertex_anthropic", "Claude Sonnet 5", "medium"),
    Model("gemini-2.5-pro", "vertex_openai", "Gemini 2.5 Pro", "large"),
    Model("gemini-2.5-flash", "vertex_openai", "Gemini 2.5 Flash", "small"),
]

MODELS = {m.id: m for m in _ROSTER}


def gateway_key() -> str:
    """Clave virtual del gateway, leída del fichero del desarrollador."""
    if not GATEWAY_KEY_PATH.exists():
        raise MissingConfig(
            f"No existe {GATEWAY_KEY_PATH}. Ahí va la clave virtual del "
            "gateway, en una sola línea y sin comillas."
        )
    return GATEWAY_KEY_PATH.read_text().strip()
