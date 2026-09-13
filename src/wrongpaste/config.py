from dataclasses import dataclass
from pathlib import Path

GCP_PROJECT = "data-science-364702"
GATEWAY_URL = "https://litellm.infra.skyc.cloud"
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
    return GATEWAY_KEY_PATH.read_text().strip()
