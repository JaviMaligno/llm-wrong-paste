import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wrongpaste import config
from wrongpaste.config import (
    GATEWAY_URL_ENV,
    GCP_PROJECT_ENV,
    MODELS,
    MissingConfig,
    gateway_url,
    gcp_project,
)


def test_roster_has_nine_models():
    assert len(MODELS) == 9


def test_every_model_has_a_known_provider():
    providers = {m.provider for m in MODELS.values()}
    assert providers == {"gateway", "vertex_anthropic", "vertex_openai"}


def test_claude_models_are_vertex_anthropic():
    assert MODELS["claude-opus-5"].provider == "vertex_anthropic"
    assert MODELS["claude-sonnet-5"].provider == "vertex_anthropic"


def test_size_ladder_tiers_are_distinct():
    ladder = [m for m in MODELS.values() if m.tier in {"large", "medium", "small"}]
    assert len(ladder) >= 3


# --- D17: endpoint y proyecto fuera del repositorio público -----------------


def test_importar_config_sin_las_variables_no_revienta():
    """Importar no puede depender del entorno: si no, la suite offline muere.

    Se hace en un intérprete aparte y con el entorno limpio de `WRONGPASTE_*`,
    que es la única forma honesta de comprobarlo: en este proceso el `conftest`
    ya las ha rellenado.
    """
    entorno = {k: v for k, v in os.environ.items() if not k.startswith("WRONGPASTE_")}
    proceso = subprocess.run(
        [sys.executable, "-c", "import wrongpaste.config"],
        env=entorno,
        capture_output=True,
        text=True,
    )
    assert proceso.returncode == 0, proceso.stderr


def test_importar_no_evalua_los_ajustes_de_entorno():
    # Si fueran constantes evaluadas al importar, aparecerían en el módulo.
    assert "GATEWAY_URL" not in vars(config)
    assert "GCP_PROJECT" not in vars(config)


def test_pedir_la_url_del_gateway_sin_variable_lanza_error_util(monkeypatch):
    monkeypatch.delenv(GATEWAY_URL_ENV, raising=False)
    with pytest.raises(MissingConfig) as exc:
        gateway_url()
    mensaje = str(exc.value)
    assert GATEWAY_URL_ENV in mensaje
    assert "export" in mensaje


def test_pedir_el_proyecto_gcp_sin_variable_lanza_error_util(monkeypatch):
    monkeypatch.delenv(GCP_PROJECT_ENV, raising=False)
    with pytest.raises(MissingConfig) as exc:
        gcp_project()
    mensaje = str(exc.value)
    assert GCP_PROJECT_ENV in mensaje
    assert "export" in mensaje


def test_una_variable_en_blanco_cuenta_como_ausente(monkeypatch):
    # `export WRONGPASTE_GATEWAY_URL=` es el fallo más fácil de cometer y el
    # más difícil de leer en el traceback si se cuela como cadena vacía.
    monkeypatch.setenv(GATEWAY_URL_ENV, "   ")
    with pytest.raises(MissingConfig):
        gateway_url()


def test_la_url_del_gateway_sale_de_la_variable(monkeypatch):
    monkeypatch.setenv(GATEWAY_URL_ENV, "https://gateway.ejemplo")
    assert gateway_url() == "https://gateway.ejemplo"


def test_la_url_del_gateway_pierde_la_barra_final(monkeypatch):
    # Las llamadas concatenan "/v1/...": una barra de más daría un 404.
    monkeypatch.setenv(GATEWAY_URL_ENV, "https://gateway.ejemplo/")
    assert gateway_url() == "https://gateway.ejemplo"


def test_el_proyecto_gcp_sale_de_la_variable(monkeypatch):
    monkeypatch.setenv(GCP_PROJECT_ENV, "proyecto-ejemplo-123")
    assert gcp_project() == "proyecto-ejemplo-123"


def test_los_ajustes_se_releen_en_cada_llamada(monkeypatch):
    # Perezosos de verdad: nada de cachear el primer valor visto.
    monkeypatch.setenv(GATEWAY_URL_ENV, "https://uno.ejemplo")
    assert gateway_url() == "https://uno.ejemplo"
    monkeypatch.setenv(GATEWAY_URL_ENV, "https://dos.ejemplo")
    assert gateway_url() == "https://dos.ejemplo"


def test_los_nombres_en_mayusculas_siguen_resolviendo_desde_el_entorno(monkeypatch):
    # Compatibilidad para los usos `config.GATEWAY_URL` que quedan en el código.
    monkeypatch.setenv(GATEWAY_URL_ENV, "https://gateway.ejemplo")
    monkeypatch.setenv(GCP_PROJECT_ENV, "proyecto-ejemplo-123")
    assert config.GATEWAY_URL == "https://gateway.ejemplo"
    assert config.GCP_PROJECT == "proyecto-ejemplo-123"

    monkeypatch.delenv(GATEWAY_URL_ENV, raising=False)
    with pytest.raises(MissingConfig):
        config.GATEWAY_URL


def test_un_atributo_inexistente_sigue_siendo_attribute_error():
    with pytest.raises(AttributeError):
        config.NO_EXISTE_ESTE_AJUSTE


def test_config_no_versiona_ningun_endpoint_interno():
    """D17: en el fuente solo pueden quedar los dominios públicos de Google.

    Escrito como lista blanca a propósito: una lista negra con el host interno
    dentro volvería a publicarlo, esta vez en los tests.
    """
    fuente = Path(config.__file__).read_text(encoding="utf-8")
    dominios = set(re.findall(r"https?://([\w.-]+)", fuente))
    assert dominios <= {
        "aiplatform.googleapis.com",
        "us-central1-aiplatform.googleapis.com",
    }


def test_config_no_versiona_el_proyecto_ni_el_gateway_como_constante():
    fuente = Path(config.__file__).read_text(encoding="utf-8")
    assert re.search(r"^GATEWAY_URL\s*=", fuente, re.MULTILINE) is None
    assert re.search(r"^GCP_PROJECT\s*=", fuente, re.MULTILINE) is None
