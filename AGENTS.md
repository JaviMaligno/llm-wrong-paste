# AGENTS.md — llm-wrong-paste

Instrucciones para agentes (Codex, Claude Code y similares). El contexto del
experimento está en `README.md`; aquí va lo operativo.

## Qué es

Arnés experimental sobre el **pegado accidental** en conversaciones con LLMs:
construye conversaciones sintéticas con un usuario simulado, inyecta un pegote
de un banco escrito a mano, deja correr turnos posteriores y guarda **todo en
crudo** en JSONL. El análisis se hace después sobre esos ficheros. El
repositorio es **público** (GitHub).

## Estructura

- `src/wrongpaste/` — paquete único (Python ≥ 3.12, `httpx` + `numpy`).
  - `config.py` — endpoints y credenciales (lectura perezosa, ver abajo).
  - `clients.py`, `conversation.py`, `simulated_user.py`, `repair.py` — llamadas
    a modelos y forma de la conversación.
  - `artifacts.py`, `topics.py`, `prefixes.py`, `similarity.py`, `tasks.py`,
    `contradictions.py`, `leak.py` — estímulos y eje de similaridad.
  - `judging.py`, `judging_ejes.py`, `rubric.py`, `rubric_ejes.py`,
    `agreement.py`, `rates.py`, `curve.py`, `annotations.py`, `records.py` —
    jueces, rúbricas y métricas.
  - `run_phase0.py`, `run_phase1a.py`, `run_phase1b.py`, `run_phase1d.py`,
    `run_phase2.py`, `run_phase2_control.py`, `run_judging.py`,
    `run_judging_f2.py`, `analysis_phase2.py` — runners por fase.
- `data/` — bancos de entrada (`artifacts/`, `artifacts-n1/`, `topics/`).
- `docs/` — rúbricas (`rubrica-v1.md`, `rubrica-v2.md`) y notas de dominios.
- `runs/` — **datos crudos versionados**: son el producto del repositorio. No
  borrar ni reescribir ficheros de tiradas existentes.
- `tests/` — suite pytest; los tests con llamadas reales llevan el marcador `live`.

El diseño y los planes por fase viven en el repositorio del blog
(`docs/superpowers/specs/` y `docs/superpowers/plans/`), no aquí.

## Entorno

```bash
uv venv && uv pip install -e ".[dev]"
```

Configuración necesaria solo para llamadas reales (sin valor por defecto a
propósito, porque el repo es público):

- `WRONGPASTE_GATEWAY_URL` — URL base del gateway LiteLLM (modelos GPT y embeddings).
- `WRONGPASTE_GCP_PROJECT` — proyecto de GCP con Vertex AI (modelos Claude y Gemini).
- Clave del gateway en `~/.acp-blog-paste-key` (una línea, fichero y no variable).
- Credenciales de Vertex: `gcloud auth application-default login`.

Nunca escribir valores reales de estas variables (ni defaults plausibles) en el
código, los tests o la documentación. `tests/conftest.py` rellena valores
obviamente falsos para la suite offline.

## Tests

```bash
uv run python -m pytest -q -m "not live"
```

- Es la suite del día a día: sin red, sin coste, sin variables.
- Usar `python -m pytest` y no `uv run pytest` a secas: los tests importan
  `tests.*` y sin el directorio actual en `sys.path` fallan con
  `ModuleNotFoundError: No module named 'tests'`.
- `-m live` hace llamadas reales a los tres proveedores: cuesta dinero, solo
  bajo petición explícita.

## Tiradas reales: cuestan dinero

Los runners (`python -m wrongpaste.run_phase0`, `run_phase1a`, `run_phase1b`,
`run_phase1d`, `run_judging <run.jsonl>`, `run_judging_f2 <run.jsonl>`; Fase 2
vía `main()` de `run_phase2` / `run_phase2_control`) gastan presupuesto real.
Antes de lanzar cualquiera:

1. Estimar y **decir en voz alta** tiempo y coste por proveedor; si pasa de 10 $,
   parar y revisar el plan con el usuario.
2. Correr antes la medida del ancho de eje (embeddings, céntimos): si la
   similaridad no separa, la tirada no mide nada.
3. Las tiradas **se reanudan** sobre el mismo JSONL (se saltan los
   `conversation_id` ya escritos; los jueces se saltan por par conversación-juez).
   No relanzar desde cero una tirada interrumpida.

## Convenciones

- Idioma: código, docstrings, tests y commits en **español**. Los nombres de
  tests son frases (`test_al_reanudar_se_...`).
- Los docstrings de módulo explican el *porqué* de cada decisión de diseño
  (D3, D6, §x del spec…). Mantener ese estilo y no borrar esas justificaciones.
- Los fallos son datos: cada celda escribe siempre su fila con `status`; no
  filtrar ni descartar filas fallidas al escribir.
- Recuentos y presupuestos se derivan del plan, nunca de números escritos a mano.
- `runs/phase2/base-*-con-duda.jsonl` está en `.gitignore`: es derivable con
  `attach_judge_duda` a partir de ficheros versionados.
