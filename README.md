# llm-wrong-paste — "perdona, eso era de otro chat"

Arnés experimental sobre el **pegado accidental**: ese bloque de texto que tenías
en el portapapeles para otra cosa y acaba enviado en medio de una conversación
con la que no tiene nada que ver. No es un ataque, no es un cambio de tema
deliberado y no es una petición mal especificada: es un error humano mundano cuya
característica definitoria es la **ambigüedad**. El pegote puede ser basura que
hay que ignorar, un cambio de tema intencionado o contexto relevante que el
usuario olvidó explicar, y el modelo no tiene forma de saber cuál de las tres.

La hipótesis que se mide aquí es que el fallo caro no es equivocarse de
interpretación —eso es inevitable— sino **resolver la ambigüedad en silencio**, y
en particular hacerlo construyendo un puente plausible entre el pegote y el tema
en curso. Como preguntar siempre sería insufrible, la conducta buena no es
"preguntar" sino "calibrar cuándo preguntar", y eso convierte la métrica en una
curva: la variable independiente es la **similaridad semántica** entre el pegote
y la conversación, medida con embeddings, y la dependiente es qué hace el modelo
justo después. El código de este repositorio construye conversaciones sintéticas
con un usuario simulado, les inyecta un artefacto de un banco de 64 pegotes
escritos a mano, deja correr dos turnos más y guarda **todo en crudo** en JSONL:
transcripción etiquetada, artefacto, las dos similaridades, el ranking completo,
los parámetros enviados y el `usage` devuelto. El análisis se hace después, sobre
esos ficheros; `runs/` se versiona porque los datos crudos son el producto.

## Configuración del entorno

```bash
uv venv && uv pip install -e ".[dev]"
```

Dos variables de entorno **sin valor por defecto**. El endpoint del gateway y el
proyecto de GCP no son secretos, pero sí material de trabajo interno, y este
repositorio es público: por eso no se versionan y por eso no hay un default
plausible que los revele.

| Variable | Qué es |
|---|---|
| `WRONGPASTE_GATEWAY_URL` | URL base del gateway LiteLLM (los modelos GPT y los embeddings) |
| `WRONGPASTE_GCP_PROJECT` | Proyecto de GCP con Vertex AI habilitado (los modelos Claude y Gemini) |

```bash
export WRONGPASTE_GATEWAY_URL=...
export WRONGPASTE_GCP_PROJECT=...
```

Se leen de forma perezosa, dentro de la función que las necesita. Si falta
alguna, la llamada falla con un mensaje que dice qué exportar; importar el
paquete sigue funcionando sin ellas.

Además:

- **Clave del gateway**: en `~/.acp-blog-paste-key`, una sola línea, sin comillas
  ni espacios. Es un fichero, no una variable, para que no acabe en el historial
  del shell ni en un `env` volcado a un log.
- **Credenciales de Vertex**: `gcloud auth application-default login`. El cliente
  saca el token con `gcloud auth print-access-token` en cada llamada.

## Suite offline

Es la única que se corre en el día a día. No toca la red, no cuesta nada y **no
necesita ninguna de las dos variables**:

```bash
uv run pytest -q -m "not live"
```

Los tests marcados `live` (`-m live`) sí hacen llamadas reales a los tres
proveedores para comprobar que responden. Cuestan céntimos, pero cuestan.

## Aviso: la tirada real cuesta dinero

`python -m wrongpaste.run_phase0` **gasta presupuesto de verdad**. La Fase 0 son
27 celdas (24 con pegote y 3 de control), varios turnos cada una contra tres
proveedores: del orden de **5 $ y unas 2 horas** de reloj. La Fase 1 es dos
órdenes de magnitud mayor.

Antes de lanzar nada:

1. Estimar y **decir en voz alta** tiempo y coste, partidos por proveedor. Si la
   estimación pasa de 10 $, parar y revisar el plan.
2. Correr antes el paso de ancho de eje (embeddings de los 64 artefactos y los 8
   prefijos, céntimos): si la similaridad no separa, la tirada no mide nada.
3. Una tirada interrumpida **se reanuda** sobre el mismo JSONL —se saltan los
   `conversation_id` ya escritos—, así que un fallo a mitad no obliga a pagar
   otra vez lo ya hecho.

El diseño y el plan de ejecución viven en el repositorio del blog, en
`docs/superpowers/specs/2026-09-13-pegado-accidental-design.md` y
`docs/superpowers/plans/2026-09-13-pegado-accidental-fase-0.md`.
