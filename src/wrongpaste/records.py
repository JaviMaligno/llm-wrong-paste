"""Esquema de las filas que se escriben en los JSONL de cada tirada.

Aquí vive todo lo que hay que poder reconstruir *después* de gastar el dinero:
identidad de la celda (D5), resultado o fallo (D6), índices de los turnos
posteriores al pegote (D7), brazo de control sin pegote (D11) y las dos
etiquetas que sostienen la interpretación del artículo — `model_label` y
`artifact_kind` (D15/D17).

El fichero no llama a nadie: es hoja del grafo de importaciones a propósito,
para que el runner, el verificador y el análisis puedan importarlo sin
arrastrar clientes HTTP.
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

# Versión del esquema de fila. Se sube cuando un cambio rompe a quien lea
# ficheros antiguos: el análisis puede así ramificar por versión en vez de
# adivinar qué campos existían.
SCHEMA_VERSION = 2

# Número de estratos de similaridad por defecto (spec §4.1, D3).
N_STRATA = 8

# Vocabulario cerrado de D6: un fallo es un dato, no una caída.
Status = Literal["ok", "http_error", "timeout", "refusal", "empty"]
STATUSES: tuple[str, ...] = ("ok", "http_error", "timeout", "refusal", "empty")

# D11: el brazo de control no lleva pegote.
Condition = Literal["paste", "no_paste"]
CONDITIONS: tuple[str, ...] = ("paste", "no_paste")

# Etiquetas de mensaje del transcript (D5).
MESSAGE_TAGS: tuple[str, ...] = (
    "opening",
    "user_sim",
    "paste",
    "repair",
    "assistant",
    "post",
)

# Brazos de reparación del spec §7. La Fase 0 corre siempre la variante (a),
# "el usuario sigue como si el pegote no existiera" (D7).
ARMS: tuple[str, ...] = ("a", "b", "c", "d")
ARM_NO_REPAIR = "a"


def make_conversation_id(
    model_id: str, topic_id: str, n_turns: int, replicate_idx: int
) -> str:
    """Identificador determinista y legible de una conversación (D5).

    Formato: ``p0-{model}-{topic}-{n_turns}-r{replicate}``. Es la clave que usa
    `main()` para saltarse celdas ya escritas al reanudar una tirada (D6), así
    que tiene que salir igual en dos ejecuciones distintas del mismo plan.
    """
    return f"p0-{model_id}-{topic_id}-{n_turns}-r{replicate_idx}"


@dataclass
class ConversationRecord:
    """Una fila del JSONL: una conversación completa, haya ido bien o mal.

    Todos los campos llevan valor por defecto para que las celdas de control
    (D11, sin artefacto) y las celdas fallidas (D6, sin reacción) se puedan
    construir sin inventar datos.
    """

    # --- identidad (D5) ---------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    conversation_id: str = ""
    cell_index: int = -1
    replicate_idx: int = 0

    # --- modelo evaluado (D17: el alias interno y la etiqueta pública) -----
    model_id: str = ""
    model_label: str = ""
    # Lo que el proveedor dice haber ejecutado, que no siempre es lo pedido.
    response_model: str = ""

    # --- celda del diseño (D3) --------------------------------------------
    topic_id: str = ""
    n_turns: int = 0
    stratum: int = -1
    n_strata: int = N_STRATA
    # D1: el prefijo se genera una vez por (tema, longitud) y se reproduce
    # idéntico a todos los modelos; esta es su huella.
    prefix_id: str = ""

    # --- brazo (D11 y spec §7) --------------------------------------------
    condition: str = "paste"
    arm: str = ARM_NO_REPAIR
    # Conversación de la que esta cuelga (mismo prefijo y pegote, otro brazo
    # de reparación). En Fase 0 siempre None.
    parent_id: str | None = None

    # --- artefacto pegado (opcional: el control no pega nada, D11) --------
    artifact_id: str | None = None
    # D15: el género del artefacto se registra en cada fila porque registro y
    # similaridad son colineales por construcción y hay que poder medirlo.
    artifact_kind: str | None = None
    artifact_text: str | None = None
    # Se copian las entidades del banco: el análisis de fuga no debe depender
    # de que el banco no haya cambiado entre tanto.
    artifact_entities: list[str] | None = None

    # --- similaridad (D2) -------------------------------------------------
    # Primaria: contra los turnos de usuario, independiente del modelo evaluado.
    similarity_user: float | None = None
    # Secundaria: contra la conversación entera. Si divergen, se verá.
    similarity_full: float | None = None
    similarity_rank: int | None = None
    similarity_pct: float | None = None
    # Los 64 pares {artifact_id, similarity} del ranking de esta celda.
    ranking: list[dict] | None = None
    # True si hubo que recortar el texto antes de embeberlo (guarda de D2).
    similarity_text_truncated: bool = False

    # --- transcripción ----------------------------------------------------
    # Índice del mensaje del pegote dentro de `transcript`; None en el control.
    paste_index: int | None = None
    # D7: índices de los dos turnos posteriores a la reacción, etiquetados
    # `post`, sobre los que se pilota el recuento de entidades.
    post_indices: list[int] = field(default_factory=list)
    # Cada mensaje lleva `tag` ∈ MESSAGE_TAGS (D5).
    transcript: list[dict] = field(default_factory=list)
    # Texto de la reacción al pegote; None si la celda falló o es control.
    reaction: str | None = None

    # --- lo que se envió (spec §4.2, D9) ----------------------------------
    # Cuerpo de la petición sin `messages`: temperature, thinking, max_tokens.
    request_params: dict = field(default_factory=dict)
    system_prompt: str | None = None
    user_model: str = ""
    # D1: modelo que hizo de asistente al fabricar el prefijo compartido.
    prefix_model: str = ""
    max_tokens: int = 0
    # Un `stop_reason` por turno, en orden de llamada (D6).
    stop_reasons: list[str | None] = field(default_factory=list)
    usages: list[dict] = field(default_factory=list)
    seed: int = 0

    # --- reloj ------------------------------------------------------------
    started_at: float | None = None
    ended_at: float | None = None
    latency_ms: int | None = None

    # --- resultado (D6) ---------------------------------------------------
    status: str = "ok"
    error_code: int | str | None = None
    error_body: str | None = None
    attempts: int = 1

    def __post_init__(self) -> None:
        if not self.conversation_id and self.model_id and self.topic_id:
            self.conversation_id = make_conversation_id(
                self.model_id, self.topic_id, self.n_turns, self.replicate_idx
            )
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"condition inválida: {self.condition!r}; esperaba {CONDITIONS}"
            )
        if self.status not in STATUSES:
            raise ValueError(
                f"status inválido: {self.status!r}; esperaba {STATUSES}"
            )

    def to_json(self) -> dict[str, Any]:
        """Diccionario listo para `json.dumps`, sin perder ningún campo."""
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Nombres de los campos, para que el verificador compruebe columnas."""
        return tuple(f.name for f in fields(cls))


@dataclass
class RunHeader:
    """Primera línea de cada JSONL: de dónde salió esta tirada (D5).

    Sirve para dos cosas: reproducir la tirada (semilla maestra, plantel, sha
    del código y de los datos) y detectar un fichero truncado — si la cabecera
    dice `planned_cells: 24` y hay 19 filas, el fichero está incompleto.
    """

    run_id: str = ""
    phase: str = "0"
    plan_path: str = ""
    spec_path: str = ""
    # sha del commit del código con el que se corrió.
    code_sha: str = ""
    # sha del contenido del banco de artefactos y del fichero de temas: si el
    # banco cambia (D12 lo contempla), las tiradas dejan de ser comparables.
    bank_sha: str = ""
    topics_sha: str = ""
    master_seed: int = 0
    roster: list[str] = field(default_factory=list)
    planned_cells: int = 0
    embedding_model: str = ""
    prefix_model: str = ""
    user_model: str = ""
    user_system_prompt: str = ""
    started_at: float | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def run_header_line(header: RunHeader | None = None, **overrides: Any) -> dict:
    """Devuelve la cabecera como dict con ``kind="run_header"`` delante.

    Se puede llamar con un `RunHeader` ya construido, con campos sueltos, o con
    ambas cosas (los campos sueltos ganan sobre los del objeto).
    """
    base = header or RunHeader()
    data = base.to_json()
    unknown = set(overrides) - set(data)
    if unknown:
        raise TypeError(
            f"campos desconocidos para RunHeader: {sorted(unknown)}"
        )
    data.update(overrides)
    return {"kind": "run_header", **data}
