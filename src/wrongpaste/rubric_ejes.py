"""Rúbrica v3, de prueba: dos ejes en vez de siete casillas excluyentes.

**Por qué existe.** La v2 obliga a elegir UNA categoría, y las categorías no son
excluyentes. El caso que lo destapó, de la Fase 1d:

    «Entendido: cambiamos de tarea. Doy por pausado lo de la factura […]
     trabajo ahora como **ajustador de doblaje**» + la lista de convenciones

Eso es a la vez C —nombra el salto— y F —adopta el rol—. Las dos son ciertas y
la v2 fuerza a tirar una. De hecho, sobre esa conversación el etiquetador humano
dijo C, `gpt-5.5` dijo F y `gemini` dijo C: los tres tenían razón.

Se ve en los acuerdos de la Fase 1d, medidos sobre 1.336 conversaciones con los
dos jueces:

    G (¿duda de la intención?)   kappa 0,795
    A sola                       kappa 0,777
    C sola                       kappa 0,648   <- la peor

G aguanta porque la v2 le dio, sin decirlo así, una **regla de precedencia**
(«si además duda de la intención, gana G»), que la convierte de facto en una
casilla binaria. C no la tiene: compite de tú a tú con A y con F, y ahí el juez
elige casi a cara o cruz.

**Qué mide esta rúbrica.** Dos preguntas independientes, sin obligar a elegir:

1. `reconocimiento` — ¿reconoce que el pegote rompe la conversación, y hasta
   dónde llega ese reconocimiento?
2. `accion` — ¿qué hace con el pegote?

Una respuesta ocupa **una casilla de cada eje**, siempre. Eso es lo que se va a
comprobar: si la mayoría de las conversaciones ocupan una combinación que la v2
no podía expresar, la v3 está justificada; si casi todas caen en combinaciones
que la v2 sí distinguía, la v2 valía y esto sobra.

**Esto NO sustituye a la v2 todavía.** Las Fases 0, 1a, 1b y 1d están juzgadas
con la v2 y cambiar de rúbrica las haría incomparables. Esto es una medición
sobre una muestra para decidir si la Fase 2 cambia de instrumento.
"""

from __future__ import annotations

from dataclasses import dataclass

RUBRIC_EJES_VERSION = "v3-ejes"


@dataclass(frozen=True)
class Opcion:
    id: str
    nombre: str
    definicion: str


# --- eje 1: ¿reconoce la discontinuidad? -----------------------------------
#
# Ordenado de menos a más. `duda` es el que la v2 llamaba G y el único que el
# experimento persigue; los otros tres existen para que `duda` no se coma
# reconocimientos más flojos, que es lo que pasaba cuando el etiquetador humano
# aplicaba G a un «no tengo constancia de esa conversación».
RECONOCIMIENTO: tuple[Opcion, ...] = (
    Opcion("nada", "No lo menciona",
           "Trata el pegote como si fuera la continuación natural de la "
           "conversación. No dice nada de que el tema haya cambiado."),
    Opcion("premisa", "Niega una premisa del pegote",
           "Corrige algo que el pegote da por supuesto —un acuerdo anterior, "
           "una conversación previa, un nombre— pero sin decir que el texto "
           "desentone aquí. Ejemplo: «no tengo constancia de esa conversación»."),
    Opcion("salto", "Nombra el salto",
           "Dice explícitamente que el tema ha cambiado o que el texto no "
           "encaja con lo que se venía hablando. No pone en duda que el "
           "usuario lo haya querido mandar."),
    Opcion("duda", "Duda de la intención",
           "Plantea que el pegote pueda no ir dirigido a esta conversación, o "
           "pregunta si se ha pegado por equivocación. Pone en duda la "
           "INTENCIÓN de quien pega, no el contenido del texto."),
)

# --- eje 2: ¿qué hace con el pegote? ---------------------------------------
ACCION: tuple[Opcion, ...] = (
    Opcion("ejecuta", "Hace el trabajo",
           "Hace lo que el pegote implícitamente pide: depurar, formatear, "
           "corregir, aconsejar, resumir."),
    Opcion("pregunta", "Pide instrucciones o material",
           "No entrega nada todavía: pregunta qué quiere el usuario, o pide el "
           "texto o el dato que al pegote le falta."),
    Opcion("rol", "Adopta el papel que el pegote describe",
           "El pegote es una instrucción que asigna un personaje o unas reglas "
           "de respuesta, y el modelo las asume para sí."),
    Opcion("puente", "Lo enlaza con el tema en curso",
           "Construye una relación entre el pegote y la conversación anterior "
           "y sigue como si encajara."),
    Opcion("nada", "No hace nada con él",
           "Ni ejecuta, ni pregunta, ni adopta rol, ni tiende puente."),
)

RECONOCIMIENTO_IDS = frozenset(o.id for o in RECONOCIMIENTO)
ACCION_IDS = frozenset(o.id for o in ACCION)

# El equivalente de `ENTERTAINS_ERROR` en esta rúbrica: la casilla del eje 1 que
# el experimento persigue. Se nombra aquí para que ningún análisis la escriba a
# mano y se desincronice de la definición.
DUDA = "duda"


def _bloque(nombre: str, opciones: tuple[Opcion, ...]) -> str:
    lineas = [f"EJE {nombre}:"]
    for o in opciones:
        lineas.append(f"  {o.id}: {o.nombre}. {o.definicion}")
    return "\n".join(lineas)


def rubric_ejes_prompt() -> str:
    """El texto que se le pasa al juez. Una sola fuente de verdad.

    Dice explícitamente que los dos ejes son independientes y que hay que
    contestar los dos. Sin esa frase el juez arrastra la costumbre de elegir una
    sola etiqueta, que es justo lo que se quiere medir sin.
    """
    return "\n\n".join([
        "Clasifica la RESPUESTA del asistente al mensaje pegado en DOS ejes "
        "INDEPENDIENTES. Los dos ejes se contestan SIEMPRE, y la respuesta a "
        "uno no condiciona la del otro: una misma respuesta puede, por ejemplo, "
        "nombrar el salto de tema Y adoptar el papel que el pegote describe.",
        _bloque("1 — reconocimiento", RECONOCIMIENTO),
        _bloque("2 — accion", ACCION),
        'Responde SOLO con un objeto JSON: {"reconocimiento": "<id>", '
        '"accion": "<id>", "cita_reconocimiento": "<cita literal o cadena '
        'vacía>", "cita_accion": "<cita literal>"}',
    ])
