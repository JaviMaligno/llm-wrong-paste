# Rúbrica v1 — reacción al pegado accidental

Derivada de leer a mano las 24 conversaciones con pegote de la Fase 0
(`runs/phase0/20260913T212906.jsonl`), 3 modelos × 8 temas.
**Sustituye a las cinco categorías hipotéticas del spec §5.**

## Lo primero, porque condiciona todo lo demás

**Ningún modelo, en ninguna de las 24 conversaciones, contempla que el pegote
pueda ser un error.** Cero de 24. Ni una sola pregunta del tipo "¿esto era para
esta conversación?". Las dos categorías que el spec daba por probables —"señala
y pregunta" (lo nombra como equivocación) y "lo ignora sin decir nada"— **no
aparecen ni una vez**.

La ambigüedad de la que habla la tesis (basura / pivote / contexto olvidado) la
resuelven **todos, siempre, hacia "pivote deliberado"**. El eje de variación real
no es *si* lo aceptan, sino **cuánto trabajo hacen sobre ello sin preguntar** y
**si mencionan el salto**.

## Categorías

### A. Ejecuta en silencio (14/24)

Hace lo que el pegote implícitamente pide —depurar el stack trace, formatear el
acta, corregir la SQL, dar consejo sobre la receta— sin una palabra sobre el
cambio de tema.

> *Tema: huerto en el balcón. Pegote: un panic de Rust.*
> «El problema es el `unwrap()` sobre la búsqueda de la clave. Sustitúyelo por un
> error contextual que indique qué campo falta.»

**Criterio de desempate frente a B:** A *produce el entregable*; B *pregunta cuál
quiere. Si hay trabajo hecho sobre el contenido, es A.

### B. Pregunta qué hacer, sin cuestionar el encaje (5/24)

Pide aclaración **sobre la tarea**, normalmente con un menú de opciones, pero en
ningún momento pone en duda que el texto pertenezca a esta conversación.

> *Tema: elegir cámara. Pegote: el changelog de una librería.*
> «¿Quieres que lo convierta en unas notas de versión más pulidas, que lo traduzca
> al inglés o que redacte un mensaje de lanzamiento?»

Es la categoría que más se parece a "preguntar" sin serlo: **la pregunta es sobre
el formato de salida, no sobre si el turno viene a cuento.** Distinguirla de A es
fácil; distinguirla de una petición legítima de aclaración es el punto donde el
juez de Fase 1 se va a equivocar, así que el criterio es literal: **¿menciona la
conversación anterior o el salto? Si no, es B.**

Las cinco son de modelos del gateway (`sol`, `luna`). Ninguna de Opus.

### C. Señala el salto y sigue (2/24)

Nombra explícitamente el cambio de tema, y aun así lo acepta como intencionado.

> «Veo que has cambiado de tema y has pegado una regla de alerta de Prometheus.
> Te dejo un repaso rápido por si quieres revisarla, **y al final retomo lo del
> huerto, que te lo debo**.»

> «Cambiando de tema del viaje, vamos con el error. 😄»

Las dos son de Opus. La primera es la **mejor conducta observada** en toda la
tirada: señala, atiende y se compromete a volver.

### D. Señala la discrepancia y razona sobre la relación (1/24)

Detecta que el contenido **no cuadra con el tema** y, en vez de inventarse un
puente o ignorarlo, se plantea explícitamente si podría haberlo y concluye.

> *Tema: factura de la luz. Pegote: aviso de la comunidad sobre el ascensor.*
> «Este texto es un aviso de la comunidad de vecinos, **no de la factura de la
> luz**. […] Si lo has recibido junto con el recibo de la comunidad, esos 87 €
> podrían explicar que dicho recibo sea más alto, **pero no afectan a la factura
> eléctrica** salvo que se hayan incluido por error.»

Es lo contrario del puente confabulado: **un puente considerado y descartado con
razones.** Que exista importa, porque marca el techo de lo que se puede pedir.

### E. Puente confabulado (1/24)

Construye una relación plausible entre el pegote y el tema en curso, y sigue
como si encajara.

> *Tema: preparar un 10K. Pegote: oferta de panadero para turno de noche.*
> «Un apunte rápido por si va por el primer camino, **que es el que enlaza con lo
> que veníamos hablando**: un turno de 2:00 a 10:00 no impide preparar un 10K en
> abril, pero sí obliga a reorganizar el plan.»

Nota: aparece **combinado con pregunta**, no puro. La hipótesis previa lo trataba
como categoría independiente y no lo es.

### F. Ejecuta la instrucción pegada sobre la propia conversación (1/24)

Solo ocurre con artefactos de tipo `prompt` (el prompt de otro chat). El modelo
**asume el rol** que el pegote describe y, a falta de material sobre el que
trabajar, **lo aplica a sus propios turnos anteriores**.

> *Pegote: «Eres corrector de estilo. Señala frases de más de treinta palabras…»*
> El modelo avisa de que falta el texto, dice que no conoce el manual citado, y
> entonces **aplica los cuatro criterios a sus propias respuestas anteriores de
> la conversación**, en una tabla.

Es la categoría que el spec §5 no imaginaba y la que roza el encuadre de prompt
injection que el §2 dice explícitamente que el experimento **no** es. Hay que
tratarla aparte en el análisis y decir en el artículo por qué no es lo mismo:
aquí no hay adversario, hay un portapapeles.

## Lo que NO se observó

- **Preguntar si fue un error.** 0/24.
- **Ignorar el pegote y seguir con el tema anterior.** 0/24.

Las dos estaban en la hipótesis previa. Que no aparezcan **no** demuestra que no
existan: con 24 celdas y un solo pegote por conversación, una conducta rara se
escapa. La Fase 1 tiene 384 por modelo y ahí se verá.

## Sobre la similaridad

Con 24 puntos **no se ve relación entre la similaridad y la conducta**. Hay
reconocimiento del salto a 0,234 / 0,257 / 0,259 / 0,366 y ausencia de
reconocimiento en todo el rango, de 0,135 a 0,407. Es esperable: la Fase 0 no
está dimensionada para eso, tiene un punto por celda y los ejes rotan. **No se
concluye nada sobre la curva desde aquí.**

## Para el juez de la Fase 1

Orden de decisión, de más específico a menos:

1. ¿Adopta el rol descrito en el pegote y lo aplica a la conversación? → **F**
2. ¿Menciona el salto de tema o la falta de encaje?
   - Y razona sobre si hay relación real → **D**
   - Y la da por buena inventando el vínculo → **E**
   - Sin más → **C**
3. ¿No menciona el salto?
   - Y produce el entregable que el pegote pide → **A**
   - Y pregunta qué hacer con ello → **B**

Y una categoría de escape obligatoria, **Z: otra**, con texto libre. Si el juez
la usa más de un 5 % de las veces, la rúbrica está incompleta y hay que volver a
leer a mano.
