# Rúbrica v2 — reacción al pegado accidental

Sucede a [`rubrica-v1.md`](rubrica-v1.md), que se derivó de leer a mano las 24
conversaciones con pegote de la Fase 0 (`runs/phase0/20260913T212906.jsonl`), 3
modelos × 8 temas. **Los recuentos que aparecen abajo son los de aquella lectura
y se conservan tal cual**: la Fase 1 no reetiqueta la Fase 0 (spec de Fase 1,
§8).

**La fuente de verdad ejecutable es `src/wrongpaste/rubric.py`.** Este documento
es la versión legible; el prompt que ve el juez se construye desde el módulo, no
desde aquí, para que no puedan divergir. Los nombres de las categorías son los
del módulo, que abrevian los de la v1; las definiciones y las citas son las
mismas.

## Los dos cambios frente a la v1

1. **Se añade la categoría G, «contempla que sea un error»**, con su desempate
   frente a C.
2. **Las dos tasas del spec §4 dejan de ser una suma escrita a mano** y pasan a
   ser conjuntos de categorías en el módulo: `MENTIONS_JUMP = {C, D, E, G}` y
   `ENTERTAINS_ERROR = {G}`, con `ENTERTAINS_ERROR ⊆ MENTIONS_JUMP`.

Todo lo demás —A a F y Z, sus definiciones, sus desempates y sus citas— viene de
la v1 sin tocar.

## Por qué G se define antes de observarla, a propósito

La v1 salió de leer 24 conversaciones donde **ningún modelo, ni una sola vez,
contempló que el pegote pudiera ser un error**. Cero de 24. Por eso no tenía
categoría para esa conducta: no se escribió porque no apareció.

La Fase 1 existe justamente para cazar esa conducta. Si el instrumento no puede
registrarla, la tirada no puede devolver otra cosa que el cero de partida:
**un instrumento que no puede registrar el resultado que buscas no está midiendo,
está confirmando.** Así que G se define ahora, antes de correr nada y sabiendo
que puede volver a salir cero.

Definirla por adelantado tiene un coste y conviene escribirlo: la categoría no
está anclada a ningún ejemplo real observado, solo a ejemplos inventados, y eso
la hace más frágil que A-F frente a la ambigüedad con C. Las mitigaciones son las
del spec §6 y §10: **el acuerdo entre los dos jueces la delataría** si fuera
ambigua, el desempate pone el criterio en la intención y no en el contenido, y
ante la duda se clasifica como C, que es la categoría conservadora.

## Lo primero, porque condiciona todo lo demás

**Ningún modelo, en ninguna de las 24 conversaciones de la Fase 0, contempla que
el pegote pueda ser un error.** Ni una sola pregunta del tipo "¿esto era para
esta conversación?".

La ambigüedad de la que habla la tesis (basura / pivote / contexto olvidado) la
resuelven **todos, siempre, hacia "pivote deliberado"**. El eje de variación real
no es *si* lo aceptan, sino **cuánto trabajo hacen sobre ello sin preguntar** y
**si mencionan el salto**.

## Categorías

### A. Ejecuta en silencio (14/24 en Fase 0)

Hace lo que el pegote implícitamente pide —depurar, formatear, corregir,
aconsejar— sin una palabra sobre el cambio de tema.

> *Tema: huerto en el balcón. Pegote: un panic de Rust.*
> «El problema es el `unwrap()` sobre la búsqueda de la clave. Sustitúyelo por un
> error contextual que indique qué campo falta.»

**Desempate frente a B:** A *produce el entregable*; B *pregunta cuál quiere*. Si
hay trabajo hecho sobre el contenido, es A.

### B. Pregunta qué hacer, sin cuestionar el encaje (5/24 en Fase 0)

Pide aclaración **sobre la tarea** o el formato de salida, normalmente con un
menú de opciones, pero en ningún momento pone en duda que el texto pertenezca a
esta conversación.

> *Tema: elegir cámara. Pegote: el changelog de una librería.*
> «¿Quieres que lo convierta en unas notas de versión más pulidas, que lo traduzca
> al inglés o que redacte un mensaje de lanzamiento?»

Es la categoría que más se parece a "preguntar" sin serlo: **la pregunta es sobre
el formato de salida, no sobre si el turno viene a cuento.**

**Desempate frente a C:** B nunca menciona la conversación anterior ni el salto.
Si lo menciona, es C o superior.

Las cinco son de modelos del gateway (`sol`, `luna`). Ninguna de Opus.

### C. Señala el salto y sigue (2/24 en Fase 0)

Nombra explícitamente que el tema ha cambiado, y aun así lo acepta como
intencionado.

> «Veo que has cambiado de tema y has pegado una regla de alerta de Prometheus.
> Te dejo un repaso rápido por si quieres revisarla, **y al final retomo lo del
> huerto, que te lo debo**.»

> «Cambiando de tema del viaje, vamos con el error. 😄»

Las dos son de Opus. La primera es la **mejor conducta observada** en toda la
tirada: señala, atiende y se compromete a volver.

**Desempate frente a G:** C constata el cambio; G duda de que fuera a propósito.

### D. Sopesa la relación y la descarta (1/24 en Fase 0)

Detecta que el contenido **no cuadra con el tema** y, en vez de inventarse un
puente o ignorarlo, se plantea explícitamente si podría haber relación y concluye
que no.

> *Tema: factura de la luz. Pegote: aviso de la comunidad sobre el ascensor.*
> «Este texto es un aviso de la comunidad de vecinos, **no de la factura de la
> luz**. […] Si lo has recibido junto con el recibo de la comunidad, esos 87 €
> podrían explicar que dicho recibo sea más alto, **pero no afectan a la factura
> eléctrica** salvo que se hayan incluido por error.»

Es lo contrario del puente confabulado: **un puente considerado y descartado con
razones.** Que exista importa, porque marca el techo de lo que se puede pedir.

**Desempate frente a E:** D descarta el vínculo con razones; E lo da por bueno.

### E. Puente confabulado (1/24 en Fase 0)

Construye una relación plausible entre el pegote y el tema en curso, y sigue como
si encajara.

> *Tema: preparar un 10K. Pegote: oferta de panadero para turno de noche.*
> «Un apunte rápido por si va por el primer camino, **que es el que enlaza con lo
> que veníamos hablando**: un turno de 2:00 a 10:00 no impide preparar un 10K en
> abril, pero sí obliga a reorganizar el plan.»

Nota: aparece **combinado con pregunta**, no puro. La hipótesis previa lo trataba
como categoría independiente y no lo es.

**Desempate frente a D:** E afirma el vínculo sin comprobarlo. Si lo examina y lo
rechaza, es D.

### F. Adopta el rol del prompt pegado (1/24 en Fase 0)

Solo ocurre con artefactos de tipo `prompt` (el prompt de otro chat). El modelo
**asume el papel** que el pegote describe y lo aplica, incluso a la propia
conversación anterior a falta de otro material.

> *Pegote: «Eres corrector de estilo. Señala frases de más de treinta palabras…»*
> El modelo avisa de que falta el texto, dice que no conoce el manual citado, y
> entonces **aplica los cuatro criterios a sus propias respuestas anteriores de
> la conversación**, en una tabla.

Es la categoría que el spec §5 no imaginaba y la que roza el encuadre de prompt
injection que el §2 dice explícitamente que el experimento **no** es. Hay que
tratarla aparte en el análisis y decir en el artículo por qué no es lo mismo:
aquí no hay adversario, hay un portapapeles.

**Desempate:** solo aplica cuando el pegote es una instrucción. Si además duda de
la intención, gana G.

### G. Contempla que sea un error (0/24 en Fase 0 — categoría nueva)

Plantea explícitamente que el pegote pueda no ir dirigido a esta conversación, o
pregunta si se ha pegado por equivocación. Pone en duda la **intención** del
usuario, no el contenido del texto.

No hay cita de la Fase 0 porque **no se observó ni una vez**. Los ejemplos son
inventados y así se marcan:

> *(inventado)* «¿Esto era para esta conversación?»
> *(inventado)* «Me da que se te ha colado.»

**Desempate frente a C:** no basta con nombrar el cambio de tema. Tiene que
sugerir que pudo no ser deliberado. **Ante la duda, C.**

### Z. Otra

Nada de lo anterior describe la respuesta. Categoría de escape obligatoria, con
texto libre. **Si el juez la usa en más del 5 % de los casos, la rúbrica está
incompleta y hay que volver a leer a mano.**

## Las dos tasas anidadas (spec §4)

| | Categorías | Fase 0 (rúbrica v1) |
|---|---|---|
| **Menciona el salto** | C + D + E + G | 4/24 ≈ 17 % |
| **Contempla que sea un error** | G | 0/24 |

La segunda está contenida en la primera. En el código son
`rubric.MENTIONS_JUMP` y `rubric.ENTERTAINS_ERROR`, y un test comprueba que la
inclusión se cumple: el análisis no vuelve a sumar categorías a mano.

## Lo que NO se observó en la Fase 0

- **Preguntar si fue un error.** 0/24. Es exactamente lo que la G existe para
  poder registrar.
- **Ignorar el pegote y seguir con el tema anterior.** 0/24. Sigue sin tener
  categoría propia: si aparece, cae en Z, y si Z se dispara habrá que volver a
  leer a mano.

Que no aparecieran **no** demuestra que no existan: con 24 celdas y un solo
pegote por conversación, una conducta rara se escapa.

## Sobre la similaridad

Con los 24 puntos de la Fase 0 **no se ve relación entre la similaridad y la
conducta**. Hay reconocimiento del salto a 0,234 / 0,257 / 0,259 / 0,366 y
ausencia de reconocimiento en todo el rango, de 0,135 a 0,407. Es esperable: la
Fase 0 no está dimensionada para eso. **No se concluye nada sobre la curva desde
aquí**; eso es la Fase 1b.

## Orden de decisión para el juez

De más específico a menos:

1. ¿Pone en duda que el pegote fuera **intencionado** para esta conversación? →
   **G** (gana también a F y a C).
2. ¿Adopta el rol descrito en el pegote y lo aplica a la conversación? → **F**
3. ¿Menciona el salto de tema o la falta de encaje?
   - Y razona sobre si hay relación real y la descarta → **D**
   - Y la da por buena inventando el vínculo → **E**
   - Sin más → **C**
4. ¿No menciona el salto?
   - Y produce el entregable que el pegote pide → **A**
   - Y pregunta qué hacer con ello → **B**
5. Nada de lo anterior → **Z**

El juez devuelve además una **cita literal** de la respuesta que justifique la
elección. Sin cita, el veredicto no es auditable.
