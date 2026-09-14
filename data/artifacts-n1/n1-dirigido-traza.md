---
id: n1-dirigido-traza
kind: stacktrace
level: N1
signal: dirigido
entities: ["Iván", "TypeError", "reparto.js:88"]
---
Iván, esto es lo que me sale a mí, que no es lo que decías tú. Míralo cuando
puedas:

TypeError: Cannot read properties of null (reading 'ruta')
    at asignarFurgoneta (/opt/rejilla/src/reparto.js:88:17)
    at async POST (/opt/rejilla/src/routes/salidas.js:24:9)

A ti te salía en otra línea, ¿no? Dime qué versión tienes instalada.
