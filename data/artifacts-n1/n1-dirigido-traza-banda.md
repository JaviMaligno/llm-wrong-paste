---
id: n1-dirigido-traza-banda
kind: stacktrace
level: N1
signal: dirigido
entities: ["Casimiro", "repertorio.py", "IndexError"]
---
Casimiro, esto es lo que sale al generar el programa de mano, a ver si te suena:

Traceback (most recent call last):
  File "/srv/banda/repertorio.py", line 88, in ordenar_piezas
    pieza = programa[posicion]
IndexError: list index out of range

Pasa solo cuando una pieza no tiene compositor asignado. Yo lo dejo por hoy.
