---
id: n1-cortado-traza
kind: stacktrace
level: N1
signal: cortado
entities: ["cerrar_dia", "calcular_comision", "liquidador/comisiones.py"]
---
…, line 178, in cerrar_dia
    total = calcular_comision(operaciones)
  File "/srv/liquidador/comisiones.py", line 63, in calcular_comision
    validar(importes)
  File "/srv/liquidador/checks.py", line 41, in validar
    raise ImporteNegativoErro
