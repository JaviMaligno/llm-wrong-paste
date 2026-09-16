---
id: n1-cortado-sql-dental
kind: sql
level: N1
signal: cortado
entities: ["tramo de la tarde", "recordatorio_enviado", "paciente_id"]
---
…y para lo de las ausencias sacaría esto, que es lo que más se repite en el
tramo de la tarde:

SELECT c.paciente_id, c.fecha, c.recordatorio_enviado
FROM citas c
WHERE c.estado = 

