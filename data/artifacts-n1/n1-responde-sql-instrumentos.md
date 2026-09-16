---
id: n1-responde-sql-instrumentos
kind: sql
level: N1
signal: responde
entities: ["sí, por número de serie", "numero_serie", "fianza_devuelta"]
---
Sí, por número de serie, no por modelo. Quedaría así:

SELECT i.numero_serie,
       i.familia,
       a.cliente_id,
       a.fecha_fin
FROM alquileres a
JOIN instrumentos i ON i.id = a.instrumento_id
WHERE a.fecha_fin < CURRENT_DATE
  AND a.fianza_devuelta IS FALSE
ORDER BY a.fecha_fin;
