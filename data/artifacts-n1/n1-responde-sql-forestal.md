---
id: n1-responde-sql-forestal
kind: sql
level: N1
signal: responde
entities: ["sí, por especie y no por lote", "plantones", "fecha_repicado"]
---
Sí, por especie y no por lote, como decías. Quedaría:

SELECT p.especie,
       COUNT(*) AS plantones,
       MIN(p.fecha_repicado) AS primer_repicado
FROM plantones p
WHERE p.fecha_salida IS NULL
GROUP BY p.especie
ORDER BY plantones DESC;
