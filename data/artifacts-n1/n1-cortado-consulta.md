---
id: n1-cortado-consulta
kind: sql
level: N1
signal: cortado
entities: ["importe_bruto", "fecha_emision", "albaranes"]
---
…ELECT proveedor, SUM(importe_bruto) AS total
FROM albaranes
WHERE fecha_emision >= '2031-01-01'
  AND estado <> 'ANULADO'
GROUP BY proveedor
HAVING SUM(importe_bruto) > 1
