---
id: n1-presupone-sql-lonja
kind: sql
level: N1
signal: presupone
entities: ["la vista que acordamos", "v_subastas_desiertas", "precio_salida"]
---
La vista que acordamos, ya con el cambio del precio de salida:

CREATE OR REPLACE VIEW v_subastas_desiertas AS
SELECT l.lote_id,
       l.especie,
       l.precio_salida,
       l.fecha
FROM lotes l
WHERE l.adjudicado_a IS NULL
  AND l.fecha >= CURRENT_DATE - 30;

Lo demás está como lo dejaste.
