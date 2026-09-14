---
id: n1-presupone-vista
kind: sql
level: N1
signal: presupone
entities: ["v_pedidos_retrasados", "dias_de_retraso", "como quedamos"]
---
La vista con el cambio que acordamos, ya con el margen en días y no en horas,
como quedamos:

CREATE OR REPLACE VIEW v_pedidos_retrasados AS
SELECT p.id,
       p.cliente_id,
       (CURRENT_DATE - p.fecha_comprometida) AS dias_de_retraso
FROM pedidos p
WHERE p.entregado_en IS NULL
  AND p.fecha_comprometida < CURRENT_DATE - 2;

El resto lo he dejado igual que la última vez que lo miraste.
