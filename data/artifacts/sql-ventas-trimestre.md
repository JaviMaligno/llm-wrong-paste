---
id: sql-ventas-trimestre
kind: sql
entities: ["ventas_trimestre", "importe_neto", "2031-04-01", "FACTURADO"]
---
-- Ventas acumuladas por región desde el cierre del trimestre anterior.
WITH ventas_trimestre AS (
    SELECT region, SUM(importe_neto) AS total
    FROM pedidos
    WHERE fecha_cierre >= '2031-04-01'
      AND estado = 'FACTURADO'
    GROUP BY region
)
SELECT region, total
FROM ventas_trimestre
ORDER BY total DESC
LIMIT 20;
