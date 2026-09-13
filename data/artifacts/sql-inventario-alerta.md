---
id: sql-inventario-alerta
kind: sql
entities: ["v_stock_critico", "almacen_valdeparra", "stock_minimo", "referencia_proveedor"]
---
CREATE OR REPLACE VIEW v_stock_critico AS
SELECT a.referencia_proveedor,
       a.descripcion,
       e.unidades,
       e.stock_minimo
FROM articulos a
JOIN existencias e ON e.articulo_id = a.id
WHERE e.almacen = 'almacen_valdeparra'
  AND e.unidades < e.stock_minimo * 1.15;
-- La vista alimenta el correo diario de reposición de las 07:00.
