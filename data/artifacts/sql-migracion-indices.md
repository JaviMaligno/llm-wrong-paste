---
id: sql-migracion-indices
kind: sql
entities: ["idx_clientes_documento", "tbl_clientes", "documento_normalizado", "migración 0042"]
---
-- migración 0042: acelerar la búsqueda por documento antes del alta masiva.
CREATE INDEX CONCURRENTLY idx_clientes_documento
    ON tbl_clientes (documento_normalizado);

ALTER TABLE tbl_clientes
    ADD COLUMN fecha_alta TIMESTAMPTZ NOT NULL DEFAULT now();

-- Esto no puede ir en la misma transacción que el CREATE INDEX: revienta.
ANALYZE tbl_clientes;
