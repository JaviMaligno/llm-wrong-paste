---
id: sql-limpieza-duplicados
kind: sql
entities: ["suscriptores_boletin", "correo_normalizado", "ROW_NUMBER", "duplicados"]
---
-- Borrar altas repetidas del boletín dejando la más antigua de cada correo.
WITH duplicados AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY correo_normalizado ORDER BY creado_en ASC
           ) AS posicion
    FROM suscriptores_boletin
)
DELETE FROM suscriptores_boletin
WHERE id IN (SELECT id FROM duplicados WHERE posicion > 1);
