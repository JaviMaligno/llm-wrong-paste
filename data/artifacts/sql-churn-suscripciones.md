---
id: sql-churn-suscripciones
kind: sql
entities: ["bajas_mensuales", "motivo_baja", "plan_bronce", "cancelada_en"]
---
-- Rápido, para la reunión de las cinco: bajas por plan del último trimestre.
SELECT plan, motivo_baja, COUNT(*) AS bajas_mensuales
FROM suscripciones
WHERE cancelada_en > CURRENT_DATE - INTERVAL '90 days'
GROUP BY plan, motivo_baja
HAVING COUNT(*) > 5
ORDER BY bajas_mensuales DESC;
-- El plan_bronce sale altísimo, pero también es el que más gente tiene.
