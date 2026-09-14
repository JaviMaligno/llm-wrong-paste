---
id: n1-responde-union
kind: sql
level: N1
signal: responde
entities: ["LEFT JOIN", "devoluciones", "sin devolución"]
---
Sí, con LEFT JOIN, que si no se te caen los pedidos sin devolución:

SELECT p.id, p.total, d.motivo
FROM pedidos p
LEFT JOIN devoluciones d ON d.pedido_id = p.id
WHERE p.cerrado_en > '2031-02-01'
ORDER BY p.id;

Y no, meter el filtro en el WHERE no vale: eso lo vuelve a convertir en un
cruce normal y estamos igual que antes.
