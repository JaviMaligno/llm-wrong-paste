---
id: n1-dirigido-miraesto
kind: sql
level: N1
signal: dirigido
entities: ["Marta", "clientes_inactivos", "ultimo_pedido_en"]
---
Marta, mira esto antes de que lo lance, que esto borra de verdad:

DELETE FROM clientes_inactivos
WHERE ultimo_pedido_en < CURRENT_DATE - INTERVAL '3 years'
  AND saldo_pendiente = 0;

¿Lo de los tres años lo dejo o lo subo a cinco? Dime algo y lo lanzo.
