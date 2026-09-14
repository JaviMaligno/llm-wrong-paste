---
id: n1-presupone-umbral
kind: config
level: N1
signal: presupone
entities: ["ColaFrigorificoLlena", "for: 20m", "guardia-almacen"]
---
Esto va ya con el umbral que acordamos subir, no con el de antes:

- alert: ColaFrigorificoLlena
  expr: frigorifico_cola_pendientes > 4000
  for: 20m
  labels:
    severidad: aviso
    equipo: guardia-almacen
  annotations:
    resumen: "La cola lleva veinte minutos por encima del umbral nuevo"

Lo dejo así hasta que decidamos lo otro que quedó pendiente.
