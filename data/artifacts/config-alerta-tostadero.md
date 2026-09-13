---
id: config-alerta-tostadero
kind: config
entities: ["ColaTostaderoLlena", "tostadero_cola_pendientes", "for: 12m", "guardia-pasteleria"]
---
- alert: ColaTostaderoLlena
  expr: tostadero_cola_pendientes > 2500
  for: 12m
  labels:
    severidad: aviso
    equipo: guardia-pasteleria
  annotations:
    resumen: "La cola del tostadero lleva doce minutos por encima de 2500 piezas"
    accion: "Mirar si el turno de noche dejó el lote grande sin partir"
