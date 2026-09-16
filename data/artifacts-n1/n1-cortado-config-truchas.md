---
id: n1-cortado-config-truchas
kind: config
level: N1
signal: cortado
entities: ["alarma_oxigeno", "estanque_cabecera", "umbral_mg_l"]
---
…y el bloque de alarmas quedó así, que es lo que más nos costó cuadrar:

[alarma_oxigeno]
estanque_cabecera=true
umbral_mg_l=6.5
reintentos=3
avisar_a=guardia
intervalo_minutos=
