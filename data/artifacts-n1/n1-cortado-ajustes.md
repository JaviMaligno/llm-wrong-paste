---
id: n1-cortado-ajustes
kind: config
level: N1
signal: cortado
entities: ["reintentos_maximos=4", "carpeta_temporal", "Almendro 2.7"]
---
…y entonces quedó así en el fichero de Almendro 2.7, que es el que vale:

[envios]
reintentos_maximos=4
espera_entre_intentos=90
carpeta_temporal=/var/spool/almendro

[registro]
nivel=aviso
rotar_cada=
