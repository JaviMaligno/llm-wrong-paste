---
id: n1-dirigido-tunel
kind: config
level: N1
signal: dirigido
entities: ["Bruno", "bastion.rioclaro.test", "ServerAliveInterval 30"]
---
Bruno, cópiate esto tal cual en tu fichero, que es el que funciona. Lo que te
pasaron el otro día es el viejo y tiene el puerto cambiado:

Host puente
    HostName bastion.rioclaro.test
    User operador
    Port 2022
    ServerAliveInterval 30
    ForwardAgent no

Si te vuelve a pedir la contraseña dos veces, avísame y lo miramos juntos.
