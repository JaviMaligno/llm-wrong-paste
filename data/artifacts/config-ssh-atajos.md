---
id: config-ssh-atajos
kind: config
entities: ["bastion.vega-nubes.test", "clave_vega_2031", "Port 2222", "ServerAliveInterval 45"]
---
# Atajos del portátil viejo. El bastión pide la contraseña de la llave, no la del usuario.
Host salto
    HostName bastion.vega-nubes.test
    User jromero
    Port 2222
    IdentityFile ~/.ssh/clave_vega_2031
    ServerAliveInterval 45

Host almacen
    HostName almacen.vega-nubes.test
    ProxyJump salto
