---
id: n1-responde-puerto
kind: config
level: N1
signal: responde
entities: ["client_max_body_size 60m", "panel.olmoseco.test", "proxy_read_timeout 180s"]
---
Sí, ese límite se toca en el bloque del servidor, no en el del sitio. Te copio
el que tengo yo funcionando:

server {
    listen 8443 ssl;
    server_name panel.olmoseco.test;
    client_max_body_size 60m;

    location /api/ {
        proxy_pass http://interno_upstream;
        proxy_read_timeout 180s;
    }
}

Y no, reiniciar no hace falta: con recargar la configuración basta.
