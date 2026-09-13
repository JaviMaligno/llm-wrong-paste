---
id: config-nginx-molino
kind: config
entities: ["panel.molinovieja.test", "cerbero_upstream", "client_max_body_size 40m", "listen 8443 ssl"]
---
server {
    listen 8443 ssl;
    server_name panel.molinovieja.test;
    client_max_body_size 40m;

    location /api/ {
        proxy_pass http://cerbero_upstream;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /estatico/ {
        alias /srv/molino/estatico/;
    }
}
