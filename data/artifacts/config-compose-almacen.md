---
id: config-compose-almacen
kind: config
entities: ["almacen-cebada", "postgres:16.2", "5433:5432", "TZ: Europe/Madrid"]
---
services:
  almacen-cebada:
    image: postgres:16.2
    ports:
      - "5433:5432"
    environment:
      POSTGRES_DB: cebada
      TZ: Europe/Madrid
    volumes:
      - ./datos:/var/lib/postgresql/data
    restart: unless-stopped
