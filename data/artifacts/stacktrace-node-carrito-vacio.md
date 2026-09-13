---
id: stacktrace-node-carrito-vacio
kind: stacktrace
entities: ["TypeError", "carrito.total", "checkout.js:57", "PedidoVacioError"]
---
TypeError: Cannot read properties of undefined (reading 'total')
    at calcularEnvio (/opt/tiendalunar/src/checkout.js:57:23)
    at async POST (/opt/tiendalunar/src/routes/pedidos.js:19:9)
    at async run (/opt/tiendalunar/node_modules/onda-router/dist/handler.js:412:5)

Nota: solo pasa cuando el carrito llega vacío desde el botón de "comprar ahora".
Habría que lanzar PedidoVacioError antes de leer carrito.total.
