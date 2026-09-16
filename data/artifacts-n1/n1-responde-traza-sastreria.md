---
id: n1-responde-traza-sastreria
kind: stacktrace
level: N1
signal: responde
entities: ["sí, al guardar el patrón", "patrones.rb", "NoMethodError"]
---
Sí, al guardar el patrón, no al abrirlo:

NoMethodError: undefined method `talla_base` for nil:NilClass
    from /opt/sastreria/patrones.rb:154:in `escalar`
    from /opt/sastreria/patrones.rb:96:in `guardar`

Solo con los patrones importados del sistema viejo. Los nuevos van bien.
