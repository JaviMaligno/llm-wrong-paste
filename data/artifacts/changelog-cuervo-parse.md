---
id: changelog-cuervo-parse
kind: changelog
entities: ["cuervo-parse 3.1.0", "parse_stream()", "Python 3.10", "DeprecationWarning"]
---
# cuervo-parse 3.1.0

Añadido
- `parse_stream()` para leer ficheros grandes sin cargarlos enteros en memoria.

Cambiado
- Se retira el soporte de Python 3.10; el mínimo pasa a ser 3.11.
- `parse()` emite un DeprecationWarning cuando recibe `strict=None`.

Arreglado
- Los separadores decimales con coma se comían el último dígito.
