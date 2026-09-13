"""Entorno mínimo para que la suite offline corra sin configuración.

`WRONGPASTE_GATEWAY_URL` y `WRONGPASTE_GCP_PROJECT` no tienen valor por defecto
en `config.py` a propósito (D17: el repositorio es público). La suite offline no
llama a nadie, pero sí construye las URLs de las peticiones antes de que el
`httpx.post` falso las intercepte, así que aquí se rellenan con valores
obviamente falsos.

Se usa `setdefault`: si el desarrollador ya tiene exportados los de verdad, no
se tocan, y los tests marcados `live` siguen apuntando a donde deben.
"""

import os

os.environ.setdefault("WRONGPASTE_GATEWAY_URL", "https://gateway.invalid")
os.environ.setdefault("WRONGPASTE_GCP_PROJECT", "proyecto-de-prueba")
