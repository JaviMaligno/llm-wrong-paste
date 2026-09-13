---
id: stacktrace-python-conciliador
kind: stacktrace
entities: ["conciliador/ledger.py", "SaldoDescuadradoError", "AG-9931", "cerrar_periodo"]
---
Traceback (most recent call last):
  File "/srv/conciliador/ledger.py", line 214, in cerrar_periodo
    validar_saldos(periodo, asientos)
  File "/srv/conciliador/checks.py", line 87, in validar_saldos
    raise SaldoDescuadradoError(mensaje)
conciliador.errors.SaldoDescuadradoError: el asiento AG-9931 descuadra en 0,07 EUR

Los 0,07 salen de redondear a dos decimales el tipo de cambio antes de sumar,
no después. Pasa solo los meses que cierran en fin de semana.
