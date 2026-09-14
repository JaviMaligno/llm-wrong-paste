---
id: n1-responde-excepcion
kind: stacktrace
level: N1
signal: responde
entities: ["NullPointerException", "AvisosService.java:96", "el lote de las 23:40"]
---
Sí, es la misma de siempre. Salta en el lote de las 23:40 y en ningún otro:

java.lang.NullPointerException: Cannot invoke "es.rejilla.avisos.Destinatario.getCorreo()" because "destinatario" is null
	at es.rejilla.avisos.AvisosService.enviar(AvisosService.java:96)
	at es.rejilla.avisos.AvisosService.procesarLote(AvisosService.java:54)
	at java.base/java.lang.Thread.run(Thread.java:1583)

Y no, lanzándolo a mano no hay manera de que salte.
