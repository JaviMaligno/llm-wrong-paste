---
id: stacktrace-java-pasarela-nulo
kind: stacktrace
entities: ["NullPointerException", "PasarelaPagosService.java:142", "reintento 3 de 3", "LoteNocturno"]
---
Excepción al procesar el lote nocturno (reintento 3 de 3):
java.lang.NullPointerException: Cannot invoke "cl.quimera.pasarela.Tarjeta.getBin()" because "tarjeta" is null
	at cl.quimera.pasarela.PasarelaPagosService.autorizar(PasarelaPagosService.java:142)
	at cl.quimera.pasarela.PasarelaPagosService.procesarLote(PasarelaPagosService.java:88)
	at cl.quimera.batch.LoteNocturno.run(LoteNocturno.java:51)
	at java.base/java.lang.Thread.run(Thread.java:1583)
