---
id: stacktrace-dotnet-timeout-informes
kind: stacktrace
entities: ["SqlException", "Timeout expired", "InformesRepository.ObtenerResumenMensual", "GeneradorPdf", "fecha_emision"]
---
Unhandled exception. Microsoft.Data.SqlClient.SqlException (0x80131904): Timeout expired. The timeout period elapsed prior to completion of the operation or the server is not responding.
   at Nubarron.Informes.Datos.InformesRepository.ObtenerResumenMensual(Int32 anio, Int32 mes) in D:\src\Nubarron.Informes\Datos\InformesRepository.cs:line 233
   at Nubarron.Informes.Servicios.GeneradorPdf.Construir(Solicitud s) in D:\src\Nubarron.Informes\Servicios\GeneradorPdf.cs:line 71
   at Nubarron.Informes.Program.<Main>$(String[] args) in D:\src\Nubarron.Informes\Program.cs:line 18

Salta siempre con el informe de diciembre: la consulta pasa de 30 segundos
porque el índice de fecha_emision deja de usarse cuando el rango cruza el año.
