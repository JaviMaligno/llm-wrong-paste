---
id: stacktrace-php-clase-faroleria
kind: stacktrace
entities: ["production.ERROR", "FacturaXml", "FacturadorController.php:64", "composer dump-autoload"]
---
[2026-03-04 02:11:47] production.ERROR: Class "App\Servicios\FacturaXml" not found
#0 /var/www/faroleria/app/Http/Controllers/FacturadorController.php:64
   App\Http\Controllers\FacturadorController->emitir()
#1 /var/www/faroleria/vendor/onda/framework/src/Routing/Controller.php:54
#2 /var/www/faroleria/public/index.php:51

Se arregló con un composer dump-autoload. La carpeta estaba subida como
"servicios/" en minúscula y en local nadie se entera porque el disco no distingue.
