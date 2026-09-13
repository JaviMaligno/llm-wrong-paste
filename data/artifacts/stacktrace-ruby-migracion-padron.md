---
id: stacktrace-ruby-migracion-padron
kind: stacktrace
entities: ["ActiveRecord::StatementInvalid", "apellido_2", "20260411_renombrar_apellidos.rb", "db:migrate"]
---
rake aborted!
ActiveRecord::StatementInvalid: PG::UndefinedColumn: ERROR: no existe la columna «apellido_2»
/home/mar/padron/db/migrate/20260411_renombrar_apellidos.rb:9:in `up'
/home/mar/padron/lib/tasks/padron.rake:14:in `block in <main>'
Caused by:
PG::UndefinedColumn: ERROR: no existe la columna «apellido_2»
Tasks: TOP => db:migrate
(Run with --trace para ver el backtrace entero.)
