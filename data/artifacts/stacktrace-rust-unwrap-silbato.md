---
id: stacktrace-rust-unwrap-silbato
kind: stacktrace
entities: ["src/config.rs:48:31", "silbato::config::cargar", "umbral_ruido", "silbato.toml"]
---
thread 'main' panicked at src/config.rs:48:31:
called `Option::unwrap()` on a `None` value
stack backtrace:
   4: silbato::config::cargar
   5: silbato::main
La clave `umbral_ruido` no está en silbato.toml y el unwrap se la traga sin decir cuál falta.
