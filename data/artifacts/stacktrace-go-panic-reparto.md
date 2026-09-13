---
id: stacktrace-go-panic-reparto
kind: stacktrace
entities: ["index out of range [7] with length 5", "goroutine 34", "almacen/reparto.go:96", "planificarRuta"]
---
panic: runtime error: index out of range [7] with length 5

goroutine 34 [running]:
almacen/reparto.planificarRuta(0xc0000b4120, 0x7)
	/home/deploy/almacen/reparto.go:96 +0x1d4
almacen/reparto.(*Despacho).Ejecutar(0xc0000ae030)
	/home/deploy/almacen/reparto.go:41 +0x88
main.main()
	/home/deploy/almacen/main.go:23 +0x5f
exit status 2
