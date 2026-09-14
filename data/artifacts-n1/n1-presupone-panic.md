---
id: n1-presupone-panic
kind: stacktrace
level: N1
signal: presupone
entities: ["index out of range [9] with length 6", "goroutine 41", "rejilla/turnos.go:73"]
---
La traza que te decía, la del caso raro que comentamos y que seguimos sin
saber reproducir:

panic: runtime error: index out of range [9] with length 6

goroutine 41 [running]:
rejilla/turnos.repartirGuardias(0xc0000c2180, 0x9)
	/home/deploy/rejilla/turnos.go:73 +0x1c8
main.main()
	/home/deploy/rejilla/main.go:19 +0x4f
exit status 2

Es la misma que salía antes de lo que cambiamos, así que no era eso.
