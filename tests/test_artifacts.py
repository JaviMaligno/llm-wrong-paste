from collections import Counter

import pytest

from wrongpaste.artifacts import _parse, entity_hits, load_artifacts

MIN_ARTIFACTS = 60


def test_bank_is_big_enough():
    assert len(load_artifacts()) >= MIN_ARTIFACTS


def test_every_artifact_declares_entities():
    for art in load_artifacts():
        assert art.entities, f"{art.id} no declara entities"


def test_kinds_are_varied():
    kinds = {a.kind for a in load_artifacts()}
    assert len(kinds) >= 8


def test_ids_are_unique():
    ids = [a.id for a in load_artifacts()]
    assert len(ids) == len(set(ids))


# --- casado de entidades (D8) ---------------------------------------------


def test_entity_hits_ignora_acentos_y_mayusculas():
    texto = "Compré pimenton de la vera en el mercado."
    assert entity_hits(texto, ["Pimentón de la Vera"]) == ["Pimentón de la Vera"]
    # y en el sentido contrario: el acento está en el texto y no en la entity
    assert entity_hits("Lleva Pimentón de la Vera.", ["pimenton de la vera"]) == [
        "pimenton de la vera"
    ]


def test_entity_hits_no_casa_dentro_de_otra_palabra():
    assert entity_hits("Miro la pantalla del horno.", ["pan"]) == []
    assert entity_hits("Se llama Trufalandia, no es el perro.", ["Trufa"]) == []
    # la misma entity sí casa cuando es una palabra suelta
    assert entity_hits("Horneé un pan de centeno.", ["pan"]) == ["pan"]


def test_entity_hits_casa_entidades_multipalabra():
    texto = "Formar, cesto enharinado y 12 horas de fermentación en frío."
    assert entity_hits(texto, ["12 horas de fermentación en frío"]) == [
        "12 horas de fermentación en frío"
    ]
    # la puntuación intermedia actúa como separador, no como obstáculo
    assert entity_hits("Factura F-2026/118, pendiente.", ["F-2026/118"]) == [
        "F-2026/118"
    ]


def test_entity_hits_respeta_el_orden_de_entrada():
    texto = "Trufa tiene cita en la Clínica Pelaires el martes."
    entities = ["Clínica Pelaires", "Trufa", "vacuna trivalente"]
    assert entity_hits(texto, entities) == ["Clínica Pelaires", "Trufa"]


def test_entity_hits_devuelve_vacio_si_no_hay_ninguna():
    assert entity_hits("Una charla sobre el huerto del balcón.", ["RabbitMQ"]) == []
    assert entity_hits("", ["cualquier cosa"]) == []
    assert entity_hits("texto sin entities que buscar", []) == []


def test_entity_hits_no_duplica():
    texto = "Masa madre, y otra vez masa madre para la segunda hornada."
    entities = ["masa madre", "masa madre", "Masa Madre"]
    assert entity_hits(texto, entities) == ["masa madre"]


def test_todas_las_entities_aparecen_en_su_propio_artefacto():
    """Cada entity del banco debe estar literalmente en el cuerpo del artefacto.

    Si falla, la métrica de fuga de la Fase 2 estaría contando entidades que
    nunca se pegaron: hay que arreglar el fichero del banco, no el test.
    """
    fallos = {}
    for art in load_artifacts():
        encontradas = set(entity_hits(art.text, art.entities))
        faltan = [e for e in art.entities if e not in encontradas]
        if faltan:
            fallos[art.id] = faltan
    assert not fallos, f"entities ausentes de su propio artefacto: {fallos}"


# --- niveles del pegote (Fase 1a, Task 2) ----------------------------------

MIN_N1 = 30


def test_el_banco_n1_existe_y_es_suficiente():
    n1 = load_artifacts(level="N1")
    assert len(n1) >= MIN_N1


def test_todo_artefacto_declara_nivel():
    for art in load_artifacts():
        assert art.level in {"N0", "N1"}, f"{art.id} sin nivel válido"


def test_n0_sigue_siendo_el_banco_original():
    assert len(load_artifacts(level="N0")) >= 64


def test_sin_filtro_devuelve_los_dos_niveles():
    assert len(load_artifacts()) == len(load_artifacts(level="N0")) + len(
        load_artifacts(level="N1")
    )


def test_los_n1_declaran_su_senal():
    # La señal intrínseca es parte del dato: sin ella no se puede analizar
    # qué tipo de pista funciona.
    validas = {"cortado", "dirigido", "responde", "presupone"}
    for art in load_artifacts(level="N1"):
        assert art.signal in validas, f"{art.id}: señal {art.signal!r} inválida"


def test_un_nivel_que_no_es_un_banco_se_rechaza():
    """N2 existe como nivel de pegote, pero no como banco en disco.

    Pedirlo aquí es un error de quien llama, y tiene que decirlo así: un
    `KeyError: 'N2'` a secas parece que el fichero no está donde debería.
    """
    with pytest.raises(ValueError, match="N2"):
        load_artifacts(level="N2")


def test_un_artefacto_n1_sin_senal_se_rechaza(tmp_path):
    fichero = tmp_path / "n1-sin-senal.md"
    fichero.write_text(
        '---\nid: n1-sin-senal\nkind: email\nlevel: N1\nentities: ["algo"]\n'
        "---\nUn texto con algo dentro.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="signal"):
        _parse(fichero, "N1")


def test_un_frontmatter_que_miente_sobre_su_nivel_se_rechaza(tmp_path):
    """El directorio manda: si no, un N0 en `artifacts-n1/` contaría como N1."""
    fichero = tmp_path / "impostor.md"
    fichero.write_text(
        '---\nid: impostor\nkind: email\nlevel: N0\nsignal: cortado\n'
        'entities: ["algo"]\n---\nUn texto con algo dentro.\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="level"):
        _parse(fichero, "N1")


# --- reparto del banco N1 (géneros y señales) ------------------------------

# El runner de la Fase 1a fuerza cobertura de `kind` (D4): en cada celda elige,
# de la ventana del estrato, el artefacto cuyo género lleve menos usos. Un
# género con uno o dos artefactos sale sobremuestreado una y otra vez, y un
# género con doce arrastra el banco entero. De ahí la horquilla.
MIN_POR_GENERO = 3
MAX_POR_GENERO = 6


def test_el_banco_n1_reparte_los_generos():
    """Ningún género de N1 puede ser tan flaco ni tan gordo que sesgue el muestreo.

    Se mide sobre el banco real en disco a propósito: un recuento construido a
    mano en el test no diría nada de los ficheros que el runner va a cargar.
    """
    reparto = Counter(a.kind for a in load_artifacts(level="N1"))
    flacos = {k: n for k, n in reparto.items() if n < MIN_POR_GENERO}
    gordos = {k: n for k, n in reparto.items() if n > MAX_POR_GENERO}
    assert not flacos, (
        f"géneros con menos de {MIN_POR_GENERO} artefactos: {flacos}; "
        f"reparto completo: {dict(sorted(reparto.items()))}"
    )
    assert not gordos, (
        f"géneros con más de {MAX_POR_GENERO} artefactos: {gordos}; "
        f"reparto completo: {dict(sorted(reparto.items()))}"
    )


def test_n1_cubre_los_mismos_generos_que_n0():
    """Los dos brazos de la puerta tienen que llevar el MISMO conjunto de géneros.

    El reparto de arriba mide el equilibrio dentro de N1 y por eso no ve un
    género ausente del todo: con `config`, `sql` y `stacktrace` fuera del banco,
    N0 y N1 difieren en composición de registro además de en nivel de pegote, y
    el registro no es el factor bajo estudio. Una tasa distinta en N1 no diría
    entonces si es por la señal intrínseca o por la falta de registro técnico
    —la colinealidad registro-similaridad de D15 entrando por la puerta de
    atrás—. Se compara contra el banco real en disco, no contra una lista
    escrita a mano, porque lo que importa es lo que el runner va a cargar.
    """
    generos_n0 = {a.kind for a in load_artifacts(level="N0")}
    generos_n1 = {a.kind for a in load_artifacts(level="N1")}
    faltan = sorted(generos_n0 - generos_n1)
    sobran = sorted(generos_n1 - generos_n0)
    assert not faltan, (
        f"géneros de N0 que N1 no cubre: {faltan}; mientras falten, los dos "
        f"brazos difieren en registro y no solo en nivel de pegote"
    )
    assert not sobran, (
        f"géneros que solo existen en N1: {sobran}; el brazo N0 no tiene con "
        f"qué compararlos"
    )


def test_el_banco_n1_reparte_las_senales():
    """Las cuatro señales del §5 del spec tienen que pesar lo mismo.

    Si una señal domina el banco, lo que se mida en la Fase 1a será el efecto de
    esa señal disfrazado de efecto del nivel N1.
    """
    senales = {"cortado", "dirigido", "responde", "presupone"}
    reparto = Counter(a.signal for a in load_artifacts(level="N1"))
    assert set(reparto) == senales, (
        f"señales del banco: {dict(sorted(reparto.items()))}"
    )
    desvio = max(reparto.values()) - min(reparto.values())
    assert desvio <= 1, (
        f"las señales están desequilibradas (diferencia de {desvio} artefactos "
        f"entre la más y la menos frecuente): {dict(sorted(reparto.items()))}"
    )
