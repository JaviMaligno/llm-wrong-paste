from wrongpaste.artifacts import entity_hits, load_artifacts

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
