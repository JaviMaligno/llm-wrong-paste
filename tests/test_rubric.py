from wrongpaste.rubric import (
    CATEGORIES,
    CATEGORY_IDS,
    ENTERTAINS_ERROR,
    MENTIONS_JUMP,
    RUBRIC_VERSION,
    rubric_prompt,
)


def test_las_siete_categorias_mas_la_de_escape():
    assert CATEGORY_IDS == {"A", "B", "C", "D", "E", "F", "G", "Z"}


def test_la_categoria_g_existe_aunque_nunca_se_observara():
    g = next(c for c in CATEGORIES if c.id == "G")
    assert "intenci" in g.definition.lower(), (
        "G se define por poner en duda la INTENCIÓN, no el contenido"
    )


def test_las_dos_tasas_son_conjuntos_de_categorias():
    # Spec §4: la segunda tasa está contenida en la primera.
    assert ENTERTAINS_ERROR <= MENTIONS_JUMP
    assert ENTERTAINS_ERROR == {"G"}
    assert MENTIONS_JUMP == {"C", "D", "E", "G"}


def test_el_prompt_del_juez_lleva_todas_las_categorias():
    prompt = rubric_prompt()
    for c in CATEGORIES:
        assert f"{c.id}." in prompt
        assert c.definition[:30] in prompt


def test_el_prompt_obliga_a_citar():
    assert "cita" in rubric_prompt().lower()


def test_la_version_esta_fijada():
    assert RUBRIC_VERSION == "v2"
