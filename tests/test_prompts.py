"""Ready-made prompts for «Probar»: factory set, user edits kept in the cache, restore."""

import pytest

from selector_ia.compare import prompts
from selector_ia.data.cache import Cache
from selector_ia.errors import SelectorError


@pytest.fixture
def cache(tmp_path):
    c = Cache(tmp_path / "c.sqlite3")
    yield c
    c.close()


def test_factory_set_is_valid():
    items = prompts.factory()
    assert len(items) == 11 and len({p.id for p in items}) == 11
    assert [p.group for p in items].count("Programación avanzada") == 10
    assert "2^3^2" in next(p for p in items if p.id == "avanzado-3").text


def test_untouched_list_is_the_factory_set(cache):
    items, custom = prompts.load(cache)
    assert not custom and items == prompts.factory()


def test_edit_add_delete_and_reset(cache):
    prompts.save(cache, {"id": "basico-1", "group": "Básico", "title": "Nuevo título", "text": "Hola"})
    items, custom = prompts.load(cache)
    assert custom and items[0].title == "Nuevo título" and len(items) == 11

    new_id = prompts.save(cache, {"group": "Básico", "title": "Otra", "text": "¿Qué es un token?"})
    items, _ = prompts.load(cache)
    assert items[1].id == new_id                       # at the end of its group, before the next group
    added = prompts.save(cache, {"group": "Mi grupo", "title": "Última", "text": "x"})
    assert prompts.load(cache)[0][-1].id == added

    prompts.delete(cache, "avanzado-1")
    assert "avanzado-1" not in {p.id for p in prompts.load(cache)[0]}

    prompts.reset(cache)
    assert prompts.load(cache) == (prompts.factory(), False)


def test_rejects_empty_and_too_long(cache):
    with pytest.raises(SelectorError):
        prompts.save(cache, {"group": "G", "title": " ", "text": "x"})
    with pytest.raises(SelectorError):
        prompts.save(cache, {"group": "G", "title": "T", "text": "x" * (prompts.MAX_TEXT + 1)})
    assert prompts.load(cache)[1] is False             # nothing was stored
