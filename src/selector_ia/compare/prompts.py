"""Ready-made test prompts for «Probar».

The factory set ships in web/prompts.json. Once the user edits anything, the whole list is kept in the
cache (setting "try_prompts") and replaces the factory set until they restore it."""

import json
import uuid
from importlib import resources

from pydantic import BaseModel, Field

from selector_ia.data.cache import Cache
from selector_ia.errors import SelectorError

SETTING = "try_prompts"
MAX_PROMPTS = 200
MAX_TEXT = 8000          # same limit as /api/probar


class TryPrompt(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    group: str = Field(min_length=1, max_length=60)
    title: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=MAX_TEXT)


def factory() -> list[TryPrompt]:
    raw = (resources.files("selector_ia") / "web" / "prompts.json").read_text(encoding="utf-8")
    return [TryPrompt.model_validate(p) for p in json.loads(raw)["prompts"]]


def load(cache: Cache) -> tuple[list[TryPrompt], bool]:
    """(prompts, custom): custom is True when the user's own list replaces the factory set."""
    saved = cache.get_setting(SETTING)
    if saved is None:
        return factory(), False
    return [TryPrompt.model_validate(p) for p in json.loads(saved)], True


def _store(cache: Cache, prompts: list[TryPrompt]) -> None:
    cache.set_setting(SETTING, json.dumps([p.model_dump() for p in prompts], ensure_ascii=False))


def save(cache: Cache, body: dict) -> str:
    """Updates the prompt with body["id"] or adds a new one at the end of its group. Returns its id."""
    group, title, text = (str(body.get(k) or "").strip() for k in ("group", "title", "text"))
    if not group or not title or not text:
        raise SelectorError("Escribe el grupo, el título y el texto de la pregunta.")
    if len(text) > MAX_TEXT:
        raise SelectorError("La pregunta es demasiado larga para una prueba.")
    prompts, _ = load(cache)
    pid = str(body.get("id") or "")
    index = next((i for i, p in enumerate(prompts) if p.id == pid), None)
    if index is not None:
        prompts[index] = TryPrompt(id=pid, group=group, title=title, text=text)
    else:
        if len(prompts) >= MAX_PROMPTS:
            raise SelectorError(f"Como máximo {MAX_PROMPTS} preguntas: borra alguna antes de añadir otra.")
        pid = uuid.uuid4().hex[:12]
        last = max((i for i, p in enumerate(prompts) if p.group == group), default=len(prompts) - 1)
        prompts.insert(last + 1, TryPrompt(id=pid, group=group, title=title, text=text))
    _store(cache, prompts)
    return pid


def delete(cache: Cache, pid: str) -> None:
    prompts, _ = load(cache)
    _store(cache, [p for p in prompts if p.id != pid])


def reset(cache: Cache) -> None:
    with cache.conn:
        cache.conn.execute("DELETE FROM settings WHERE key = ?", (SETTING,))
