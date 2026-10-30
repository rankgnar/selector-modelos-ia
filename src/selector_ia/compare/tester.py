"""«Probar»: sends one test prompt to up to 3 models via OpenRouter and collects the answers.

Every id is re-checked against the current catalog before sending anything."""

import time
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from selector_ia.data.models import OpenRouterModel
from selector_ia.data.openrouter import OpenRouterClient
from selector_ia.errors import SelectorError

MAX_MODELS = 3
MAX_TOKENS = 4000  # room for reasoning models; they also get effort "low"


class CompareTarget(BaseModel):
    openrouter_id: str        # what is called
    label: str                # what the user sees (model id + where it comes from)
    accepts_text: bool = True


class CompareAnswer(BaseModel):
    openrouter_id: str
    content: str = ""
    seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


def ask_one(client: OpenRouterClient, model_id: str, prompt: str, *, reasoning: bool = False,
            max_tokens: int = MAX_TOKENS) -> CompareAnswer:
    start = time.monotonic()
    try:
        reply = client.chat(model_id, [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                            reasoning_effort="low" if reasoning else None)
    except SelectorError as exc:
        return CompareAnswer(openrouter_id=model_id, error=exc.user_message, seconds=round(time.monotonic() - start, 1))
    return CompareAnswer(openrouter_id=model_id, content=reply.content or "(respuesta vacía)",
                      seconds=round(time.monotonic() - start, 1), prompt_tokens=reply.prompt_tokens,
                      completion_tokens=reply.completion_tokens, cost_usd=reply.cost_usd)


def check_ids(model_ids: list[str], catalog_ids) -> None:
    if not model_ids:
        raise SelectorError("Elige al menos un modelo para probar.")
    if len(model_ids) > MAX_MODELS:
        raise SelectorError(f"Se pueden comparar como máximo {MAX_MODELS} modelos a la vez.")
    unknown = [m for m in model_ids if m not in catalog_ids]
    if unknown:
        raise SelectorError(f"Estos modelos ya no están en OpenRouter: {', '.join(unknown)}.")


def compare(client: OpenRouterClient, model_ids: list[str], prompt: str,
            catalog: dict[str, OpenRouterModel]) -> list[CompareAnswer]:
    """All models in parallel; one failing model never hides the others' answers.
    `catalog`: current OpenRouter models by id (the ids must be there)."""
    check_ids(model_ids, catalog)
    if not prompt.strip():
        raise SelectorError("Escribe el mensaje de prueba.")
    with ThreadPoolExecutor(max_workers=MAX_MODELS) as pool:
        return list(pool.map(lambda m: ask_one(client, m, prompt, reasoning=catalog[m].supports_reasoning),
                             model_ids))


def total_cost(answers: list[CompareAnswer]) -> float | None:
    costs = [a.cost_usd for a in answers if a.cost_usd is not None]
    return sum(costs) if costs else None
