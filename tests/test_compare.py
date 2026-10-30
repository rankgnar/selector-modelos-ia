"""«Probar»: same prompt to up to 3 OpenRouter models."""

import httpx
import pytest
import respx

from datetime import datetime, timezone

from selector_ia.compare.tester import compare, total_cost
from selector_ia.data.openrouter import BASE_URL, OpenRouterClient, parse_model
from selector_ia.errors import NotFoundError, SelectorError
from tests.conftest import load_fixture

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def catalog_models():
    return {m.id: m for m in (parse_model(r, NOW) for r in load_fixture("openrouter_models.json")["data"])}


def test_compare_refuses_ids_outside_catalog_and_limits():
    client = OpenRouterClient(api_key="k")
    with pytest.raises(SelectorError, match="ya no están"):
        compare(client, ["openai/gpt-invented"], "hola", catalog_models())
    with pytest.raises(SelectorError, match="máximo"):
        compare(client, ["a", "b", "c", "d"], "hola", catalog_models())
    with pytest.raises(SelectorError, match="al menos"):
        compare(client, [], "hola", catalog_models())


@respx.mock
def test_compare_runs_all_and_one_failure_does_not_hide_others():
    def reply(request):
        model = __import__("json").loads(request.content)["model"]
        if model == "minimax/minimax-m3.1-flash":
            return httpx.Response(404, json={"error": {"message": "No endpoints found matching your data policy"}})
        return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": f"hola desde {model}"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.001}})
    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=reply)
    answers = compare(OpenRouterClient(api_key="k"), ["moonshotai/kimi-k3.1-pro", "minimax/minimax-m3.1-flash"],
                      "hola", catalog_models())
    ok, failed = answers
    assert ok.content == "hola desde moonshotai/kimi-k3.1-pro" and ok.cost_usd == 0.001 and ok.seconds is not None
    assert "privacidad" in failed.error and failed.content == ""
    assert total_cost(answers) == 0.001


@respx.mock
def test_data_policy_404_explains_the_setting():
    respx.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(
        404, json={"error": {"message": "0 endpoints ... matching your guardrail restrictions and data policy"}}))
    with pytest.raises(NotFoundError, match="openrouter.ai/settings/privacy"):
        OpenRouterClient(api_key="k").chat("x/y", [{"role": "user", "content": "hola"}])
