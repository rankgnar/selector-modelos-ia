from datetime import date, datetime, timezone
from decimal import Decimal

import httpx
import pytest
import respx

from selector_ia.data.openrouter import BASE_URL, OpenRouterClient, parse_model
from selector_ia.errors import (
    ApiError, InsufficientCreditsError, InvalidKeyError, OfflineError, RateLimitError,
)
from tests.conftest import load_fixture

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def models_by_id():
    return {m.id: m for m in (parse_model(r, NOW) for r in load_fixture("openrouter_models.json")["data"])}


def test_parses_real_catalog_entries():
    models = models_by_id()
    assert len(models) == 6
    m = models["minimax/minimax-m3.1-flash"]
    assert m.hugging_face_id == "MiniMaxAI/MiniMax-M3.1-Flash"
    assert m.is_open_weights
    assert m.context_length == 1048576
    assert m.pricing["prompt"] == "0.00000015"  # copied as published
    assert m.url == "https://openrouter.ai/minimax/minimax-m3.1-flash"


def test_price_per_million_is_exact():
    m = models_by_id()["minimax/minimax-m3.1-flash"]
    assert m.price_per_million("prompt") == Decimal("0.15")
    assert m.price_per_million("completion") == Decimal("0.6")
    assert m.price_per_million("image") is None  # not published -> "sin datos"


def test_flags_alias_router_free_and_retirement():
    models = models_by_id()
    assert models["~unbiased/pareto-latest"].is_alias
    assert models["typesafe/jev-router"].is_router
    assert models["typesafe/jev-router"].price_per_million("prompt") is None
    assert models["recursive/flint-v1.0:free"].is_free
    assert not models["moonshotai/kimi-k3.1-pro"].is_free
    expiring = models["dots-studio/dots-3-note-preview:free"]
    assert expiring.expiration_date == date(2026, 12, 31)
    assert not expiring.is_retired(today=date(2026, 10, 30))
    assert expiring.is_retired(today=date(2027, 1, 1))


def test_benchmarks_keep_source_and_skip_nulls():
    kimi = models_by_id()["moonshotai/kimi-k3.1-pro"]
    aa = [b for b in kimi.benchmarks if b.source.startswith("Artificial Analysis")]
    assert [(b.name, b.value) for b in aa] == [("intelligence_index", 46)]  # null indexes not invented
    assert len([b for b in kimi.benchmarks if b.source.startswith("Design Arena")]) == 7


def test_malformed_entry_is_skipped():
    assert parse_model({"name": "no id"}, NOW) is None
    assert parse_model({"id": "x/y", "context_length": "not-a-number"}, NOW) is None


@respx.mock
def test_fetch_models():
    respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    assert len(OpenRouterClient().fetch_models()) == 6


@pytest.mark.parametrize("status,error", [
    (401, InvalidKeyError), (402, InsufficientCreditsError), (429, RateLimitError), (503, ApiError),
])
@respx.mock
def test_http_errors_are_mapped(status, error):
    route = respx.get(f"{BASE_URL}/key").mock(return_value=httpx.Response(status, headers={"retry-after": "7"}))
    with pytest.raises(error) as exc_info:
        OpenRouterClient(api_key="bad").key_info()
    assert exc_info.value.user_message  # always a Spanish message for the UI
    if status == 429:
        assert exc_info.value.retry_after == 7
    if status == 503:
        assert route.call_count == 3  # retried


@respx.mock
def test_offline_is_reported():
    respx.get(f"{BASE_URL}/models").mock(side_effect=httpx.ConnectError("no network"))
    with pytest.raises(OfflineError, match="conexión"):
        OpenRouterClient().fetch_models()


@respx.mock
def test_key_is_sent_as_bearer():
    route = respx.get(f"{BASE_URL}/key").mock(return_value=httpx.Response(200, json={"data": {"usage": 1.5}}))
    assert OpenRouterClient(api_key="sk-test").key_info() == {"usage": 1.5}
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-test"


@respx.mock
def test_catalog_asks_for_every_modality_and_merges_video_prices():
    from datetime import datetime
    raw = {"data": [{"id": "x/video", "name": "V", "architecture": {"input_modalities": ["text"], "output_modalities": ["video"]},
                     "pricing": {"prompt": "0", "completion": "0"}, "created": 1790000000},
                    {"id": "x/chat", "name": "C", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                     "pricing": {"prompt": "0", "completion": "0"}, "created": 1790000000}]}
    models_route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=raw))
    respx.get(f"{BASE_URL}/videos/models").mock(return_value=httpx.Response(200, json={"data": [
        {"id": "x/video", "pricing_skus": {"cents_per_video_output_second_720p": "3"}}]}))
    by_id = {m.id: m for m in OpenRouterClient().fetch_models()}
    assert models_route.calls[0].request.url.params["output_modalities"] == "all"
    assert by_id["x/video"].video_pricing == {"cents_per_video_output_second_720p": "3"}
    assert not by_id["x/video"].is_free and by_id["x/chat"].is_free     # 0/0 means free only for chat models
