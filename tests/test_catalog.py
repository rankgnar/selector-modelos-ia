from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from selector_ia.data import huggingface, openrouter
from selector_ia.data.cache import Cache
from selector_ia.data.catalog import Catalog
from selector_ia.data.huggingface import HuggingFaceClient
from selector_ia.data.openrouter import OpenRouterClient
from selector_ia.errors import OfflineError
from tests.conftest import load_fixture

OR_MODELS = f"{openrouter.BASE_URL}/models"
HF_MODELS = f"{huggingface.BASE_URL}/models"


@pytest.fixture
def catalog(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    yield Catalog(cache, OpenRouterClient(), HuggingFaceClient())
    cache.close()


def make_stale(cache: Cache):
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    with cache.conn:
        cache.conn.execute("UPDATE meta SET value = ?", (old,))
        cache.conn.execute("UPDATE hf_queries SET fetched_at = ?", (old,))
        cache.conn.execute("UPDATE hf_models SET fetched_at = ?", (old,))


@respx.mock
def test_openrouter_cached_within_refresh_window(catalog):
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    first = catalog.openrouter_models()
    second = catalog.openrouter_models()
    assert route.call_count == 1
    assert not first.from_cache and second.from_cache
    assert {m.id for m in second.items} == {m.id for m in first.items}
    assert second.fetched_at is not None


@respx.mock
def test_stale_cache_is_refreshed(catalog):
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    catalog.openrouter_models()
    make_stale(catalog.cache)
    assert not catalog.openrouter_models().from_cache
    assert route.call_count == 2


@respx.mock
def test_refresh_interval_is_configurable(catalog):
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    catalog.openrouter_models()
    catalog.cache.set_setting("refresh_hours", "0")
    catalog.openrouter_models()
    assert route.call_count == 2


@respx.mock
def test_offline_falls_back_to_cache_with_warning(catalog):
    respx.get(OR_MODELS).mock(side_effect=[
        httpx.Response(200, json=load_fixture("openrouter_models.json")),
        httpx.ConnectError("down"), httpx.ConnectError("down"), httpx.ConnectError("down"),
    ])
    catalog.openrouter_models()
    make_stale(catalog.cache)
    result = catalog.openrouter_models()
    assert result.from_cache and len(result.items) == 6
    assert "datos guardados" in result.warning


@respx.mock
def test_offline_without_cache_raises(catalog):
    respx.get(OR_MODELS).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(OfflineError):
        catalog.openrouter_models()


@respx.mock
def test_hf_queries_are_cached_per_params(catalog):
    route = respx.get(HF_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("hf_search_asr_sv.json")))
    catalog.search_hf(pipeline_tag="automatic-speech-recognition", languages=["sv"])
    cached = catalog.search_hf(pipeline_tag="automatic-speech-recognition", languages=["sv"])
    catalog.search_hf(pipeline_tag="automatic-speech-recognition", languages=["fi"])
    assert route.call_count == 2
    assert cached.from_cache
    assert [m.id for m in cached.items][0] == "openai/whisper-large-v3-turbo"  # order preserved


@respx.mock
def test_cross_openrouter_hugging_face(catalog):
    respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    mapping = catalog.openrouter_by_hf_id()
    assert [m.id for m in mapping["minimaxai/minimax-m3.1-flash"]] == ["minimax/minimax-m3.1-flash"]
    assert "moonshotai/kimi-k3.1-pro" not in {m.id for ms in mapping.values() for m in ms}  # proprietary


@respx.mock
def test_hf_id_known_only_for_real_repos(catalog):
    respx.get(HF_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("hf_search_asr_sv.json")))
    respx.get(f"{HF_MODELS}/invented/model-xl").mock(return_value=httpx.Response(401))
    catalog.search_hf(pipeline_tag="automatic-speech-recognition")
    assert catalog.hf_id_known("openai/whisper-small")       # came from the API
    assert not catalog.hf_id_known("invented/model-xl")      # does not exist


@respx.mock
def test_refresh_all_forces_download_and_clears_hf(catalog):
    route = respx.get(OR_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("openrouter_models.json")))
    respx.get(HF_MODELS).mock(return_value=httpx.Response(200, json=load_fixture("hf_search_asr_sv.json")))
    catalog.openrouter_models()
    catalog.search_hf(pipeline_tag="automatic-speech-recognition")
    catalog.refresh_all()
    assert route.call_count == 2
    assert catalog.cache.load_hf_model("openai/whisper-small") is None
