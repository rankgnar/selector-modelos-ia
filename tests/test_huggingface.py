from datetime import datetime, timezone

import httpx
import pytest
import respx

from selector_ia.data.huggingface import BASE_URL, HuggingFaceClient, parse_model
from selector_ia.errors import InvalidKeyError, NotFoundError, RateLimitError
from tests.conftest import load_fixture

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def asr_models():
    return {m.id: m for m in (parse_model(r, NOW) for r in load_fixture("hf_search_asr_sv.json"))}


def test_parses_real_search_results():
    m = asr_models()["openai/whisper-large-v3-turbo"]
    assert m.pipeline_tag == "automatic-speech-recognition"
    assert m.license == "mit"
    assert "sv" in m.languages
    assert m.params_total == 808878080 and m.params_source == "safetensors"
    assert m.downloads == 5780450
    assert m.url == "https://huggingface.co/openai/whisper-large-v3-turbo"


def test_missing_param_count_stays_none():
    # CTranslate2 repo publishes no safetensors metadata: never estimated.
    m = asr_models()["Systran/faster-whisper-small"]
    assert m.params_total is None and m.params_source is None


def test_quant_formats_and_gguf_params():
    models = {m.id: m for m in (parse_model(r, NOW) for r in load_fixture("hf_quantized_qwen3_8b.json"))}
    assert models["Qwen/Qwen3-8B-AWQ"].quant_format == "AWQ"
    assert models["Qwen/Qwen3-8B-FP8"].quant_format == "FP8"
    gguf = models["Qwen/Qwen3-8B-GGUF"]
    assert gguf.quant_format == "GGUF"
    assert gguf.params_total == 8190735360 and gguf.params_source == "gguf"
    assert gguf.gguf.context_length == 40960


def test_eval_results_keep_date_source_and_verification():
    m = parse_model(load_fixture("hf_model_with_evals.json"), NOW)
    assert len(m.eval_results) == 10
    first = m.eval_results[0]
    assert first.date == "2026-10-25"
    assert first.url == "https://huggingface.co/Nanbeige/Nanbeige4.3-33B"
    assert first.verified is False


@respx.mock
def test_search_sends_filters_as_and():
    route = respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=load_fixture("hf_search_asr_sv.json")))
    result = HuggingFaceClient().search(pipeline_tag="automatic-speech-recognition", languages=["sv"],
                                        license="mit", limit=3)
    assert len(result) == 3
    params = route.calls[0].request.url.params
    assert params.get_list("filter") == ["sv", "license:mit"]
    assert params["pipeline_tag"] == "automatic-speech-recognition"
    assert "evalResults" in params.get_list("expand[]")


@respx.mock
def test_unknown_repo_is_not_found():
    # Hugging Face answers 401 (not 404) for repos that do not exist.
    respx.get(f"{BASE_URL}/models/fake/model-123").mock(return_value=httpx.Response(401))
    with pytest.raises(NotFoundError):
        HuggingFaceClient().get_model("fake/model-123")


@respx.mock
def test_invalid_token_is_reported_as_invalid_key():
    respx.get(f"{BASE_URL}/models/fake/model-123").mock(return_value=httpx.Response(401))
    respx.get(f"{BASE_URL}/whoami-v2").mock(return_value=httpx.Response(401))
    with pytest.raises(InvalidKeyError):
        HuggingFaceClient(token="hf_bad").get_model("fake/model-123")


@respx.mock
def test_rate_limit_reads_hf_header():
    respx.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(429, headers={"ratelimit": '"api";r=0;t=120'}))
    with pytest.raises(RateLimitError) as exc_info:
        HuggingFaceClient().search(limit=1)
    assert exc_info.value.retry_after == 120
