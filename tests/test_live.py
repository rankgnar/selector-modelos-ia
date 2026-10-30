"""Real API checks. Run with: pytest -m live"""

import pytest

from selector_ia.data.huggingface import HuggingFaceClient
from selector_ia.data.openrouter import OpenRouterClient
from selector_ia.errors import NotFoundError

pytestmark = pytest.mark.live


def test_openrouter_catalog_is_real_and_parses():
    models = OpenRouterClient().fetch_models()
    assert len(models) > 50
    assert len({m.id for m in models}) == len(models)
    assert any(m.hugging_face_id for m in models)
    assert any(m.benchmarks for m in models)


def test_hf_search_and_existence():
    hf = HuggingFaceClient()
    results = hf.search(pipeline_tag="automatic-speech-recognition", languages=["sv"], limit=5)
    assert results and all(m.pipeline_tag == "automatic-speech-recognition" for m in results)
    assert hf.get_model(results[0].id).id == results[0].id
    with pytest.raises(NotFoundError):
        hf.get_model("this-org-does-not-exist-xyz/invented-model-9000")


def test_openrouter_hf_ids_exist_on_hugging_face():
    """Sample of the cross-reference: OpenRouter's hugging_face_id must point to real repos."""
    hf = HuggingFaceClient()
    linked = [m for m in OpenRouterClient().fetch_models() if m.hugging_face_id][:5]
    for model in linked:
        hf.get_model(model.hugging_face_id)


def test_fit_check_on_real_model(tmp_path):
    from selector_ia.data.cache import Cache
    from selector_ia.data.catalog import Catalog
    from selector_ia.hardware.check import check_model
    from selector_ia.hardware.profile import GIB, HardwareProfile

    catalog = Catalog(Cache(tmp_path / "c.sqlite3"), OpenRouterClient(), HuggingFaceClient())
    profile = HardwareProfile(name="cpu16", kind="remote_manual", ram_bytes=16 * GIB, os="Ubuntu 24.04")
    check = check_model(catalog, "Qwen/Qwen3-8B", profile)
    assert check.fit.recommended is not None
    rec = check.fit.recommended.variant
    assert rec.url and rec.repo_id and not rec.estimated
    HuggingFaceClient().get_model(rec.repo_id)  # recommended download repo really exists


def test_finder_catalog_is_real_and_complete(tmp_path):
    """Free (public catalogs only): the snapshot the search page uses, built from the live APIs."""
    from selector_ia.data.cache import Cache
    from selector_ia.data.catalog import Catalog
    from selector_ia.finder.catalog import build_snapshot

    catalog = Catalog(Cache(tmp_path / "c.sqlite3"), OpenRouterClient(), HuggingFaceClient())
    snap = build_snapshot(catalog)
    caps = {c for r in snap.records for c in r.capabilities}
    assert {"chat", "code", "tools", "audio_in", "image_gen", "video_gen", "voice", "embeddings"} <= caps
    assert any(r.benchmarks.asr_wer for r in snap.records)            # Open ASR Leaderboard via HF evalResults
    assert any(r.benchmarks.coding for r in snap.records)              # Artificial Analysis via OpenRouter
    fresh_or = {m.id for m in OpenRouterClient().fetch_models()}
    assert all(r.openrouter.id in fresh_or for r in snap.records if r.openrouter)
