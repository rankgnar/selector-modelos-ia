"""Catalog for the search page: one record per original model, capabilities only from the data."""

from datetime import datetime, timezone

from selector_ia.data.catalog import CatalogResult
from selector_ia.data.models import Benchmark, HFModel, InferenceProvider, OpenRouterModel
from selector_ia.errors import NotFoundError
from selector_ia.finder.catalog import build_snapshot, is_original, license_class

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
OLD = datetime(2023, 1, 1, tzinfo=timezone.utc)


def offer(model_id, *, inputs=("text",), outputs=("text",), params=("tools",), hf_id=None, prompt="0.000001",
          completion="0.000002", aa=None, created=NOW, moderated=None, **extra) -> OpenRouterModel:
    benchmarks = [Benchmark(source="Artificial Analysis (vía OpenRouter)", name=n, value=v)
                  for n, v in (aa or {}).items()]
    return OpenRouterModel(id=model_id, name=f"Org: {model_id}", input_modalities=list(inputs),
                           output_modalities=list(outputs), supported_parameters=list(params), hugging_face_id=hf_id,
                           pricing={"prompt": prompt, "completion": completion}, benchmarks=benchmarks,
                           created=created, fetched_at=NOW, is_moderated=moderated, **extra)


def hf(model_id, *, tag="text-generation", likes=10, tags=(), base=None, created=NOW, **extra) -> HFModel:
    extra.setdefault("last_modified", created)
    return HFModel(id=model_id, pipeline_tag=tag, likes=likes, downloads=100, tags=list(tags),
                   base_models=[base] if base else [], created_at=created, fetched_at=NOW, **extra)


class MemoryBlobs:
    def __init__(self):
        self.blobs = {}

    def load_blob(self, key):
        return (self.blobs[key], NOW) if key in self.blobs else None

    def save_blob(self, key, data):
        self.blobs[key] = data


class FakeCatalog:
    def __init__(self, or_models=(), hf_models=(), cards=(), configs=None):
        self.or_models, self.hf_models = list(or_models), list(hf_models)
        self.cards = {c.id: c for c in cards}
        self.configs = configs or {}
        self.config_calls = []
        self.queries = []
        self.cache = MemoryBlobs()

    def model_config(self, repo):
        self.config_calls.append(repo)
        return self.configs.get(repo)

    def openrouter_models(self, force=False):
        return CatalogResult(self.or_models, NOW, from_cache=True)

    def search_hf(self, force=False, **params):
        self.queries.append(params)
        return CatalogResult([m for m in self.hf_models if m.pipeline_tag == params.get("pipeline_tag")], NOW, True)

    def hf_model(self, model_id, force=False):
        if model_id not in self.cards:
            raise NotFoundError("no")
        return self.cards[model_id]


def records(*args, **kwargs):
    return {r.id: r for r in build_snapshot(FakeCatalog(*args, **kwargs), now=NOW).records}


def test_openrouter_capabilities_come_from_modalities_and_parameters():
    recs = records([
        offer("a/reader", inputs=("text", "image", "file"), params=("tools", "reasoning")),
        offer("a/listener", inputs=("text", "audio"), params=()),
        offer("a/painter", inputs=("text",), outputs=("image",), params=()),
        offer("a/editor", inputs=("text", "image"), outputs=("image", "text"), params=()),
    ])
    assert set(recs["a/reader"].capabilities) == {"chat", "pdf", "vision", "tools", "reasoning"}
    assert set(recs["a/listener"].capabilities) == {"chat", "audio_in"}
    assert set(recs["a/painter"].capabilities) == {"image_gen"}
    assert {"image_gen", "image_edit", "vision"} <= set(recs["a/editor"].capabilities)


def test_reasoning_flag_and_prices_per_million():
    m = offer("a/thinker", params=("reasoning",))
    rec = records([m])["a/thinker"]
    assert "reasoning" in rec.capabilities
    assert rec.openrouter.prices == {"prompt": 1.0, "completion": 2.0}


def test_variants_merge_and_cross_with_hugging_face():
    recs = records(
        [offer("org/m", hf_id="Org/M", aa={"coding_index": 40.0}),
         offer("org/m:free", hf_id="Org/M", prompt="0", completion="0")],
        [hf("Org/M", license="apache-2.0", languages=["es"], params_total=8_000_000_000,
            inference_providers=[InferenceProvider(provider="together", status="live"),
                                 InferenceProvider(provider="old", status="error")])])
    rec = recs["Org/M"]
    assert len(recs) == 1 and rec.openrouter.id == "org/m" and rec.openrouter.free_id == "org/m:free"
    assert rec.license_class == "permissive" and rec.params == 8_000_000_000 and rec.providers == ["together"]
    assert rec.benchmarks.coding == 40.0


def test_only_original_models():
    assert is_original(hf("Org/M"))
    assert is_original(hf("Org/M-Instruct", tags=["base_model:finetune:Org/M-Base"]))     # same author
    assert not is_original(hf("x/M-finetune", tags=["base_model:finetune:Org/M"]))         # third party
    assert not is_original(hf("x/M-GGUF", tags=["gguf", "base_model:quantized:Org/M"]))
    assert not is_original(hf("x/style", tags=["lora"]))
    assert not is_original(hf("pyannote/diarization", tags=["pyannote-audio"]))
    assert not is_original(hf("x/whisper-coreml", library_name="coreml"))


def test_repackaged_files_are_dropped_and_undeclared_copies_are_community():
    recs = records([], [hf("Maker/Big-Thing", tag="text-to-image", likes=50),
                        hf("Repack/Big-Thing-Diffusers", tag="text-to-image", likes=1),
                        hf("other/FastBig-Thing-v2", tag="text-to-image", likes=2),
                        hf("Maker/Big-Thing-Base", tag="text-to-image", likes=3)])
    assert set(recs) == {"Maker/Big-Thing", "Maker/Big-Thing-Base", "other/FastBig-Thing-v2"}
    assert not recs["Maker/Big-Thing"].community and not recs["Maker/Big-Thing-Base"].community
    assert recs["other/FastBig-Thing-v2"].community and recs["other/FastBig-Thing-v2"].base_model == "Maker/Big-Thing"


def test_community_versions_are_kept_and_marked():
    recs = records([], [
        hf("fan/M-abliterated-GGUF", tags=["gguf", "base_model:quantized:Org/M"], base="Org/M"),     # uncensored build
        hf("fan/M-GGUF", tags=["gguf", "base_model:quantized:Org/M"], base="Org/M"),                 # plain copy
        hf("lab/M-medical", tags=["base_model:finetune:Org/M"], base="Org/M"),                       # third-party tune
        hf("fan/Img-Uncensored", tag="text-to-image", tags=["not-for-all-audiences", "uncensored"]),
    ])
    assert set(recs) == {"fan/M-abliterated-GGUF", "lab/M-medical", "fan/Img-Uncensored"}
    unc = recs["fan/M-abliterated-GGUF"]
    assert unc.community and unc.uncensored and unc.base_model == "Org/M"
    assert recs["lab/M-medical"].community and not recs["lab/M-medical"].uncensored
    img = recs["fan/Img-Uncensored"]
    assert img.uncensored and img.adult and not img.community      # no declared base: its own model


def test_openrouter_models_declared_uncensored():
    recs = records([offer("venice/uncensored", description="An uncensored model", moderated=False),
                    offer("v/plain")])
    assert recs["venice/uncensored"].uncensored and not recs["v/plain"].uncensored
    assert recs["venice/uncensored"].openrouter.moderated is False


def test_live_search_records_classify_like_the_catalog():
    from selector_ia.finder.catalog import SNAPSHOT_VERSION, hf_records
    out = {r.id: r for r in hf_records([hf("x/new-model"), hf("x/style", tags=["lora"]),
                                         hf("y/M-heretic", tags=["base_model:finetune:Org/M"], base="Org/M")], NOW)}
    assert set(out) == {"x/new-model", "y/M-heretic"} and out["y/M-heretic"].uncensored
    assert build_snapshot(FakeCatalog(), now=NOW).version == SNAPSHOT_VERSION


def test_hf_capabilities_by_pipeline_tag_and_asr_benchmark():
    recs = records([], [
        hf("a/asr", tag="automatic-speech-recognition", eval_results=[
            Benchmark(source="open-asr-leaderboard", name="mean_wer", value=7.4, date="2025-01-01",
                      dataset="hf-audio/open-asr-leaderboard"),
            Benchmark(source="open-asr-leaderboard", name="mean_wer", value=6.1, date="2026-01-01",
                      dataset="hf-audio/open-asr-leaderboard")]),
        hf("a/vid", tag="image-to-video"), hf("a/emb", tag="sentence-similarity")])
    assert recs["a/asr"].capabilities == ["audio_in"] and recs["a/asr"].benchmarks.asr_wer == 6.1
    assert recs["a/vid"].capabilities == ["video_gen"] and recs["a/emb"].capabilities == ["embeddings"]


def test_old_means_published_and_updated_more_than_two_years_ago():
    recs = records([offer("v/legacy", created=OLD)],
                   [hf("a/old", created=OLD), hf("a/refreshed", created=OLD, last_modified=NOW)])
    assert recs["v/legacy"].old and recs["a/old"].old
    assert not recs["a/refreshed"].old


def test_retired_aliases_and_routers_are_not_listed():
    recs = records([offer("v/gone", expiration_date="2026-01-01"), offer("~v/latest"),
                    offer("openrouter/auto", prompt="-1", completion="-1"), offer("v/ok")])
    assert set(recs) == {"v/ok"}


def test_open_openrouter_models_get_their_hf_card():
    recs = records([offer("org/m", hf_id="Org/M")], [], cards=[hf("Org/M", license="cc-by-nc-4.0")])
    assert recs["Org/M"].license_class == "non_commercial"


def test_newest_query_skips_repos_without_likes():
    catalog = FakeCatalog([], [hf("x/brand-new", likes=0)])
    snap = build_snapshot(catalog, now=NOW)
    assert any(q.get("sort") == "createdAt" for q in catalog.queries)
    assert "x/brand-new" in {r.id for r in snap.records}       # found by the other queries anyway


def test_license_classes():
    assert license_class("mit") == "permissive"
    assert license_class("llama3.3") == "conditions"
    assert license_class("cc-by-nc-sa-4.0") == "non_commercial"
    assert license_class(None) == "unknown"


def test_all_openrouter_modalities_map_to_capabilities():
    recs = records([
        offer("a/video", inputs=("text", "image"), outputs=("video",), params=(), prompt="0", completion="0",
              video_pricing={"cents_per_image_input": "1", "cents_per_video_output_second_480p": "2",
                             "cents_per_video_output_second_1080p": "14"}),
        offer("a/video2", outputs=("video",), params=(), prompt="0", completion="0",
              video_pricing={"duration_seconds_480p": "0.05", "cents_per_second_output": "3"}),
        offer("a/video3", outputs=("video",), params=(), prompt="0", completion="0",
              video_pricing={"duration_seconds_480p": "0.05"}),
        offer("a/tts", outputs=("speech",), params=(), prompt="0.000015", completion="0"),
        offer("a/stt", inputs=("audio",), outputs=("transcription",), params=()),
        offer("a/embed", outputs=("embeddings",), params=()),
        offer("a/rerank", outputs=("rerank",), params=(), prompt="0", completion="0"),
        offer("a/rerank:free", outputs=("rerank",), params=(), prompt="0", completion="0"),
        offer("a/extract", inputs=("text", "image"), outputs=("decisions",), params=()),
    ])
    assert recs["a/video"].capabilities == ["video_gen"]
    assert recs["a/video"].openrouter.video_usd_per_second == 0.02 and recs["a/video"].openrouter.video_resolution == "480p"
    assert recs["a/video"].openrouter.prices == {} and recs["a/video"].openrouter.free_id is None   # 0/0 ≠ free
    assert recs["a/video2"].openrouter.video_usd_per_second == 0.03                 # explicit cents/second only
    assert recs["a/video3"].openrouter.video_usd_per_second is None                 # undocumented units: no price
    assert recs["a/video3"].openrouter.video_skus == {"duration_seconds_480p": "0.05"}   # shown as published
    assert recs["a/tts"].capabilities == ["voice"] and recs["a/stt"].capabilities == ["audio_in"]
    assert recs["a/embed"].capabilities == ["embeddings"]
    assert recs["a/rerank"].capabilities == ["rerank"] and recs["a/rerank"].openrouter.free_id == "a/rerank:free"
    assert "a/extract" not in recs                                       # 'decisions': fits no checkbox


def test_architecture_mtp_and_max_context_from_config():
    from selector_ia.data.models import Benchmark as B
    qwen = {"text_config": {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256,
                            "max_position_embeddings": 262144, "mtp_num_hidden_layers": 1,
                            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}}
    catalog = FakeCatalog([], [
        hf("Qwen/Q", eval_results=[B(source="x", name="pass@1", value=60.0, dataset="ScaleAI/SWE-bench_Pro"),
                                   B(source="x", name="acc", value=80.0, dataset="Idavidrein/gpqa", verified=True)]),
        hf("fan/Q-abliterated-GGUF", tags=["gguf", "base_model:quantized:Qwen/Q"], base="Qwen/Q"),
        hf("Maker/image", tag="text-to-image")], configs={"Qwen/Q": qwen})
    recs = {r.id: r for r in build_snapshot(catalog, now=NOW).records}
    q = recs["Qwen/Q"]
    assert q.arch["layer_counts"] == {"linear_attention": 48, "full_attention": 16}
    assert q.max_context == 262144 and q.mtp
    assert {(e.category, e.verified) for e in q.evals} == {("agentic_coding", False), ("reasoning", True)}
    assert recs["fan/Q-abliterated-GGUF"].arch == q.arch            # GGUF repo: its base model's architecture
    assert recs["Maker/image"].arch is None                          # only text models need the KV cache
    build_snapshot(catalog, now=NOW)
    assert catalog.config_calls.count("Qwen/Q") == 1                 # configs are cached for good
    assert catalog.config_calls.count("fan/Q-abliterated-GGUF") == 1  # including "not published"


def test_programming_needs_evidence():
    recs = records([
        offer("v/indexed", aa={"coding_index": 40.0}),
        offer("v/described", description="Strong at large codebases and fixing bugs."),
        offer("v/devstral-small"),
        offer("v/poet", description="Writes beautiful poems."),
    ])
    assert recs["v/indexed"].code_evidence == "índice de programación de Artificial Analysis"
    assert recs["v/described"].code_evidence == "la descripción de su proveedor"
    assert "code" in recs["v/devstral-small"].capabilities
    assert "code" not in recs["v/poet"].capabilities and "chat" in recs["v/poet"].capabilities


def test_parameter_count_that_contradicts_the_name_is_corrected():
    recs = records([offer("prism-ml/ternary-bonsai-2-9b", hf_id="prism-ml/Ternary-Bonsai-2-9B-gguf")],
                   [hf("prism-ml/Ternary-Bonsai-2-9B-gguf", params_total=1_470_621_184),
                    hf("Qwen/Qwen3-30B-A3B", params_total=30_500_000_000), hf("x/model", params_total=7)])
    bonsai = recs["prism-ml/Ternary-Bonsai-2-9B-gguf"]
    assert bonsai.params == 9_000_000_000 and "1,5 mil millones" in bonsai.params_note
    assert recs["Qwen/Qwen3-30B-A3B"].params == 30_500_000_000 and recs["Qwen/Qwen3-30B-A3B"].params_note is None
    assert recs["x/model"].params == 7                                   # no size in the name: as published
