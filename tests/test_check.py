from datetime import datetime, timezone

from selector_ia.data.catalog import CatalogResult
from selector_ia.data.models import HFModel, RepoFile
from selector_ia.hardware.check import check_model
from selector_ia.hardware.profile import GIB, HardwareProfile
from tests.conftest import load_fixture

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
CPU16 = HardwareProfile(name="pc", kind="local", os="Windows 11", ram_bytes=16 * GIB)


class FakeCatalog:
    def __init__(self, models, files, quantized=(), config=None):
        self.models, self.files, self.quantized, self.config = models, files, list(quantized), config
        self.listed = []

    def hf_model(self, model_id):
        return self.models[model_id]

    def model_config(self, _repo):
        return self.config

    def repo_files(self, repo):
        self.listed.append(repo)
        return self.files.get(repo, [])

    def quantized_versions(self, _base):
        return CatalogResult(self.quantized, NOW, from_cache=True)


def tree():
    return [RepoFile(path=f["path"], size=(f.get("lfs") or {}).get("size", f.get("size")))
            for f in load_fixture("tree_unsloth_qwen3_8b_gguf.json") if f["type"] == "file"]


def test_llm_uses_published_gguf_and_suggests_ollama():
    base = HFModel(id="Qwen/Qwen3-8B", pipeline_tag="text-generation", params_total=8_190_735_360, fetched_at=NOW)
    gguf = HFModel(id="unsloth/Qwen3-8B-GGUF", tags=["gguf"], downloads=10, fetched_at=NOW)
    catalog = FakeCatalog({base.id: base},
                          {base.id: [RepoFile(path="model.safetensors", size=16_381_516_776)],
                           gguf.id: tree()},
                          quantized=[gguf], config=load_fixture("config_qwen3_8b.json"))
    check = check_model(catalog, base.id, CPU16)
    assert check.fit.recommended.variant.label == "Q4_K_M"
    assert check.fit.recommended.variant.repo_id == "unsloth/Qwen3-8B-GGUF"
    assert check.runners[0].command == "ollama run hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"
    assert all(not r.variant.estimated for r in check.fit.results)


def test_llm_without_gguf_falls_back_to_labelled_estimates():
    base = HFModel(id="o/new-llm", pipeline_tag="text-generation", params_total=4 * 10 ** 9, fetched_at=NOW)
    check = check_model(FakeCatalog({base.id: base}, {}), base.id, CPU16)
    assert any("No hay versiones GGUF" in n for n in check.notes)
    assert check.fit.results and all(r.variant.estimated and r.variant.url is None for r in check.fit.results)


def test_non_llm_does_not_scan_gguf_conversions():
    asr = HFModel(id="openai/whisper-small", pipeline_tag="automatic-speech-recognition",
                  library_name="transformers", fetched_at=NOW)
    catalog = FakeCatalog({asr.id: asr}, {asr.id: [RepoFile(path="model.safetensors", size=967_000_000)]})
    check = check_model(catalog, asr.id, CPU16)
    assert catalog.listed == [asr.id]
    assert check.fit.recommended.verdict.value == "ok"
    assert check.runners[0].tool == "Transformers"
