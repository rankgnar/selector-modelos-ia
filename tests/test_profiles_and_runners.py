from datetime import datetime, timezone

from selector_ia.data.cache import Cache
from selector_ia.data.models import HFModel
from selector_ia.hardware.profile import GIB, GPU, HardwareProfile, SSHTarget
from selector_ia.hardware.runners import suggest_runners
from selector_ia.hardware.variants import Variant

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def test_profiles_roundtrip_including_ones_saved_by_older_versions(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    old = HardwareProfile(name="VPS Hetzner", kind="remote_ssh", ram_bytes=64 * GIB,
                          ssh=SSHTarget(host="1.2.3.4", username="root"))
    cache.save_profile(old)
    cache.save_profile(HardwareProfile(name="Mi PC", kind="local", machine="pc", ram_bytes=16 * GIB))
    assert [x.name for x in cache.list_profiles()] == ["Mi PC", "VPS Hetzner"]
    assert cache.load_profile("Mi PC").machine == "pc"
    assert cache.delete_profile("Mi PC") and not cache.delete_profile("Mi PC")
    cache.close()


LINUX_NVIDIA = HardwareProfile(name="srv", kind="remote_ssh", os="Ubuntu 24.04", ram_bytes=64 * GIB,
                               gpus=[GPU(name="L40S", vendor="nvidia", vram_bytes=48 * GIB)])
WINDOWS_CPU = HardwareProfile(name="pc", kind="local", os="Windows 11", ram_bytes=16 * GIB)


def test_gguf_chat_commands_use_exact_repo_and_quant():
    model = HFModel(id="Qwen/Qwen3-8B", pipeline_tag="text-generation", fetched_at=NOW)
    v = Variant(label="Q4_K_M", format="GGUF", repo_id="Qwen/Qwen3-8B-GGUF")
    tips = {t.tool: t for t in suggest_runners(model, v, WINDOWS_CPU, 8192)}
    assert tips["Ollama"].command == "ollama run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
    assert tips["llama.cpp"].command == "llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M -c 8192"  # no -ngl on CPU
    assert "LM Studio" in tips
    gpu_tips = {t.tool: t for t in suggest_runners(model, v, LINUX_NVIDIA, 4096)}
    assert gpu_tips["llama.cpp"].command.endswith("-ngl 99")


def test_vllm_only_for_nvidia_linux():
    model = HFModel(id="Qwen/Qwen3-8B-AWQ", pipeline_tag="text-generation", fetched_at=NOW, tags=["awq"])
    v = Variant(label="AWQ", format="AWQ", repo_id="Qwen/Qwen3-8B-AWQ")
    assert any(t.tool == "vLLM" for t in suggest_runners(model, v, LINUX_NVIDIA))
    assert not any(t.tool == "vLLM" for t in suggest_runners(model, v, WINDOWS_CPU))


def test_asr_and_embeddings():
    fw = HFModel(id="Systran/faster-whisper-small", pipeline_tag="automatic-speech-recognition",
                 tags=["ctranslate2"], fetched_at=NOW)
    assert suggest_runners(fw, None, WINDOWS_CPU)[0].tool == "faster-whisper"
    emb = HFModel(id="BAAI/bge-m3", pipeline_tag="sentence-similarity", library_name="sentence-transformers",
                  fetched_at=NOW)
    assert "BAAI/bge-m3" in suggest_runners(emb, None, WINDOWS_CPU)[0].command
