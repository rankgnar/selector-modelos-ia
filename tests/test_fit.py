"""Memory estimate correctness, checked against hand-computed numbers from real Qwen3-8B data."""

import pytest

from selector_ia.data.models import RepoFile
from selector_ia.hardware.fit import (
    Verdict, assess_model, choose, estimate_memory, kv_cache, assess,
)
from selector_ia.hardware.profile import GIB, GPU, HardwareProfile
from selector_ia.hardware.variants import BYTES_PER_WEIGHT, Variant, formula_variants, gguf_variants
from tests.conftest import load_fixture

QWEN3_8B_PARAMS = 8_190_735_360          # safetensors total published on Hugging Face
Q4_K_M_SIZE = 5_027_784_512              # real file size, unsloth/Qwen3-8B-GGUF
QWEN_CONFIG = load_fixture("config_qwen3_8b.json")


def qwen_variants():
    files = [RepoFile(path=f["path"], size=(f.get("lfs") or {}).get("size", f.get("size")))
             for f in load_fixture("tree_unsloth_qwen3_8b_gguf.json") if f["type"] == "file"]
    return gguf_variants("unsloth/Qwen3-8B-GGUF", files)


def profile(ram_gb, vram_gb=None, *, unified=False, integrated=False, disk_gb=None):
    gpus = []
    if unified:
        gpus = [GPU(name="Apple M3", vendor="apple", vram_bytes=int(ram_gb * GIB), unified_memory=True)]
    elif vram_gb is not None:
        gpus = [GPU(name="GPU", vendor="amd" if integrated else "nvidia", vram_bytes=int(vram_gb * GIB),
                    integrated=integrated)]
    return HardwareProfile(name="t", kind="remote_manual", ram_bytes=int(ram_gb * GIB), gpus=gpus,
                           disk_free_bytes=int(disk_gb * GIB) if disk_gb else None, os="Ubuntu 24.04")


def test_kv_cache_gqa_formula():
    kv = kv_cache(QWEN_CONFIG, 8192)
    # 2 (K,V) × 36 layers × 8 kv heads × 128 head_dim × 8192 tokens × 2 bytes
    assert kv.bytes == 2 * 36 * 8 * 128 * 8192 * 2 == 1_207_959_552
    assert "36 capas" in kv.formula and "8 cabezas_kv" in kv.formula


def test_kv_cache_scales_with_context():
    assert kv_cache(QWEN_CONFIG, 32768).bytes == 4 * kv_cache(QWEN_CONFIG, 8192).bytes


def test_kv_cache_mla_formula():
    kv = kv_cache(load_fixture("config_deepseek_v3.json"), 8192)
    assert kv.bytes == 61 * (512 + 64) * 8192 * 2 == 575_668_224


def test_kv_cache_head_dim_from_hidden_size_and_nested_text_config():
    config = {"text_config": {"num_hidden_layers": 10, "num_attention_heads": 16, "hidden_size": 2048}}
    assert kv_cache(config, 1000).bytes == 2 * 10 * 16 * 128 * 1000 * 2  # kv heads default to heads


def test_kv_cache_missing_data_is_none():
    assert kv_cache(None, 8192) is None
    assert kv_cache({"model_type": "whisper"}, 8192) is None


def test_total_uses_real_file_size_plus_kv_plus_margin():
    q4 = next(v for v in qwen_variants() if v.label == "Q4_K_M")
    assert q4.size_bytes == Q4_K_M_SIZE and not q4.estimated
    mem = estimate_memory(q4, QWEN3_8B_PARAMS, kv_cache(QWEN_CONFIG, 8192))
    assert mem.weights_bytes == Q4_K_M_SIZE
    assert mem.total_bytes == int((Q4_K_M_SIZE + 1_207_959_552) * 1.15) == 7_171_105_673
    assert "tamaño real" in mem.weights_source


def test_formula_fallback_is_labelled_estimate():
    variants = {v.label: v for v in formula_variants(QWEN3_8B_PARAMS)}
    q4 = variants["Q4_K_M"]
    assert q4.estimated and q4.url is None and q4.repo_id is None
    assert q4.size_bytes == int(QWEN3_8B_PARAMS * BYTES_PER_WEIGHT["Q4_K_M"])
    assert "estimación" in estimate_memory(q4, QWEN3_8B_PARAMS, None).weights_source
    # Measured constant reproduces the real file within 1 %.
    assert abs(q4.size_bytes - Q4_K_M_SIZE) / Q4_K_M_SIZE < 0.01


def test_unknown_size_and_params_is_unknown_not_guessed():
    v = Variant(label="AWQ", format="AWQ", repo_id="x/y")
    mem = estimate_memory(v, None, None)
    assert mem.total_bytes is None and mem.weights_source == "sin datos"
    assert assess(v, mem, profile(32, 24)).verdict is Verdict.UNKNOWN


def fit_q4(p):
    fit = assess_model(qwen_variants(), p, params_total=QWEN3_8B_PARAMS, config=QWEN_CONFIG)
    return fit, next(r for r in fit.results if r.variant.label == "Q4_K_M")


@pytest.mark.parametrize("prof,expected", [
    (profile(64, 24), Verdict.OK),                      # fits entirely in a 24 GB GPU
    (profile(32, 4), Verdict.LIMITED),                  # partial offload to RAM
    (profile(16), Verdict.LIMITED),                     # CPU only, fits in RAM
    (profile(16, unified=True), Verdict.OK),            # Apple: 75 % of 16 GB = 12 GB
    (profile(6), Verdict.NO),                           # 6 GB × 0.8 = 4.8 GB < 6.7 GB
    (profile(64, 24, disk_gb=2), Verdict.NO),           # not enough disk
])
def test_verdicts(prof, expected):
    _fit, q4 = fit_q4(prof)
    assert q4.verdict is expected, q4.reason


def test_integrated_gpu_is_not_counted_as_vram():
    _fit, q4 = fit_q4(profile(16, 0.5, integrated=True))
    assert q4.verdict is Verdict.LIMITED and "sin GPU" in q4.reason


def test_recommends_q8_on_big_gpu_not_16_bit():
    fit, _ = fit_q4(profile(64, 24))
    assert fit.recommended.variant.label == "Q8_0"


def test_recommends_q4_k_m_on_cpu():
    fit, _ = fit_q4(profile(16))
    assert fit.recommended.variant.label == "Q4_K_M"


def test_never_recommends_below_3_bits_when_better_fits():
    fit, _ = fit_q4(profile(16))
    assert fit.recommended.memory.weights_bytes / QWEN3_8B_PARAMS >= 0.45


def test_nothing_fits_explains_hardware_needed():
    fit = assess_model(qwen_variants(), profile(2), params_total=QWEN3_8B_PARAMS, config=QWEN_CONFIG)
    assert fit.nothing_fits and fit.recommended is None
    assert "Q4_K_M" in fit.hardware_needed and "VRAM" in fit.hardware_needed


def test_choose_handles_empty():
    assert choose([], None) is None
