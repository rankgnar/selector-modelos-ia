from selector_ia.data.models import HFModel, RepoFile
from selector_ia.hardware.profile import GIB, GPU, HardwareProfile
from selector_ia.hardware.variants import (
    Variant, gguf_quant_label, gguf_variants, runs_on, weights_variant,
)
from datetime import datetime, timezone

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def test_gguf_labels():
    assert gguf_quant_label("Qwen3-8B-Q4_K_M.gguf") == "Q4_K_M"
    assert gguf_quant_label("model.q8_0.gguf") == "Q8_0"
    assert gguf_quant_label("Q5_K_M/Big-Q5_K_M-00002-of-00003.gguf") == "Q5_K_M"
    assert gguf_quant_label("mmproj-F16.gguf") is None      # vision projector, not the model
    assert gguf_quant_label("README.md") is None


def test_split_files_are_summed_and_link_is_exact():
    files = [RepoFile(path=f"Q4_K_M/M-Q4_K_M-0000{i}-of-00002.gguf", size=10 * GIB) for i in (1, 2)]
    [v] = gguf_variants("org/M-GGUF", files)
    assert v.size_bytes == 20 * GIB and len(v.files) == 2
    assert v.url == "https://huggingface.co/org/M-GGUF/resolve/main/Q4_K_M/M-Q4_K_M-00001-of-00002.gguf"


def test_missing_file_size_stays_unknown():
    [v] = gguf_variants("org/M", [RepoFile(path="M-Q4_K_M.gguf", size=None)])
    assert v.size_bytes is None


def test_standard_filter_keeps_exotic_when_nothing_standard():
    files = [RepoFile(path="a-IQ2_XXS.gguf", size=1), RepoFile(path="a-Q4_K_M.gguf", size=2)]
    assert [v.label for v in gguf_variants("o/a", files)] == ["Q4_K_M"]
    assert [v.label for v in gguf_variants("o/a", files[:1])] == ["IQ2_XXS"]
    assert len(gguf_variants("o/a", files, standard_only=False)) == 2


def test_weights_variant_prefers_safetensors():
    model = HFModel(id="o/m", fetched_at=NOW, tags=["awq"])
    files = [RepoFile(path="model.safetensors", size=5), RepoFile(path="pytorch_model.bin", size=9),
             RepoFile(path="sub/extra.safetensors", size=100)]
    v = weights_variant(model, files)
    assert v.label == "AWQ" and v.size_bytes == 5 and v.files == ["model.safetensors"]


def test_runs_on():
    cpu = HardwareProfile(name="c", kind="local", ram_bytes=16 * GIB)
    gpu = HardwareProfile(name="g", kind="local", ram_bytes=16 * GIB,
                          gpus=[GPU(name="RTX", vendor="nvidia", vram_bytes=24 * GIB)])
    mac = HardwareProfile(name="m", kind="local", ram_bytes=16 * GIB,
                          gpus=[GPU(name="M3", vendor="apple", vram_bytes=16 * GIB, unified_memory=True)])
    awq, mlx, gguf = (Variant(label=x, format=x) for x in ("AWQ", "MLX", "GGUF"))
    assert not runs_on(awq, cpu) and runs_on(awq, gpu)
    assert runs_on(mlx, mac) and not runs_on(mlx, gpu)
    assert all(runs_on(gguf, p) for p in (cpu, gpu, mac))


def test_quantized_copies_detected_by_name():
    for repo, fmt in [("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "NVFP4"),
                      ("mlx-community/Qwen3-8B-4bit", "4-bit"), ("org/Model-FP8", "FP8")]:
        assert HFModel(id=repo, fetched_at=NOW).quant_format == fmt
    assert HFModel(id="Qwen/Qwen3-8B", fetched_at=NOW).quant_format is None


def test_diffusers_components_are_summed_preferring_fp16():
    model = HFModel(id="Wan-AI/Wan-Diffusers", library_name="diffusers", fetched_at=NOW)
    files = [RepoFile(path="transformer/diffusion_pytorch_model-00001-of-00002.safetensors", size=10),
             RepoFile(path="transformer/diffusion_pytorch_model-00002-of-00002.safetensors", size=5),
             RepoFile(path="vae/diffusion_pytorch_model.safetensors", size=4),
             RepoFile(path="vae/diffusion_pytorch_model.fp16.safetensors", size=2),
             RepoFile(path="model_index.json", size=1)]
    v = weights_variant(model, files)
    assert v.size_bytes == 10 + 5 + 2 and len(v.files) == 3


def test_single_file_repos_have_no_computable_size():
    model = HFModel(id="Comfy-Org/X", library_name="diffusion-single-file", fetched_at=NOW)
    files = [RepoFile(path="split_files/diffusion_models/a.safetensors", size=10),
             RepoFile(path="split_files/diffusion_models/b.safetensors", size=20)]
    assert weights_variant(model, files) is None  # alternatives, not parts: never summed
