"""Downloadable versions of a model (GGUF quantizations, AWQ/GPTQ/FP8 repos, full weights).

Real file sizes from Hugging Face are used whenever they exist. Only when no file is published do
we fall back to `params × bytes per weight`, and that variant is flagged as an estimate.
"""

import re

from pydantic import BaseModel, Field

from selector_ia.data.models import HFModel, RepoFile
from selector_ia.hardware.profile import HardwareProfile

# Bytes per weight measured on unsloth/Qwen3-8B-GGUF (file size / 8,190,735,360 params, 2026-10-30).
# Used ONLY when a model has no published file to measure.
BYTES_PER_WEIGHT = {
    "BF16": 2.0,
    "F16": 2.0,
    "Q8_0": 1.063,
    "Q6_K": 0.821,
    "Q5_K_M": 0.714,
    "Q4_K_M": 0.614,
    "IQ4_NL": 0.585,
    "IQ4_XS": 0.559,   # imatrix: spends bits by weight importance
    "Q3_K_M": 0.504,
}
BYTES_PER_WEIGHT_SOURCE = "medido en unsloth/Qwen3-8B-GGUF (30/10/2026)"

# GGUF levels shown by default (others like IQ1/IQ2/UD-* exist but add noise for non-experts).
STANDARD_GGUF = ("Q3_K_M", "IQ3_M", "IQ3_S", "Q4_K_S", "Q4_K_M", "IQ4_XS", "IQ4_NL", "Q5_K_S", "Q5_K_M", "Q6_K",
                 "Q8_0", "BF16", "F16", "MXFP4")

# Below ~3.5 bits per weight quality drops a lot; never recommended unless nothing else fits.
MIN_RECOMMENDED_BYTES_PER_WEIGHT = 0.45

_GGUF_QUANT = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?:[-_.]|$)"
)
_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class Variant(BaseModel):
    label: str                      # "Q4_K_M", "AWQ", "BF16"...
    format: str                     # "GGUF", "AWQ", "GPTQ", "FP8", "safetensors"...
    repo_id: str | None = None      # None only for pure formula estimates
    files: list[str] = Field(default_factory=list)
    size_bytes: int | None = None   # real total size of the files (dato de Hugging Face)
    estimated: bool = False         # True when size comes from the formula
    url: str | None = None          # exact download link

    @property
    def page_url(self) -> str | None:
        return f"https://huggingface.co/{self.repo_id}" if self.repo_id else None


def gguf_quant_label(filename: str) -> str | None:
    name = filename.rsplit("/", 1)[-1]
    name = _SPLIT.sub(".gguf", name)
    if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
        return None
    match = _GGUF_QUANT.search(name[:-5])
    return match.group(1).upper() if match else None


def gguf_variants(repo_id: str, files: list[RepoFile], standard_only: bool = True) -> list[Variant]:
    """One variant per quantization; split files (-00001-of-00003) are summed.
    With `standard_only`, keeps STANDARD_GGUF levels (unless the repo has none of them)."""
    groups: dict[str, list[RepoFile]] = {}
    for f in files:
        label = gguf_quant_label(f.path)
        if label:
            groups.setdefault(label, []).append(f)
    variants = []
    for label, group in groups.items():
        group.sort(key=lambda f: f.path)
        sizes = [f.size for f in group]
        variants.append(Variant(
            label=label, format="GGUF", repo_id=repo_id, files=[f.path for f in group],
            size_bytes=sum(sizes) if all(s is not None for s in sizes) else None,
            url=f"https://huggingface.co/{repo_id}/resolve/main/{group[0].path}",
        ))
    if standard_only and any(v.label in STANDARD_GGUF for v in variants):
        variants = [v for v in variants if v.label in STANDARD_GGUF]
    return sorted(variants, key=lambda v: v.size_bytes or 0)


def runs_on(variant: Variant, profile: HardwareProfile) -> bool:
    """MLX needs Apple Silicon; AWQ/GPTQ/FP8 need a dedicated GPU (vLLM / Transformers)."""
    if variant.format == "MLX":
        return profile.has_unified_memory
    if variant.format in ("AWQ", "GPTQ", "FP8", "EXL2", "EXL3", "NVFP4", "bitsandbytes", "4-bit", "8-bit",
                          "INT4", "INT8", "W4A16", "W8A8"):
        return bool(profile.dedicated_vram_bytes)
    return True


def weights_variant(model: HFModel, files: list[RepoFile]) -> Variant | None:
    """The repo's own weights (safetensors / AWQ / GPTQ / FP8 / MLX), with their real size.
    Diffusers repos keep each component in a subfolder (transformer/, text_encoder/, vae/...):
    all of them are needed, so they are summed (using the .fp16 copies when a folder has both).
    Single-file repos (ComfyUI style) hold alternative checkpoints: their size is left unknown."""
    weights = [f for f in files if f.path.endswith((".safetensors", ".bin")) and "/" not in f.path]
    if any(f.path.endswith(".safetensors") for f in weights):
        weights = [f for f in weights if f.path.endswith(".safetensors")]
    if not weights and model.library_name == "diffusers":
        weights = _diffusers_weights(files)
    if not weights:
        return None
    sizes = [f.size for f in weights]
    fmt = model.quant_format or "safetensors"
    return Variant(
        label=fmt if model.quant_format else "original (16 bits)",
        format=fmt, repo_id=model.id, files=[f.path for f in weights],
        size_bytes=sum(sizes) if all(s is not None for s in sizes) else None,
        url=f"https://huggingface.co/{model.id}/tree/main",
    )


def _diffusers_weights(files: list[RepoFile]) -> list[RepoFile]:
    by_folder: dict[str, list[RepoFile]] = {}
    for f in files:
        if f.path.endswith(".safetensors") and f.path.count("/") == 1:
            by_folder.setdefault(f.path.split("/")[0], []).append(f)
    chosen = []
    for group in by_folder.values():
        fp16 = [f for f in group if ".fp16." in f.path]
        chosen += fp16 or [f for f in group if ".fp32." not in f.path and ".bf16." not in f.path] or group
    return chosen


def formula_variants(params_total: int) -> list[Variant]:
    """Estimated sizes when no quantized file is published (no download link: it doesn't exist)."""
    return [
        Variant(label=label, format="GGUF (estimación)", size_bytes=int(params_total * bpw), estimated=True)
        for label, bpw in BYTES_PER_WEIGHT.items() if label != "F16"
    ]
