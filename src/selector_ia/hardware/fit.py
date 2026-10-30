"""Does a model fit on a hardware profile? Memory ESTIMATE (always labelled as such, with formula).

    memoria ≈ (pesos + caché KV del contexto) × 1,15
    pesos    = tamaño real de los archivos publicados  (o parámetros × bytes/peso si no hay archivos)
    caché KV = 2 × capas × cabezas_kv × dim_cabeza × tokens_contexto × 2 bytes (FP16)
"""

from enum import Enum

from pydantic import BaseModel

from selector_ia.hardware.profile import GIB, HardwareProfile
from selector_ia.hardware.variants import (
    BYTES_PER_WEIGHT, BYTES_PER_WEIGHT_SOURCE, MIN_RECOMMENDED_BYTES_PER_WEIGHT, Variant,
)

MARGIN = 0.15               # runtime buffers, activations
RAM_USABLE_SHARE = 0.80     # leave room for the OS and other programs
UNIFIED_GPU_SHARE = 0.75    # macOS lets the GPU use ~75 % of unified memory by default
SMALL_FOR_CPU_BYTES = 2 * GIB
KV_BYTES_PER_ELEMENT = 2    # FP16 cache (llama.cpp, vLLM default)
DEFAULT_CONTEXT = 8192

FORMULA = ("memoria ≈ (pesos + caché KV) × 1,15 · caché KV = 2 × capas × cabezas_kv × dim_cabeza × "
           "tokens × 2 bytes")
FORMULA_NO_KV = "memoria ≈ pesos × 1,15 (sin caché de contexto: no aplica o sin datos)"


class Verdict(str, Enum):
    OK = "ok"
    LIMITED = "limited"
    NO = "no"
    UNKNOWN = "unknown"

    @property
    def icon(self) -> str:
        return {"ok": "✅", "limited": "⚠️", "no": "❌", "unknown": "❔"}[self.value]

    @property
    def text(self) -> str:
        return {"ok": "funciona bien", "limited": "funciona con limitaciones", "no": "no cabe",
                "unknown": "no se puede calcular"}[self.value]


class KVCache(BaseModel):
    bytes: int
    context_tokens: int
    formula: str


class MemoryEstimate(BaseModel):
    weights_bytes: int | None
    weights_source: str            # where the weight size comes from
    kv: KVCache | None
    total_bytes: int | None
    formula: str = FORMULA


class FitResult(BaseModel):
    variant: Variant
    memory: MemoryEstimate
    verdict: Verdict
    reason: str


class ModelFit(BaseModel):
    results: list[FitResult]
    recommended: FitResult | None
    nothing_fits: bool
    hardware_needed: str | None    # what would be required when nothing fits


def gb(n: int | float) -> str:
    return f"{n / GIB:.1f} GB".replace(".", ",")


# --- KV cache ---

def kv_cache(config: dict | None, context_tokens: int) -> KVCache | None:
    """From the model's config.json. None when the fields are missing (shown as 'sin datos')."""
    if not config:
        return None
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    if not isinstance(layers, int):
        return None
    # Multi-head latent attention (DeepSeek-style): compressed cache.
    if isinstance(cfg.get("kv_lora_rank"), int):
        per_token = cfg["kv_lora_rank"] + (cfg.get("qk_rope_head_dim") or 0)
        total = layers * per_token * context_tokens * KV_BYTES_PER_ELEMENT
        return KVCache(bytes=total, context_tokens=context_tokens,
                       formula=f"{layers} capas × ({cfg['kv_lora_rank']} + {cfg.get('qk_rope_head_dim') or 0}) × "
                               f"{context_tokens} tokens × 2 bytes (atención latente)")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim")
    if not head_dim and isinstance(cfg.get("hidden_size"), int) and isinstance(heads, int) and heads:
        head_dim = cfg["hidden_size"] // heads
    if not (isinstance(kv_heads, int) and isinstance(head_dim, int)):
        return None
    total = 2 * layers * kv_heads * head_dim * context_tokens * KV_BYTES_PER_ELEMENT
    return KVCache(bytes=total, context_tokens=context_tokens,
                   formula=f"2 × {layers} capas × {kv_heads} cabezas_kv × {head_dim} × {context_tokens} tokens × 2 bytes")


# --- memory ---

def estimate_memory(variant: Variant, params_total: int | None, kv: KVCache | None) -> MemoryEstimate:
    if variant.size_bytes is not None and not variant.estimated:
        weights, source = variant.size_bytes, f"tamaño real de los archivos ({variant.repo_id})"
    elif variant.size_bytes is not None:
        weights, source = variant.size_bytes, (f"estimación: parámetros × {BYTES_PER_WEIGHT[variant.label]} "
                                                f"bytes/peso ({BYTES_PER_WEIGHT_SOURCE})")
    elif params_total and variant.label in BYTES_PER_WEIGHT:
        bpw = BYTES_PER_WEIGHT[variant.label]
        weights, source = int(params_total * bpw), f"estimación: parámetros × {bpw} bytes/peso"
    else:
        return MemoryEstimate(weights_bytes=None, weights_source="sin datos", kv=kv, total_bytes=None)
    total = int((weights + (kv.bytes if kv else 0)) * (1 + MARGIN))
    return MemoryEstimate(weights_bytes=weights, weights_source=source, kv=kv, total_bytes=total,
                          formula=FORMULA if kv else FORMULA_NO_KV)


# --- verdict ---

def gpu_memory(profile: HardwareProfile) -> int:
    if profile.has_unified_memory and profile.ram_bytes:
        return int(profile.ram_bytes * UNIFIED_GPU_SHARE)
    return profile.dedicated_vram_bytes


def assess(variant: Variant, memory: MemoryEstimate, profile: HardwareProfile) -> FitResult:
    def result(verdict: Verdict, reason: str) -> FitResult:
        return FitResult(variant=variant, memory=memory, verdict=verdict, reason=reason)

    need = memory.total_bytes
    if need is None:
        return result(Verdict.UNKNOWN, "No se publica el número de parámetros ni el tamaño de los archivos.")
    if profile.ram_bytes is None:
        return result(Verdict.UNKNOWN, "Falta la RAM del equipo en el perfil.")
    if profile.disk_free_bytes is not None and memory.weights_bytes and memory.weights_bytes > profile.disk_free_bytes:
        return result(Verdict.NO, f"No hay espacio en disco: ocupa {gb(memory.weights_bytes)} y quedan "
                                  f"{gb(profile.disk_free_bytes)} libres.")
    vram = gpu_memory(profile)
    ram = int(profile.ram_bytes * RAM_USABLE_SHARE)
    where = "la memoria unificada" if profile.has_unified_memory else "la GPU"
    if vram and need <= vram:
        return result(Verdict.OK, f"Cabe entero en {where}: necesita ~{gb(need)} de {gb(vram)}.")
    if vram and not profile.has_unified_memory and need <= vram + ram:
        return result(Verdict.LIMITED, f"No cabe entero en la GPU ({gb(vram)}): una parte irá a la RAM. "
                                       f"Funcionará, pero bastante más lento. Necesita ~{gb(need)}.")
    if not vram and need <= ram:
        if need <= SMALL_FOR_CPU_BYTES:
            return result(Verdict.OK, f"Sin GPU, pero es pequeño (~{gb(need)}): irá fluido en CPU.")
        return result(Verdict.LIMITED, f"Cabe en la RAM (~{gb(need)} de {gb(ram)} usables) pero sin GPU irá lento.")
    available = vram + (ram if not profile.has_unified_memory else 0)
    return result(Verdict.NO, f"No cabe: necesita ~{gb(need)} y el equipo tiene ~{gb(available)} utilizables.")


def _bytes_per_param(r: FitResult, params_total: int | None) -> float | None:
    if params_total and r.memory.weights_bytes:
        return r.memory.weights_bytes / params_total
    return BYTES_PER_WEIGHT.get(r.variant.label)


def choose(results: list[FitResult], params_total: int | None) -> FitResult | None:
    """✅ first: the best quality up to 8-bit (Q8 ≈ lossless, half the memory of 16-bit).
    Only ⚠️: the smallest variant that keeps 4-bit quality or better (faster when offloading)."""
    def bpp(r):
        return _bytes_per_param(r, params_total) or 0

    usable = [r for r in results if bpp(r) >= MIN_RECOMMENDED_BYTES_PER_WEIGHT] or results
    ok = [r for r in usable if r.verdict is Verdict.OK]
    if ok:
        up_to_8bit = [r for r in ok if bpp(r) <= 1.1]
        return max(up_to_8bit or ok, key=bpp)
    limited = [r for r in usable if r.verdict is Verdict.LIMITED]
    if limited:
        four_bit = [r for r in limited if bpp(r) >= 0.55]
        return min(four_bit, key=bpp) if four_bit else max(limited, key=bpp)
    return None


def hardware_needed(results: list[FitResult]) -> str | None:
    """Smallest reasonable variant -> memory a GPU (or RAM for CPU) would need."""
    sized = [r for r in results if r.memory.total_bytes]
    if not sized:
        return None
    four_bit = [r for r in sized if r.variant.label.upper().startswith(("Q4", "AWQ", "GPTQ"))]
    target = min(four_bit or sized, key=lambda r: r.memory.total_bytes)
    need = target.memory.total_bytes
    return (f"Para la versión {target.variant.label} haría falta una GPU con al menos {gb(need)} de VRAM "
            f"(o {gb(need / RAM_USABLE_SHARE)} de RAM para ejecutarlo en CPU, mucho más lento).")


def assess_model(
    variants: list[Variant],
    profile: HardwareProfile,
    *,
    params_total: int | None,
    config: dict | None,
    context_tokens: int = DEFAULT_CONTEXT,
) -> ModelFit:
    kv = kv_cache(config, context_tokens)
    results = [assess(v, estimate_memory(v, params_total, kv), profile) for v in variants]
    recommended = choose(results, params_total)
    nothing = recommended is None and any(r.verdict is Verdict.NO for r in results)
    return ModelFit(results=results, recommended=recommended, nothing_fits=nothing,
                    hardware_needed=hardware_needed(results) if nothing else None)
