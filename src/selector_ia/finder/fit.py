"""«¿Cabe?» for every open model of the catalog at once, on one saved computer and one way of using it.

An ESTIMATE from the parameter count (always labelled as such), with the same rules as hardware/fit.py:
memory = weights + KV cache (context × agents, f16 or q8_0, only the layers that keep a growing cache)
+ margin. Text models in GGUF quantizations (sizes measured on unsloth/Qwen3-8B-GGUF), image/video/voice
models in 16 bits. The exact check with the real files is hardware/check.check_model (one model, on demand).
"""

from pydantic import BaseModel

from selector_ia.finder.catalog import LLM_CAPS, ModelRecord
from selector_ia.hardware.fit import Verdict, assess_model, gb
from selector_ia.hardware.profile import GIB, HardwareProfile
from selector_ia.hardware.variants import Variant, formula_variants

CONTEXTS = (8192, 32768, 131072, 262144)
AGENTS = (1, 2, 4, 8)


class Usage(BaseModel):
    """How the student will run it: drives the KV cache and the quantization floor."""
    context: int = 8192          # tokens each agent needs
    agents: int = 1              # parallel requests / agents on the same model
    kv_q8: bool = False          # quantized KV cache (q8_0): about half the cache memory
    agentic: bool = False        # tool calling: Q5_K_M or better (IQ4_XS as exception)

    @property
    def kv_bytes(self) -> int:
        return 1 if self.kv_q8 else 2


def _variants(rec: ModelRecord) -> list[Variant]:
    if LLM_CAPS & set(rec.capabilities):
        return formula_variants(rec.params)
    return [Variant(label="F16", format="16 bits (estimación)", size_bytes=rec.params * 2, estimated=True)]


def _gb(n: int | None) -> float | None:
    return round(n / GIB, 1) if n else None


def estimate(rec: ModelRecord, profile: HardwareProfile, use: Usage | None = None) -> dict:
    use = use or Usage()
    if not rec.hf_id:
        return {"verdict": "unknown", "text": "Solo se usa por API: no se descarga."}
    llm = bool(LLM_CAPS & set(rec.capabilities))
    if llm and rec.max_context and use.context > rec.max_context:
        return {"verdict": "no", "estimate": True, "context_limit": rec.max_context,
                "text": f"Admite como máximo {rec.max_context:,} tokens de contexto y pides {use.context:,}."
                        .replace(",", ".")}
    if not rec.params:
        return {"verdict": "unknown", "text": "Su ficha no publica el número de parámetros: no se puede estimar."}
    fit = assess_model(_variants(rec), profile, params_total=rec.params, config=rec.arch if llm else None,
                       context_tokens=use.context, kv_bytes_per_element=use.kv_bytes, agents=use.agents,
                       agentic=use.agentic and llm)
    chosen = fit.recommended
    kv_note = None if not llm or rec.arch else "Sin datos de su arquitectura: no incluye la memoria del contexto."
    if chosen is None:
        below = fit.below_floor
        if below is not None:     # fits, but only under the agentic quality floor
            mem = below.memory
            return {"verdict": "limited", "estimate": True, "variant": below.variant.label, "below_floor": True,
                    "need_gb": _gb(mem.total_bytes), "weights_gb": _gb(mem.weights_bytes),
                    "kv_gb": _gb(mem.kv.bytes if mem.kv else None), "kv_note": kv_note,
                    "text": f"Para agentes conviene Q5_K_M o mejor (o IQ4_XS) y no cabe; la mejor que cabe es "
                            f"{below.variant.label}, por debajo de lo recomendado para usar herramientas."}
        sized = [r for r in fit.results if r.memory.total_bytes]
        smallest = min(sized, key=lambda r: r.memory.total_bytes) if sized else None
        return {"verdict": Verdict.NO.value if fit.nothing_fits else Verdict.UNKNOWN.value, "estimate": True,
                "text": smallest.reason if smallest else "No se puede calcular.", "kv_note": kv_note,
                "need_gb": _gb(smallest.memory.total_bytes) if smallest else None,
                "variant": smallest.variant.label if smallest else None}
    mem = chosen.memory
    return {"verdict": chosen.verdict.value, "estimate": True, "variant": chosen.variant.label,
            "need_gb": _gb(mem.total_bytes), "weights_gb": _gb(mem.weights_bytes),
            "kv_gb": _gb(mem.kv.bytes if mem.kv else None), "margin_gb": _gb(mem.margin_bytes), "kv_note": kv_note,
            "text": f"{chosen.reason} Versión {chosen.variant.label}." if mem.total_bytes else chosen.reason,
            "need_text": gb(mem.total_bytes) if mem.total_bytes else None}


def estimate_all(records: list[ModelRecord], profile: HardwareProfile, use: Usage | None = None) -> dict[str, dict]:
    return {r.id: estimate(r, profile, use) for r in records if r.hf_id}
