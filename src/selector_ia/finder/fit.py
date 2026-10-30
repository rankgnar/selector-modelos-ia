"""«¿Cabe?» for every open model of the catalog at once, on one saved computer.

An ESTIMATE from the parameter count (always labelled as such), with the same rules as hardware/fit.py:
text models in GGUF quantizations (sizes measured on unsloth/Qwen3-8B-GGUF), image/video/voice models
in 16 bits. The exact check with the real files is hardware/check.check_model (one model, on demand).
"""

from selector_ia.finder.catalog import ModelRecord
from selector_ia.hardware.fit import Verdict, assess_model, gb
from selector_ia.hardware.profile import HardwareProfile
from selector_ia.hardware.variants import Variant, formula_variants

LLM_CAPS = {"chat", "code", "vision"}


def _variants(rec: ModelRecord) -> list[Variant]:
    if LLM_CAPS & set(rec.capabilities):
        return formula_variants(rec.params)
    return [Variant(label="F16", format="16 bits (estimación)", size_bytes=rec.params * 2, estimated=True)]


def estimate(rec: ModelRecord, profile: HardwareProfile) -> dict:
    if not rec.hf_id:
        return {"verdict": "unknown", "text": "Solo se usa por API: no se descarga."}
    if not rec.params:
        return {"verdict": "unknown", "text": "Su ficha no publica el número de parámetros: no se puede estimar."}
    fit = assess_model(_variants(rec), profile, params_total=rec.params, config=None)
    chosen = fit.recommended
    if chosen is None:
        sized = [r for r in fit.results if r.memory.total_bytes]
        smallest = min(sized, key=lambda r: r.memory.total_bytes) if sized else None
        need = smallest.memory.total_bytes if smallest else None
        return {"verdict": Verdict.NO.value if fit.nothing_fits else Verdict.UNKNOWN.value, "estimate": True,
                "text": smallest.reason if smallest else "No se puede calcular.",
                "need_gb": round(need / 1024 ** 3, 1) if need else None,
                "variant": smallest.variant.label if smallest else None}
    need = chosen.memory.total_bytes
    return {"verdict": chosen.verdict.value, "estimate": True, "variant": chosen.variant.label,
            "need_gb": round(need / 1024 ** 3, 1) if need else None,
            "text": f"{chosen.reason} Versión {chosen.variant.label}." if need else chosen.reason,
            "need_text": gb(need) if need else None}


def estimate_all(records: list[ModelRecord], profile: HardwareProfile) -> dict[str, dict]:
    return {r.id: estimate(r, profile) for r in records if r.hf_id}
