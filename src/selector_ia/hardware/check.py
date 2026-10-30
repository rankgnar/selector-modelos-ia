"""Puts it together for one model: real downloadable variants -> fit on a profile -> runner tips."""

from pydantic import BaseModel

from selector_ia.data.catalog import Catalog
from selector_ia.data.models import HFModel
from selector_ia.errors import SelectorError
from selector_ia.hardware.fit import DEFAULT_CONTEXT, ModelFit, assess_model
from selector_ia.hardware.profile import HardwareProfile
from selector_ia.hardware.runners import CHAT_TASKS, RunnerSuggestion, suggest_runners
from selector_ia.hardware.variants import Variant, formula_variants, gguf_variants, runs_on, weights_variant

GGUF_REPOS_TO_SCAN = 2
GENERATION_TASKS = {"text-to-image", "image-to-image", "text-to-video", "image-to-video", "image-text-to-video",
                    "image-text-to-image", "video-to-video", "text-to-3d", "image-to-3d"}


class ModelCheck(BaseModel):
    model: HFModel
    fit: ModelFit
    runners: list[RunnerSuggestion]
    context_tokens: int
    notes: list[str]


def collect_variants(catalog: Catalog, model: HFModel, notes: list[str]) -> list[Variant]:
    variants: list[Variant] = []
    files = catalog.repo_files(model.id)
    if model.quant_format == "GGUF":
        variants += gguf_variants(model.id, files)
    else:
        own = weights_variant(model, files)
        if own:
            variants.append(own)
        if model.pipeline_tag not in CHAT_TASKS:
            return variants  # GGUF/AWQ conversions are for LLM runtimes (llama.cpp, vLLM)
        quantized = catalog.quantized_versions(model.id).items
        gguf_repos = sorted((q for q in quantized if q.quant_format == "GGUF"),
                            key=lambda q: q.downloads or 0, reverse=True)[:GGUF_REPOS_TO_SCAN]
        seen = set()
        for repo in gguf_repos:
            try:
                repo_files = catalog.repo_files(repo.id)
            except SelectorError:
                continue
            for v in gguf_variants(repo.id, repo_files):
                if v.label not in seen:
                    seen.add(v.label)
                    variants.append(v)
        for q in sorted(quantized, key=lambda q: q.downloads or 0, reverse=True):
            if q.quant_format in ("AWQ", "GPTQ", "FP8", "MLX") and q.quant_format not in seen:
                seen.add(q.quant_format)
                try:
                    v = weights_variant(q, catalog.repo_files(q.id))
                except SelectorError:
                    continue
                if v:
                    variants.append(v)
        if not gguf_repos:
            notes.append("No hay versiones GGUF publicadas de este modelo.")
            if model.params_total and model.pipeline_tag in (None, "text-generation", "image-text-to-text"):
                notes.append("Los tamaños GGUF mostrados son una estimación por fórmula (no existen archivos).")
                variants += formula_variants(model.params_total)
    return variants


def check_model(catalog: Catalog, model_id: str, profile: HardwareProfile,
                context_tokens: int | None = None) -> ModelCheck:
    model = catalog.hf_model(model_id)
    notes: list[str] = []
    config = catalog.model_config(model.base_models[0] if model.quant_format and model.base_models else model.id)
    max_ctx = _max_context(config) or (model.gguf.context_length if model.gguf else None)
    context = context_tokens or min(DEFAULT_CONTEXT, max_ctx or DEFAULT_CONTEXT)
    if config is None and model.pipeline_tag in CHAT_TASKS:
        notes.append("Sin config.json accesible: la memoria del contexto (caché KV) no se incluye.")
    variants = collect_variants(catalog, model, notes)
    if model.pipeline_tag in GENERATION_TASKS:
        notes.append("Generar imágenes o vídeo necesita memoria extra durante el proceso (resolución, duración): "
                     "la estimación puede quedarse corta.")
    if not variants:
        notes.append("Este repositorio no publica un tamaño de pesos que se pueda calcular (p. ej. archivos "
                     "alternativos para ComfyUI): revisa su ficha.")
    skipped = sorted({v.format for v in variants if not runs_on(v, profile)})
    if skipped:
        notes.append(f"No se muestran las versiones {', '.join(skipped)}: no funcionan en este equipo.")
    variants = [v for v in variants if runs_on(v, profile)]
    fit = assess_model(variants, profile, params_total=model.params_total, config=config, context_tokens=context)
    chosen = fit.recommended.variant if fit.recommended else None
    return ModelCheck(model=model, fit=fit, runners=suggest_runners(model, chosen, profile, context),
                      context_tokens=context, notes=notes)


def _max_context(config: dict | None) -> int | None:
    if not config:
        return None
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    value = cfg.get("max_position_embeddings")
    return value if isinstance(value, int) else None
