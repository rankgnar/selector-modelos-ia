"""One catalog of current AI models for the search page, built from OpenRouter + Hugging Face.

One `ModelRecord` per model (never per task): an API model from OpenRouter (':free'/':batch'
variants merged), an open model from Hugging Face, or both crossed via `hugging_face_id`.
LoRAs/adapters, plain quantized copies and repackaged files are left out. Versions made by someone
other than the original author (fine-tunes, merges, uncensored/abliterated builds) are kept but marked
`community` (hidden in the page unless the student asks for them). Every field comes from the sources;
unknown = None.

This module only builds data. Filtering, sorting and the wording shown to students live in
web/finder.js, so every click in the page is instant.
"""

import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from pydantic import BaseModel, Field

from selector_ia.data.catalog import Catalog
from selector_ia.data.models import HFModel, OpenRouterModel
from selector_ia.errors import SelectorError

log = logging.getLogger(__name__)

# Hugging Face pipeline tag -> capabilities (checked against GET /api/tasks, 2026-10-30)
HF_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "text-generation": ("chat",),          # "code" only with evidence (see _code_evidence)
    "image-text-to-text": ("chat", "vision"),
    "automatic-speech-recognition": ("audio_in",),
    "audio-text-to-text": ("audio_in",),
    "text-to-speech": ("voice",),
    "text-to-image": ("image_gen",),
    "image-to-image": ("image_edit",),
    "image-text-to-image": ("image_edit",),
    "text-to-video": ("video_gen",),
    "image-to-video": ("video_gen",),
    "image-text-to-video": ("video_gen",),
    "text-to-audio": ("music",),
    "feature-extraction": ("embeddings",),
    "sentence-similarity": ("embeddings",),
    "text-ranking": ("rerank",),
}
EXCLUDED_TAGS = {"pyannote-audio",     # speaker diarization: tagged as speech recognition, does not transcribe
                 "coreml", "mlx"}      # Apple-only formats: this app is for Linux

SNAPSHOT_VERSION = 5       # bump when ModelRecord changes: a saved snapshot of another version is rebuilt
HF_TOP_LIMIT = 50          # per pipeline tag, most downloaded
HF_TRENDING_LIMIT = 30
HF_PROVIDER_LIMIT = 30     # most downloaded among those served by a third-party provider
HF_NEWEST_LIMIT = 30       # newest, so recent models with few downloads are not missed
NEWEST_MIN_LIKES = 5       # ...but skip brand-new repos nobody has liked yet (mostly tests and spam)
OPENROUTER_HF_FETCH = 60   # open-weight OpenRouter models whose HF card is fetched (licence, languages, size)
OLD_AFTER_DAYS = 730       # "antiguo": published and last updated more than 2 years ago
HF_UNCENSORED_LIMIT = 30   # declared uncensored / abliterated versions, per pipeline tag below
UNCENSORED_TAGS = ("text-generation", "image-text-to-text", "text-to-image", "image-to-image",
                   "image-text-to-image", "text-to-video", "image-to-video", "text-to-speech")
UNCENSORED_WORDS = ("uncensored", "abliterated", "unfiltered", "heretic", "obliterated")
ADULT_TAGS = {"not-for-all-audiences", "nsfw"}
# Every chat model can write code; «Programar» only lists those with some evidence of being good at it.
_CODE_NAME = re.compile(r"(?i)cod(e|er|ing|estral)|devstral|starcoder|\bswe\b")
_CODE_TEXT = re.compile(r"(?i)\b(cod(e|es|ing|ebases?)|programming|software|developers?|debugg?ing|bugs?)\b")
_NAME_SIZE = re.compile(r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*b(?![a-z])")
CONFIG_FETCH_LIMIT = 450   # config.json downloads per build (cached for good afterwards: they do not change)
LLM_CAPS = {"chat", "code", "vision"}
ARCH_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "num_attention_heads", "n_head", "num_key_value_heads",
             "head_dim", "hidden_size", "kv_lora_rank", "qk_rope_head_dim", "sliding_window",
             "full_attention_interval", "max_position_embeddings", "num_nextn_predict_layers",
             "mtp_num_hidden_layers")

# Benchmark dataset / task names -> category (our grouping; values are copied as published).
EVAL_CATEGORIES = (
    ("agentic_coding", "Programación con agentes", ("swe-bench", "swe_bench", "swebench", "terminal-bench",
                                                    "terminal_bench", "deep-swe", "livecodebench", "aider")),
    ("tools", "Uso de herramientas y agentes", ("bfcl", "tau-bench", "tau2", "tau_bench", "claw-eval", "toolbench",
                                                "agents-last-exam", "ask-my-agent", "mcp", "agent")),
    ("instructions", "Seguir instrucciones", ("ifeval", "ifbench", "ifstruct", "multichallenge", "arena-hard")),
    ("computer_use", "Uso del ordenador", ("osworld", "webarena", "browsecomp", "screenspot", "computer")),
    ("reasoning", "Razonamiento y conocimiento", ("gpqa", "hle", "mmlu", "aime", "math", "lexam", "omnicompass",
                                                  "supergpqa")),
    ("documents", "Documentos", ("parsebench", "extractbench", "docbench", "ocr", "chartqa", "docvqa")),
    ("vision", "Visión", ("mmmu", "mathvista", "videomme", "mmi")),
    ("speech", "Voz", ("open-asr-leaderboard", "wer", "vaani", "fleurs", "librispeech")),
)

ASR_DATASET = "hf-audio/open-asr-leaderboard"
_VARIANT = re.compile(r":[a-z0-9-]+$")
_REPACK_SUFFIX = re.compile(r"[-_.](diffusers|comfy|comfyui|gguf|fp8|fp16|bf16|hf|safetensors)$")

_PERMISSIVE = {"apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc-by-sa-4.0",
               "cc-by-3.0", "cc0-1.0", "openrail", "openrail++", "creativeml-openrail-m", "bigscience-openrail-m",
               "unlicense", "mpl-2.0", "isc", "artistic-2.0", "ecl-2.0", "afl-3.0", "bsl-1.0", "zlib", "wtfpl"}
_NON_COMMERCIAL = re.compile(r"(^|-)nc(-|$)|non-?commercial|research|academic|cc-by-nc", re.IGNORECASE)


def license_class(license_id: str | None) -> str:
    """permissive | conditions (llama, gemma, qwen, other: allowed with terms) | non_commercial | unknown."""
    if not license_id:
        return "unknown"
    lic = license_id.lower()
    if lic in _PERMISSIVE:
        return "permissive"
    if _NON_COMMERCIAL.search(lic):
        return "non_commercial"
    return "conditions"


class OpenRouterOffer(BaseModel):
    id: str
    url: str
    free_id: str | None = None          # ':free' variant, if any
    context: int | None = None
    prices: dict[str, float] = Field(default_factory=dict)   # USD per million tokens, as published
    tools: bool = False
    reasoning: bool = False
    moderated: bool | None = None
    video_usd_per_second: float | None = None   # cheapest resolution, only from SKUs explicitly in cents/second
    video_resolution: str | None = None
    video_skus: dict[str, str] = Field(default_factory=dict)   # every SKU as published (units not documented)
    expires: str | None = None          # ISO date when OpenRouter retires it
    preview: bool = False


class EvalResult(BaseModel):
    category: str                       # key of EVAL_CATEGORIES or "other"
    name: str                           # dataset id (and task when it adds information)
    value: float
    date: str | None = None
    verified: bool = False              # False = published by the model's author (not independently checked)


class Benchmarks(BaseModel):
    intelligence: float | None = None   # Artificial Analysis (inside the OpenRouter catalog)
    coding: float | None = None
    agentic: float | None = None
    asr_wer: float | None = None        # Open ASR Leaderboard mean WER (HF evalResults), lower is better
    asr_date: str | None = None
    arena_image: float | None = None    # Design Arena ELO (inside the OpenRouter catalog)
    arena_edit: float | None = None


class ModelRecord(BaseModel):
    id: str                             # HF id for open models, OpenRouter id otherwise
    name: str
    author: str
    capabilities: list[str]
    hf_id: str | None = None
    hf_pipeline: str | None = None
    openrouter: OpenRouterOffer | None = None
    providers: list[str] = Field(default_factory=list)       # third-party providers serving it now (HF)
    license: str | None = None
    license_class: str = "unknown"
    languages: list[str] = Field(default_factory=list)
    params: int | None = None
    downloads: int | None = None        # last month, Hugging Face
    likes: int | None = None
    trending: int | None = None
    gated: bool = False
    community: bool = False             # made by someone other than the original author
    base_model: str | None = None       # for community versions: the model it derives from
    uncensored: bool = False            # its author / provider declares it uncensored (name, tags, description)
    adult: bool = False                 # tagged not-for-all-audiences / nsfw by its author
    code_evidence: str | None = None    # why it counts for «Programar»
    params_note: str | None = None      # when the published parameter count contradicts the model's name
    evals: list[EvalResult] = Field(default_factory=list)   # model card results, by category
    arch: dict | None = None            # attention layout from config.json (KV cache per context)
    max_context: int | None = None      # tokens the model accepts (config.json or OpenRouter)
    mtp: bool = False                   # multi-token prediction (speculative decoding) declared
    created: str | None = None          # ISO
    updated: str | None = None
    old: bool = False                   # published and last updated more than OLD_AFTER_DAYS ago
    benchmarks: Benchmarks = Field(default_factory=Benchmarks)

    @property
    def is_open(self) -> bool:
        return self.hf_id is not None


class CatalogSnapshot(BaseModel):
    version: int = SNAPSHOT_VERSION
    records: list[ModelRecord]
    openrouter_date: str | None = None
    hf_date: str | None = None
    built_at: str
    warnings: list[str] = Field(default_factory=list)


# --- classification of Hugging Face repos ---

def _repack_stem(model_id: str) -> str:
    name = model_id.split("/", 1)[-1].lower()
    while True:
        stripped = _REPACK_SUFFIX.sub("", name)
        if stripped == name:
            return name
        name = stripped


def _derived_from(m: HFModel) -> tuple[str | None, str | None]:
    """(relation, base) from tags like base_model:finetune:org/x. relation: quantized/finetune/merge/adapter."""
    for tag in m.tags:
        parts = tag.split(":", 2)
        if len(parts) == 3 and parts[0] == "base_model":
            return parts[1], parts[2]
    return (None, m.base_models[0]) if m.base_models else (None, None)


def declares_uncensored(*texts: str) -> bool:
    joined = " ".join(texts).lower()
    return any(w in joined for w in UNCENSORED_WORDS)


def classify(m: HFModel) -> tuple[str | None, str | None]:
    """("original" | "community" | None to drop, base model). Dropped: adapters, Apple-only formats,
    diarization, and plain quantized copies (a GGUF of a model is the same model; it is offered as a
    download of the original). Quantized repos stay only when they are a different model: uncensored."""
    tags = {t.lower() for t in m.tags}
    if m.library_name in ("peft", "coreml", "mlx") or {"lora", "adapter"} & tags or tags & EXCLUDED_TAGS:
        return None, None
    relation, base = _derived_from(m)
    uncensored = declares_uncensored(m.id, *tags)
    if relation == "adapter":
        return None, None
    if relation == "quantized" or m.quant_format:
        return ("community", base) if uncensored else (None, None)
    if base is None:
        return "original", None
    if base.split("/")[0].lower() == m.id.split("/")[0].lower():
        return "original", None                       # e.g. official instruct model on the org's own base
    return "community", base


def is_original(m: HFModel) -> bool:
    return classify(m)[0] == "original"


def _org(model_id: str) -> str:
    return model_id.split("/", 1)[0].lower()


def _split_undeclared_copies(models: list[HFModel], served: list[tuple[str, str, int]],
                             ) -> tuple[list[HFModel], dict[str, str]]:
    """Repackaged files (same name + "-Diffusers", "-GGUF"…) are dropped, keeping the most liked. Repos that
    carry another author's model name without declaring it (e.g. "x/FastWan2.2-TI2V-5B" next to
    "Wan-AI/Wan2.2-TI2V-5B") are community versions of it. `served`: (org, stem, likes) of models already
    listed via OpenRouter. Returns (originals, {community id: base id})."""
    served_stems = {stem for _org_, stem, _likes in served}
    by_stem: dict[str, HFModel] = {}
    for m in models:
        stem = _repack_stem(m.id)
        if stem in served_stems:
            continue
        current = by_stem.get(stem)
        if current is None or (m.likes or 0) > (current.likes or 0):
            by_stem[stem] = m
    kept = list(by_stem.values())
    originals = [(m.id, _org(m.id), _repack_stem(m.id), m.likes or 0) for m in kept]
    originals += [(None, org, stem, likes) for org, stem, likes in served]
    result, copies = [], {}
    for m in kept:
        org, name, own, likes = _org(m.id), m.id.split("/", 1)[-1].lower(), _repack_stem(m.id), m.likes or 0
        base = next((other_id or stem for other_id, other_org, stem, other_likes in originals
                     if len(stem) >= 6 and stem != own and stem in name and other_org != org and other_likes > likes),
                    None)
        if base:
            copies[m.id] = base
        else:
            result.append(m)
    return result, copies


# --- record building ---

def _per_million(m: OpenRouterModel, kind: str) -> float | None:
    value = m.price_per_million(kind)
    return float(value) if isinstance(value, Decimal) else None


def _openrouter_capabilities(m: OpenRouterModel) -> set[str]:
    """From input/output modalities. Besides text and image, OpenRouter publishes (output_modalities=all)
    'video', 'speech', 'transcription', 'embeddings', 'rerank' and 'decisions' (extraction/classification
    models that fit no checkbox)."""
    inputs, outputs = set(m.input_modalities), set(m.output_modalities)
    caps = set()
    if "text" in inputs and "text" in outputs:
        caps.add("chat")
    if "file" in inputs and "text" in outputs:
        caps.add("pdf")
    if "image" in inputs and "text" in outputs:
        caps.add("vision")
    if "audio" in inputs and ("text" in outputs or "transcription" in outputs):
        caps.add("audio_in")
    if "video" in inputs and "text" in outputs:
        caps.add("video_in")
    if "image" in outputs:
        caps.add("image_edit" if "image" in inputs else "image_gen")
        if "text" in inputs:
            caps.add("image_gen")
    if "audio" in outputs or "speech" in outputs:
        caps.add("voice")
    if "video" in outputs:
        caps.add("video_gen")
    if "embeddings" in outputs:
        caps.add("embeddings")
    if "rerank" in outputs:
        caps.add("rerank")
    if "tools" in m.supported_parameters and "chat" in caps:
        caps.add("tools")
    if (m.supports_reasoning or "reasoning" in m.supported_parameters) and "chat" in caps:
        caps.add("reasoning")
    return caps


def _bench(m: OpenRouterModel, source: str, name: str) -> float | None:
    return next((b.value for b in m.benchmarks if b.source.startswith(source) and b.name == name), None)


def _offer(offers: list[OpenRouterModel]) -> OpenRouterOffer:
    main = next((o for o in offers if ":" not in o.id), offers[0])
    free = next((o for o in offers if o.is_free and o.id != main.id), None)
    if main.is_free and free is None:
        free = main
    prices = {k: v for k in ("prompt", "completion", "audio", "image", "image_output")
              if (v := _per_million(main, k)) is not None}
    if "text" not in main.output_modalities and not main.is_free and not any(prices.values()):
        prices = {}       # 0/0 on a video, speech, rerank… model means "billed another way", not free
    # OpenRouter: "Pricing SKU units can differ by provider" and the keys are not documented. Only SKUs that
    # say so in their name (cents per second of output video) become a price; the rest is shown as published.
    video = {}
    for key, value in main.video_pricing.items():
        for prefix in ("cents_per_video_output_second", "cents_per_second_output"):
            if key.startswith(prefix) and _is_number(value):
                video[key[len(prefix):].lstrip("_") or "cualquier resolución"] = float(value) / 100
    cheapest = min(video.items(), key=lambda kv: kv[1]) if video else (None, None)
    return OpenRouterOffer(
        id=main.id, url=main.url, free_id=free.id if free else None, context=main.context_length, prices=prices,
        tools="tools" in main.supported_parameters,
        reasoning=main.supports_reasoning or "reasoning" in main.supported_parameters,
        video_usd_per_second=cheapest[1], video_resolution=cheapest[0], video_skus=dict(main.video_pricing),
        moderated=main.is_moderated,
        expires=main.expiration_date.isoformat() if main.expiration_date else None,
        preview="preview" in main.id.lower() or "preview" in main.name.lower())


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _code_evidence(name: str, tags: list[str], description: str, evals: list["EvalResult"],
                   coding_index: float | None) -> str | None:
    if coding_index is not None:
        return "índice de programación de Artificial Analysis"
    if any(e.category == "agentic_coding" for e in evals):
        return "benchmarks de programación en su ficha"
    if _CODE_NAME.search(name) or {"code", "coding"} & set(tags):
        return "su nombre o sus etiquetas"
    if _CODE_TEXT.search(description):
        return "la descripción de su proveedor"
    return None


def _checked_params(model_id: str, params: int | None) -> tuple[int | None, str | None]:
    """Some repos (packed / ternary GGUF) publish a parameter count that contradicts the size in their own
    name (e.g. «…-9B» with 1.5 billion): then the name wins and the discrepancy is shown."""
    match = _NAME_SIZE.search(model_id.split("/")[-1])
    if not params or not match:
        return params, None
    named = float(match.group(1)) * 1e9
    if 0.5 <= params / named <= 2:
        return params, None
    published = f"{params / 1e9:.1f}".replace(".", ",")
    return int(named), f"Tamaño según su nombre: sus metadatos publican {published} mil millones de parámetros"


def eval_category(dataset: str, task: str) -> str:
    text = f"{dataset} {task}".lower()
    for key, _label, words in EVAL_CATEGORIES:
        if any(w in text for w in words):
            return key
    return "other"


def _evals(hf: HFModel | None) -> list[EvalResult]:
    out = []
    for b in (hf.eval_results if hf else []):
        dataset = b.dataset or b.source
        name = dataset if b.name in ("", "?", dataset) else f"{dataset} · {b.name}"
        out.append(EvalResult(category=eval_category(dataset, b.name), name=name, value=b.value, date=b.date,
                              verified=bool(b.verified)))
    return out


def slim_config(config: dict | None) -> dict | None:
    """Only what the KV-cache estimate needs, with `layer_types` summarized as counts."""
    if not isinstance(config, dict):
        return None
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    out = {k: cfg[k] for k in ARCH_KEYS if isinstance(cfg.get(k), int)}
    if isinstance(cfg.get("layer_types"), list):
        types = [str(x) for x in cfg["layer_types"]]
        out["layer_counts"] = {t: types.count(t) for t in set(types)}
    return out if any(k in out for k in ("num_hidden_layers", "n_layer", "num_layers")) else None


def _config(catalog: Catalog, repo: str) -> tuple[dict | None, bool]:
    """(config, downloaded now). Cached for good, including "not available" (stored as {})."""
    cached = catalog.cache.load_blob(f"config:{repo}")
    if cached is not None:
        return (cached[0] or None), False
    config = catalog.model_config(repo)
    catalog.cache.save_blob(f"config:{repo}", config or {})
    return config, True


def _add_architectures(catalog: Catalog, records: list["ModelRecord"], say) -> None:
    llms = [r for r in records if r.hf_id and LLM_CAPS & set(r.capabilities)]
    llms.sort(key=lambda r: (r.community, -(r.downloads or 0)))
    fetched = 0
    for i, rec in enumerate(llms, 1):
        if i % 25 == 0:
            say(f"Leyendo la arquitectura de los modelos ({i} de {len(llms)})…")
        arch = None
        for repo in (rec.hf_id, rec.base_model):
            if not repo or fetched >= CONFIG_FETCH_LIMIT:
                continue
            try:
                config, downloaded = _config(catalog, repo)
            except SelectorError:
                continue
            fetched += downloaded
            arch = slim_config(config)
            if arch:
                break
        if arch:
            rec.arch = arch
            rec.max_context = arch.get("max_position_embeddings") or rec.max_context
            rec.mtp = rec.mtp or bool(arch.get("mtp_num_hidden_layers") or arch.get("num_nextn_predict_layers"))


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _is_old(created: datetime | None, updated: datetime | None, now: datetime) -> bool:
    newest = max((d for d in (created, updated) if d), default=None)
    return newest is not None and (now - newest) > timedelta(days=OLD_AFTER_DAYS)


def _record(offers: list[OpenRouterModel], hf: HFModel | None, hf_id: str | None, now: datetime, *,
            community: bool = False, base_model: str | None = None) -> ModelRecord:
    main = next((o for o in offers if ":" not in o.id), offers[0]) if offers else None
    caps = set(_openrouter_capabilities(main)) if main else set()
    if hf and hf.pipeline_tag in HF_CAPABILITIES:
        caps |= set(HF_CAPABILITIES[hf.pipeline_tag])
    benchmarks = Benchmarks()
    if main:
        benchmarks = Benchmarks(
            intelligence=_bench(main, "Artificial Analysis", "intelligence_index"),
            coding=_bench(main, "Artificial Analysis", "coding_index"),
            agentic=_bench(main, "Artificial Analysis", "agentic_index"),
            arena_image=_bench(main, "Design Arena", "models/image elo"),
            arena_edit=_bench(main, "Design Arena", "models/imageediting elo"))
    if hf:
        wer = [b for b in hf.eval_results if b.dataset == ASR_DATASET and b.name == "mean_wer"]
        if wer:
            latest = max(wer, key=lambda b: b.date or "")
            benchmarks.asr_wer, benchmarks.asr_date = latest.value, latest.date
    model_id = hf_id or main.id.split(":")[0]
    created = (hf.created_at if hf and hf.created_at else None) or (main.created if main else None)
    updated = hf.last_modified if hf else None
    if main:
        name = main.name.split(": ", 1)[-1]
    else:
        name = model_id.split("/", 1)[-1]
    tags = [x.lower() for x in hf.tags] if hf else []
    uncensored = declares_uncensored(model_id, name, *tags, *(o.description or "" for o in offers[:1]))
    evals = _evals(hf)
    code = None
    if "chat" in caps:
        code = _code_evidence(f"{model_id} {name}", tags, main.description or "" if main else "", evals,
                              benchmarks.coding)
        if code:
            caps.add("code")
    params, params_note = _checked_params(model_id, hf.params_total if hf else None)
    return ModelRecord(
        id=model_id, name=name, author=model_id.split("/", 1)[0], capabilities=sorted(caps), hf_id=hf_id,
        hf_pipeline=hf.pipeline_tag if hf else None, openrouter=_offer(offers) if offers else None,
        providers=sorted({p.provider for p in (hf.inference_providers if hf else []) if p.status == "live"}),
        license=hf.license if hf else None, license_class=license_class(hf.license) if hf else "unknown",
        languages=hf.languages if hf else [], params=params, params_note=params_note,
        downloads=hf.downloads if hf else None, likes=hf.likes if hf else None,
        trending=hf.trending_score if hf else None, gated=bool(hf.gated) if hf else False,
        community=community, base_model=base_model, uncensored=uncensored, adult=bool(ADULT_TAGS & set(tags)),
        evals=evals, code_evidence=code, max_context=main.context_length if main else None,
        mtp="mtp" in model_id.lower() or "mtp" in tags,
        created=_iso(created), updated=_iso(updated), old=_is_old(created, updated, now), benchmarks=benchmarks)


def build_snapshot(catalog: Catalog, progress: Callable[[str], None] | None = None,
                   now: datetime | None = None) -> CatalogSnapshot:
    say = progress or (lambda _msg: None)
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []

    # 1) OpenRouter: concrete, not retired; ':free'/':batch' variants grouped under the base id
    say("Leyendo el catálogo de OpenRouter…")
    or_result = catalog.openrouter_models()
    if or_result.warning:
        warnings.append(or_result.warning)
    groups: dict[str, list[OpenRouterModel]] = {}
    for m in or_result.items:
        if not (m.is_alias or m.is_router or m.is_retired()):
            groups.setdefault(_VARIANT.sub("", m.id), []).append(m)
    or_by_key: dict[str, list[OpenRouterModel]] = {}
    for base, offers in groups.items():
        offers.sort(key=lambda o: (o.id != base, o.id))
        hf_id = next((o.hugging_face_id for o in offers if o.hugging_face_id), None)
        or_by_key.setdefault(hf_id.lower() if hf_id else base, []).extend(offers)

    # 2) Hugging Face: every pipeline tag, four views of it
    found: dict[str, HFModel] = {}
    dates: list[datetime] = []
    queries = []
    for tag in HF_CAPABILITIES:
        queries += [dict(pipeline_tag=tag, sort="downloads", limit=HF_TOP_LIMIT),
                    dict(pipeline_tag=tag, sort="trendingScore", limit=HF_TRENDING_LIMIT),
                    dict(pipeline_tag=tag, inference_provider="all", sort="downloads", limit=HF_PROVIDER_LIMIT),
                    dict(pipeline_tag=tag, sort="createdAt", limit=HF_NEWEST_LIMIT)]
        if tag in UNCENSORED_TAGS:
            queries += [dict(pipeline_tag=tag, filters=["uncensored"], sort="downloads", limit=HF_UNCENSORED_LIMIT),
                        dict(pipeline_tag=tag, text="abliterated", sort="downloads", limit=HF_UNCENSORED_LIMIT)]
    for i, params in enumerate(queries, 1):
        say(f"Consultando Hugging Face ({i} de {len(queries)})…")
        try:
            res = catalog.search_hf(**params)
        except SelectorError as exc:
            if exc.user_message not in warnings:
                warnings.append(exc.user_message)
            continue
        if res.warning and res.warning not in warnings:
            warnings.append(res.warning)
        if res.fetched_at:
            dates.append(res.fetched_at)
        newest = params["sort"] == "createdAt"
        for m in res.items:
            if newest and (m.likes or 0) < NEWEST_MIN_LIKES:
                continue
            found.setdefault(m.id.lower(), m)

    # 3) HF cards of open models served on OpenRouter that the searches did not bring
    missing = [(k, offers) for k, offers in or_by_key.items()
               if offers[0].hugging_face_id and k not in found]
    missing.sort(key=lambda item: -(item[1][0].created.timestamp() if item[1][0].created else 0))
    for i, (key, offers) in enumerate(missing[:OPENROUTER_HF_FETCH], 1):
        say(f"Leyendo fichas de modelos abiertos ({i} de {min(len(missing), OPENROUTER_HF_FETCH)})…")
        hf_id = next(o.hugging_face_id for o in offers if o.hugging_face_id)
        try:
            found[key] = catalog.hf_model(hf_id)
        except SelectorError:
            log.info("hugging_face_id %s not found on HF", hf_id)

    # 4) records: OpenRouter ones (with their HF card when known), then HF-only originals
    records = []
    for key, offers in or_by_key.items():
        hf_id = next((o.hugging_face_id for o in offers if o.hugging_face_id), None)
        hf = found.get(key)
        if hf and hf.tags and {t.lower() for t in hf.tags} & EXCLUDED_TAGS:
            continue
        rec = _record(offers, hf, hf.id if hf else hf_id, now)
        if rec.capabilities:              # e.g. 'decisions' (extraction) models fit no checkbox
            records.append(rec)
    served = [(_org(r.hf_id), _repack_stem(r.hf_id), r.likes or 0) for r in records if r.hf_id]
    originals, community = [], []
    for k, m in found.items():
        if k in or_by_key or m.pipeline_tag not in HF_CAPABILITIES:
            continue
        kind, base = classify(m)
        if kind == "original":
            originals.append(m)
        elif kind == "community":
            community.append((m, base))
    kept, copies = _split_undeclared_copies(originals, served)
    for m in kept:
        records.append(_record([], m, m.id, now))
    for m in originals:
        if m.id in copies:
            records.append(_record([], m, m.id, now, community=True, base_model=copies[m.id]))
    for m, base in community:
        records.append(_record([], m, m.id, now, community=True, base_model=base))

    _add_architectures(catalog, records, say)
    return CatalogSnapshot(records=records, openrouter_date=_iso(or_result.fetched_at),
                           hf_date=_iso(max(dates, default=None)), built_at=now.isoformat(), warnings=warnings)


def hf_records(models: list[HFModel], now: datetime | None = None) -> list[ModelRecord]:
    """Records for Hugging Face search results outside the snapshot (live search by name)."""
    now = now or datetime.now(timezone.utc)
    out = []
    for m in models:
        if m.pipeline_tag not in HF_CAPABILITIES:
            continue
        kind, base = classify(m)
        if kind:
            out.append(_record([], m, m.id, now, community=kind == "community", base_model=base))
    return out
