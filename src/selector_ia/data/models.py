"""Normalized catalog records. Values are copied from the source as-is; missing ones stay None."""

import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

OPENROUTER_SOURCE = "OpenRouter /api/v1/models"
HF_SOURCE = "Hugging Face Hub API"


class Benchmark(BaseModel):
    source: str              # e.g. "Artificial Analysis (vía OpenRouter)"
    name: str                # e.g. "intelligence_index" or a dataset id
    value: float
    date: str | None = None  # date of the result if the source gives one
    url: str | None = None
    verified: bool | None = None
    notes: str | None = None
    dataset: str | None = None  # HF dataset id of the benchmark, when the source gives one


class OpenRouterModel(BaseModel):
    id: str
    name: str
    canonical_slug: str | None = None
    hugging_face_id: str | None = None
    created: datetime | None = None
    description: str | None = None
    context_length: int | None = None
    max_completion_tokens: int | None = None
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    pricing: dict[str, str] = Field(default_factory=dict)  # USD per token, strings as published
    video_pricing: dict[str, str] = Field(default_factory=dict)  # /videos/models pricing_skus (cents), as published
    supported_parameters: list[str] = Field(default_factory=list)
    expiration_date: date | None = None
    knowledge_cutoff: str | None = None
    alias_target: str | None = None
    supports_reasoning: bool = False
    is_moderated: bool | None = None   # top_provider.is_moderated: the provider filters content itself
    benchmarks: list[Benchmark] = Field(default_factory=list)
    fetched_at: datetime

    @property
    def url(self) -> str:
        return f"https://openrouter.ai/{self.id}"

    @property
    def is_open_weights(self) -> bool:
        return bool(self.hugging_face_id)

    @property
    def is_alias(self) -> bool:
        return self.alias_target is not None or self.id.startswith("~")

    @property
    def is_router(self) -> bool:
        """Meta-models (auto routers) publish price -1: not a concrete model."""
        return self.pricing.get("prompt") == "-1" or self.pricing.get("completion") == "-1"

    @property
    def is_free(self) -> bool:
        """':free' variants, or chat models priced 0/0. Video, speech, embedding, rerank… models often publish
        0/0 because they are billed another way: for them 0/0 means "no price here", not free."""
        if self.id.endswith(":free"):
            return True
        chat = "text" in self.output_modalities
        return chat and self.price_per_million("prompt") == 0 and self.price_per_million("completion") == 0

    def is_retired(self, today: date | None = None) -> bool:
        today = today or datetime.now(timezone.utc).date()
        return self.expiration_date is not None and self.expiration_date <= today

    def price_per_million(self, kind: str) -> Decimal | None:
        """USD per 1M tokens (exact arithmetic on the published string). None = sin datos."""
        raw = self.pricing.get(kind)
        if raw is None:
            return None
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return None
        if value < 0:
            return None
        return value * 1_000_000


class GGUFInfo(BaseModel):
    total_params: int | None = None
    architecture: str | None = None
    context_length: int | None = None
    total_file_size: int | None = None


class InferenceProvider(BaseModel):
    """A third-party provider serving the model (HF inferenceProviderMapping), e.g. fal-ai, replicate."""
    provider: str
    status: str | None = None     # "live" | "staging" | "error"
    task: str | None = None
    provider_id: str | None = None


class HFModel(BaseModel):
    id: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    license: str | None = None
    languages: list[str] = Field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    last_modified: datetime | None = None
    created_at: datetime | None = None
    params_total: int | None = None       # from safetensors metadata, or GGUF header
    params_source: str | None = None      # "safetensors" | "gguf"
    gguf: GGUFInfo | None = None
    base_models: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    gated: bool | str | None = None
    eval_results: list[Benchmark] = Field(default_factory=list)
    trending_score: int | None = None
    inference_providers: list[InferenceProvider] = Field(default_factory=list)
    fetched_at: datetime

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.id}"

    @property
    def quant_format(self) -> str | None:
        return quant_format_from_tags(self.tags) or quant_format_from_name(self.id)


# Order matters: first match wins.
_QUANT_TAGS = [
    ("gguf", "GGUF"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("exl2", "EXL2"), ("exl3", "EXL3"),
    ("mlx", "MLX"), ("fp8", "FP8"), ("nvfp4", "NVFP4"), ("bitsandbytes", "bitsandbytes"),
]


_QUANT_NAME = re.compile(r"[-_.](GGUF|AWQ|GPTQ|EXL2|EXL3|MLX|FP8|NVFP4|MXFP4|INT4|INT8|W4A16|W8A8|[48]bit|bnb-4bit)$",
                         re.IGNORECASE)


def quant_format_from_name(model_id: str) -> str | None:
    """Repos named like 'Model-NVFP4' / 'Model-4bit' are quantized copies even when untagged."""
    match = _QUANT_NAME.search(model_id)
    if not match:
        return None
    suffix = match.group(1).upper()
    return {"4BIT": "4-bit", "8BIT": "8-bit", "BNB-4BIT": "bitsandbytes"}.get(suffix, suffix)


def quant_format_from_tags(tags: list[str]) -> str | None:
    lowered = {t.lower() for t in tags}
    for tag, label in _QUANT_TAGS:
        if tag in lowered:
            return label
    return None


class RepoFile(BaseModel):
    path: str
    size: int | None = None
