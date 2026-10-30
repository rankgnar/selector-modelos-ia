"""Hugging Face Hub client (plain REST, so responses are easy to cache and test)."""

import logging
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError

from selector_ia.data.http import make_client, request
from selector_ia.data.models import Benchmark, GGUFInfo, HFModel, InferenceProvider, RepoFile
from selector_ia.errors import InvalidKeyError, NotFoundError

log = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/api"
SERVICE = "Hugging Face"

EXPAND = [
    "downloads", "likes", "lastModified", "createdAt", "safetensors", "cardData", "gguf",
    "pipeline_tag", "tags", "library_name", "gated", "evalResults", "trendingScore", "inferenceProviderMapping",
]


class HuggingFaceClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        self.token = token
        self.client = client or make_client()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get(self, path: str, params: list[tuple[str, str]] | None = None) -> httpx.Response:
        return request(self.client, "GET", f"{BASE_URL}{path}", service=SERVICE,
                       params=params, headers=self._headers())

    def search(
        self,
        *,
        pipeline_tag: str | None = None,
        languages: list[str] | None = None,
        license: str | None = None,
        filters: list[str] | None = None,
        text: str | None = None,
        inference_provider: str | None = None,
        sort: str = "downloads",
        limit: int = 50,
    ) -> list[HFModel]:
        """All `filters` (language tags, license:x, base_model:..., gguf...) must match (AND).
        `inference_provider="all"` keeps only models served by at least one third-party provider."""
        params: list[tuple[str, str]] = [("sort", sort), ("limit", str(limit))]
        if pipeline_tag:
            params.append(("pipeline_tag", pipeline_tag))
        if inference_provider:
            params.append(("inference_provider", inference_provider))
        if text:
            params.append(("search", text))
        for lang in languages or []:
            params.append(("filter", lang))
        if license:
            params.append(("filter", f"license:{license}"))
        for f in filters or []:
            params.append(("filter", f))
        params += [("expand[]", e) for e in EXPAND]
        fetched_at = datetime.now(timezone.utc)
        return [m for m in (parse_model(raw, fetched_at) for raw in self._get("/models", params).json()) if m]

    def get_model(self, model_id: str) -> HFModel:
        """Raises NotFoundError if the repo does not exist (HF answers 401 for unknown repos)."""
        try:
            raw = self._get(f"/models/{model_id}", [("expand[]", e) for e in EXPAND]).json()
        except InvalidKeyError as exc:
            if self.token and not self.token_is_valid():
                raise
            raise NotFoundError(f"El modelo «{model_id}» no existe en Hugging Face.", detail=exc.detail) from exc
        except NotFoundError as exc:
            raise NotFoundError(f"El modelo «{model_id}» no existe en Hugging Face.", detail=exc.detail) from exc
        model = parse_model(raw, datetime.now(timezone.utc))
        if model is None:
            raise NotFoundError(f"Hugging Face devolvió datos no válidos para «{model_id}».")
        return model

    def find_quantized(self, base_model_id: str, limit: int = 30) -> list[HFModel]:
        """Repos that declare themselves quantized versions of `base_model_id` (GGUF, AWQ, GPTQ...)."""
        return self.search(filters=[f"base_model:quantized:{base_model_id}"], limit=limit)

    def list_files(self, repo_id: str) -> list[RepoFile]:
        """Every file in the repo with its real size in bytes."""
        try:
            raw = self._get(f"/models/{repo_id}/tree/main", [("recursive", "true")]).json()
        except (InvalidKeyError, NotFoundError) as exc:
            raise NotFoundError(f"No se pueden listar los archivos de «{repo_id}».", detail=exc.detail) from exc
        return [RepoFile(path=f["path"], size=(f.get("lfs") or {}).get("size", f.get("size")))
                for f in raw if f.get("type") == "file"]

    def get_config(self, repo_id: str) -> dict | None:
        """The repo's config.json (layers, heads... for the KV-cache estimate). None if missing or gated."""
        try:
            response = request(self.client, "GET", f"https://huggingface.co/{repo_id}/resolve/main/config.json",
                               service=SERVICE, headers=self._headers())
            data = response.json()
        except (InvalidKeyError, NotFoundError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def task_ids(self) -> list[str]:
        """Official task ids (pipeline tags) from GET /api/tasks."""
        data = self._get("/tasks").json()
        return sorted(data) if isinstance(data, dict) else []

    def token_is_valid(self) -> bool:
        if not self.token:
            return False
        try:
            self._get("/whoami-v2")
            return True
        except InvalidKeyError:
            return False


def parse_model(raw: dict, fetched_at: datetime) -> HFModel | None:
    try:
        card = raw.get("cardData") or {}
        tags = raw.get("tags") or []
        safetensors = raw.get("safetensors") or {}
        gguf_raw = raw.get("gguf") or {}
        gguf = GGUFInfo(
            total_params=gguf_raw.get("total"),
            architecture=gguf_raw.get("architecture"),
            context_length=gguf_raw.get("context_length"),
            total_file_size=gguf_raw.get("totalFileSize"),
        ) if gguf_raw else None

        params_total, params_source = None, None
        if isinstance(safetensors.get("total"), int):
            params_total, params_source = safetensors["total"], "safetensors"
        elif gguf and gguf.total_params:
            params_total, params_source = gguf.total_params, "gguf"

        return HFModel(
            id=raw["id"],
            pipeline_tag=raw.get("pipeline_tag") or card.get("pipeline_tag"),
            library_name=raw.get("library_name") or card.get("library_name"),
            license=_license(card, tags),
            languages=_as_list(card.get("language")),
            downloads=raw.get("downloads"),
            likes=raw.get("likes"),
            last_modified=raw.get("lastModified"),
            created_at=raw.get("createdAt"),
            params_total=params_total,
            params_source=params_source,
            gguf=gguf,
            base_models=_as_list(card.get("base_model")),
            tags=tags,
            gated=raw.get("gated"),
            eval_results=_eval_results(raw.get("evalResults") or []),
            trending_score=raw.get("trendingScore") if isinstance(raw.get("trendingScore"), int) else None,
            inference_providers=_inference_providers(raw.get("inferenceProviderMapping")),
            fetched_at=fetched_at,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        log.warning("Skipping malformed HF entry %r: %s", raw.get("id"), exc)
        return None


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v is not None]


def _license(card: dict, tags: list[str]) -> str | None:
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if lic:
        return str(lic)
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _eval_results(entries: list) -> list[Benchmark]:
    """Results published in the repo's .eval_results/ files (self-reported unless verified=True)."""
    results = []
    for entry in entries:
        data = (entry or {}).get("data") or {}
        value = data.get("value")
        if not isinstance(value, (int, float)):
            continue
        dataset = data.get("dataset") or {}
        source = data.get("source") or {}
        results.append(Benchmark(
            source=source.get("name") or "Ficha del modelo en Hugging Face",
            name=dataset.get("task_id") or dataset.get("id") or "?",
            value=value,
            date=str(data["date"]) if data.get("date") else None,
            url=source.get("url"),
            verified=entry.get("verified"),
            notes=data.get("notes"),
            dataset=dataset.get("id"),
        ))
    return results


def _inference_providers(raw) -> list[InferenceProvider]:
    """The search endpoint returns a list of {provider, ...}; the model endpoint a dict keyed by provider."""
    if isinstance(raw, dict):
        raw = [{"provider": name, **(info or {})} for name, info in raw.items()]
    if not isinstance(raw, list):
        return []
    return [InferenceProvider(provider=e["provider"], status=e.get("status"), task=e.get("task"),
                              provider_id=e.get("providerId"))
            for e in raw if isinstance(e, dict) and e.get("provider")]
