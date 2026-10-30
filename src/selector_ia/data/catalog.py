"""Catalog service: serves OpenRouter and Hugging Face data from the cache or the network,
and crosses both sources via OpenRouter's `hugging_face_id`."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

from selector_ia.data.cache import Cache
from selector_ia.data.huggingface import HuggingFaceClient
from selector_ia.data.models import HFModel, OpenRouterModel, RepoFile
from selector_ia.data.openrouter import OpenRouterClient
from selector_ia.errors import OfflineError, RateLimitError, SelectorError

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CatalogResult(Generic[T]):
    items: list[T]
    fetched_at: datetime | None
    from_cache: bool
    warning: str | None = None  # set when we fell back to old cached data


class Catalog:
    def __init__(self, cache: Cache, openrouter: OpenRouterClient, hf: HuggingFaceClient):
        self.cache = cache
        self.openrouter = openrouter
        self.hf = hf

    # --- OpenRouter ---

    def openrouter_models(self, force: bool = False) -> CatalogResult[OpenRouterModel]:
        refreshed_at = self.cache.openrouter_refreshed_at()
        if not force and self.cache.is_fresh(refreshed_at):
            return CatalogResult(self.cache.load_openrouter(), refreshed_at, from_cache=True)
        try:
            models = self.openrouter.fetch_models()
        except (OfflineError, RateLimitError) as exc:
            if refreshed_at is None:
                raise
            return CatalogResult(self.cache.load_openrouter(), refreshed_at, from_cache=True,
                                 warning=f"{exc.user_message} Se usan los datos guardados del "
                                         f"{refreshed_at:%d/%m/%Y %H:%M} UTC.")
        self.cache.save_openrouter(models)
        return CatalogResult(models, self.cache.openrouter_refreshed_at(), from_cache=False)

    def openrouter_by_hf_id(self) -> dict[str, list[OpenRouterModel]]:
        """hugging_face_id (lowercase) -> OpenRouter models serving those weights."""
        mapping: dict[str, list[OpenRouterModel]] = {}
        for model in self.openrouter_models().items:
            if model.hugging_face_id:
                mapping.setdefault(model.hugging_face_id.lower(), []).append(model)
        return mapping

    # --- Hugging Face ---

    def search_hf(self, force: bool = False, **params) -> CatalogResult[HFModel]:
        """Same keyword arguments as HuggingFaceClient.search; results cached per query."""
        cached = self.cache.load_hf_query(params)
        if cached and not force and self.cache.is_fresh(cached[1]):
            return CatalogResult(cached[0], cached[1], from_cache=True)
        try:
            models = self.hf.search(**params)
        except (OfflineError, RateLimitError) as exc:
            if not cached:
                raise
            return CatalogResult(cached[0], cached[1], from_cache=True,
                                 warning=f"{exc.user_message} Se usan los datos guardados del "
                                         f"{cached[1]:%d/%m/%Y %H:%M} UTC.")
        self.cache.save_hf_query(params, models)
        return CatalogResult(models, models[0].fetched_at if models else None, from_cache=False)

    def quantized_versions(self, base_model_id: str, force: bool = False) -> CatalogResult[HFModel]:
        return self.search_hf(force=force, filters=[f"base_model:quantized:{base_model_id}"], limit=30)

    def hf_model(self, model_id: str, force: bool = False) -> HFModel:
        cached = self.cache.load_hf_model(model_id)
        if cached and not force and self.cache.is_fresh(cached.fetched_at):
            return cached
        try:
            model = self.hf.get_model(model_id)
        except (OfflineError, RateLimitError):
            if cached:
                return cached
            raise
        self.cache.save_hf_models([model])
        return model

    def repo_files(self, repo_id: str) -> list[RepoFile]:
        key = f"files:{repo_id}"
        cached = self.cache.load_blob(key)
        if cached and self.cache.is_fresh(cached[1]):
            return [RepoFile(**f) for f in cached[0]]
        try:
            files = self.hf.list_files(repo_id)
        except (OfflineError, RateLimitError):
            if cached:
                return [RepoFile(**f) for f in cached[0]]
            raise
        self.cache.save_blob(key, [f.model_dump() for f in files])
        return files

    def model_config(self, repo_id: str) -> dict | None:
        key = f"config:{repo_id}"
        cached = self.cache.load_blob(key)
        if cached and self.cache.is_fresh(cached[1]):
            return cached[0]
        try:
            config = self.hf.get_config(repo_id)
        except (OfflineError, RateLimitError):
            return cached[0] if cached else None
        self.cache.save_blob(key, config)
        return config

    def readme(self, repo_id: str) -> str | None:
        key = f"readme:{repo_id}"
        cached = self.cache.load_blob(key)
        if cached and self.cache.is_fresh(cached[1]):
            return cached[0] or None
        try:
            text = self.hf.get_readme(repo_id)
        except (OfflineError, RateLimitError):
            return cached[0] if cached else None
        self.cache.save_blob(key, text or "")
        return text

    # --- existence checks (used by the anti-invention guard) ---

    def openrouter_ids(self) -> set[str]:
        return {m.id for m in self.openrouter_models().items}

    def hf_id_known(self, model_id: str) -> bool:
        """True only if the id came back from Hugging Face (cached or confirmed live now)."""
        if self.cache.load_hf_model(model_id):
            return True
        try:
            self.hf_model(model_id)
            return True
        except SelectorError:
            return False

    # --- "Actualizar ahora" ---

    def refresh_all(self) -> CatalogResult[OpenRouterModel]:
        """Re-downloads OpenRouter now and drops cached HF queries (they refetch on next use)."""
        result = self.openrouter_models(force=True)
        if not result.warning:
            self.cache.clear_hf()
        return result


def build_catalog() -> Catalog:
    """Catalog wired to the user's cache and keyring credentials. Create one per thread."""
    from selector_ia import credentials
    return Catalog(
        Cache(),
        OpenRouterClient(api_key=credentials.get_secret(credentials.OPENROUTER_KEY)),
        HuggingFaceClient(token=credentials.get_secret(credentials.HF_TOKEN)),
    )
