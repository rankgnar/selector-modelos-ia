"""OpenRouter client: model catalog and key status."""

import json
import logging
import re
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, ValidationError

from selector_ia.data.http import make_client, request
from selector_ia.data.models import Benchmark, OpenRouterModel
from selector_ia.errors import ApiError, InvalidKeyError, SelectorError

log = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
SERVICE = "OpenRouter"
CHAT_TIMEOUT_SECONDS = 300.0  # reasoning models can think for a while


class ChatResult(BaseModel):
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None   # usage.cost as returned by OpenRouter

    def json(self) -> dict:
        """Parses the reply as JSON (tolerates ```json fences)."""
        text = self.content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ApiError("El modelo no ha devuelto una respuesta con el formato esperado.", detail=self.content[:500])
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ApiError("El modelo ha devuelto un JSON no válido.", detail=self.content[:500]) from exc


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None):
        self.api_key = api_key
        self.client = client or make_client()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def fetch_models_raw(self) -> list[dict]:
        """The public catalog; no key needed. Without `output_modalities=all` OpenRouter only lists models that
        answer with text (no video, embeddings, speech, transcription or rerank models)."""
        response = request(self.client, "GET", f"{BASE_URL}/models", service=SERVICE,
                           params={"output_modalities": "all"})
        return response.json().get("data", [])

    def fetch_video_pricing(self) -> dict[str, dict[str, str]]:
        """Video models are priced per second of output (cents, per resolution) in /videos/models."""
        response = request(self.client, "GET", f"{BASE_URL}/videos/models", service=SERVICE)
        return {m["id"]: {k: str(v) for k, v in (m.get("pricing_skus") or {}).items()}
                for m in response.json().get("data", []) if isinstance(m, dict) and m.get("id")}

    def fetch_models(self) -> list[OpenRouterModel]:
        fetched_at = datetime.now(timezone.utc)
        models = []
        for raw in self.fetch_models_raw():
            model = parse_model(raw, fetched_at)
            if model is not None:
                models.append(model)
        if any("video" in m.output_modalities for m in models):
            try:
                skus = self.fetch_video_pricing()
            except SelectorError as exc:   # the catalog is still useful without video prices ("sin datos")
                log.warning("Video pricing unavailable: %s", exc)
                skus = {}
            for m in models:
                m.video_pricing = skus.get(m.id, {})
        return models

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        schema_name: str = "answer",
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        if not self.api_key:
            raise InvalidKeyError("Falta la clave de OpenRouter. Guárdala en Ajustes (o con: selector-ia clave).")
        body: dict = {"model": model, "messages": messages, "usage": {"include": True}}
        if json_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": schema_name, "strict": True, "schema": json_schema}}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        response = request(self.client, "POST", f"{BASE_URL}/chat/completions", service=SERVICE,
                           headers={**self._headers(), "X-Title": "Selector de Modelos IA"},
                           json=body, timeout=CHAT_TIMEOUT_SECONDS)
        data = response.json()
        if data.get("error"):
            error = data["error"]
            raise ApiError(f"OpenRouter ha devuelto un error con el modelo {model}: {error.get('message', error)}",
                           detail=str(error))
        choices = data.get("choices") or []
        if not choices:
            raise ApiError(f"El modelo {model} no ha devuelto respuesta.", detail=str(data)[:500])
        usage = data.get("usage") or {}
        return ChatResult(
            content=(choices[0].get("message") or {}).get("content") or "",
            model=data.get("model") or model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
        )

    def credits(self) -> dict:
        """total_credits / total_usage in USD as published (GET /credits)."""
        response = request(self.client, "GET", f"{BASE_URL}/credits", service=SERVICE, headers=self._headers())
        return response.json().get("data", {})

    def key_info(self) -> dict:
        """Validates the key and returns usage/limit as published (GET /key)."""
        response = request(self.client, "GET", f"{BASE_URL}/key", service=SERVICE, headers=self._headers())
        return response.json().get("data", {})


def parse_model(raw: dict, fetched_at: datetime) -> OpenRouterModel | None:
    """Normalize one catalog entry. Returns None (and logs) if the entry is malformed."""
    try:
        arch = raw.get("architecture") or {}
        top = raw.get("top_provider") or {}
        alias = raw.get("alias_target") or {}
        created = raw.get("created")
        return OpenRouterModel(
            id=raw["id"],
            name=raw.get("name") or raw["id"],
            canonical_slug=raw.get("canonical_slug"),
            hugging_face_id=raw.get("hugging_face_id") or None,
            created=datetime.fromtimestamp(created, timezone.utc) if created else None,
            description=raw.get("description"),
            context_length=raw.get("context_length") or top.get("context_length"),
            max_completion_tokens=top.get("max_completion_tokens"),
            input_modalities=arch.get("input_modalities") or [],
            output_modalities=arch.get("output_modalities") or [],
            pricing={k: str(v) for k, v in (raw.get("pricing") or {}).items() if isinstance(v, (str, int, float))},
            supported_parameters=raw.get("supported_parameters") or [],
            expiration_date=raw.get("expiration_date"),
            knowledge_cutoff=raw.get("knowledge_cutoff"),
            alias_target=alias.get("slug") if isinstance(alias, dict) else None,
            supports_reasoning=bool(raw.get("reasoning")) or "reasoning" in (raw.get("supported_parameters") or []),
            is_moderated=top.get("is_moderated"),
            benchmarks=parse_benchmarks(raw.get("benchmarks") or {}),
            fetched_at=fetched_at,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        log.warning("Skipping malformed OpenRouter entry %r: %s", raw.get("id"), exc)
        return None


def parse_benchmarks(raw: dict) -> list[Benchmark]:
    """OpenRouter embeds Artificial Analysis indexes and Design Arena results. No own date is published."""
    results = []
    for name, value in (raw.get("artificial_analysis") or {}).items():
        if isinstance(value, (int, float)):
            results.append(Benchmark(source="Artificial Analysis (vía OpenRouter)", name=name, value=value,
                                     url="https://artificialanalysis.ai"))
    for entry in raw.get("design_arena") or []:
        if isinstance(entry, dict) and isinstance(entry.get("elo"), (int, float)):
            results.append(Benchmark(
                source="Design Arena (vía OpenRouter)",
                name=f"{entry.get('arena', 'models')}/{entry.get('category', '?')} elo",
                value=entry["elo"],
                url="https://www.designarena.ai",
                notes=f"rank {entry.get('rank')}, win rate {entry.get('win_rate')}%",
            ))
    return results
