"""Shared HTTP helper: retries transient failures and maps errors to Spanish messages."""

import ssl
import threading
import time

import certifi
import httpx

from selector_ia.config import HTTP_TIMEOUT_SECONDS, USER_AGENT
from selector_ia.errors import (
    ApiError,
    InsufficientCreditsError,
    InvalidKeyError,
    NotFoundError,
    OfflineError,
    RateLimitError,
)

RETRIES = 2


_ssl_lock = threading.Lock()
_ssl_context: ssl.SSLContext | None = None


def shared_ssl_context() -> ssl.SSLContext:
    """One SSL context for the whole app. Building contexts concurrently from several threads
    (UI workers) crashes the interpreter, so it is created once, under a lock."""
    global _ssl_context
    with _ssl_lock:
        if _ssl_context is None:
            _ssl_context = ssl.create_default_context(cafile=certifi.where())
        return _ssl_context


def make_client() -> httpx.Client:
    return httpx.Client(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                        verify=shared_ssl_context())


def request(client: httpx.Client, method: str, url: str, *, service: str, **kwargs) -> httpx.Response:
    """Send a request; raise a SelectorError subclass on failure."""
    last_exc: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            response = client.request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt < RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise OfflineError(
                f"No se puede conectar con {service}. Revisa tu conexión a internet.", detail=str(exc)
            ) from exc
        if response.status_code >= 500 and attempt < RETRIES:
            time.sleep(1.5 * (attempt + 1))
            continue
        _raise_for_status(response, service)
        return response
    raise OfflineError(f"No se puede conectar con {service}.", detail=str(last_exc))


def _raise_for_status(response: httpx.Response, service: str) -> None:
    code = response.status_code
    if code < 400:
        return
    detail = response.text[:500]
    if code == 401:
        raise InvalidKeyError(f"La clave de {service} no es válida o falta. Revísala en Ajustes.", detail=detail)
    if code == 402:
        raise InsufficientCreditsError(
            f"No tienes saldo suficiente en {service}. Recarga créditos para continuar.", detail=detail
        )
    if code == 404:
        message = _api_message(response).lower()
        if "data policy" in message or "guardrail" in message:
            raise NotFoundError(
                f"Tu configuración de privacidad de {service} bloquea este modelo (sus proveedores pueden usar los "
                "datos para entrenar o no cumplen tus restricciones). Puedes cambiarlo en "
                "https://openrouter.ai/settings/privacy", detail=detail)
        if "no endpoints" in message or "not a valid model" in message:
            raise NotFoundError(f"Este modelo no está disponible ahora mismo en {service}.", detail=detail)
        raise NotFoundError(f"{service} no encuentra lo que se ha pedido.", detail=detail)
    if code == 429:
        retry_after = _parse_retry_after(response)
        wait = f" Espera unos {int(retry_after)} s y vuelve a intentarlo." if retry_after else " Espera un poco."
        raise RateLimitError(f"Has alcanzado el límite de peticiones de {service}.{wait}",
                             retry_after=retry_after, detail=detail)
    raise ApiError(f"{service} ha devuelto un error ({code}). Inténtalo más tarde.", detail=detail)


def _api_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return response.text[:500]
    return str(error.get("message", "")) if isinstance(error, dict) else str(error)


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            return None
    # Hugging Face: `ratelimit: "api";r=0;t=292` (t = seconds until the window resets)
    for part in response.headers.get("ratelimit", "").split(";"):
        if part.strip().startswith("t="):
            try:
                return float(part.strip()[2:])
            except ValueError:
                return None
    return None
