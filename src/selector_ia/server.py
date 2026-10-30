"""Local web server for the search page: serves web/ and a small JSON API, only on 127.0.0.1.

The catalog snapshot is kept in memory and in the cache (blob "finder_snapshot") so the page opens
instantly; it is rebuilt in the background when older than `refresh_hours` (at start and while open).
Requests from other sites are refused (Host and Origin checks), so a web page cannot drive this API.
"""

import argparse
import json
import logging
import mimetypes
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from selector_ia import credentials
from selector_ia.compare import prompts as try_prompts
from selector_ia.compare.tester import MAX_MODELS, ask_one, check_ids
from selector_ia.data.cache import Cache
from selector_ia.data.catalog import build_catalog
from selector_ia.data.huggingface import HuggingFaceClient
from selector_ia.data.openrouter import OpenRouterClient
from selector_ia.errors import SelectorError
from selector_ia.finder.catalog import SNAPSHOT_VERSION, CatalogSnapshot, build_snapshot, hf_records
from selector_ia.finder.card import read_card
from selector_ia.finder.equipment import TEMPLATES, EquipmentForm, profile_from_form, profile_json
from selector_ia.finder.fit import AGENTS, CONTEXTS, Usage, estimate_all
from selector_ia.hardware.profile import GIB
from selector_ia.hardware.check import check_model
from selector_ia.hardware.fit import AGENTIC_OK
from selector_ia.hardware.local import detect_local

log = logging.getLogger(__name__)

APP = "selector"                 # also the host name: selector.localhost
BASE_PORT = 8790
PORT_TRIES = 15                  # the launcher looks for the server in BASE_PORT … BASE_PORT + 14
SNAPSHOT_BLOB = "finder_snapshot"
WATCH_SECONDS = 60               # how often the background thread checks the catalog's age
RETRY_SECONDS = 30 * 60          # after a failed update (e.g. offline), try again this often
MAX_BODY = 64 * 1024
HF_LIVE_LIMIT = 40               # live search by name in all of Hugging Face
WEB_DIR = Path(str(resources.files("selector_ia") / "web"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class State:
    """Catalog snapshot + build status, shared by request threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot: CatalogSnapshot | None = None
        self.building = False
        self.progress = ""
        self.error: str | None = None
        self.fit_cache: dict[tuple[str, str], dict] = {}
        self.last_attempt = 0.0

    # --- snapshot ---

    def load_saved(self) -> None:
        cache = Cache()
        try:
            saved = cache.load_blob(SNAPSHOT_BLOB)
        finally:
            cache.close()
        if saved and saved[0].get("version") == SNAPSHOT_VERSION:
            try:
                self.snapshot = CatalogSnapshot.model_validate(saved[0])
            except ValidationError:
                log.warning("Saved snapshot has an old format: rebuilding")

    def age_hours(self) -> float | None:
        if not self.snapshot:
            return None
        built = datetime.fromisoformat(self.snapshot.built_at)
        return (_now() - built).total_seconds() / 3600

    def stale(self) -> bool:
        cache = Cache()
        try:
            hours = cache.refresh_hours()
        finally:
            cache.close()
        age = self.age_hours()
        return age is None or age >= hours

    def rebuild(self, force: bool = False) -> bool:
        """Starts a background rebuild unless one is running. `force`: download everything again."""
        with self.lock:
            if self.building:
                return False
            self.building, self.error, self.progress = True, None, "Empezando…"
            self.last_attempt = time.monotonic()
        threading.Thread(target=self._build, args=(force,), daemon=True).start()
        return True

    def _build(self, force: bool) -> None:
        catalog = build_catalog()
        try:
            if force:
                catalog.refresh_all()
            snapshot = build_snapshot(catalog, progress=self._set_progress)
            catalog.cache.save_blob(SNAPSHOT_BLOB, snapshot.model_dump(mode="json", exclude_none=True))
            with self.lock:
                self.snapshot, self.fit_cache = snapshot, {}
        except SelectorError as exc:
            self.error = exc.user_message
        except Exception as exc:  # keep serving the old snapshot
            log.exception("Catalog build failed")
            self.error = f"Error inesperado al actualizar el catálogo: {exc}"
        finally:
            catalog.cache.close()
            with self.lock:
                self.building, self.progress = False, ""

    def _set_progress(self, text: str) -> None:
        self.progress = text

    def watch(self) -> None:
        """Background thread: keeps the catalog fresh while the app is open."""
        while True:
            try:
                retry_due = not self.error or time.monotonic() - self.last_attempt >= RETRY_SECONDS
                if self.stale() and retry_due:
                    self.rebuild()
            except Exception:  # never let the watcher die
                log.exception("Watcher failed")
            time.sleep(WATCH_SECONDS)

    # --- derived data ---

    def fit(self, profile_name: str, use: Usage | None = None) -> dict:
        snapshot = self.snapshot
        if not snapshot:
            return {}
        use = use or Usage()
        key = (profile_name, snapshot.built_at, use.model_dump_json())
        if key not in self.fit_cache:
            cache = Cache()
            try:
                profile = cache.load_profile(profile_name)
            finally:
                cache.close()
            if profile is None:
                raise SelectorError(f"No existe el equipo «{profile_name}».")
            self.fit_cache[key] = estimate_all(snapshot.records, profile, use)
        return self.fit_cache[key]

    def openrouter_ids(self) -> dict[str, bool]:
        """Every OpenRouter id in the snapshot (plain and ':free') -> supports reasoning."""
        ids = {}
        for r in self.snapshot.records if self.snapshot else []:
            if r.openrouter:
                ids[r.openrouter.id] = r.openrouter.reasoning
                if r.openrouter.free_id:
                    ids[r.openrouter.free_id] = r.openrouter.reasoning
        return ids


STATE = State()


def usage_from(params: dict) -> Usage:
    """?ctx=32768&agents=4&kvq=1&agentic=1 (only the offered values are accepted)."""
    def pick(name, allowed, default):
        try:
            value = int(params.get(name, default))
        except ValueError:
            return default
        return value if value in allowed else default
    return Usage(context=pick("ctx", CONTEXTS, 8192), agents=pick("agents", AGENTS, 1),
                 kv_q8=params.get("kvq") == "1", agentic=params.get("agentic") == "1")


def app_version() -> dict:
    """Installed commit (written by the installer) or, in development, the repo's HEAD."""
    stamp = WEB_DIR.parent / "VERSION.json"
    if stamp.exists():
        try:
            return json.loads(stamp.read_text())
        except ValueError:
            pass
    try:
        out = subprocess.run(["git", "-C", str(WEB_DIR), "log", "-1", "--format=%h|%cI"], capture_output=True,
                             text=True, timeout=5)
        commit, date = out.stdout.strip().split("|")
        return {"commit": commit, "date": date, "source": "desarrollo"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"commit": None, "date": None, "source": "desconocida"}


# --- request handling ---

class Handler(BaseHTTPRequestHandler):
    server_version = "SelectorModelosIA"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: only errors go to the log
        log.debug(fmt, *args)

    # security

    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}", f"{APP}.localhost:{port}"}

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self._allowed_hosts()

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc in self._allowed_hosts()

    # responses

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise SelectorError("Petición no válida.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise SelectorError("Petición demasiado grande.")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise SelectorError("Petición no válida.")
        return data

    # routing

    def do_GET(self):
        if not self._host_ok():
            return self._error("Host no permitido.", 403)
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        routes = {
            "/api/ocupado": self.get_busy, "/api/status": self.get_status, "/api/catalog": self.get_catalog,
            "/api/profiles": self.get_profiles, "/api/fit": self.get_fit, "/api/check": self.get_check,
            "/api/settings": self.get_settings, "/api/version": self.get_version,
            "/api/hfsearch": self.get_hf_search, "/api/card": self.get_card, "/api/prompts": self.get_prompts,
        }
        try:
            if url.path in routes:
                return routes[url.path](params)
            return self.static(url.path)
        except SelectorError as exc:
            return self._error(exc.user_message)
        except Exception as exc:
            log.exception("GET %s failed", url.path)
            return self._error(f"Error inesperado: {exc}", 500)

    def do_POST(self):
        if not self._host_ok() or not self._origin_ok():
            return self._error("Origen no permitido.", 403)
        routes = {
            "/api/profiles": self.post_profile, "/api/profiles/detect": self.post_detect,
            "/api/profiles/delete": self.post_delete_profile, "/api/settings": self.post_settings,
            "/api/refresh": self.post_refresh, "/api/probar": self.post_try,
            "/api/prompts": self.post_prompt, "/api/prompts/delete": self.post_delete_prompt,
            "/api/prompts/reset": self.post_reset_prompts,
        }
        path = urlparse(self.path).path
        if path not in routes:
            return self._error("No encontrado.", 404)
        try:
            return routes[path](self._body())
        except (SelectorError, ValidationError, ValueError) as exc:
            message = exc.user_message if isinstance(exc, SelectorError) else "Datos no válidos."
            return self._error(message)
        except Exception as exc:
            log.exception("POST %s failed", path)
            return self._error(f"Error inesperado: {exc}", 500)

    # static files

    def static(self, path: str) -> None:
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB_DIR / name).resolve()
        if WEB_DIR.resolve() not in target.parents or not target.is_file() or target.name == "package.json":
            return self._error("No encontrado.", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            ctype = "text/javascript"
        self._send(200, target.read_bytes(), f"{ctype}; charset=utf-8" if ctype.startswith("text") else ctype)

    # --- GET ---

    def get_busy(self, _params):
        self._json({"app": APP, "ocupado": STATE.building,
                    "motivo": "actualizando el catálogo" if STATE.building else ""})

    def get_status(self, _params):
        snap = STATE.snapshot
        cache = Cache()
        try:
            hours = cache.refresh_hours()
        finally:
            cache.close()
        self._json({"ready": snap is not None, "building": STATE.building, "progress": STATE.progress,
                    "error": STATE.error, "built_at": snap.built_at if snap else None,
                    "openrouter_date": snap.openrouter_date if snap else None,
                    "hf_date": snap.hf_date if snap else None, "refresh_hours": hours})

    def get_catalog(self, _params):
        if not STATE.snapshot:
            raise SelectorError("El catálogo todavía se está preparando.")
        self._json(STATE.snapshot.model_dump(mode="json", exclude_none=True))

    def get_profiles(self, _params):
        cache = Cache()
        try:
            profiles = cache.list_profiles()
        finally:
            cache.close()
        self._json({"profiles": [profile_json(p) for p in profiles],
                    "templates": [t.model_dump() for t in TEMPLATES]})

    def get_fit(self, params):
        self._json({"fit": STATE.fit(params.get("profile", ""), usage_from(params))})

    def get_check(self, params):
        model_id, profile_name = params.get("id", ""), params.get("profile", "")
        cache = Cache()
        try:
            profile = cache.load_profile(profile_name)
        finally:
            cache.close()
        if profile is None:
            raise SelectorError(f"No existe el equipo «{profile_name}».")
        use = usage_from(params)
        catalog = build_catalog()
        try:
            check = check_model(catalog, model_id, profile, use.context, use.agents, use.kv_q8, use.agentic)
        finally:
            catalog.cache.close()
        fit = check.fit
        rec = fit.recommended or fit.below_floor

        def size(n):
            return round(n / GIB, 1) if n else None

        mem = rec.memory if rec else None
        self._json({
            "verdict": ("limited" if fit.below_floor else rec.verdict.value) if rec
            else ("no" if fit.nothing_fits else "unknown"),
            "reason": rec.reason if rec else (fit.hardware_needed or "No se puede calcular."),
            "variant": rec.variant.label if rec else None,
            "below_floor": bool(fit.below_floor),
            "variant_url": rec.variant.url if rec else None,
            "memory_source": mem.weights_source if mem else None,
            "formula": mem.formula if mem else None,
            "breakdown": {"weights_gb": size(mem.weights_bytes), "kv_gb": size(mem.kv.bytes if mem.kv else None),
                          "margin_gb": size(mem.margin_bytes), "total_gb": size(mem.total_bytes),
                          "kv_formula": mem.kv.formula if mem.kv else None} if mem else None,
            "variants": [{"label": r.variant.label, "format": r.variant.format, "repo": r.variant.repo_id,
                          "size_gb": size(r.memory.weights_bytes), "need_gb": size(r.memory.total_bytes),
                          "verdict": r.verdict.value, "estimated": r.variant.estimated,
                          "agentic_ok": r.variant.label in AGENTIC_OK,
                          "chosen": bool(rec) and r.variant.label == rec.variant.label and
                          r.variant.repo_id == rec.variant.repo_id}
                         for r in sorted(fit.results, key=lambda r: r.memory.weights_bytes or 0)],
            "context": check.context_tokens,
            "runners": [r.model_dump() for r in check.runners],
            "notes": check.notes,
        })

    def get_card(self, params):
        """Evaluation facts quoted from a model's card (uncensored / community versions)."""
        model_id = params.get("id", "")
        if "/" not in model_id:
            raise SelectorError("Modelo no válido.")
        catalog = build_catalog()
        try:
            readme = catalog.readme(model_id)
        finally:
            catalog.cache.close()
        facts = read_card(model_id, readme)
        self._json(facts.model_dump() | {"documented": facts.documented, "found": readme is not None})

    def get_hf_search(self, params):
        """Models of all Hugging Face matching a name, beyond the snapshot (e.g. «abliterated», «dolphin»)."""
        text = params.get("q", "").strip()
        if len(text) < 3:
            raise SelectorError("Escribe al menos 3 letras para buscar en Hugging Face.")
        catalog = build_catalog()
        try:
            found = catalog.search_hf(text=text, sort="downloads", limit=HF_LIVE_LIMIT)
            profile = catalog.cache.load_profile(params["profile"]) if params.get("profile") else None
        finally:
            catalog.cache.close()
        known = {r.id.lower() for r in STATE.snapshot.records} if STATE.snapshot else set()
        records = [r for r in hf_records(found.items) if r.id.lower() not in known]
        self._json({"records": [r.model_dump(mode="json", exclude_none=True) for r in records],
                    "fit": estimate_all(records, profile, usage_from(params)) if profile else {}})

    def get_settings(self, _params):
        cache = Cache()
        try:
            hours = cache.refresh_hours()
        finally:
            cache.close()
        self._json({"openrouter_key": bool(credentials.get_secret(credentials.OPENROUTER_KEY)),
                    "hf_token": bool(credentials.get_secret(credentials.HF_TOKEN)), "refresh_hours": hours,
                    "version": app_version()})

    def get_version(self, _params):
        self._json(app_version())

    def get_prompts(self, _params, saved: str | None = None):
        cache = Cache()
        try:
            prompts, custom = try_prompts.load(cache)
        finally:
            cache.close()
        self._json({"prompts": [p.model_dump() for p in prompts], "custom": custom, "saved": saved})

    # --- POST ---

    def post_profile(self, body):
        form = EquipmentForm.model_validate(body)
        profile = profile_from_form(form)
        cache = Cache()
        try:
            original = body.get("original_name")
            if original and original != profile.name:
                cache.delete_profile(original)
            elif not original and cache.load_profile(profile.name):
                raise SelectorError(f"Ya hay un equipo llamado «{profile.name}». Elige otro nombre.")
            cache.save_profile(profile)
            profiles = cache.list_profiles()
        finally:
            cache.close()
        STATE.fit_cache.clear()
        self._json({"profiles": [profile_json(p) for p in profiles], "saved": profile.name})

    def post_detect(self, _body):
        profile = detect_local("Este ordenador")
        profile.machine = "pc"
        cache = Cache()
        try:
            cache.save_profile(profile)
            profiles = cache.list_profiles()
        finally:
            cache.close()
        STATE.fit_cache.clear()
        self._json({"profiles": [profile_json(p) for p in profiles], "saved": profile.name})

    def post_delete_profile(self, body):
        cache = Cache()
        try:
            cache.delete_profile(str(body.get("name", "")))
            profiles = cache.list_profiles()
        finally:
            cache.close()
        self._json({"profiles": [profile_json(p) for p in profiles]})

    def post_settings(self, body):
        messages = []
        if body.get("openrouter_key"):
            key = str(body["openrouter_key"]).strip()
            OpenRouterClient(api_key=key).key_info()            # raises InvalidKeyError if wrong
            credentials.set_secret(credentials.OPENROUTER_KEY, key)
            messages.append("Clave de OpenRouter comprobada y guardada.")
        if body.get("hf_token"):
            token = str(body["hf_token"]).strip()
            if not HuggingFaceClient(token=token).token_is_valid():
                raise SelectorError("El token de Hugging Face no es válido.")
            credentials.set_secret(credentials.HF_TOKEN, token)
            messages.append("Token de Hugging Face guardado.")
        for name in body.get("delete") or []:
            if name in (credentials.OPENROUTER_KEY, credentials.HF_TOKEN):
                credentials.delete_secret(name)
                messages.append("Clave borrada.")
        if body.get("refresh_hours") is not None:
            hours = float(body["refresh_hours"])
            if not 1 <= hours <= 720:
                raise SelectorError("Las horas deben estar entre 1 y 720.")
            cache = Cache()
            try:
                cache.set_setting("refresh_hours", str(hours))
            finally:
                cache.close()
            messages.append(f"El catálogo se actualizará cada {hours:g} h.")
        self._json({"message": " ".join(messages) or "Sin cambios."})

    def post_refresh(self, _body):
        started = STATE.rebuild(force=True)
        self._json({"started": started, "message": "Actualizando el catálogo…" if started else "Ya se está actualizando."})

    def post_try(self, body):
        model, prompt = str(body.get("model", "")), str(body.get("prompt", "")).strip()
        if not prompt:
            raise SelectorError("Escribe el mensaje de prueba.")
        if len(prompt) > 8000:
            raise SelectorError("El mensaje es demasiado largo para una prueba.")
        known = STATE.openrouter_ids()
        check_ids([model], known)
        key = credentials.get_secret(credentials.OPENROUTER_KEY)
        if not key:
            raise SelectorError("Falta tu clave de OpenRouter: añádela en Ajustes para poder probar.")
        answer = ask_one(OpenRouterClient(api_key=key), model, prompt, reasoning=known[model])
        self._json(answer.model_dump())

    def post_prompt(self, body):
        cache = Cache()
        try:
            saved = try_prompts.save(cache, body)
        finally:
            cache.close()
        self.get_prompts({}, saved)

    def post_delete_prompt(self, body):
        cache = Cache()
        try:
            try_prompts.delete(cache, str(body.get("id", "")))
        finally:
            cache.close()
        self.get_prompts({})

    def post_reset_prompts(self, _body):
        cache = Cache()
        try:
            try_prompts.reset(cache)
        finally:
            cache.close()
        self.get_prompts({})


# --- start ---

def bind(port: int | None) -> ThreadingHTTPServer:
    ports = [port] if port else range(BASE_PORT, BASE_PORT + PORT_TRIES)
    last = None
    for p in ports:
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Handler)
        except OSError as exc:
            last = exc
    raise SystemExit(f"No hay ningún puerto libre entre {BASE_PORT} y {BASE_PORT + PORT_TRIES - 1}: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Servidor local del Selector de Modelos IA")
    parser.add_argument("--port", type=int, default=None, help=f"puerto (por defecto el primero libre desde {BASE_PORT})")
    parser.add_argument("--smoke-test", action="store_true", help="arranca, responde a /api/ocupado y sale")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    server = bind(args.port)
    port = server.server_address[1]
    if args.smoke_test:
        import urllib.request
        threading.Thread(target=server.serve_forever, daemon=True).start()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ocupado", headers={"Host": f"127.0.0.1:{port}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            ok = json.loads(res.read()).get("app") == APP
        server.shutdown()
        print("El servidor arranca correctamente." if ok else "El servidor no responde bien.")
        return 0 if ok else 1
    STATE.load_saved()
    threading.Thread(target=STATE.watch, daemon=True).start()
    print(f"Selector de Modelos IA en http://{APP}.localhost:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
