"""Local web server: real HTTP on a free port, fake catalog, no network."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from selector_ia import credentials, server
from selector_ia.compare.tester import CompareAnswer
from selector_ia.finder.catalog import CatalogSnapshot, ModelRecord, OpenRouterOffer


def snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(built_at="2099-01-01T00:00:00+00:00", openrouter_date="2026-10-30", records=[
        ModelRecord(id="v/chat", name="Chat", author="v", capabilities=["chat", "code"],
                    openrouter=OpenRouterOffer(id="v/chat", url="https://openrouter.ai/v/chat", free_id="v/chat:free",
                                               reasoning=True, prices={"prompt": 1.0, "completion": 2.0})),
        ModelRecord(id="o/small", name="Small", author="o", capabilities=["chat"], hf_id="o/small", params=1_000_000_000,
                    max_context=32768, arch={"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": 128}),
        ModelRecord(id="o/huge", name="Huge", author="o", capabilities=["chat"], hf_id="o/huge", params=700_000_000_000),
    ])


@pytest.fixture
def app(monkeypatch):
    state = server.State()
    state.snapshot = snapshot()
    monkeypatch.setattr(server, "STATE", state)
    httpd = server.bind(None)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    yield f"127.0.0.1:{port}", state
    httpd.shutdown()
    httpd.server_close()


def call(host, path, body=None, *, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://{host}{path}", data=data, method="POST" if body is not None else "GET",
                                 headers={"Host": host, **({"Content-Type": "application/json"} if data else {}),
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            raw = res.read()
            return res.status, (json.loads(raw) if res.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_launcher_endpoint_and_status(app):
    host, _ = app
    assert call(host, "/api/ocupado") == (200, {"app": "selector", "ocupado": False, "motivo": ""})
    status, data = call(host, "/api/status")
    assert data["ready"] and data["refresh_hours"] == 6


def test_catalog_and_static_files(app):
    host, _ = app
    status, data = call(host, "/api/catalog")
    assert status == 200 and [r["id"] for r in data["records"]] == ["v/chat", "o/small", "o/huge"]
    status, html = call(host, "/")
    assert status == 200 and b"Selector de Modelos IA" in html
    assert call(host, "/../pyproject.toml")[0] == 404
    assert call(host, "/package.json")[0] == 404


def test_other_hosts_and_origins_are_refused(app):
    host, _ = app
    assert call(host, "/api/status", headers={"Host": "evil.example"})[0] == 403
    assert call(host, "/api/refresh", {}, headers={"Origin": "https://evil.example"})[0] == 403
    port = host.split(":")[1]
    assert call(host, "/api/status", headers={"Host": f"selector.localhost:{port}"})[0] == 200


def test_equipment_from_template_is_saved_and_fit_is_estimated(app):
    host, _ = app
    status, data = call(host, "/api/profiles")
    vps = next(t for t in data["templates"] if t["id"] == "vps-8")
    form = {k: vps[k] for k in ("machine", "ram_gb", "gpu", "vram_gb", "gpu_count")} | {"name": "Mi VPS"}
    status, data = call(host, "/api/profiles", form)
    assert status == 200 and data["profiles"][0]["name"] == "Mi VPS" and data["profiles"][0]["machine"] == "vps"
    assert call(host, "/api/profiles", form)[1]["error"].startswith("Ya hay un equipo")
    status, data = call(host, "/api/fit?profile=Mi%20VPS")
    fit = data["fit"]
    assert fit["o/small"]["verdict"] == "ok" and fit["o/huge"]["verdict"] == "no" and "v/chat" not in fit
    assert fit["o/small"]["estimate"] is True


def test_equipment_edit_rename_and_delete(app):
    host, _ = app
    call(host, "/api/profiles", {"name": "GPU", "machine": "gpu_server", "ram_gb": 64, "gpu": "NVIDIA L40S", "vram_gb": 48})
    status, data = call(host, "/api/profiles", {"name": "GPU alquilada", "machine": "gpu_server", "ram_gb": 64,
                                               "gpu": "NVIDIA L40S", "vram_gb": 48, "original_name": "GPU"})
    assert [p["name"] for p in data["profiles"]] == ["GPU alquilada"] and data["profiles"][0]["vram_gb"] == 48
    status, data = call(host, "/api/profiles/delete", {"name": "GPU alquilada"})
    assert data["profiles"] == []
    assert "error" in call(host, "/api/profiles", {"name": "", "ram_gb": 8})[1]


def test_settings_never_return_the_key(app, monkeypatch):
    host, _ = app
    monkeypatch.setattr(server.OpenRouterClient, "key_info", lambda self: {"label": "x"})
    status, data = call(host, "/api/settings", {"openrouter_key": "sk-or-secret", "refresh_hours": 12})
    assert status == 200 and "guardada" in data["message"]
    status, data = call(host, "/api/settings")
    assert data["openrouter_key"] is True and data["refresh_hours"] == 12 and "sk-or-secret" not in json.dumps(data)
    assert credentials.get_secret(credentials.OPENROUTER_KEY) == "sk-or-secret"


def test_try_needs_a_key_and_a_known_model(app, monkeypatch):
    host, _ = app
    assert "clave" in call(host, "/api/probar", {"model": "v/chat", "prompt": "hola"})[1]["error"]
    credentials.set_secret(credentials.OPENROUTER_KEY, "k")
    assert "ya no están" in call(host, "/api/probar", {"model": "x/invented", "prompt": "hola"})[1]["error"]
    seen = {}

    def fake_ask(client, model_id, prompt, reasoning=False):
        seen.update(model=model_id, reasoning=reasoning)
        return CompareAnswer(openrouter_id=model_id, content="hola", seconds=1.0, cost_usd=0.001)

    monkeypatch.setattr(server, "ask_one", fake_ask)
    status, data = call(host, "/api/probar", {"model": "v/chat:free", "prompt": "hola"})
    assert status == 200 and data["content"] == "hola" and seen == {"model": "v/chat:free", "reasoning": True}


def test_stale_catalog_is_rebuilt_in_background(app, monkeypatch):
    host, state = app
    state.snapshot = state.snapshot.model_copy(update={"built_at": "2020-01-01T00:00:00+00:00"})
    built = threading.Event()

    def fake_build(catalog, progress=None):
        progress("Consultando…")
        built.set()
        return snapshot()

    monkeypatch.setattr(server, "build_snapshot", fake_build)
    assert state.stale()
    assert state.rebuild() is True
    assert built.wait(5)
    for _ in range(50):
        if not state.building:
            break
        threading.Event().wait(0.05)
    assert not state.stale() and state.error is None
    assert server.Cache().load_blob(server.SNAPSHOT_BLOB) is not None


def test_failed_update_keeps_the_old_catalog(app, monkeypatch):
    host, state = app
    from selector_ia.errors import OfflineError

    def offline(catalog, progress=None):
        raise OfflineError("Sin conexión a internet.")

    monkeypatch.setattr(server, "build_snapshot", offline)
    state.rebuild()
    for _ in range(50):
        if not state.building:
            break
        threading.Event().wait(0.05)
    assert state.error == "Sin conexión a internet." and state.snapshot is not None
    assert call(host, "/api/status")[1]["error"] == "Sin conexión a internet."


def test_smoke_test_flag():
    assert server.main(["--smoke-test"]) == 0


def test_live_hugging_face_search_adds_models_outside_the_catalog(app, monkeypatch):
    host, _ = app
    from datetime import datetime, timezone
    from selector_ia.data.catalog import CatalogResult
    from selector_ia.data.models import HFModel
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    found = [HFModel(id="o/small", pipeline_tag="text-generation", fetched_at=now),          # already in the catalog
             HFModel(id="fan/Small-abliterated", pipeline_tag="text-generation", params_total=1_000_000_000,
                     tags=["base_model:finetune:o/small"], base_models=["o/small"], fetched_at=now)]

    class FakeCatalog:
        def __init__(self):
            self.cache = server.Cache()          # created in the request thread, like build_catalog()

        def search_hf(self, **params):
            assert params["text"] == "small"
            return CatalogResult(found, now, from_cache=False)

    monkeypatch.setattr(server, "build_catalog", FakeCatalog)
    call(host, "/api/profiles", {"name": "PC", "ram_gb": 16})
    status, data = call(host, "/api/hfsearch?q=small&profile=PC")
    assert [r["id"] for r in data["records"]] == ["fan/Small-abliterated"]
    rec = data["records"][0]
    assert rec["community"] and rec["uncensored"] and data["fit"]["fan/Small-abliterated"]["verdict"] == "ok"
    assert "error" in call(host, "/api/hfsearch?q=ab")[1]


def test_snapshot_saved_by_another_version_is_ignored(monkeypatch):
    cache = server.Cache()
    cache.save_blob(server.SNAPSHOT_BLOB, {"records": [], "built_at": "2099-01-01T00:00:00+00:00"})   # no version
    cache.close()
    state = server.State()
    state.load_saved()
    assert state.snapshot is None


def test_fit_follows_context_agents_and_kv_cache(app):
    host, _ = app
    call(host, "/api/profiles", {"name": "VPS", "machine": "vps", "ram_gb": 32})
    base = call(host, "/api/fit?profile=VPS")[1]["fit"]["o/small"]
    big = call(host, "/api/fit?profile=VPS&ctx=32768&agents=4")[1]["fit"]["o/small"]
    q8 = call(host, "/api/fit?profile=VPS&ctx=32768&agents=4&kvq=1")[1]["fit"]["o/small"]
    assert base["kv_gb"] < q8["kv_gb"] < big["kv_gb"] and abs(q8["kv_gb"] * 2 - big["kv_gb"]) < 0.2
    too_long = call(host, "/api/fit?profile=VPS&ctx=131072")[1]["fit"]["o/small"]
    assert too_long["verdict"] == "no" and "32.768" in too_long["text"]          # beyond its max context
    assert call(host, "/api/fit?profile=VPS&ctx=12345")[1]["fit"]["o/small"] == base   # unknown value -> default


def test_card_quotes_the_model_card(app, monkeypatch):
    host, _ = app

    class FakeCatalog:
        def __init__(self):
            self.cache = server.Cache()

        def readme(self, repo):
            return "| **Refusals** | 6/100 | 100/100 |\nThis is the Balanced variant."

    monkeypatch.setattr(server, "build_catalog", FakeCatalog)
    data = call(host, "/api/card?id=fan/M-Uncensored")[1]
    assert data["refusals"] == ["Refusals 6/100 100/100"] and data["documented"] and data["variant"] == "balanced"
    assert "error" in call(host, "/api/card?id=nada")[1]
