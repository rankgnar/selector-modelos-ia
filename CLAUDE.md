# selector-modelos-ia

**Start every session by reading `docs/ESTADO.md`** (status, decisions, pending work; in Spanish).
Update it at the end of every significant change.

Local web app for **Linux/Ubuntu only** (the user never uses Windows or Mac): a search engine of
current AI models for the user's students. Python serves a JSON API + static page on 127.0.0.1;
the launcher opens it in its own Chrome `--app` window (pattern of `~/proyectos/herramientas/iconos/README.md`).
No LLM in the loop: filtering/sorting is deterministic code over the catalog.

## Hard rule: zero invention
Every model/price/licence/benchmark shown must come from OpenRouter or Hugging Face data fetched
now or cached; agent setup steps come from each agent's official docs (URL + check date in
`web/harnesses.json`). Missing values stay `None` -> shown as "sin datos". Only hardware memory figures
may be estimated, labelled "estimación" with the formula.

## Commands
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # dev install
.venv/bin/pytest                         # offline tests (respx mocks + real-response fixtures + real HTTP server)
.venv/bin/pytest -m live                 # tests against the real APIs (free)
node --test tests/js/*.test.mjs          # search logic (web/finder.js)
SELECTOR_IA_DATA_DIR=/tmp/x .venv/bin/selector-ia-server --port 8799   # dev server with a throwaway cache
./packaging/linux/install.sh             # install/refresh the menu app from this repo's LAST COMMIT
```
The installed app (`~/.local/opt/selector-modelos-ia`) is a git clone + venv: commit before running
`install.sh`. On every launch `actualizar.sh` fast-forwards it from GitHub `main` (rolls back if the new
version fails to install; a failed commit is remembered in `version-fallida` and not retried).

## Structure
- `src/selector_ia/data/` — `openrouter.py`, `huggingface.py` (plain REST via httpx), `models.py`
  (pydantic records), `cache.py` (SQLite; `DATA_SCHEMA_VERSION` forces a re-download when parsed fields
  change), `catalog.py` (cache/TTL/offline fallback), `http.py` (retries + Spanish error messages).
- `src/selector_ia/finder/` — `catalog.py` (`build_snapshot`: one `ModelRecord` per ORIGINAL model from
  OpenRouter + every HF pipeline tag; capabilities only from data; `old` = published and not updated for
  2 years; third-party fine-tunes/merges/uncensored builds kept as `community` with `base_model`;
  `uncensored`/`adult` flags; `evals` by category; `arch` = slim config.json (fetched once, cached for good,
  GGUF repos use their base model), `max_context`, `mtp`; `hf_records` for the live search; `SNAPSHOT_VERSION`),
  `card.py` (verbatim quotes from model cards: refusals, KL, tool-call tests, aggressive/balanced), `fit.py` (fit estimate for the whole catalog on one profile), `equipment.py` (saved computers:
  form, templates VPS / PC GPU / GPU server).
- `src/selector_ia/hardware/` — `profile.py`, `local.py` (psutil, nvidia-smi, amdgpu sysfs), `parsers.py`,
  `variants.py` (GGUF incl. imatrix IQ4_XS/IQ4_NL, AWQ... with REAL file sizes), `fit.py` (memory = weights +
  KV cache of the layers that grow with the context × agents, f16/q8_0 + 15 %; ✅/⚠️/❌; `AGENTIC_OK` floor),
  `runners.py` (Ollama/llama.cpp with context, -np and q8_0 flags; vLLM/SGLang for several agents; llama-bench),
  `check.py` (exact check of one model with real files and the student's usage).
- `src/selector_ia/server.py` — `ThreadingHTTPServer` on 127.0.0.1 (ports 8790…8804, first free), Host +
  Origin checks, CSP. Snapshot kept in memory + cache blob `finder_snapshot`; a watcher thread rebuilds it
  when older than `refresh_hours` (default 6, retry every 30 min after a failure). API: `/api/ocupado`
  (launcher), `status`, `catalog`, `fit`, `check`, `profiles` (+`/detect`, `/delete`), `settings` (keys go to
  the keyring and are never returned), `refresh`, `probar`, `version`, `hfsearch` (live search by name in
  all of Hugging Face).
- `src/selector_ia/web/` — `finder.js` (pure search logic: AND filters, counts, 6 sorts, quality = benchmarks
  of the ticked capabilities, suggestions that never drop a capability first, harness requirements and
  setup), `app.js` (rendering; without the server it reads `snapshot.json` as a labelled mock-up),
  `styles.css` (tokens, dark/light), `harnesses.json` (coding agents for the OpenRouter API).
- `src/selector_ia/compare/tester.py` — «Probar»: `ask_one` / `check_ids` (ids re-checked, max 3).
- `packaging/linux/` — `install.sh`, `uninstall.sh`, `lanzar.sh`, `parar.sh`, `actualizar.sh`, icon.
- `tests/fixtures/` — trimmed real API responses (2026-10-30). `tests/js/` — Node tests.

## API facts (verified 2026-10-30)
- **`/api/v1/models` without `?output_modalities=all` lists only text-output models** (456 of 692 on
  2026-10-30): video (31), embeddings (42), speech (38), transcription (29), rerank (10) and image models were
  missing. Always request `output_modalities=all`. 'decisions' models (extraction) fit no checkbox: dropped.
- Video prices live in `/api/v1/videos/models` → `pricing_skus`; OpenRouter documents neither keys nor units
  ("pricing SKU units can differ by provider"). Only `cents_per_video_output_second_*` / `cents_per_second_output*`
  become a price (USD/s); other SKUs are shown verbatim in the detail panel. Non-chat models priced 0/0 are
  billed another way: not "free" unless the id ends in `:free`.
- OpenRouter `/api/v1/models` is public; entries carry `benchmarks` (artificial_analysis intelligence/
  coding/agentic indexes, design_arena ELO), `top_provider.is_moderated`, `expiration_date`, `alias_target`
  (`~...-latest` aliases) and price `-1` for routers.
- Hugging Face returns **401, not 404**, for repos that don't exist. Rate limit: 500 req / 5 min.
  `expand[]=inferenceProviderMapping` (list in search, dict in model) gives third-party providers;
  `inference_provider=all` filters by them. Open ASR Leaderboard results come in `evalResults`
  (dataset `hf-audio/open-asr-leaderboard`, `mean_wer`). Open LLM Leaderboard is retired.
- `pyannote-audio` repos are tagged speech recognition but only do diarization: excluded. CoreML/MLX repos
  are Apple-only: excluded.
- Claude Code + OpenRouter: only guaranteed with Anthropic models (OpenRouter docs) → the harness filter
  shows Anthropic models unless «ver también los modelos sin garantía».
- Bytes/weight constants in `variants.py` were measured on unsloth/Qwen3-8B-GGUF; fallback only.
- AMD APUs report a ~512 MB VRAM carve-out in sysfs: < 2 GB is treated as integrated (no VRAM).
- Threads: build SSL contexts only via `http.shared_ssl_context()` (concurrent creation segfaults);
  SQLite connections are per thread (each request opens its own `Cache()`).
- OpenRouter 404 "data policy/guardrail" = the user's privacy settings exclude every provider of that model.
- Port 8790 is also used by the user's «Subtítulos» app: the launcher finds this server by `/api/ocupado`
  (`"app": "selector"`), never by port alone.
