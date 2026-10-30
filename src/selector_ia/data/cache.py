"""SQLite cache for catalogs, Hugging Face queries and settings."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from selector_ia.config import DEFAULT_REFRESH_HOURS, db_path
from selector_ia.data.models import HFModel, OpenRouterModel
from selector_ia.hardware.profile import HardwareProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS openrouter_models (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hf_models (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hf_queries (
    key TEXT PRIMARY KEY,
    params TEXT NOT NULL,
    model_ids TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    name TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

OPENROUTER_REFRESHED = "openrouter_refreshed_at"
DATA_SCHEMA = "data_schema"
DATA_SCHEMA_VERSION = "3"   # bump when parsed fields change: cached catalogs are then downloaded again


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Cache:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else db_path()
        self.conn = sqlite3.connect(self.path, timeout=30)  # one connection per thread; wait on locks
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if self._get_meta(DATA_SCHEMA) != DATA_SCHEMA_VERSION:
            self.clear_hf()
            with self.conn:
                self.conn.execute("DELETE FROM meta WHERE key = ?", (OPENROUTER_REFRESHED,))
                self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                                  (DATA_SCHEMA, DATA_SCHEMA_VERSION))

    def close(self) -> None:
        self.conn.close()

    # --- settings / meta ---

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

    def refresh_hours(self) -> float:
        return float(self.get_setting("refresh_hours", str(DEFAULT_REFRESH_HOURS)))

    def _get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def is_fresh(self, fetched_at: datetime | None) -> bool:
        return fetched_at is not None and _now() - fetched_at < timedelta(hours=self.refresh_hours())

    # --- OpenRouter catalog (replaced as a whole on each refresh) ---

    def save_openrouter(self, models: list[OpenRouterModel]) -> None:
        refreshed = _now().isoformat()
        with self.conn:
            self.conn.execute("DELETE FROM openrouter_models")
            self.conn.executemany(
                "INSERT INTO openrouter_models (id, data, fetched_at) VALUES (?, ?, ?)",
                [(m.id, m.model_dump_json(), m.fetched_at.isoformat()) for m in models],
            )
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                              (OPENROUTER_REFRESHED, refreshed))

    def load_openrouter(self) -> list[OpenRouterModel]:
        rows = self.conn.execute("SELECT data FROM openrouter_models ORDER BY id").fetchall()
        return [OpenRouterModel.model_validate_json(r[0]) for r in rows]

    def openrouter_refreshed_at(self) -> datetime | None:
        value = self._get_meta(OPENROUTER_REFRESHED)
        return datetime.fromisoformat(value) if value else None

    # --- Hugging Face models and queries ---

    def save_hf_models(self, models: list[HFModel]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO hf_models (id, data, fetched_at) VALUES (?, ?, ?)",
                [(m.id, m.model_dump_json(), m.fetched_at.isoformat()) for m in models],
            )

    def load_hf_model(self, model_id: str) -> HFModel | None:
        row = self.conn.execute("SELECT data FROM hf_models WHERE id = ?", (model_id,)).fetchone()
        return HFModel.model_validate_json(row[0]) if row else None

    def load_hf_models(self, model_ids: list[str]) -> list[HFModel]:
        found = {m.id: m for m in (self.load_hf_model(i) for i in model_ids) if m}
        return [found[i] for i in model_ids if i in found]

    @staticmethod
    def query_key(params: dict) -> str:
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def save_hf_query(self, params: dict, models: list[HFModel]) -> None:
        self.save_hf_models(models)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO hf_queries (key, params, model_ids, fetched_at) VALUES (?, ?, ?, ?)",
                (self.query_key(params), json.dumps(params, sort_keys=True),
                 json.dumps([m.id for m in models]), _now().isoformat()),
            )

    def load_hf_query(self, params: dict) -> tuple[list[HFModel], datetime] | None:
        row = self.conn.execute("SELECT model_ids, fetched_at FROM hf_queries WHERE key = ?",
                                (self.query_key(params),)).fetchone()
        if not row:
            return None
        return self.load_hf_models(json.loads(row[0])), datetime.fromisoformat(row[1])

    def hf_last_fetched_at(self) -> datetime | None:
        row = self.conn.execute("SELECT MAX(fetched_at) FROM hf_queries").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def clear_hf(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM hf_queries")
            self.conn.execute("DELETE FROM hf_models")
            self.conn.execute("DELETE FROM blobs")

    # --- generic JSON blobs (repo file lists, config.json) ---

    def save_blob(self, key: str, data) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO blobs (key, data, fetched_at) VALUES (?, ?, ?)",
                              (key, json.dumps(data), _now().isoformat()))

    def load_blob(self, key: str) -> tuple[object, datetime] | None:
        row = self.conn.execute("SELECT data, fetched_at FROM blobs WHERE key = ?", (key,)).fetchone()
        return (json.loads(row[0]), datetime.fromisoformat(row[1])) if row else None

    # --- hardware profiles ---

    def save_profile(self, profile: HardwareProfile) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO profiles (name, data, updated_at) VALUES (?, ?, ?)",
                              (profile.name, profile.model_dump_json(), _now().isoformat()))

    def load_profile(self, name: str) -> HardwareProfile | None:
        row = self.conn.execute("SELECT data FROM profiles WHERE name = ?", (name,)).fetchone()
        return HardwareProfile.model_validate_json(row[0]) if row else None

    def list_profiles(self) -> list[HardwareProfile]:
        rows = self.conn.execute("SELECT data FROM profiles ORDER BY name").fetchall()
        return [HardwareProfile.model_validate_json(r[0]) for r in rows]

    def delete_profile(self, name: str) -> bool:
        with self.conn:
            return self.conn.execute("DELETE FROM profiles WHERE name = ?", (name,)).rowcount > 0

    # --- search history ---

    def save_search(self, request: str, data_json: str) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO searches (created_at, request, data) VALUES (?, ?, ?)",
                                    (_now().isoformat(), request, data_json))
            return cur.lastrowid

    def list_searches(self, limit: int = 50) -> list[tuple[int, datetime, str]]:
        rows = self.conn.execute("SELECT id, created_at, request FROM searches ORDER BY id DESC LIMIT ?",
                                 (limit,)).fetchall()
        return [(r[0], datetime.fromisoformat(r[1]), r[2]) for r in rows]

    def load_search(self, search_id: int) -> str | None:
        row = self.conn.execute("SELECT data FROM searches WHERE id = ?", (search_id,)).fetchone()
        return row[0] if row else None
