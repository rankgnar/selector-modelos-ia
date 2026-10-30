import json
import os

from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise PasswordDeleteError(username)
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def isolated_env(request, tmp_path, monkeypatch):
    """Never touch the real cache, keyring or wait on retries.
    Live tests keep the real keyring so they can read the user's OpenRouter key."""
    monkeypatch.setenv("SELECTOR_IA_DATA_DIR", str(tmp_path / "data"))
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    previous = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    monkeypatch.setattr("selector_ia.data.http.time.sleep", lambda _s: None)
    yield
    keyring.set_keyring(previous)
