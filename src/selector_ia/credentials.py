"""API keys in the system keyring (never in plain text files).

Environment variables (OPENROUTER_API_KEY, HF_TOKEN) take precedence, for tests/CI.
"""

import os

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE = "selector-modelos-ia"

OPENROUTER_KEY = "openrouter_api_key"
HF_TOKEN = "huggingface_token"

_ENV_OVERRIDES = {OPENROUTER_KEY: "OPENROUTER_API_KEY", HF_TOKEN: "HF_TOKEN"}


def get_secret(name: str) -> str | None:
    env_var = _ENV_OVERRIDES.get(name)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    try:
        return keyring.get_password(SERVICE, name)
    except KeyringError:
        return None


def set_secret(name: str, value: str) -> None:
    keyring.set_password(SERVICE, name, value)


def delete_secret(name: str) -> None:
    try:
        keyring.delete_password(SERVICE, name)
    except PasswordDeleteError:
        pass

