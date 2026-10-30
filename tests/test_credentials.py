from selector_ia import credentials


def test_roundtrip_in_keyring():
    assert credentials.get_secret(credentials.OPENROUTER_KEY) is None
    credentials.set_secret(credentials.OPENROUTER_KEY, "sk-or-123")
    assert credentials.get_secret(credentials.OPENROUTER_KEY) == "sk-or-123"
    credentials.delete_secret(credentials.OPENROUTER_KEY)
    credentials.delete_secret(credentials.OPENROUTER_KEY)  # deleting twice is fine
    assert credentials.get_secret(credentials.OPENROUTER_KEY) is None


def test_env_var_overrides_keyring(monkeypatch):
    credentials.set_secret(credentials.HF_TOKEN, "from-keyring")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert credentials.get_secret(credentials.HF_TOKEN) == "from-env"


def test_nothing_written_to_data_dir(tmp_path):
    credentials.set_secret(credentials.OPENROUTER_KEY, "sk-or-secret")
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"sk-or-secret" not in path.read_bytes()
