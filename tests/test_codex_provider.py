from pathlib import Path


def _base_settings():
    return {
        "chat_model_provider": "openrouter",
        "chat_model_name": "model-chat",
        "chat_model_api_base": "https://chat.example",
        "chat_model_kwargs": {"temperature": 0.1},
        "util_model_provider": "openai",
        "util_model_name": "model-util",
        "util_model_api_base": "https://util.example",
        "util_model_kwargs": {"temperature": 0.2},
        "browser_model_provider": "anthropic",
        "browser_model_name": "model-browser",
        "browser_model_api_base": "https://browser.example",
        "browser_model_kwargs": {"temperature": 0.3},
        "api_keys": {},
    }


def test_codex_config_round_trips(tmp_path: Path, monkeypatch) -> None:
    from python.helpers import codex_provider

    monkeypatch.setattr(codex_provider, "CONFIG_PATH", tmp_path / "codex_provider.json")

    saved = codex_provider.save_config(
        {
            "auth_mode": "oauth",
            "oauth_access_token": "token",
            "oauth_refresh_token": "refresh",
            "token_expires_at": 123,
            "chatgpt_account_id": "acct",
            "proxy_port": 9999,
            "auto_configure": True,
            "chat_model": "gpt-5.3-codex",
            "util_model": "bad-model",
            "browser_model": "gpt-5.1",
            "saved_previous_settings": None,
        }
    )

    assert saved["util_model"] == codex_provider.DEFAULT_UTILITY_MODEL
    loaded = codex_provider.load_config()
    assert loaded["oauth_access_token"] == "token"
    assert loaded["proxy_port"] == 9999
    assert loaded["browser_model"] == "gpt-5.1"


def test_apply_and_restore_codex_settings(tmp_path: Path, monkeypatch) -> None:
    from python.helpers import codex_provider

    monkeypatch.setattr(codex_provider, "CONFIG_PATH", tmp_path / "codex_provider.json")

    persisted_env = {}
    current_settings = _base_settings()

    def fake_get_settings():
        return current_settings.copy()

    def fake_set_settings(new_settings, apply=True):
        current_settings.clear()
        current_settings.update(new_settings)
        return current_settings.copy()

    monkeypatch.setattr(codex_provider.settings, "get_settings", fake_get_settings)
    monkeypatch.setattr(codex_provider.settings, "set_settings", fake_set_settings)
    monkeypatch.setattr(
        codex_provider.dotenv,
        "save_dotenv_value",
        lambda key, value: persisted_env.__setitem__(key, value),
    )

    codex_provider.save_config(codex_provider.load_config())

    config, updated = codex_provider.apply_codex_settings(
        "gpt-5.2-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1",
    )

    assert updated["chat_model_provider"] == codex_provider.PROVIDER_ID
    assert updated["util_model_provider"] == codex_provider.PROVIDER_ID
    assert updated["browser_model_provider"] == codex_provider.PROVIDER_ID
    assert updated["chat_model_name"] == "gpt-5.2-codex"
    assert updated["api_keys"][codex_provider.PROVIDER_ID] == codex_provider.DUMMY_API_KEY
    assert persisted_env["API_KEY_CODEX_PROXY"] == codex_provider.DUMMY_API_KEY
    assert config["saved_previous_settings"]["chat"]["provider"] == "openrouter"

    _, restored = codex_provider.restore_previous_settings()

    assert restored["chat_model_provider"] == "openrouter"
    assert restored["chat_model_name"] == "model-chat"
    assert restored["chat_model_kwargs"] == {"temperature": 0.1}
    assert restored["browser_model_provider"] == "anthropic"
