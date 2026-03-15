import json
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from python.helpers import dotenv, files, settings


SUPPORTED_MODELS = [
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.1",
]
DEFAULT_CHAT_MODEL = "gpt-5.3-codex"
DEFAULT_UTILITY_MODEL = "gpt-5.1-codex-mini"
DEFAULT_BROWSER_MODEL = "gpt-5.1-codex-mini"
DEFAULT_PROXY_PORT = 8400
PROVIDER_ID = "codex_proxy"
DUMMY_API_KEY = "sk-codex-proxy-local"
CONFIG_PATH = Path(files.get_abs_path("usr/codex_provider.json"))


class SavedModeSettings(TypedDict):
    provider: str
    name: str
    api_base: str
    kwargs: dict[str, Any]


class SavedPreviousSettings(TypedDict):
    chat: SavedModeSettings
    utility: SavedModeSettings
    browser: SavedModeSettings


class CodexAccountInfo(TypedDict):
    email: str
    plan: str
    account_id: str
    user_id: str
    expires_at: int | None


class CodexConfig(TypedDict):
    auth_mode: Literal["none", "oauth"]
    oauth_access_token: str
    oauth_refresh_token: str
    token_expires_at: int | None
    chatgpt_account_id: str
    proxy_port: int
    auto_configure: bool
    chat_model: str
    util_model: str
    browser_model: str
    saved_previous_settings: Optional[SavedPreviousSettings]


_config_lock = threading.RLock()


def _default_config() -> CodexConfig:
    return CodexConfig(
        auth_mode="none",
        oauth_access_token="",
        oauth_refresh_token="",
        token_expires_at=None,
        chatgpt_account_id="",
        proxy_port=DEFAULT_PROXY_PORT,
        auto_configure=False,
        chat_model=DEFAULT_CHAT_MODEL,
        util_model=DEFAULT_UTILITY_MODEL,
        browser_model=DEFAULT_BROWSER_MODEL,
        saved_previous_settings=None,
    )


def _normalize_mode_model(model: str | None, fallback: str) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return fallback


def normalize_config(data: dict[str, Any] | None) -> CodexConfig:
    config = _default_config()
    if isinstance(data, dict):
        config.update(data)

    config["auth_mode"] = "oauth" if config.get("oauth_access_token") else "none"
    config["proxy_port"] = int(config.get("proxy_port") or DEFAULT_PROXY_PORT)
    config["auto_configure"] = bool(config.get("auto_configure", False))
    config["token_expires_at"] = (
        int(config["token_expires_at"])
        if config.get("token_expires_at")
        else None
    )
    config["chat_model"] = _normalize_mode_model(
        str(config.get("chat_model") or ""), DEFAULT_CHAT_MODEL
    )
    config["util_model"] = _normalize_mode_model(
        str(config.get("util_model") or ""), DEFAULT_UTILITY_MODEL
    )
    config["browser_model"] = _normalize_mode_model(
        str(config.get("browser_model") or ""), DEFAULT_BROWSER_MODEL
    )
    saved = config.get("saved_previous_settings")
    if not isinstance(saved, dict):
        config["saved_previous_settings"] = None
    return config


def load_config() -> CodexConfig:
    with _config_lock:
        if CONFIG_PATH.exists():
            try:
                return normalize_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                return _default_config()
        return _default_config()


def save_config(config: CodexConfig) -> CodexConfig:
    normalized = normalize_config(config)
    with _config_lock:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(normalized, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return normalized


def update_config(delta: dict[str, Any]) -> CodexConfig:
    config = load_config()
    config.update(delta)
    return save_config(config)


def clear_tokens() -> CodexConfig:
    return update_config(
        {
            "auth_mode": "none",
            "oauth_access_token": "",
            "oauth_refresh_token": "",
            "token_expires_at": None,
            "chatgpt_account_id": "",
            "auto_configure": False,
        }
    )


def has_credentials(config: CodexConfig | None = None) -> bool:
    current = config or load_config()
    return bool(current.get("oauth_access_token"))


def token_is_expired(config: CodexConfig | None = None) -> bool:
    current = config or load_config()
    expires_at = current.get("token_expires_at")
    if not expires_at:
        return False
    return expires_at <= int(time.time()) + 30


def get_proxy_base_url(config: CodexConfig | None = None) -> str:
    current = config or load_config()
    return f"http://127.0.0.1:{current['proxy_port']}/v1"


def get_supported_models() -> list[dict[str, str]]:
    labels = {
        "gpt-5.3-codex": "GPT-5.3 Codex",
        "gpt-5.2-codex": "GPT-5.2 Codex",
        "gpt-5.1-codex": "GPT-5.1 Codex",
        "gpt-5.1-codex-mini": "GPT-5.1 Codex Mini",
        "gpt-5.2": "GPT-5.2",
        "gpt-5.1": "GPT-5.1",
    }
    return [{"id": model, "name": labels[model]} for model in SUPPORTED_MODELS]


def extract_account_info(access_token: str) -> CodexAccountInfo | None:
    try:
        import base64

        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        auth_info = data.get("https://api.openai.com/auth", {})
        profile_info = data.get("https://api.openai.com/profile", {})
        return CodexAccountInfo(
            email=profile_info.get("email", ""),
            plan=auth_info.get("chatgpt_plan_type", "unknown"),
            account_id=auth_info.get("chatgpt_account_id", ""),
            user_id=auth_info.get("chatgpt_user_id", ""),
            expires_at=data.get("exp"),
        )
    except Exception:
        return None


def _snapshot_mode(provider: str, name: str, api_base: str, kwargs: dict[str, Any]) -> SavedModeSettings:
    return SavedModeSettings(
        provider=provider,
        name=name,
        api_base=api_base,
        kwargs=kwargs or {},
    )


def _current_mode_snapshots(current_settings: settings.Settings) -> SavedPreviousSettings:
    return SavedPreviousSettings(
        chat=_snapshot_mode(
            current_settings["chat_model_provider"],
            current_settings["chat_model_name"],
            current_settings["chat_model_api_base"],
            current_settings["chat_model_kwargs"],
        ),
        utility=_snapshot_mode(
            current_settings["util_model_provider"],
            current_settings["util_model_name"],
            current_settings["util_model_api_base"],
            current_settings["util_model_kwargs"],
        ),
        browser=_snapshot_mode(
            current_settings["browser_model_provider"],
            current_settings["browser_model_name"],
            current_settings["browser_model_api_base"],
            current_settings["browser_model_kwargs"],
        ),
    )


def apply_codex_settings(
    chat_model: str | None = None,
    util_model: str | None = None,
    browser_model: str | None = None,
) -> tuple[CodexConfig, settings.Settings]:
    config = load_config()
    current_settings = settings.get_settings()

    config["chat_model"] = _normalize_mode_model(chat_model, config["chat_model"])
    config["util_model"] = _normalize_mode_model(util_model, config["util_model"])
    config["browser_model"] = _normalize_mode_model(browser_model, config["browser_model"])

    if (
        current_settings["chat_model_provider"] != PROVIDER_ID
        or current_settings["util_model_provider"] != PROVIDER_ID
        or current_settings["browser_model_provider"] != PROVIDER_ID
    ):
        config["saved_previous_settings"] = _current_mode_snapshots(current_settings)

    updated = current_settings.copy()
    updated["chat_model_provider"] = PROVIDER_ID
    updated["chat_model_name"] = config["chat_model"]
    updated["chat_model_api_base"] = get_proxy_base_url(config)
    updated["chat_model_kwargs"] = {}

    updated["util_model_provider"] = PROVIDER_ID
    updated["util_model_name"] = config["util_model"]
    updated["util_model_api_base"] = get_proxy_base_url(config)
    updated["util_model_kwargs"] = {}

    updated["browser_model_provider"] = PROVIDER_ID
    updated["browser_model_name"] = config["browser_model"]
    updated["browser_model_api_base"] = get_proxy_base_url(config)
    updated["browser_model_kwargs"] = {}

    api_keys = dict(updated["api_keys"])
    api_keys[PROVIDER_ID] = DUMMY_API_KEY
    updated["api_keys"] = api_keys

    dotenv.save_dotenv_value("API_KEY_CODEX_PROXY", DUMMY_API_KEY)
    saved_settings = settings.set_settings(updated)
    saved_config = save_config(config)
    return saved_config, saved_settings


def restore_previous_settings() -> tuple[CodexConfig, settings.Settings]:
    config = load_config()
    previous = config.get("saved_previous_settings")
    current_settings = settings.get_settings()
    updated = current_settings.copy()

    if previous:
        updated["chat_model_provider"] = previous["chat"]["provider"]
        updated["chat_model_name"] = previous["chat"]["name"]
        updated["chat_model_api_base"] = previous["chat"]["api_base"]
        updated["chat_model_kwargs"] = previous["chat"]["kwargs"]

        updated["util_model_provider"] = previous["utility"]["provider"]
        updated["util_model_name"] = previous["utility"]["name"]
        updated["util_model_api_base"] = previous["utility"]["api_base"]
        updated["util_model_kwargs"] = previous["utility"]["kwargs"]

        updated["browser_model_provider"] = previous["browser"]["provider"]
        updated["browser_model_name"] = previous["browser"]["name"]
        updated["browser_model_api_base"] = previous["browser"]["api_base"]
        updated["browser_model_kwargs"] = previous["browser"]["kwargs"]

    saved_settings = settings.set_settings(updated)
    config["saved_previous_settings"] = None
    saved_config = save_config(config)
    return saved_config, saved_settings
