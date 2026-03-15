from python.helpers.api import ApiHandler, Request, Response
from python.helpers import codex_oauth_manager, codex_provider, settings
from python.helpers.codex_proxy_server import ensure_running, get_proxy


class CodexConfigure(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = input.get("action", "apply")

        if action == "apply":
            return await self._apply(input)
        if action == "disconnect":
            return await self._disconnect()
        if action == "stop":
            return await self._stop()
        if action == "get_models":
            return {"ok": True, "models": codex_provider.get_supported_models()}
        if action == "settings_snapshot":
            current = settings.get_settings()
            return {
                "ok": True,
                "chat_model_provider": current["chat_model_provider"],
                "util_model_provider": current["util_model_provider"],
                "browser_model_provider": current["browser_model_provider"],
            }
        return {"ok": False, "error": f"Unknown action: {action}"}

    async def _apply(self, input: dict) -> dict:
        config = codex_provider.load_config()
        if not codex_provider.has_credentials(config):
            return {"ok": False, "error": "No credentials configured"}

        config = codex_provider.save_config(
            {
                **config,
                "chat_model": input.get("chat_model", config["chat_model"]),
                "util_model": input.get("util_model", config["util_model"]),
                "browser_model": input.get("browser_model", config["browser_model"]),
                "auto_configure": bool(input.get("auto_configure", True)),
            }
        )

        proxy = await ensure_running(config)
        saved_config, saved_settings = codex_provider.apply_codex_settings(
            config["chat_model"],
            config["util_model"],
            config["browser_model"],
        )

        return {
            "ok": True,
            "message": "Agent Zero configured to use Codex via proxy",
            "proxy_url": proxy.base_url,
            "chat_model": saved_config["chat_model"],
            "util_model": saved_config["util_model"],
            "browser_model": saved_config["browser_model"],
            "settings": settings.convert_out(saved_settings),
        }

    async def _stop(self) -> dict:
        proxy = get_proxy()
        if proxy and proxy._running:
            await proxy.stop()

        restored_config, restored_settings = codex_provider.restore_previous_settings()

        return {
            "ok": True,
            "message": "Codex proxy stopped and previous model settings restored",
            "settings": settings.convert_out(restored_settings),
        }

    async def _disconnect(self) -> dict:
        config = codex_provider.load_config()

        if config.get("oauth_refresh_token"):
            await codex_oauth_manager.revoke_token(config["oauth_refresh_token"])
        if config.get("oauth_access_token"):
            await codex_oauth_manager.revoke_token(config["oauth_access_token"])

        proxy = get_proxy()
        if proxy and proxy._running:
            await proxy.stop()

        restored_config, restored_settings = codex_provider.restore_previous_settings()
        cleared_config = codex_provider.clear_tokens()
        cleared_config["saved_previous_settings"] = restored_config.get("saved_previous_settings")
        codex_provider.save_config(cleared_config)

        return {
            "ok": True,
            "message": "Codex disconnected and previous model settings restored",
            "settings": settings.convert_out(restored_settings),
        }
