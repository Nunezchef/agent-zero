import aiohttp

from python.helpers.api import ApiHandler, Request, Response
from python.helpers import codex_provider
from python.helpers.codex_proxy_server import CODEX_BASE_URL, ensure_running, get_proxy


class CodexStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = input.get("action", "status")

        if action == "status":
            return await self._status()
        if action == "test":
            return await self._test_connection()
        if action == "start_proxy":
            return await self._start_proxy()
        return {"ok": False, "error": f"Unknown action: {action}"}

    async def _status(self) -> dict:
        config = codex_provider.load_config()
        proxy = get_proxy()
        account_info = codex_provider.extract_account_info(config.get("oauth_access_token", ""))
        return {
            "ok": True,
            "connected": codex_provider.has_credentials(config),
            "expired": codex_provider.token_is_expired(config),
            "proxy_running": bool(proxy and proxy._running),
            "proxy_port": config["proxy_port"],
            "proxy_url": codex_provider.get_proxy_base_url(config),
            "upstream": CODEX_BASE_URL,
            "auth_mode": config["auth_mode"],
            "auto_configure": config["auto_configure"],
            "models": {
                "chat": config["chat_model"],
                "utility": config["util_model"],
                "browser": config["browser_model"],
            },
            "account_info": account_info,
        }

    async def _test_connection(self) -> dict:
        config = codex_provider.load_config()
        if not codex_provider.has_credentials(config):
            return {"ok": False, "error": "No credentials configured"}

        proxy = await ensure_running(config)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{proxy.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return {"ok": False, "error": await resp.text()}
                    data = await resp.json()
                    models = [item.get("id") for item in data.get("data", [])]
                    return {"ok": True, "models": models, "count": len(models)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _start_proxy(self) -> dict:
        config = codex_provider.load_config()
        if not codex_provider.has_credentials(config):
            return {"ok": False, "error": "No credentials configured"}
        proxy = await ensure_running(config)
        return {"ok": True, "proxy_url": proxy.base_url, "message": "Proxy started"}
