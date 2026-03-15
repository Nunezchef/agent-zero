import json

from python.helpers.api import ApiHandler, Request, Response
from python.helpers import codex_oauth_manager, codex_provider
from python.helpers.codex_proxy_server import get_proxy


class CodexOauth(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        action = input.get("action", "")

        if action == "start_device_flow":
            return await self._start_device_flow()
        if action == "poll":
            return await self._poll(input)
        if action == "save_tokens":
            return await self._save_tokens(input)
        if action == "refresh":
            return await self._refresh()
        return {"ok": False, "error": f"Unknown action: {action}"}

    async def _start_device_flow(self) -> dict:
        codex_oauth_manager.cleanup_expired()
        session = await codex_oauth_manager.start_device_flow()
        if session.status == "error":
            return {"ok": False, "error": session.error}
        return {"ok": True, **session.to_dict()}

    async def _poll(self, input: dict) -> dict:
        session_id = input.get("session_id", "")
        if not session_id:
            return {"ok": False, "error": "Missing session_id"}

        session = await codex_oauth_manager.poll_device_flow(session_id)
        if not session:
            return {"ok": False, "error": "Session not found"}

        if session.status == "complete" and session.access_token:
            account_info = session.account_info or {}
            config = codex_provider.save_config(
                {
                    **codex_provider.load_config(),
                    "auth_mode": "oauth",
                    "oauth_access_token": session.access_token,
                    "oauth_refresh_token": session.refresh_token or "",
                    "token_expires_at": session.expires_at,
                    "chatgpt_account_id": account_info.get("account_id", ""),
                    "auto_configure": True,
                }
            )
            proxy = get_proxy()
            if proxy:
                proxy.config = config

        return {"ok": True, **session.to_dict()}

    async def _save_tokens(self, input: dict) -> dict:
        access_token = input.get("access_token", "")
        refresh_token = input.get("refresh_token", "")
        auth_json = input.get("auth_json")

        if auth_json:
            if isinstance(auth_json, str):
                auth_json = json.loads(auth_json)
            tokens = auth_json.get("tokens", auth_json)
            access_token = tokens.get("access_token", access_token)
            refresh_token = tokens.get("refresh_token", refresh_token)

        if not access_token:
            return {"ok": False, "error": "No access token provided"}

        account_info = codex_provider.extract_account_info(access_token)
        config = codex_provider.save_config(
            {
                **codex_provider.load_config(),
                "auth_mode": "oauth",
                "oauth_access_token": access_token,
                "oauth_refresh_token": refresh_token or "",
                "token_expires_at": account_info.get("expires_at") if account_info else None,
                "chatgpt_account_id": account_info.get("account_id", "") if account_info else "",
                "auto_configure": True,
            }
        )

        proxy = get_proxy()
        if proxy:
            proxy.config = config

        return {
            "ok": True,
            "message": "Tokens saved successfully",
            "account_info": account_info,
            "expires_at": config.get("token_expires_at"),
        }

    async def _refresh(self) -> dict:
        config = codex_provider.load_config()
        refresh_token = config.get("oauth_refresh_token", "")
        if not refresh_token:
            return {"ok": False, "error": "No refresh token stored"}

        result = await codex_oauth_manager.refresh_access_token(refresh_token)
        if not result:
            return {"ok": False, "error": "Token refresh failed. Please reconnect."}

        account_info = result.get("account_info") or {}
        updated = codex_provider.save_config(
            {
                **config,
                "auth_mode": "oauth",
                "oauth_access_token": result["access_token"],
                "oauth_refresh_token": result["refresh_token"],
                "token_expires_at": result["expires_at"],
                "chatgpt_account_id": account_info.get("account_id", ""),
                "auto_configure": True,
            }
        )
        proxy = get_proxy()
        if proxy:
            proxy.config = updated

        return {
            "ok": True,
            "message": "Token refreshed",
            "expires_at": updated.get("token_expires_at"),
            "account_info": account_info,
        }
