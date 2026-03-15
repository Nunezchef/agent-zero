import asyncio
import hashlib
import json
import logging
import secrets
import time

import aiohttp


logger = logging.getLogger("codex-provider.oauth")

_HEADERS = {"User-Agent": "codex-cli/1.0"}
DEVICE_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REVOKE_URL = "https://auth.openai.com/oauth/revoke"
DEVICE_VERIFICATION_URI = "https://auth.openai.com/codex/device"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"

_active_flows: dict[str, "OAuthSession"] = {}


class OAuthSession:
    def __init__(self):
        self.session_id = secrets.token_urlsafe(16)
        self.created_at = time.time()
        self.status = "pending"
        self.error: str | None = None
        self.device_code = ""
        self.user_code = ""
        self.verification_uri = DEVICE_VERIFICATION_URI
        self.verification_uri_complete = DEVICE_VERIFICATION_URI
        self.interval = 5
        self.expires_in = 900
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.expires_at: int | None = None
        self.account_info: dict | None = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.expires_in

    def to_dict(self) -> dict:
        payload = {
            "session_id": self.session_id,
            "status": self.status,
            "error": self.error,
            "interval": self.interval,
        }
        if self.status in ("pending", "polling"):
            payload["user_code"] = self.user_code
            payload["verification_uri"] = self.verification_uri
            payload["verification_uri_complete"] = self.verification_uri_complete
            payload["expires_in"] = max(0, int(self.expires_in - (time.time() - self.created_at)))
        if self.status == "complete":
            payload["account_info"] = self.account_info
            payload["expires_at"] = self.expires_at
        return payload


def _extract_account_info(access_token: str) -> dict | None:
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
        return {
            "email": profile_info.get("email", ""),
            "plan": auth_info.get("chatgpt_plan_type", "unknown"),
            "account_id": auth_info.get("chatgpt_account_id", ""),
            "user_id": auth_info.get("chatgpt_user_id", ""),
            "expires_at": data.get("exp"),
        }
    except Exception as exc:
        logger.warning("Failed to extract account info: %s", exc)
        return None


async def start_device_flow() -> OAuthSession:
    session = OAuthSession()
    async with aiohttp.ClientSession() as http:
        async with http.post(
            DEVICE_CODE_URL,
            json={"client_id": CLIENT_ID},
            headers={**_HEADERS, "Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                session.status = "error"
                session.error = f"Device code request failed ({resp.status}): {text[:200]}"
                return session

            data = await resp.json()
            session.device_code = data.get("device_auth_id", "")
            session.user_code = data.get("user_code", "")
            session.interval = int(data.get("interval", 5))
            expires_at = data.get("expires_at", "")
            if expires_at:
                from datetime import datetime, timezone

                try:
                    exp_dt = datetime.fromisoformat(expires_at)
                    session.expires_in = max(
                        60, int((exp_dt - datetime.now(timezone.utc)).total_seconds())
                    )
                except Exception:
                    session.expires_in = int(data.get("expires_in", 900))
            else:
                session.expires_in = int(data.get("expires_in", 900))
    _active_flows[session.session_id] = session
    return session


def _get_error_code(data: dict) -> str:
    err = data.get("error")
    if isinstance(err, dict):
        return err.get("code", err.get("type", ""))
    return str(err or "")


async def poll_device_flow(session_id: str) -> OAuthSession | None:
    session = _active_flows.get(session_id)
    if not session:
        return None
    if session.is_expired:
        session.status = "expired"
        session.error = "Authentication session expired. Please start again."
        return session
    if session.status in ("complete", "error", "expired"):
        return session

    session.status = "polling"
    async with aiohttp.ClientSession() as http:
        async with http.post(
            DEVICE_TOKEN_URL,
            json={
                "client_id": CLIENT_ID,
                "device_auth_id": session.device_code,
                "user_code": session.user_code,
            },
            headers={**_HEADERS, "Content-Type": "application/json"},
        ) as resp:
            data = await resp.json()
            error_code = _get_error_code(data)

            if resp.status == 200:
                auth_code = data.get("authorization_code", "")
                code_verifier = data.get("code_verifier", "")
                if not auth_code:
                    if data.get("access_token"):
                        session.access_token = data["access_token"]
                        session.refresh_token = data.get("refresh_token")
                        session.account_info = _extract_account_info(session.access_token)
                        session.expires_at = (
                            session.account_info.get("expires_at") if session.account_info else None
                        )
                        session.status = "complete"
                        return session
                    session.status = "error"
                    session.error = "Unexpected response from auth server"
                    return session

                exchange_body = (
                    f"grant_type=authorization_code"
                    f"&code={auth_code}"
                    f"&redirect_uri={DEVICE_REDIRECT_URI}"
                    f"&client_id={CLIENT_ID}"
                    f"&code_verifier={code_verifier}"
                )
                async with http.post(
                    TOKEN_URL,
                    data=exchange_body,
                    headers={**_HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
                ) as token_resp:
                    if token_resp.status == 200:
                        tokens = await token_resp.json()
                        session.access_token = tokens.get("access_token")
                        session.refresh_token = tokens.get("refresh_token")
                        session.account_info = _extract_account_info(session.access_token or "")
                        expires_in = tokens.get("expires_in", 86400)
                        session.expires_at = int(time.time() + expires_in)
                        session.status = "complete"
                    else:
                        text = await token_resp.text()
                        session.status = "error"
                        session.error = f"Token exchange failed ({token_resp.status}): {text[:200]}"
            elif error_code in ("deviceauth_authorization_unknown", "authorization_pending"):
                session.status = "pending"
            elif error_code == "slow_down":
                session.interval = min(session.interval + 2, 30)
                session.status = "pending"
            elif error_code in ("expired_token", "deviceauth_expired"):
                session.status = "expired"
                session.error = "Session expired. Please start a new sign-in."
            elif error_code == "access_denied":
                session.status = "error"
                session.error = "Access denied. Please try again."
            else:
                err = data.get("error", {})
                session.status = "error"
                session.error = err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return session


async def refresh_access_token(refresh_token: str) -> dict | None:
    async with aiohttp.ClientSession() as http:
        async with http.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            headers=_HEADERS,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error("Token refresh failed (%s): %s", resp.status, text[:200])
                return None
            data = await resp.json()
            access_token = data.get("access_token")
            if not access_token:
                return None
            expires_at = int(time.time() + data.get("expires_in", 86400))
            return {
                "access_token": access_token,
                "refresh_token": data.get("refresh_token", refresh_token),
                "expires_at": expires_at,
                "account_info": _extract_account_info(access_token),
            }


async def revoke_token(token: str) -> bool:
    async with aiohttp.ClientSession() as http:
        async with http.post(
            REVOKE_URL,
            data={"client_id": CLIENT_ID, "token": token},
            headers=_HEADERS,
        ) as resp:
            return resp.status == 200


def cleanup_expired() -> None:
    expired = [sid for sid, session in _active_flows.items() if session.is_expired]
    for sid in expired:
        del _active_flows[sid]
