import asyncio
import json
import logging
import threading
import time
import urllib.request
import uuid

from aiohttp import ClientSession, ClientTimeout, web

from python.helpers import codex_provider


logger = logging.getLogger("codex-provider")

_proxy_instance = None
_proxy_lock = threading.Lock()

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"
TOKEN_URL = "https://auth.openai.com/oauth/token"
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


class CodexProxy:
    def __init__(self, config: codex_provider.CodexConfig):
        self.config = config
        self.app = None
        self.runner = None
        self.session = None
        self.port = int(config.get("proxy_port", codex_provider.DEFAULT_PROXY_PORT))
        self._running = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def start(self):
        if self._running:
            return

        self.app = web.Application()
        self.app.router.add_get("/health", self._health)
        self.app.router.add_get("/v1/models", self._models)
        self.app.router.add_post("/v1/responses", self._responses)
        self.app.router.add_post("/v1/chat/completions", self._chat_completions)
        self.app.router.add_route("*", "/v1/{path:.*}", self._passthrough)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await site.start()
        self._running = True
        logger.info("Codex proxy listening on 127.0.0.1:%s", self.port)

    async def stop(self):
        if self.session:
            await self.session.close()
            self.session = None
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
        self._running = False

    def _get_session(self) -> ClientSession:
        if not self.session or self.session.closed:
            self.session = ClientSession(timeout=ClientTimeout(total=300, connect=10))
        return self.session

    def _get_access_token(self) -> str:
        return self.config.get("oauth_access_token", "")

    async def _refresh_token(self) -> bool:
        refresh_token = self.config.get("oauth_refresh_token", "")
        if not refresh_token:
            return False
        try:
            data = json.dumps(
                {
                    "grant_type": "refresh_token",
                    "client_id": OAUTH_CLIENT_ID,
                    "refresh_token": refresh_token,
                }
            ).encode()
            req = urllib.request.Request(
                TOKEN_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "codex-cli/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
            access_token = result.get("access_token", "")
            if not access_token:
                return False
            account_info = codex_provider.extract_account_info(access_token)
            self.config = codex_provider.save_config(
                {
                    **self.config,
                    "oauth_access_token": access_token,
                    "oauth_refresh_token": result.get("refresh_token", refresh_token),
                    "token_expires_at": int(time.time() + result.get("expires_in", 86400)),
                    "chatgpt_account_id": (
                        account_info.get("account_id", "") if account_info else ""
                    ),
                }
            )
            return True
        except Exception as exc:
            logger.error("Token refresh failed: %s", exc)
            return False

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "accept": "text/event-stream",
        }
        account_id = self.config.get("chatgpt_account_id", "")
        if not account_id:
            info = codex_provider.extract_account_info(self._get_access_token())
            account_id = info.get("account_id", "") if info else ""
            if account_id:
                self.config["chatgpt_account_id"] = account_id
                codex_provider.save_config(self.config)
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "running": self._running,
                "upstream": CODEX_BASE_URL,
                "proxy_url": self.base_url,
                "account_id": bool(self.config.get("chatgpt_account_id")),
            }
        )

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": model["id"], "object": "model", "owned_by": "openai"}
                    for model in codex_provider.get_supported_models()
                ],
            }
        )

    async def _chat_completions(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
                status=400,
            )
        codex_body = _chat_to_responses(body)
        target_url = f"{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}"
        session = self._get_session()
        stream = bool(body.get("stream", False))

        for attempt in range(2):
            try:
                async with session.post(
                    target_url,
                    json=codex_body,
                    headers=self._build_headers(),
                ) as resp:
                    if resp.status == 401 and attempt == 0 and await self._refresh_token():
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        return web.json_response(
                            {"error": {"message": f"Codex API error ({resp.status}): {text[:200]}", "type": "upstream_error"}},
                            status=resp.status,
                        )
                    if stream:
                        return await self._stream_responses_to_chat(request, resp, body)
                    return await self._collect_responses_to_chat(resp, body)
            except asyncio.TimeoutError:
                return web.json_response(
                    {"error": {"message": "Upstream timeout", "type": "timeout"}},
                    status=504,
                )
            except Exception as exc:
                logger.exception("Proxy error")
                return web.json_response(
                    {"error": {"message": str(exc), "type": "proxy_error"}},
                    status=502,
                )
        return web.json_response(
            {"error": {"message": "Authentication failed", "type": "upstream_error"}},
            status=401,
        )

    async def _responses(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
                status=400,
            )

        body.setdefault("store", False)
        body.setdefault("include", ["reasoning.encrypted_content"])
        body.setdefault("reasoning", {"effort": "medium", "summary": "auto"})
        body.setdefault("instructions", "You are a helpful assistant.")

        streaming_requested = bool(body.get("stream", True))
        body["stream"] = True
        target_url = f"{CODEX_BASE_URL}{CODEX_RESPONSES_PATH}"
        session = self._get_session()

        for attempt in range(2):
            try:
                async with session.post(
                    target_url,
                    json=body,
                    headers=self._build_headers(),
                ) as resp:
                    if resp.status == 401 and attempt == 0 and await self._refresh_token():
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        return web.json_response(
                            {"error": {"message": f"Codex API error ({resp.status}): {text[:200]}", "type": "upstream_error"}},
                            status=resp.status,
                        )
                    if streaming_requested:
                        return await self._stream_passthrough(request, resp)
                    return await self._collect_response(resp)
            except asyncio.TimeoutError:
                return web.json_response(
                    {"error": {"message": "Upstream timeout", "type": "timeout"}},
                    status=504,
                )
            except Exception as exc:
                logger.exception("Proxy error in /v1/responses")
                return web.json_response(
                    {"error": {"message": str(exc), "type": "proxy_error"}},
                    status=502,
                )
        return web.json_response(
            {"error": {"message": "Authentication failed", "type": "upstream_error"}},
            status=401,
        )

    async def _stream_passthrough(self, request: web.Request, resp) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        async for chunk in resp.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        return response

    async def _collect_response(self, resp) -> web.Response:
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                item = await resp.json()
                return web.json_response(normalize_openai_response(item))
            except Exception:
                return web.Response(text=await resp.text(), content_type="application/json")

        final_response = None
        collected_text = ""
        async for chunk in resp.content.iter_any():
            for line in chunk.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.completed":
                    final_response = event.get("response", {})
                elif event.get("type") == "response.output_text.delta":
                    collected_text += event.get("delta", "")

        if final_response:
            return web.json_response(normalize_openai_response(final_response))
        if collected_text:
            return web.json_response(
                normalize_openai_response(
                    {
                        "id": f"resp-{uuid.uuid4().hex[:24]}",
                        "object": "response",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": collected_text}],
                            }
                        ],
                        "status": "completed",
                    }
                )
            )
        return web.json_response(
            {"error": {"message": "No response.completed event received", "type": "upstream_error"}},
            status=502,
        )

    async def _stream_responses_to_chat(self, request: web.Request, resp, orig_body: dict) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model = orig_body.get("model", codex_provider.DEFAULT_CHAT_MODEL)
        sent_role = False
        buffer = b""

        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    stop_chunk = _make_chat_chunk(chat_id, model, finish_reason="stop")
                    await response.write(f"data: {json.dumps(stop_chunk)}\n\n".encode())
                    await response.write(b"data: [DONE]\n\n")
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "response.output_item.added" and not sent_role:
                    await response.write(
                        f"data: {json.dumps(_make_chat_chunk(chat_id, model, role='assistant'))}\n\n".encode()
                    )
                    sent_role = True
                elif event.get("type") == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        await response.write(
                            f"data: {json.dumps(_make_chat_chunk(chat_id, model, content=delta))}\n\n".encode()
                        )
                elif event.get("type") == "response.completed":
                    if not sent_role:
                        text = _extract_text_from_response(event.get("response", {}))
                        if text:
                            await response.write(
                                f"data: {json.dumps(_make_chat_chunk(chat_id, model, role='assistant'))}\n\n".encode()
                            )
                            await response.write(
                                f"data: {json.dumps(_make_chat_chunk(chat_id, model, content=text))}\n\n".encode()
                            )
                    await response.write(
                        f"data: {json.dumps(_make_chat_chunk(chat_id, model, finish_reason='stop'))}\n\n".encode()
                    )
                    await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def _collect_responses_to_chat(self, resp, orig_body: dict) -> web.Response:
        full_text = ""
        async for chunk in resp.content.iter_any():
            for line in chunk.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.output_text.delta":
                    full_text += event.get("delta", "")
                elif event.get("type") == "response.completed":
                    text = _extract_text_from_response(event.get("response", {}))
                    if text and not full_text:
                        full_text = text

        model = orig_body.get("model", codex_provider.DEFAULT_CHAT_MODEL)
        return web.json_response(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    async def _passthrough(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"message": "Only /v1/chat/completions, /v1/responses, and /v1/models are supported", "type": "not_supported"}},
            status=404,
        )


def normalize_openai_response(resp: dict) -> dict:
    if "created_at" not in resp:
        resp["created_at"] = int(time.time())
    resp.setdefault("status", "completed")
    resp.setdefault("output", [])
    return resp


def _chat_to_responses(body: dict) -> dict:
    messages = body.get("messages", [])
    model = body.get("model", codex_provider.DEFAULT_CHAT_MODEL)
    instructions = ""
    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        if role == "system":
            instructions = content
        elif role == "user":
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        elif role == "assistant":
            input_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            )

    return {
        "model": model,
        "store": False,
        "stream": True,
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"verbosity": "medium"},
        "instructions": instructions or "You are a helpful assistant.",
    }


def _make_chat_chunk(chat_id: str, model: str, role: str | None = None, content: str | None = None, finish_reason: str | None = None) -> dict:
    delta = {}
    if role:
        delta["role"] = role
        delta["content"] = ""
    if content is not None:
        delta["content"] = content
    if finish_reason:
        delta = {}
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _extract_text_from_response(response: dict) -> str:
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    texts.append(part.get("text", ""))
    return "".join(texts)


def get_proxy() -> CodexProxy | None:
    return _proxy_instance


async def ensure_running(config: codex_provider.CodexConfig | None = None) -> CodexProxy:
    global _proxy_instance
    config = config or codex_provider.load_config()
    with _proxy_lock:
        if _proxy_instance is None:
            _proxy_instance = CodexProxy(config)
        else:
            _proxy_instance.config = config
            _proxy_instance.port = int(config.get("proxy_port", codex_provider.DEFAULT_PROXY_PORT))
    if not _proxy_instance._running:
        await _proxy_instance.start()
    return _proxy_instance
