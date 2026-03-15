from python.helpers.extension import Extension
from python.helpers import codex_provider
from python.helpers.codex_proxy_server import ensure_running, get_proxy


class CodexProxyStart(Extension):
    async def execute(self, **kwargs):
        config = codex_provider.load_config()
        if not config.get("auto_configure", False):
            return
        if not codex_provider.has_credentials(config):
            return

        proxy = get_proxy()
        if proxy and proxy._running:
            return

        await ensure_running(config)
