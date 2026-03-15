# a0 Codex 0.1.1 Release Notes

## What's Changed

* **LiteLLM Compatibility Fix**: Added a `normalize_openai_response` adapter function to the `codex_proxy_server.py`. This resolves an issue where missing `created_at` fields in proxy responses caused crashes during LiteLLM response transformation when Codex outputs were successful.
  * Added corresponding regression tests for the normalization adapter.
* **UI Stop Proxy Button**: Added a "Stop Proxy" action to the Web UI (`codex.html` / `codex-store.js`) and API (`codex_configure.py`). This allows users to cleanly stop the proxy and restore their previous LLM settings (such as local Ollama models or Cloud LLMs) when running out of Codex API limits, without having to completely disconnect and lose their Codex OAuth tokens.
