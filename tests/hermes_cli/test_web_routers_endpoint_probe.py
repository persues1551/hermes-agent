"""Endpoint-probe contract for the Desktop local/custom endpoint validators (#63472).

httpx honours ``HTTP(S)_PROXY`` (and the Windows system proxy) but never the proxy bypass list,
so a system proxy answered ``127.0.0.1`` probes with its own error page. The GUI then reported
"advertised no models" for a llama.cpp server the CLI (urllib, honours the bypass) saw fine.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.parametrize(
    "url, trusts_env",
    [
        ("http://127.0.0.1:8080/v1/models", False),
        ("http://localhost:11434/v1/models", False),
        ("http://192.168.1.20:8000/v1/models", False),
        ("https://api.example.com/v1/models", True),
    ],
)
def test_local_endpoint_probes_bypass_env_proxy(url, trusts_env, monkeypatch):
    from hermes_cli.web_routers.config_env import _endpoint_probe_client

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    client = _endpoint_probe_client(url, 1.0)
    assert client.trust_env is trusts_env


def test_openai_base_url_probe_names_the_http_status_instead_of_no_models(monkeypatch):
    """A reachable endpoint answering non-2xx with no model list is a failure the user can act on,
    not an empty catalog the GUI turns into 'start a model on that endpoint'."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import EnvVarUpdate

    class _Resp:
        status_code = 502
        is_success = False

        def json(self):
            return {"error": "proxy upstream unavailable"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())
    monkeypatch.setattr(mod, "_require_token", lambda request: None)

    body = EnvVarUpdate(key="OPENAI_BASE_URL", value="http://127.0.0.1:8080/v1", api_key="")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is False and out["reachable"] is True
    assert "HTTP 502" in out["message"]


def test_cloudflare_browser_integrity_block_is_not_a_bad_key(monkeypatch):
    """A Cloudflare 1010 block (403 + browser-signature body) must be reported
    as a client-signature block, not as 'The endpoint rejected the API key'
    (fork port of the 2026-08-27 dashboard-probe fix)."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate

    class _Resp:
        status_code = 403
        is_success = False
        text = "<html><title>Error 1010</title>blocked based on your browser signature</html>"

        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())
    body = CustomEndpointUpdate(
        name="Relay", base_url="https://relay.example.com/v1", model="m", api_key="sk-x",
    )
    out = asyncio.run(mod.validate_custom_endpoint(body))

    assert out["ok"] is False and out["reachable"] is False
    assert "1010" in out["message"]
