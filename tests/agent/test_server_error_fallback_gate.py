from pathlib import Path
from types import SimpleNamespace

from agent import conversation_loop
from agent.error_classifier import FailoverReason


def test_conversation_loop_honors_classifier_fallback_signal() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
    ).read_text(encoding="utf-8")

    gate = """_should_fallback = (
                    is_rate_limited
                    or (_is_transport_failure and retry_count >= 2)
                    or classified.should_fallback
                )"""

    assert gate in source


def test_rate_limit_pool_guard_does_not_block_other_fallback_reasons(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        conversation_loop,
        "_ra",
        lambda: SimpleNamespace(
            _pool_may_recover_from_rate_limit=lambda pool: calls.append(pool) or True,
        ),
    )
    pool = object()

    assert conversation_loop._credential_pool_may_recover_from_fallback_reason(
        FailoverReason.server_error,
        pool,
    ) is False
    assert calls == []

    assert conversation_loop._credential_pool_may_recover_from_fallback_reason(
        FailoverReason.rate_limit,
        pool,
    ) is True
    assert calls == [pool]


def test_stale_copilot_error_defers_to_same_provider_recovery() -> None:
    agent = SimpleNamespace(
        provider="github-copilot",
        _is_copilot_provider=lambda: True,
    )

    assert conversation_loop._should_defer_to_copilot_credential_recovery(
        agent,
        False,
        400,
        "model_not_available_for_integrator",
    ) is True
    assert conversation_loop._should_defer_to_copilot_credential_recovery(
        agent,
        True,
        400,
        "model_not_available_for_integrator",
    ) is False


def test_non_copilot_format_error_does_not_defer_fallback() -> None:
    agent = SimpleNamespace(
        provider="custom",
        _is_copilot_provider=lambda: False,
    )

    assert conversation_loop._should_defer_to_copilot_credential_recovery(
        agent,
        False,
        400,
        "model_not_supported",
    ) is False


def test_server_error_fallback_status_is_not_labeled_rate_limited() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
    ).read_text(encoding="utf-8")

    assert "Upstream server error — switching to fallback provider..." in source