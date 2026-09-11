"""Fork-port guards (bd55032f8): a plain upstream 500/502 — classified
``server_error`` + ``should_fallback`` by ``error_classifier._status_5xx`` —
must switch to the next provider on the first failure, and the credential-pool
guard must not block that switch.

The live gate lives in ``agent.turn_recovery.route_classified_error`` since the
v0.21.1 turn-module split; it was ``agent.conversation_loop`` at the time of
the original patch.
"""

from types import SimpleNamespace

import pytest

from agent import conversation_loop
from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_retry_state import TurnRetryState
from agent.turn_recovery import route_classified_error


class _FakeAPIError(Exception):
    """Minimal error surface for the routing helpers."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.body = None
        self.status_code = status_code


class _PoolProbe:
    """Records whether the credential-pool guard was consulted."""

    def __init__(self) -> None:
        self.calls = 0
        self.result = True

    def _pool_may_recover_from_rate_limit(self, pool):  # noqa: ARG002
        self.calls += 1
        return self.result


@pytest.fixture
def pool_probe(monkeypatch):
    probe = _PoolProbe()
    # route_classified_error imports both helpers from agent.conversation_loop
    # at call time — stub the restart arm and observe the pool guard.
    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *a, **k: None)
    monkeypatch.setattr(conversation_loop, "_ra", lambda: probe)
    return probe


def _route(classified, *, retry_count=0, fallback_index=0, chain_len=1):
    statuses: list = []
    activations: list = []
    agent = SimpleNamespace(
        _fallback_index=fallback_index,
        _fallback_chain=[SimpleNamespace() for _ in range(chain_len)],
        _buffer_status=statuses.append,
        _try_activate_fallback=lambda reason=None: activations.append(reason) or True,
        _credential_pool=object(),
        provider="custom",
        log_prefix="[fallback-gate] ",
    )
    verdict = route_classified_error(
        agent,
        _FakeAPIError("Bad Gateway", 502),
        classified,
        TurnRetryState(),
        error_msg="Bad Gateway",
        error_context=None,
        recovered_with_pool=False,
        base_url="https://relay.example.com/v1",
        model="gpt-6-astra",
        messages=[],
        api_messages=[],
        system_message=None,
        active_system_prompt=None,
        conversation_history=None,
        retry_count=retry_count,
        max_retries=5,
        compression_attempts=0,
        max_compression_attempts=3,
        api_call_count=1,
        effective_task_id="fallback-gate-test",
    )
    return verdict, statuses, activations


def test_plain_502_falls_back_on_first_failure(pool_probe):
    classified = ClassifiedError(reason=FailoverReason.server_error, should_fallback=True)
    verdict, statuses, activations = _route(classified)
    assert verdict.action == "break"
    assert activations == [FailoverReason.server_error]
    assert any("server error" in line for line in statuses)
    # A 500/502 must never be held back by the credential-pool guard.
    assert pool_probe.calls == 0


def test_pool_guard_cannot_block_the_server_error_fallback(pool_probe):
    pool_probe.result = True  # pool "may recover" — irrelevant for a 500/502
    classified = ClassifiedError(reason=FailoverReason.server_error, should_fallback=True)
    verdict, _, activations = _route(classified)
    assert verdict.action == "break"
    assert activations == [FailoverReason.server_error]


def test_empty_response_5xx_does_not_fall_back(pool_probe):
    classified = ClassifiedError(reason=FailoverReason.server_error, should_fallback=False)
    verdict, _, activations = _route(classified)
    assert verdict.action == "fallthrough"
    assert activations == []


def test_transport_failure_still_waits_for_the_second_retry(pool_probe):
    pool_probe.result = False
    classified = ClassifiedError(reason=FailoverReason.timeout)
    _, _, before = _route(classified, retry_count=1)
    assert before == []
    verdict, _, after = _route(classified, retry_count=2)
    assert verdict.action == "break"


def test_exhausted_fallback_chain_does_not_activate(pool_probe):
    classified = ClassifiedError(reason=FailoverReason.server_error, should_fallback=True)
    verdict, _, activations = _route(classified, fallback_index=1, chain_len=1)
    assert verdict.action == "fallthrough"
    assert activations == []
