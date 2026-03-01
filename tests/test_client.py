"""Tests for VantinelClient."""

import pytest
from vantinel_sdk import VantinelConfig
from vantinel_sdk.client import VantinelClient, CircuitBreaker, CircuitBreakerState
from vantinel_sdk.types import VantinelEvent, Decision
import time


@pytest.mark.asyncio
async def test_dry_run_mode():
    """Test that dry-run mode doesn't make network calls."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    client = VantinelClient(config)

    event = VantinelEvent(
        event_type="tool_call",
        client_id="test_client",
        session_id="test_session",
        agent_id="test_agent",
        tool_name="test_tool",
        tool_args_hash="abc123",
        timestamp=int(time.time() * 1000),
    )

    response = await client.send_event(event)
    assert response.decision == Decision.ALLOW

    await client.close()


def test_circuit_breaker_closed():
    """Test circuit breaker in closed state."""
    cb = CircuitBreaker(threshold=3)
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.can_attempt() is True
    assert cb.is_open() is False


def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold."""
    cb = CircuitBreaker(threshold=3)

    # Record failures
    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED

    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.is_open() is True
    assert cb.can_attempt() is False


def test_circuit_breaker_resets():
    """Test circuit breaker resets after timeout."""
    cb = CircuitBreaker(threshold=2, reset_timeout=0.1)

    # Open the circuit
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    # Wait for reset timeout
    time.sleep(0.15)

    # Should transition to half-open
    assert cb.can_attempt() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # Success should close it
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_circuit_breaker_success_resets_count():
    """Test that success resets failure count."""
    cb = CircuitBreaker(threshold=3)

    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2

    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state == CircuitBreakerState.CLOSED
