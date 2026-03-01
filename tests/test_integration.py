"""Integration tests for Vantinel SDK with real collector.

These tests require the Vantinel Collector to be running on localhost:8000.
Run with: pytest tests/test_integration.py

To skip integration tests: pytest -m "not integration"
"""

import pytest
import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig, Decision
from vantinel_sdk.errors import ToolCallBlockedError


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_with_collector():
    """Test full integration with running collector.

    Prerequisites:
    - Collector running on http://localhost:8000
    - Database and Redis configured
    """
    config = VantinelConfig(
        api_key="test_integration_key",
        client_id="test_integration_client",
    ).with_collector_url("http://localhost:8000")

    async with VantinelMonitor(config) as monitor:
        # Send a test event
        execution = await monitor.watch_tool(
            "integration_test_tool", '{"test": "data"}'
        )

        # Simulate work
        await asyncio.sleep(0.1)

        # Report success
        await execution.success()

        # Verify stats
        assert await monitor.total_calls() == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_zombie_loop_detection():
    """Test that zombie loop detection blocks duplicate calls.

    This test sends identical tool calls rapidly to trigger the
    zombie loop detector.
    """
    config = VantinelConfig(
        api_key="test_zombie_key",
        client_id="test_zombie_client",
    ).with_collector_url("http://localhost:8000")

    async with VantinelMonitor(config) as monitor:
        # Send identical calls rapidly
        for i in range(5):
            try:
                execution = await monitor.watch_tool(
                    "zombie_test_tool",
                    '{"same": "args"}',  # Same args every time
                )
                await execution.success()
                await asyncio.sleep(0.01)
            except ToolCallBlockedError as e:
                # Expected after 3 identical calls
                print(f"Blocked on call {i + 1}: {e}")
                assert i >= 3  # Should be blocked by 4th call
                break


@pytest.mark.integration
@pytest.mark.asyncio
async def test_budget_cap_enforcement():
    """Test that budget cap blocks expensive calls."""
    config = VantinelConfig(
        api_key="test_budget_key",
        client_id="test_budget_client",
    ).with_collector_url("http://localhost:8000").with_session_budget(0.10)  # Very low budget

    async with VantinelMonitor(config) as monitor:
        try:
            # Send expensive calls until budget exceeded
            for i in range(20):
                execution = await monitor.watch_tool(
                    "expensive_tool", f'{{"call": {i}}}', estimated_cost=0.01
                )
                await execution.success()
                await asyncio.sleep(0.01)
        except ToolCallBlockedError as e:
            # Expected when budget exceeded
            print(f"Budget exceeded: {e}")
            cost = await monitor.session_cost()
            assert cost >= 0.10


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dangerous_tool_blocking():
    """Test that dangerous tools are blocked or require approval."""
    config = VantinelConfig(
        api_key="test_danger_key",
        client_id="test_danger_client",
    ).with_collector_url("http://localhost:8000")

    async with VantinelMonitor(config) as monitor:
        # Try to call a dangerous tool
        with pytest.raises(ToolCallBlockedError):
            execution = await monitor.watch_tool(
                "delete_all_users",  # Should trigger dangerous pattern
                '{"confirm": true}',
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_circuit_breaker_with_unreachable_collector():
    """Test circuit breaker when collector is unreachable."""
    config = VantinelConfig(
        api_key="test_cb_key",
        client_id="test_cb_client",
    ).with_collector_url(
        "http://localhost:9999"  # Wrong port
    ).with_circuit_breaker(
        threshold=2, reset_timeout=1.0
    )

    async with VantinelMonitor(config) as monitor:
        # These should succeed (fail open) even though collector is down
        for i in range(5):
            execution = await monitor.watch_tool("test_tool", f'{{"i": {i}}}')
            await execution.success()

        # Circuit breaker should be open
        assert monitor.client.circuit_breaker.is_open()

        # But calls still go through (fail open)
        assert await monitor.total_calls() == 5


if __name__ == "__main__":
    print("Integration tests require the Vantinel Collector to be running.")
    print("Start the collector with: cd ../../gateway && cargo run")
    print("\nRun tests with: pytest tests/test_integration.py")
