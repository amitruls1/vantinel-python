"""Tests for VantinelMonitor."""

import pytest
import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig, Decision
from vantinel_sdk.errors import ToolCallBlockedError


@pytest.mark.asyncio
async def test_monitor_creation():
    """Test creating a monitor."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    monitor = VantinelMonitor(config)
    assert monitor.session_id is not None
    assert await monitor.total_calls() == 0
    assert await monitor.session_cost() == 0.0

    await monitor.close()


@pytest.mark.asyncio
async def test_watch_tool_success():
    """Test watching a successful tool execution."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:
        execution = await monitor.watch_tool("test_tool", '{"arg": "value"}')
        assert execution is not None

        await execution.success()
        assert await monitor.total_calls() == 1


@pytest.mark.asyncio
async def test_watch_tool_error():
    """Test watching a failed tool execution."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:
        execution = await monitor.watch_tool("failing_tool", "{}")
        await execution.error("Something went wrong")

        stats = await monitor.tool_stats("failing_tool")
        assert stats is not None
        calls, avg_latency, errors = stats
        assert calls == 1
        assert errors == 1


def test_decorator_sync():
    """Test decorator with sync function."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    monitor = VantinelMonitor(config)

    @monitor.watch_tool_decorator()
    def my_sync_tool(x: int) -> int:
        return x * 2

    result = my_sync_tool(5)
    assert result == 10
    assert asyncio.run(monitor.total_calls()) == 1

    asyncio.run(monitor.close())


@pytest.mark.asyncio
async def test_decorator_async():
    """Test decorator with async function."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:

        @monitor.watch_tool_decorator()
        async def my_async_tool(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3

        result = await my_async_tool(5)
        assert result == 15
        assert await monitor.total_calls() == 1


@pytest.mark.asyncio
async def test_sampling():
    """Test sampling reduces event count."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run().with_sampling_rate(0.1)

    async with VantinelMonitor(config) as monitor:
        # Send 100 events with 10% sampling - expect ~10 to go through
        for i in range(100):
            execution = await monitor.watch_tool("test_tool", f'{{"i": {i}}}')
            await execution.success()

        # Due to randomness, this is approximate
        total = await monitor.total_calls()
        # Should be around 10, but allow range 1-30
        assert 1 <= total <= 30


@pytest.mark.asyncio
async def test_session_cost_tracking():
    """Test session cost tracking."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:
        execution1 = await monitor.watch_tool("tool1", "{}", estimated_cost=0.01)
        await execution1.success()

        execution2 = await monitor.watch_tool("tool2", "{}", estimated_cost=0.02)
        await execution2.success()

        cost = await monitor.session_cost()
        assert cost == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_tool_stats():
    """Test per-tool statistics."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:
        # Call tool multiple times
        for _ in range(5):
            execution = await monitor.watch_tool("my_tool", "{}")
            await execution.success()

        # Call with one error
        execution = await monitor.watch_tool("my_tool", "{}")
        await execution.error("test error")

        stats = await monitor.tool_stats("my_tool")
        assert stats is not None
        calls, avg_latency, errors = stats
        assert calls == 6
        assert errors == 1
        assert avg_latency > 0


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    async with VantinelMonitor(config) as monitor:
        execution = await monitor.watch_tool("test_tool", "{}")
        await execution.success()
        assert await monitor.total_calls() == 1

    # Monitor should be closed after exiting context


@pytest.mark.asyncio
async def test_custom_session_id():
    """Test providing a custom session ID."""
    config = VantinelConfig(
        api_key="test_key", client_id="test_client"
    ).with_dry_run()

    session_id = "my-custom-session-123"
    monitor = VantinelMonitor(config, session_id=session_id)
    assert monitor.session_id == session_id

    await monitor.close()


def test_config_from_env(monkeypatch):
    """Test configuration from environment variables."""
    monkeypatch.setenv("VANTINEL_API_KEY", "env_key")
    monkeypatch.setenv("VANTINEL_CLIENT_ID", "env_client")
    monkeypatch.setenv("VANTINEL_AGENT_ID", "env_agent")
    monkeypatch.setenv("VANTINEL_DRY_RUN", "true")

    config = VantinelConfig.from_env()
    assert config.api_key == "env_key"
    assert config.client_id == "env_client"
    assert config.agent_id == "env_agent"
    assert config.dry_run is True


def test_config_builder():
    """Test configuration builder pattern."""
    config = (
        VantinelConfig(api_key="test_key", client_id="test_client")
        .with_agent_id("my_agent")
        .with_session_budget(10.0)
        .with_timeout(3.0)
        .with_sampling_rate(0.5)
        .with_dry_run()
        .with_verbose()
    )

    assert config.agent_id == "my_agent"
    assert config.session_budget == 10.0
    assert config.timeout == 3.0
    assert config.sampling_rate == 0.5
    assert config.dry_run is True
    assert config.verbose is True
