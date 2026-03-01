# Vantinel SDK - Quick Start Guide

Get your AI agent monitored in 3 minutes.

## Prerequisites

- Python 3.9 or higher
- Vantinel Collector running (or use dry-run mode for testing)

## Installation

```bash
pip install vantinel-sdk
```

## 1. Basic Setup (30 seconds)

```python
from vantinel_sdk import VantinelMonitor, VantinelConfig

# Configure
config = VantinelConfig(
    api_key="your_api_key",
    client_id="your_company"
)

# Create monitor
monitor = VantinelMonitor(config)
```

## 2. Monitor a Tool Call (1 minute)

```python
import asyncio

async def my_tool(query: str):
    # Your tool logic here
    return f"Results for: {query}"

async def main():
    # Start watching
    execution = await monitor.watch_tool("my_tool", '{"query": "test"}')

    # Run your tool
    result = await my_tool("test")

    # Report success
    await execution.success()

    await monitor.close()

asyncio.run(main())
```

## 3. Use the Decorator (Easiest!)

```python
@monitor.watch_tool_decorator()
async def my_tool(query: str):
    return f"Results for: {query}"

# That's it! Every call is now monitored automatically
result = await my_tool("test")
```

## Test Without a Collector

```python
# Use dry-run mode for testing
config = VantinelConfig(
    api_key="test_key",
    client_id="test_client"
).with_dry_run().with_verbose()

# Now you can test without a running collector
monitor = VantinelMonitor(config)
```

## Check Statistics

```python
# Get session statistics
total_calls = await monitor.total_calls()
total_cost = await monitor.session_cost()

# Get per-tool statistics
stats = await monitor.tool_stats("my_tool")
if stats:
    calls, avg_latency, errors = stats
    print(f"Calls: {calls}, Latency: {avg_latency:.2f}ms, Errors: {errors}")
```

## Set a Budget Cap

```python
config = VantinelConfig(
    api_key="your_api_key",
    client_id="your_company"
).with_session_budget(5.0)  # USD

# Now if session cost exceeds $5, tools will be blocked
```

## Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Decorator pattern
python examples/decorator_example.py

# High-volume with sampling
python examples/high_volume_sampling.py
```

## Next Steps

1. **Set up the Collector**: Follow the deployment guide to run the Collector
2. **Configure Policies**: Use the dashboard to set budget caps and blocked tools
3. **Enable Alerts**: Configure Slack/email alerts for threats
4. **Production Deploy**: Remove dry-run mode and point to your Collector URL

## Configuration Options

```python
config = VantinelConfig(
    api_key="key",
    client_id="client"
).with_agent_id("my_agent") \
 .with_session_budget(10.0) \
 .with_collector_url("https://collector.vantinel.com") \
 .with_timeout(5.0) \
 .with_sampling_rate(0.1) \
 .with_dry_run() \
 .with_verbose()
```

## Environment Variables

```bash
export VANTINEL_API_KEY="your_key"
export VANTINEL_CLIENT_ID="your_company"
export VANTINEL_COLLECTOR_URL="http://localhost:8000"
export VANTINEL_SESSION_BUDGET="10.0"
```

```python
config = VantinelConfig.from_env()
```

## Common Patterns

### Context Manager (Recommended)

```python
async with VantinelMonitor(config) as monitor:
    execution = await monitor.watch_tool("tool", "{}")
    result = await my_tool()
    await execution.success()
# Automatically closed
```

### Error Handling

```python
from vantinel_sdk.errors import ToolCallBlockedError

try:
    execution = await monitor.watch_tool("risky_tool", "{}")
    result = await risky_tool()
    await execution.success()
except ToolCallBlockedError as e:
    print(f"Tool blocked: {e.reason}")
```

### Cost Estimation

```python
execution = await monitor.watch_tool(
    "llm_call",
    '{"prompt": "..."}',
    estimated_cost=0.002  # USD
)
```

## Troubleshooting

### "Collector unavailable" messages

This is normal! The SDK fails open - your agent keeps running even if the Collector is down.

### No events appearing in dashboard

1. Check the Collector is running: `curl http://localhost:8000/health`
2. Check dry-run mode is disabled
3. Check API key is correct
4. Enable verbose mode: `.with_verbose()`

### High network overhead

Use sampling to reduce overhead:

```python
config = VantinelConfig(
    api_key="key",
    client_id="client"
).with_sampling_rate(0.1)  # Monitor only 10% of calls
```

## Support

- [Full Documentation](README.md)
- [Examples](examples/)
- [GitHub Issues](https://github.com/vantinel/vantinel-sdk/issues)
- Email: team@vantinel.com
