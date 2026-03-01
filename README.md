# Vantinel SDK for Python

[![PyPI version](https://img.shields.io/pypi/v/vantinel-sdk.svg)](https://pypi.org/project/vantinel-sdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/vantinel-sdk.svg)](https://pypi.org/project/vantinel-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight observability and guardrails SDK for AI agents. Monitor, protect, and optimize your autonomous AI systems in real-time.

## Features

- **🚨 Zombie Loop Detection** - Automatically detect and prevent infinite loops and retry storms
- **💰 Cost Control** - Real-time budget tracking and forecasting to prevent runaway costs
- **🛡️ Safety Guardrails** - Block dangerous operations and require human approval
- **📊 Performance Monitoring** - Track latency, error rates, and usage patterns
- **⚡ Zero Overhead** - Async fire-and-forget telemetry with circuit breaker
- **🎯 Simple API** - 3 lines of code to get started

## Quick Start

### Installation

```bash
pip install vantinel-sdk
```

### Basic Usage

```python
import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig

async def main():
    # Configure the SDK
    config = VantinelConfig(
        api_key="vantinel_abc123",
        client_id="my_company"
    ).with_session_budget(5.0)

    # Create a monitor
    async with VantinelMonitor(config) as monitor:
        # Watch a tool execution
        execution = await monitor.watch_tool(
            "search_database",
            '{"query": "users"}'
        )

        # Execute your tool
        result = await search_database("users")

        # Report the result
        await execution.success()

asyncio.run(main())
```

### Using the Decorator

The easiest way to monitor your tools:

```python
from vantinel_sdk import VantinelMonitor, VantinelConfig

config = VantinelConfig(api_key="key", client_id="company")
monitor = VantinelMonitor(config)

@monitor.watch_tool_decorator()
async def search_database(query: str):
    # Your tool logic
    return results

# Now every call is automatically monitored!
result = await search_database("users")
```

## Architecture

The SDK sends telemetry events to the Vantinel Collector, which runs real-time anomaly detection algorithms:

```
┌─────────────────┐
│   Your Agent    │
│    (Python)     │
└────────┬────────┘
         │ Vantinel SDK
         ▼
┌─────────────────┐      ┌──────────────┐
│    Collector    │─────▶│  Dashboard   │
│  (Algorithms)   │      │  (Metrics)   │
└─────────────────┘      └──────────────┘
```

### What Gets Sent

The SDK only sends **metadata**, never actual data:

```json
{
  "event": "tool_call",
  "client_id": "your_company",
  "session_id": "sess_abc123",
  "tool_name": "search_database",
  "tool_args_hash": "md5:a3f8b9c2...",
  "timestamp": 1675432100000,
  "latency_ms": 45,
  "estimated_cost": 0.002
}
```

We **never** send:
- Request/response bodies
- User queries or prompts
- Tool arguments (only MD5 hash)
- Sensitive data

## Configuration

### From Environment Variables

```bash
export VANTINEL_API_KEY="your_api_key"
export VANTINEL_CLIENT_ID="your_company"
export VANTINEL_AGENT_ID="my_agent"
export VANTINEL_SESSION_BUDGET="10.0"
```

```python
from vantinel_sdk import VantinelConfig

config = VantinelConfig.from_env()
monitor = VantinelMonitor(config)
```

### Builder Pattern

```python
config = VantinelConfig(
    api_key="test_key",
    client_id="test_client"
).with_agent_id("my_agent") \
 .with_session_budget(10.0) \
 .with_collector_url("https://collector.vantinel.com") \
 .with_timeout(5.0) \
 .with_batching(100, 1.0) \
 .with_sampling_rate(0.1)  # Sample 10% of events
```

## Advanced Features

### Sampling for High-Volume Scenarios

```python
# Monitor only 10% of traffic to reduce overhead
config = VantinelConfig(
    api_key="key",
    client_id="client"
).with_sampling_rate(0.1)
```

### Session Management

```python
# Create a new session
monitor = VantinelMonitor(config)
session_id = monitor.session_id

# Resume an existing session
monitor = VantinelMonitor(config, session_id="existing-session-id")
```

### Statistics

```python
# Get session statistics
total_calls = await monitor.total_calls()
session_cost = await monitor.session_cost()

# Get per-tool statistics
stats = await monitor.tool_stats("my_tool")
if stats:
    calls, avg_latency, errors = stats
    print(f"Tool called {calls} times")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Errors: {errors}")
```

## Testing

For local development, use dry-run mode:

```python
config = VantinelConfig(
    api_key="test_key",
    client_id="test_client"
).with_dry_run() \
 .with_verbose()  # Print debug info
```

## Examples

### Basic Usage
```bash
python examples/basic_usage.py
```

Demonstrates:
- Creating a monitor
- Watching tool executions
- Reporting success and errors
- Checking statistics

### Decorator Pattern
```bash
python examples/decorator_example.py
```

Shows:
- Using the `@watch_tool_decorator()`
- Sync and async function support
- Custom tool naming

### LangChain Integration
```bash
python examples/langchain_integration.py
```

Example of monitoring a LangChain agent.

### High-Volume Sampling
```bash
python examples/high_volume_sampling.py
```

Demonstrates sampling for reducing overhead in high-traffic scenarios.

## Circuit Breaker

The SDK includes a built-in circuit breaker that fails gracefully if the Collector is unavailable:

```python
config = VantinelConfig(
    api_key="key",
    client_id="client"
).with_circuit_breaker(
    threshold=3,      # Open after 3 failures
    reset_timeout=30  # Reset after 30 seconds
)
```

States:
- **Closed**: Normal operation
- **Open**: Collector unavailable, allow all operations
- **Half-Open**: Testing if Collector recovered

## Error Handling

The SDK fails gracefully by default. If the Collector is unreachable, tool calls are allowed to proceed:

```python
from vantinel_sdk.errors import ToolCallBlockedError

try:
    execution = await monitor.watch_tool("my_tool", "{}")
    result = await my_tool()
    await execution.success()
except ToolCallBlockedError as e:
    # Tool call blocked by policy
    print(f"Blocked: {e}")
```

## Performance

- **Latency**: < 1ms overhead per tool call (async fire-and-forget)
- **Throughput**: 10,000+ events/sec per monitor
- **Memory**: ~1MB per monitor instance
- **Network**: ~200 bytes per event

## Development

### Install in Development Mode

```bash
cd sdk/python
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Format Code

```bash
black vantinel_sdk/ tests/ examples/
ruff check vantinel_sdk/ tests/ examples/
```

### Type Checking

```bash
mypy vantinel_sdk/
```

## Requirements

- Python 3.9+
- httpx >= 0.24.0
- tiktoken >= 0.5.0 (for token counting)

## OpenAI & LangChain Wrappers

### OpenAI (3 lines)

```python
from openai import AsyncOpenAI
from vantinel_sdk import VantinelMonitor, VantinelConfig

monitor = VantinelMonitor(VantinelConfig.from_env())
client = monitor.wrap_openai(AsyncOpenAI())
# All client.chat.completions.create() calls are now monitored with auto cost tracking
```

### LangChain (3 lines)

```python
from langchain_openai import ChatOpenAI

monitor = VantinelMonitor(VantinelConfig.from_env())
llm = monitor.wrap_langchain(ChatOpenAI())
result = llm.invoke("What is 2+2?")  # invoke and ainvoke are both monitored
```

## Error Capture

```python
from vantinel_sdk import VantinelMonitor, VantinelConfig

monitor = VantinelMonitor(VantinelConfig.from_env())

try:
    await my_tool()
except Exception as e:
    await monitor.capture_error("my_tool", e, metadata={"retry": 1})
    raise
```

## Roadmap

- [ ] Automatic LlamaIndex integration
- [ ] Cost estimation for all popular models
- [ ] Local-first mode with SQLite storage
- [ ] Java SDK

## Contributing

Contributions are welcome! Please open an issue or PR.

## License

Licensed under the MIT license. See [LICENSE](LICENSE) for details.

## Support

- GitHub Issues: https://github.com/vantinel/vantinel-sdk/issues
- Email: team@vantinel.com
- Discord: https://discord.gg/vantinel

## Resources

- [Documentation](https://docs.vantinel.com)
- [Examples](examples/)
- [Dashboard](https://dashboard.vantinel.com)
- [Project Documentation](../../CLAUDE.md)
