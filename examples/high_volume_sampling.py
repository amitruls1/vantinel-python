"""
High-volume sampling example for Vantinel SDK.

This example demonstrates how to use sampling to reduce overhead
in high-traffic scenarios while still getting meaningful insights.
"""

import asyncio
import random
from vantinel_sdk import VantinelMonitor, VantinelConfig


async def simulate_tool_call(tool_name: str, duration: float = 0.01):
    """Simulate a tool execution."""
    await asyncio.sleep(duration)
    return f"Result from {tool_name}"


async def main():
    """Main example."""
    print("\n=== High-Volume Sampling Example ===\n")

    # Configure with 10% sampling
    # This means only 1 in 10 events are sent to the collector
    config = VantinelConfig(
        api_key="vantinel_test_key",
        client_id="high_volume_client",
    ).with_agent_id("production_agent").with_sampling_rate(0.1).with_dry_run().with_verbose()

    async with VantinelMonitor(config) as monitor:
        print("Simulating 1000 tool calls with 10% sampling...\n")

        # Simulate high-volume tool calls
        tasks = []
        for i in range(1000):
            # Create monitoring task
            async def make_call(iteration):
                tool_name = random.choice(
                    ["search", "database_query", "api_call", "llm_request"]
                )

                execution = await monitor.watch_tool(
                    tool_name, f'{{"iteration": {iteration}}}'
                )

                try:
                    await simulate_tool_call(tool_name)
                    await execution.success()
                except Exception as e:
                    await execution.error(str(e))

            tasks.append(make_call(i))

        # Execute all calls
        await asyncio.gather(*tasks)

        # Display results
        print("\n=== Results ===")
        total_calls = await monitor.total_calls()
        print(f"Total calls instrumented: 1000")
        print(f"Calls sent to collector: {total_calls} (~10%)")
        print(f"Sampling rate: {config.sampling_rate * 100}%")
        print(f"\nThis reduced network overhead by {100 - (total_calls / 10):.0f}%")

        # Even with sampling, we get meaningful statistics
        print("\n=== Per-Tool Statistics (from sampled data) ===")
        for tool_name in ["search", "database_query", "api_call", "llm_request"]:
            stats = await monitor.tool_stats(tool_name)
            if stats:
                calls, avg_latency, errors = stats
                print(
                    f"{tool_name}: {calls} samples, {avg_latency:.2f}ms avg latency"
                )


if __name__ == "__main__":
    asyncio.run(main())
