"""
Basic usage example for Vantinel SDK.

This example demonstrates the core functionality:
- Creating a monitor
- Watching tool executions
- Reporting success and errors
- Checking statistics
"""

import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig


async def search_database(query: str) -> list:
    """Simulate a database search tool."""
    print(f"Searching database for: {query}")
    await asyncio.sleep(0.1)  # Simulate API latency
    return [{"id": 1, "name": "Result 1"}, {"id": 2, "name": "Result 2"}]


async def send_email(to: str, subject: str) -> bool:
    """Simulate an email sending tool."""
    print(f"Sending email to {to}: {subject}")
    await asyncio.sleep(0.05)
    return True


async def main():
    """Main example."""
    # Configure the SDK
    config = VantinelConfig(
        api_key="vantinel_test_key",
        client_id="demo_company",
    ).with_agent_id("customer_support_bot").with_session_budget(5.0).with_dry_run().with_verbose()

    # Create a monitor
    async with VantinelMonitor(config) as monitor:
        print(f"\n=== Vantinel SDK Demo ===")
        print(f"Session ID: {monitor.session_id}\n")

        # Example 1: Successful tool execution
        print("Example 1: Successful database search")
        execution = await monitor.watch_tool(
            "search_database",
            '{"query": "recent orders"}',
            estimated_cost=0.001,
        )

        try:
            result = await search_database("recent orders")
            await execution.success(result)
            print(f"✓ Success: Found {len(result)} results\n")
        except Exception as e:
            await execution.error(str(e))
            print(f"✗ Error: {e}\n")

        # Example 2: Another successful tool
        print("Example 2: Sending email")
        execution = await monitor.watch_tool(
            "send_email",
            '{"to": "customer@example.com", "subject": "Order confirmation"}',
            estimated_cost=0.0005,
        )

        try:
            result = await send_email("customer@example.com", "Order confirmation")
            await execution.success(result)
            print(f"✓ Success: Email sent\n")
        except Exception as e:
            await execution.error(str(e))
            print(f"✗ Error: {e}\n")

        # Example 3: Simulated error
        print("Example 3: Failed tool execution")
        execution = await monitor.watch_tool("failing_tool", "{}")

        try:
            raise Exception("Simulated failure")
        except Exception as e:
            await execution.error(str(e))
            print(f"✗ Error: {e}\n")

        # Display statistics
        print("=== Session Statistics ===")
        print(f"Total calls: {await monitor.total_calls()}")
        print(f"Total cost: ${await monitor.session_cost():.4f}")

        # Per-tool statistics
        for tool_name in ["search_database", "send_email", "failing_tool"]:
            stats = await monitor.tool_stats(tool_name)
            if stats:
                calls, avg_latency, errors = stats
                print(
                    f"\n{tool_name}:"
                    f"\n  Calls: {calls}"
                    f"\n  Avg latency: {avg_latency:.2f}ms"
                    f"\n  Errors: {errors}"
                )


if __name__ == "__main__":
    asyncio.run(main())
