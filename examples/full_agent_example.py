"""
Complete AI agent example with Vantinel SDK.

This example demonstrates a realistic AI agent with:
- Multiple tools (LLM, database, API, email)
- Error handling
- Cost tracking
- Statistics reporting
"""

import asyncio
import random
from vantinel_sdk import VantinelMonitor, VantinelConfig
from vantinel_sdk.errors import ToolCallBlockedError


class AIAgent:
    """Example AI agent with multiple tools."""

    def __init__(self, monitor: VantinelMonitor):
        self.monitor = monitor

    @property
    def tools(self):
        """Return decorated tools."""
        # Decorate tools at runtime
        return {
            "query_database": self.monitor.watch_tool_decorator()(
                self.query_database
            ),
            "call_llm": self.monitor.watch_tool_decorator()(self.call_llm),
            "send_notification": self.monitor.watch_tool_decorator()(
                self.send_notification
            ),
            "fetch_api_data": self.monitor.watch_tool_decorator()(
                self.fetch_api_data
            ),
        }

    async def query_database(self, query: str) -> list:
        """Query the database."""
        print(f"  📊 Querying database: {query}")
        await asyncio.sleep(0.05)  # Simulate DB latency

        # Simulate occasional errors
        if random.random() < 0.1:
            raise Exception("Database connection timeout")

        return [
            {"id": 1, "name": "Result 1"},
            {"id": 2, "name": "Result 2"},
        ]

    async def call_llm(self, prompt: str, model: str = "gpt-4") -> str:
        """Call an LLM."""
        print(f"  🤖 Calling LLM ({model}): {prompt[:50]}...")
        await asyncio.sleep(0.2)  # Simulate LLM latency

        return f"LLM response to: {prompt[:30]}..."

    async def send_notification(self, recipient: str, message: str) -> bool:
        """Send a notification."""
        print(f"  📧 Sending notification to {recipient}")
        await asyncio.sleep(0.03)
        return True

    async def fetch_api_data(self, endpoint: str) -> dict:
        """Fetch data from external API."""
        print(f"  🌐 Fetching from API: {endpoint}")
        await asyncio.sleep(0.1)
        return {"status": "success", "data": [1, 2, 3]}

    async def run_workflow(self):
        """Execute a complete agent workflow."""
        print("\n🚀 Starting AI agent workflow...\n")

        try:
            # Step 1: Fetch initial data
            print("Step 1: Fetch data from API")
            data = await self.tools["fetch_api_data"]("/api/v1/users")
            print(f"  ✓ Fetched: {data}\n")

            # Step 2: Query database
            print("Step 2: Query database")
            db_results = await self.tools["query_database"](
                "SELECT * FROM orders WHERE status = 'pending'"
            )
            print(f"  ✓ Found {len(db_results)} results\n")

            # Step 3: Process with LLM
            print("Step 3: Process with LLM")
            llm_response = await self.tools["call_llm"](
                f"Analyze these results: {db_results}", model="gpt-4"
            )
            print(f"  ✓ Analysis complete\n")

            # Step 4: Send notification
            print("Step 4: Send notification")
            await self.tools["send_notification"](
                "admin@example.com", "Workflow completed"
            )
            print(f"  ✓ Notification sent\n")

            print("✅ Workflow completed successfully!")

        except ToolCallBlockedError as e:
            print(f"\n🚫 Tool call blocked: {e}")
            print(f"   Reason: {e.reason}")

        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")


async def main():
    """Main execution."""
    print("=" * 60)
    print("  Vantinel SDK - Complete AI Agent Example")
    print("=" * 60)

    # Configure Vantinel
    config = (
        VantinelConfig(api_key="demo_key", client_id="demo_company")
        .with_agent_id("example_agent")
        .with_session_budget(10.0)
        .with_dry_run()
        .with_verbose()
    )

    # Create monitor
    async with VantinelMonitor(config) as monitor:
        print(f"\n📊 Session ID: {monitor.session_id}")
        print(f"💰 Budget Cap: $10.00\n")

        # Create and run agent
        agent = AIAgent(monitor)
        await agent.run_workflow()

        # Display comprehensive statistics
        print("\n" + "=" * 60)
        print("  Session Statistics")
        print("=" * 60)

        total_calls = await monitor.total_calls()
        session_cost = await monitor.session_cost()

        print(f"\n📈 Overview:")
        print(f"   Total tool calls: {total_calls}")
        print(f"   Session cost: ${session_cost:.4f}")
        print(f"   Budget remaining: ${10.0 - session_cost:.4f}")
        print(f"   Budget used: {(session_cost / 10.0) * 100:.1f}%")

        print(f"\n📊 Per-Tool Breakdown:")
        print(f"   {'Tool Name':<25} {'Calls':<8} {'Avg Latency':<15} {'Errors'}")
        print(f"   {'-' * 25} {'-' * 8} {'-' * 15} {'-' * 8}")

        for tool_name in [
            "query_database",
            "call_llm",
            "send_notification",
            "fetch_api_data",
        ]:
            stats = await monitor.tool_stats(tool_name)
            if stats:
                calls, avg_latency, errors = stats
                error_rate = (errors / calls * 100) if calls > 0 else 0
                print(
                    f"   {tool_name:<25} {calls:<8} {avg_latency:>8.2f}ms      "
                    f"{errors} ({error_rate:.0f}%)"
                )

        print("\n" + "=" * 60)
        print("\n✨ Demo complete! Check your Vantinel dashboard for insights.\n")


if __name__ == "__main__":
    asyncio.run(main())
