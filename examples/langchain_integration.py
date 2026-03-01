"""
LangChain integration example for Vantinel SDK.

This example shows how to monitor a LangChain agent with Vantinel.
Note: This is a conceptual example - adjust imports based on your LangChain version.
"""

import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig

# Note: These imports are examples - adjust for your LangChain version
# from langchain.agents import initialize_agent, Tool
# from langchain.llms import OpenAI


async def main():
    """Main example."""
    print("\n=== LangChain + Vantinel Integration ===\n")

    # Configure Vantinel
    vantinel_config = VantinelConfig(
        api_key="vantinel_test_key",
        client_id="langchain_demo",
    ).with_agent_id("langchain_agent").with_session_budget(10.0).with_dry_run().with_verbose()

    async with VantinelMonitor(vantinel_config) as monitor:

        # Define tools wrapped with Vantinel monitoring
        @monitor.watch_tool_decorator("langchain_search")
        async def search_tool(query: str) -> str:
            """Search tool with Vantinel monitoring."""
            print(f"Searching for: {query}")
            await asyncio.sleep(0.1)
            return f"Search results for: {query}"

        @monitor.watch_tool_decorator("langchain_calculator")
        def calculator_tool(expression: str) -> str:
            """Calculator tool with Vantinel monitoring."""
            print(f"Calculating: {expression}")
            try:
                result = eval(expression)  # In production, use a safe eval
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        # Simulate agent workflow
        print("Agent starting workflow...\n")

        # Step 1: Search
        search_result = await search_tool("latest AI news")
        print(f"Search result: {search_result}\n")

        # Step 2: Calculate
        calc_result = calculator_tool("25 * 4 + 10")
        print(f"Calculation result: {calc_result}\n")

        # Step 3: Another search
        search_result2 = await search_tool("weather forecast")
        print(f"Search result: {search_result2}\n")

        # Display monitoring results
        print("=== Vantinel Monitoring Results ===")
        print(f"Total tool calls: {await monitor.total_calls()}")
        print(f"Session cost: ${await monitor.session_cost():.4f}")

        # Per-tool breakdown
        for tool_name in ["langchain_search", "langchain_calculator"]:
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
    print("Note: This is a conceptual example.")
    print("Adjust LangChain imports based on your version.\n")
    asyncio.run(main())
