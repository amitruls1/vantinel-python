"""
Decorator example for Vantinel SDK.

This example shows how to use the @watch_tool_decorator() to automatically
instrument your tool functions.
"""

import asyncio
from vantinel_sdk import VantinelMonitor, VantinelConfig


async def main():
    """Main example."""
    # Configure the SDK
    config = VantinelConfig(
        api_key="vantinel_test_key",
        client_id="demo_company",
    ).with_dry_run().with_verbose()

    monitor = VantinelMonitor(config)

    # Define tools with decorator
    @monitor.watch_tool_decorator()
    async def analyze_sentiment(text: str) -> str:
        """Analyze sentiment of text."""
        print(f"Analyzing: {text}")
        await asyncio.sleep(0.05)
        return "positive"

    @monitor.watch_tool_decorator("custom_tool_name")
    async def fetch_data(url: str) -> dict:
        """Fetch data from URL."""
        print(f"Fetching: {url}")
        await asyncio.sleep(0.1)
        return {"status": "success", "data": [1, 2, 3]}

    @monitor.watch_tool_decorator()
    def calculate_total(items: list) -> float:
        """Synchronous function example."""
        print(f"Calculating total for {len(items)} items")
        return sum(items)

    # Use the decorated tools
    print("\n=== Decorator Example ===\n")

    # Call async tools
    sentiment = await analyze_sentiment("I love this product!")
    print(f"Result: {sentiment}\n")

    data = await fetch_data("https://api.example.com/data")
    print(f"Result: {data}\n")

    # Call sync tool
    total = calculate_total([10.5, 20.3, 15.7])
    print(f"Result: ${total:.2f}\n")

    # Display statistics
    print("=== Statistics ===")
    print(f"Total calls: {await monitor.total_calls()}")

    for tool_name in ["analyze_sentiment", "custom_tool_name", "calculate_total"]:
        stats = await monitor.tool_stats(tool_name)
        if stats:
            calls, avg_latency, errors = stats
            print(f"{tool_name}: {calls} calls, {avg_latency:.2f}ms avg")

    await monitor.close()


if __name__ == "__main__":
    asyncio.run(main())
