"""Core monitoring functionality."""

import time
import uuid
import random
import asyncio
import functools
from typing import Optional, Dict, Any, Callable, TypeVar, Tuple, ClassVar
from collections import defaultdict

from .config import VantinelConfig
from .client import VantinelClient
from .types import VantinelEvent, VantinelResponse, Decision
from .cost import hash_tool_args, estimate_cost
from .errors import ToolCallBlockedError

T = TypeVar("T")


class ToolExecution:
    """Context manager for tracking a tool execution."""

    def __init__(
        self,
        monitor: "VantinelMonitor",
        event: VantinelEvent,
        response: VantinelResponse,
    ):
        self.monitor = monitor
        self.event = event
        self.response = response
        self.start_time = time.time()

    async def success(self, result: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """Report successful execution.

        Args:
            result: Optional result value
            metadata: Optional additional metadata
        """
        latency_ms = (time.time() - self.start_time) * 1000
        self.event.latency_ms = latency_ms
        self.event.status = "success"

        if metadata:
            self.event.metadata.update(metadata)

        # Update statistics
        self.monitor._update_stats(self.event.tool_name, latency_ms, success=True)

        if self.monitor.trace and self.event.trace_payload is not None:
             self.event.trace_payload["result"] = str(result)

        # Send completion event (fire-and-forget)
        if not self.monitor.config.dry_run:
            asyncio.create_task(self._send_completion())

    async def error(self, error_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Report failed execution.

        Args:
            error_message: Error description
            metadata: Optional additional metadata
        """
        latency_ms = (time.time() - self.start_time) * 1000
        self.event.latency_ms = latency_ms
        self.event.status = "error"
        self.event.error_message = error_message

        if metadata:
            self.event.metadata.update(metadata)

        # Update statistics
        self.monitor._update_stats(self.event.tool_name, latency_ms, success=False)

        # Send completion event (fire-and-forget)
        if not self.monitor.config.dry_run:
            asyncio.create_task(self._send_completion())

    async def _send_completion(self):
        """Send completion event to collector."""
        try:
            await self.monitor.client.send_event(self.event)
        except Exception as e:
            if self.monitor.config.verbose:
                print(f"[ERROR] Failed to send completion event: {e}")


class VantinelMonitor:
    """Main SDK class for monitoring AI agents."""

    # Singleton instance (for get_singleton pattern)
    _singleton: ClassVar[Optional["VantinelMonitor"]] = None

    def __init__(self, config: VantinelConfig, session_id: Optional[str] = None, trace: bool = False):
        """Initialize the monitor.

        Args:
            config: Configuration object
            session_id: Optional existing session ID (default: generate new UUID)
            trace: Enable full payload tracing (Opt-in Debug Mode)
        """
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.trace = trace
        self.client = VantinelClient(config)

        # Global metadata merged into every event
        self._global_metadata: Dict[str, Any] = {}

        # Statistics tracking
        self._stats_lock = asyncio.Lock()
        self._total_calls = 0
        self._session_cost = 0.0
        self._tool_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "total_latency": 0.0, "errors": 0}
        )

    @classmethod
    def get_singleton(cls, config: Optional[VantinelConfig] = None) -> "VantinelMonitor":
        """Return the singleton VantinelMonitor instance, creating it if needed.

        Args:
            config: Configuration to use when creating the singleton (only used on first call)

        Returns:
            The shared VantinelMonitor instance

        Raises:
            ValueError: If no config is provided and the singleton has not been created yet
        """
        if cls._singleton is None:
            if config is None:
                raise ValueError(
                    "config is required when creating the singleton for the first time."
                )
            cls._singleton = cls(config)
        return cls._singleton

    def set_global_metadata(self, metadata: dict) -> None:
        """Set metadata that is automatically merged into every event.

        Args:
            metadata: Key-value pairs to include in every event's metadata field
        """
        self._global_metadata.update(metadata)

    async def watch_tool(
        self,
        tool_name: str,
        tool_args: str = "{}",
        estimated_cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        skip: bool = False,
    ) -> ToolExecution:
        """Watch a tool execution.

        Args:
            tool_name: Name of the tool being called
            tool_args: JSON string of tool arguments
            estimated_cost: Estimated cost in USD (optional)
            metadata: Additional metadata (optional)
            trace_id: Optional trace ID added to event metadata as "trace_id"
            skip: If True, return a dummy execution immediately without sending an event

        Returns:
            ToolExecution context manager

        Raises:
            ToolCallBlockedError: If the tool call is blocked by policy
        """
        # If skip=True, return a dummy execution immediately
        if skip:
            return self._create_dummy_execution()

        # Apply sampling
        if self.config.sampling_rate < 1.0:
            if random.random() > self.config.sampling_rate:
                # Skip this event (return a dummy execution)
                return self._create_dummy_execution()

        # Build merged metadata: global metadata + per-call metadata
        merged_metadata: Dict[str, Any] = {}
        merged_metadata.update(self._global_metadata)
        if metadata:
            merged_metadata.update(metadata)

        # Attach trace_id if provided
        if trace_id is not None:
            merged_metadata["trace_id"] = trace_id

        # Create event
        event = VantinelEvent(
            event_type="tool_call",
            client_id=self.config.client_id,
            session_id=self.session_id,
            agent_id=self.config.agent_id,
            tool_name=tool_name,
            tool_args_hash=hash_tool_args(tool_name, tool_args),
            timestamp=int(time.time() * 1000),
            estimated_cost=estimated_cost,
            metadata=merged_metadata,
        )

        # Attach trace payload if enabled
        if self.trace:
            try:
                # We attempt to parse the args string if it looks like JSON, otherwise keep raw
                import json
                try:
                    parsed_args = json.loads(tool_args)
                except Exception:
                    parsed_args = tool_args

                event.trace_payload = {
                    "args": parsed_args
                }
            except Exception as e:
                if self.config.verbose:
                    print(f"[WARNING] Failed to attach trace payload: {e}")

        # Send to collector
        response = await self.client.send_event(event)

        # Check decision
        if response.decision == Decision.BLOCK:
            raise ToolCallBlockedError(
                f"Tool call blocked: {tool_name}",
                reason=response.message or "Policy violation",
            )

        if response.decision == Decision.REQUIRE_APPROVAL:
            if self.config.verbose:
                print(
                    f"[WARNING] Tool {tool_name} requires approval: {response.message}"
                )
            # For now, block tools requiring approval
            # In production, this would pause and wait for approval
            raise ToolCallBlockedError(
                f"Tool call requires approval: {tool_name}",
                reason=response.message or "Requires human approval",
            )

        if response.decision == Decision.WARN:
            if self.config.verbose:
                print(f"[WARNING] {response.message}")

        # Increment total calls
        async with self._stats_lock:
            self._total_calls += 1
            if estimated_cost:
                self._session_cost += estimated_cost

        return ToolExecution(self, event, response)

    def _create_dummy_execution(self) -> ToolExecution:
        """Create a dummy execution for sampled-out events."""
        dummy_event = VantinelEvent(
            event_type="tool_call",
            client_id=self.config.client_id,
            session_id=self.session_id,
            agent_id=self.config.agent_id,
            tool_name="sampled_out",
            tool_args_hash="",
            timestamp=int(time.time() * 1000),
        )
        dummy_response = VantinelResponse(decision=Decision.ALLOW)
        return ToolExecution(self, dummy_event, dummy_response)

    def watch_tool_decorator(self, tool_name: Optional[str] = None):
        """Decorator for automatically watching tool functions.

        Args:
            tool_name: Optional tool name (default: function name)

        Example:
            @monitor.watch_tool_decorator()
            def my_tool(arg1, arg2):
                return "result"

            @monitor.watch_tool_decorator("custom_name")
            async def my_async_tool():
                return "result"
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            name = tool_name or func.__name__

            if asyncio.iscoroutinefunction(func):
                # Async function
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs) -> T:
                    # Start watching
                    execution = await self.watch_tool(name, "{}")

                    try:
                        # Execute the function
                        result = await func(*args, **kwargs)
                        await execution.success(result)
                        return result
                    except Exception as e:
                        await execution.error(str(e))
                        raise

                return async_wrapper
            else:
                # Sync function - need to handle async watch_tool
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs) -> T:
                    # Get or create event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Start watching
                    execution = loop.run_until_complete(self.watch_tool(name, "{}"))

                    try:
                        # Execute the function
                        result = func(*args, **kwargs)
                        loop.run_until_complete(execution.success(result))
                        return result
                    except Exception as e:
                        loop.run_until_complete(execution.error(str(e)))
                        raise

                return sync_wrapper

        return decorator

    def wrap_openai(self, openai_client):
        """Wrap an OpenAI client to auto-monitor all chat completions.

        Usage (3 lines):
            from openai import OpenAI
            monitor = VantinelMonitor(VantinelConfig.from_env())
            client = monitor.wrap_openai(OpenAI())
        """
        original_create = openai_client.chat.completions.create

        @functools.wraps(original_create)
        async def monitored_create(**kwargs):
            model = kwargs.get("model", "unknown")
            tool_name = f"openai_chat_{model}"
            args_str = f'{{"model": "{model}", "messages_count": {len(kwargs.get("messages", []))}}}'

            is_stream = kwargs.get("stream", False)
            if is_stream:
                if "stream_options" not in kwargs:
                    kwargs["stream_options"] = {"include_usage": True}

            execution = await self.watch_tool(
                tool_name=tool_name,
                tool_args=args_str,
                metadata={"model": model, "messages_count": len(kwargs.get("messages", []))},
            )

            start = time.time()
            try:
                result = await original_create(**kwargs)

                if is_stream:
                    async def wrapper():
                        final_usage = None
                        try:
                            async for chunk in result:
                                if hasattr(chunk, "usage") and chunk.usage:
                                    final_usage = chunk.usage
                                yield chunk
                        finally:
                            latency_ms = (time.time() - start) * 1000
                            cost = None
                            if final_usage:
                                cached_tokens = 0
                                if hasattr(final_usage, "prompt_tokens_details") and final_usage.prompt_tokens_details:
                                    if isinstance(final_usage.prompt_tokens_details, dict):
                                        cached_tokens = final_usage.prompt_tokens_details.get("cached_tokens", 0)
                                    else:
                                        cached_tokens = getattr(final_usage.prompt_tokens_details, "cached_tokens", 0)
                                cost = estimate_cost(
                                    model_name=model,
                                    input_tokens=getattr(final_usage, "prompt_tokens", 0),
                                    output_tokens=getattr(final_usage, "completion_tokens", 0),
                                    cached_tokens=cached_tokens
                                )
                            # Send completion event
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            loop.create_task(execution.success(result=None, metadata={"latency_ms": latency_ms, "cost_usd": cost}))
                    return wrapper()

                latency_ms = (time.time() - start) * 1000

                # Extract cost if usage available
                cost = None
                if hasattr(result, "usage") and result.usage:
                    cached_tokens = 0
                    if hasattr(result.usage, "prompt_tokens_details") and result.usage.prompt_tokens_details:
                        if isinstance(result.usage.prompt_tokens_details, dict):
                            cached_tokens = result.usage.prompt_tokens_details.get("cached_tokens", 0)
                        else:
                            cached_tokens = getattr(result.usage.prompt_tokens_details, "cached_tokens", 0)
                            
                    cost = estimate_cost(
                        model_name=model,
                        input_tokens=getattr(result.usage, "prompt_tokens", 0),
                        output_tokens=getattr(result.usage, "completion_tokens", 0),
                        cached_tokens=cached_tokens
                    )

                await execution.success(result, metadata={"latency_ms": latency_ms, "cost_usd": cost})
                return result
            except Exception as e:
                await execution.error(str(e))
                raise

        openai_client.chat.completions.create = monitored_create
        return openai_client

    def wrap_langchain(self, llm):
        """Wrap a LangChain LLM to auto-monitor all invocations.

        Usage (3 lines):
            from langchain_openai import ChatOpenAI
            monitor = VantinelMonitor(VantinelConfig.from_env())
            llm = monitor.wrap_langchain(ChatOpenAI())
        """
        original_invoke = llm.invoke
        original_ainvoke = getattr(llm, 'ainvoke', None)

        @functools.wraps(original_invoke)
        def monitored_invoke(input, **kwargs):
            loop = asyncio.get_event_loop()
            tool_name = f"langchain_{llm.__class__.__name__}"
            execution = loop.run_until_complete(
                self.watch_tool(tool_name=tool_name, tool_args='{}')
            )
            try:
                result = original_invoke(input, **kwargs)
                loop.run_until_complete(execution.success(result))
                return result
            except Exception as e:
                loop.run_until_complete(execution.error(str(e)))
                raise

        llm.invoke = monitored_invoke

        if original_ainvoke:
            @functools.wraps(original_ainvoke)
            async def monitored_ainvoke(input, **kwargs):
                tool_name = f"langchain_{llm.__class__.__name__}"
                execution = await self.watch_tool(tool_name=tool_name, tool_args='{}')
                try:
                    result = await original_ainvoke(input, **kwargs)
                    await execution.success(result)
                    return result
                except Exception as e:
                    await execution.error(str(e))
                    raise
            llm.ainvoke = monitored_ainvoke

        return llm

    async def ping(self) -> dict:
        """Check connectivity to the collector. Returns {"ok": bool, "latency_ms": float}."""
        start = time.time()
        try:
            response = await self.client.client.get(
                f"{self.config.collector_url}/health",
                timeout=5.0
            )
            latency_ms = (time.time() - start) * 1000
            return {"ok": response.status_code == 200, "latency_ms": latency_ms}
        except Exception:
            return {"ok": False, "latency_ms": (time.time() - start) * 1000}

    async def capture_error(
        self,
        tool_name: str,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an error event to the collector.

        Args:
            tool_name: Name of the tool that errored
            error: The exception that occurred
            metadata: Optional additional metadata
        """
        merged_metadata: Dict[str, Any] = {}
        merged_metadata.update(self._global_metadata)
        if metadata:
            merged_metadata.update(metadata)

        event = VantinelEvent(
            event_type="tool_error",
            client_id=self.config.client_id,
            session_id=self.session_id,
            agent_id=self.config.agent_id,
            tool_name=tool_name,
            tool_args_hash="",
            timestamp=int(time.time() * 1000),
            status="error",
            error_message=str(error),
            metadata=merged_metadata,
        )

        if not self.config.dry_run:
            try:
                await self.client.send_event(event)
            except Exception as e:
                if self.config.verbose:
                    print(f"[ERROR] Failed to send error event: {e}")

    def _update_stats(self, tool_name: str, latency_ms: float, success: bool):
        """Update internal statistics."""
        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        stats["total_latency"] += latency_ms
        if not success:
            stats["errors"] += 1

    async def total_calls(self) -> int:
        """Get total number of tool calls in this session."""
        async with self._stats_lock:
            return self._total_calls

    async def session_cost(self) -> float:
        """Get total estimated cost for this session in USD."""
        async with self._stats_lock:
            return self._session_cost

    async def tool_stats(
        self, tool_name: str
    ) -> Optional[Tuple[int, float, int]]:
        """Get statistics for a specific tool.

        Returns:
            Tuple of (call_count, avg_latency_ms, error_count) or None
        """
        async with self._stats_lock:
            if tool_name not in self._tool_stats:
                return None

            stats = self._tool_stats[tool_name]
            calls = stats["calls"]
            avg_latency = stats["total_latency"] / calls if calls > 0 else 0.0
            errors = stats["errors"]

            return (calls, avg_latency, errors)

    async def close(self):
        """Close the monitor and cleanup resources."""
        await self.client.close()

    def __enter__(self):
        """Context manager support (sync)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup (sync)."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.close())
        except Exception:
            pass

    async def __aenter__(self):
        """Async context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.close()


# Convenience function for one-off tool watching
async def watch_tool_fn(
    monitor: VantinelMonitor,
    tool_name: str,
    tool_args: str,
    func: Callable[[], T],
) -> T:
    """Helper function to watch a single tool execution.

    Args:
        monitor: VantinelMonitor instance
        tool_name: Name of the tool
        tool_args: JSON string of arguments
        func: Function to execute

    Returns:
        Result from the function

    Example:
        result = await watch_tool_fn(
            monitor,
            "my_tool",
            '{"arg": "value"}',
            lambda: my_tool("value")
        )
    """
    execution = await monitor.watch_tool(tool_name, tool_args)

    try:
        if asyncio.iscoroutinefunction(func):
            result = await func()
        else:
            result = func()
        await execution.success(result)
        return result
    except Exception as e:
        await execution.error(str(e))
        raise
