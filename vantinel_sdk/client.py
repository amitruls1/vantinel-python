"""HTTP client for communicating with the Vantinel Collector."""

import json
import httpx
import asyncio
import time
from typing import Optional
from enum import Enum

from .config import VantinelConfig
from .types import VantinelEvent, VantinelResponse, Decision
from .errors import CollectorUnavailableError, CircuitBreakerOpenError
from .security import hmac_sign, validate_collector_url, generate_nonce, redact_api_key


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Collector unavailable
    HALF_OPEN = "half_open"  # Testing if collector recovered


class CircuitBreaker:
    """Circuit breaker for handling collector unavailability."""

    def __init__(self, threshold: int = 3, reset_timeout: float = 30.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED

    def record_success(self):
        """Record a successful request."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = CircuitBreakerState.OPEN

    def can_attempt(self) -> bool:
        """Check if we can attempt a request."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to try again
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.reset_timeout
            ):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False

        # HALF_OPEN state
        return True

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN


class VantinelClient:
    """Async HTTP client for the Vantinel Collector."""

    def __init__(self, config: VantinelConfig):
        self.config = config
        self.config.collector_url = validate_collector_url(config.collector_url)
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset,
        )

    async def send_event(self, event: VantinelEvent) -> VantinelResponse:
        """Send a telemetry event to the collector.

        Args:
            event: The event to send

        Returns:
            Response from the collector with policy decision

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            CollectorUnavailableError: If collector cannot be reached
        """
        # Dry-run mode: always allow
        if self.config.dry_run:
            if self.config.verbose:
                print(f"[DRY-RUN] Would send event: {event.tool_name}")
            return VantinelResponse(decision=Decision.ALLOW)

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            if self.config.verbose:
                print(
                    f"[CIRCUIT-BREAKER] Open - allowing {event.tool_name} without check"
                )
            # When circuit breaker is open, allow all operations
            return VantinelResponse(decision=Decision.ALLOW)

        try:
            url = f"{self.config.collector_url}/v1/events"
            body = json.dumps(event.to_dict())
            timestamp = int(time.time() * 1000)
            nonce = generate_nonce()
            signature = hmac_sign(self.config.api_key, timestamp, body)

            headers = {
                "Content-Type": "application/json",
                "X-Vantinel-API-Key": self.config.api_key,
                "X-Vantinel-Signature": signature,
                "X-Vantinel-Timestamp": str(timestamp),
                "X-Vantinel-Nonce": nonce,
                "X-Vantinel-Client": self.config.client_id,
            }

            response = await self.client.post(
                url, content=body.encode("utf-8"), headers=headers
            )

            if response.status_code == 200:
                self.circuit_breaker.record_success()
                data = response.json()
                vantinel_response = VantinelResponse.from_dict(data)

                # Shadow mode: run detection but never actually block
                if self.config.shadow_mode and vantinel_response.decision in (
                    Decision.BLOCK, Decision.REQUIRE_APPROVAL
                ):
                    cost = getattr(event, "estimated_cost", None)
                    cost_str = f"${cost:.2f}" if cost is not None else "unknown"
                    print(
                        f"[Vantinel Shadow] Would have blocked: {event.tool_name}. "
                        f"Estimated savings: {cost_str}"
                    )
                    vantinel_response.decision = Decision.ALLOW

                return vantinel_response
            else:
                # Non-200 response - treat as failure
                self.circuit_breaker.record_failure()
                if self.config.verbose:
                    print(
                        f"[ERROR] Collector returned {response.status_code}: {response.text}"
                    )
                if self.config.fail_mode == "closed":
                    return VantinelResponse(
                        decision=Decision.BLOCK,
                        message=f"Collector error {response.status_code} and fail_mode is closed",
                    )
                # Fail open - allow the operation
                return VantinelResponse(
                    decision=Decision.ALLOW,
                    message=f"Collector error: {response.status_code}",
                )

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            self.circuit_breaker.record_failure()
            if self.config.verbose:
                redacted_key = redact_api_key(self.config.api_key)
                error_msg = str(e).replace(self.config.api_key, redacted_key)
                print(f"[ERROR] Failed to reach collector: {error_msg}")
            if self.config.fail_mode == "closed":
                return VantinelResponse(
                    decision=Decision.BLOCK, message=f"Collector unavailable and fail_mode is closed: {e}"
                )
            # Fail open - allow the operation
            return VantinelResponse(
                decision=Decision.ALLOW, message=f"Collector unavailable: {e}"
            )

        except Exception as e:
            self.circuit_breaker.record_failure()
            if self.config.verbose:
                redacted_key = redact_api_key(self.config.api_key)
                error_msg = str(e).replace(self.config.api_key, redacted_key)
                print(f"[ERROR] Unexpected error: {error_msg}")
            if self.config.fail_mode == "closed":
                return VantinelResponse(
                    decision=Decision.BLOCK, message=f"Unexpected error and fail_mode is closed: {e}"
                )
            # Fail open
            return VantinelResponse(
                decision=Decision.ALLOW, message=f"Error: {e}"
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Try to close the client if event loop is available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup
                loop.create_task(self.client.aclose())
            else:
                loop.run_until_complete(self.client.aclose())
        except Exception:
            # Ignore errors during cleanup
            pass
