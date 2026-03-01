"""Configuration for Vantinel SDK."""

import os
from dataclasses import dataclass
from typing import Optional

from .security import validate_collector_url


@dataclass
class VantinelConfig:
    """Configuration for the Vantinel SDK."""

    api_key: str
    client_id: str
    agent_id: str = "default_agent"
    collector_url: str = "http://localhost:8000"
    session_budget: Optional[float] = None  # USD
    timeout: float = 5.0  # seconds
    batch_size: int = 1  # Events to batch before sending (1 = no batching)
    batch_interval: float = 1.0  # seconds
    sampling_rate: float = 1.0  # 1.0 = 100%, 0.1 = 10%
    dry_run: bool = False  # If True, no network calls
    verbose: bool = False  # If True, print debug info
    shadow_mode: bool = False  # Run detection but never block; log "would have blocked" alerts
    fail_mode: str = "open"  # 'open' or 'closed'
    circuit_breaker_threshold: int = 3  # Failures before opening circuit
    circuit_breaker_reset: float = 30.0  # seconds

    @classmethod
    def from_env(cls, **kwargs) -> "VantinelConfig":
        """Create config from environment variables.

        Environment variables:
        - VANTINEL_API_KEY: API key for authentication
        - VANTINEL_CLIENT_ID: Client identifier
        - VANTINEL_AGENT_ID: Agent identifier
        - VANTINEL_COLLECTOR_URL: URL of the Vantinel Collector
        - VANTINEL_SESSION_BUDGET: Session budget in USD
        - VANTINEL_TIMEOUT: Request timeout in seconds
        - VANTINEL_DRY_RUN: Set to "true" for dry-run mode
        - VANTINEL_VERBOSE: Set to "true" for verbose logging
        """
        api_key = kwargs.get("api_key") or os.getenv("VANTINEL_API_KEY")
        client_id = kwargs.get("client_id") or os.getenv("VANTINEL_CLIENT_ID")

        if not api_key:
            raise ValueError(
                "api_key is required. Set VANTINEL_API_KEY or pass api_key parameter."
            )
        if not client_id:
            raise ValueError(
                "client_id is required. Set VANTINEL_CLIENT_ID or pass client_id parameter."
            )

        agent_id = kwargs.get("agent_id") or os.getenv("VANTINEL_AGENT_ID", "default_agent")
        collector_url = kwargs.get("collector_url") or os.getenv(
            "VANTINEL_COLLECTOR_URL", "http://localhost:8000"
        )

        session_budget = kwargs.get("session_budget")
        if session_budget is None and os.getenv("VANTINEL_SESSION_BUDGET"):
            session_budget = float(os.getenv("VANTINEL_SESSION_BUDGET"))

        timeout = float(kwargs.get("timeout", os.getenv("VANTINEL_TIMEOUT", "5.0")))
        dry_run = kwargs.get("dry_run", os.getenv("VANTINEL_DRY_RUN", "").lower() == "true")
        verbose = kwargs.get("verbose", os.getenv("VANTINEL_VERBOSE", "").lower() == "true")
        shadow_mode = kwargs.get("shadow_mode", os.getenv("VANTINEL_SHADOW_MODE", "").lower() == "true")
        fail_mode = kwargs.get("fail_mode", os.getenv("VANTINEL_FAIL_MODE", "open").lower())

        collector_url = validate_collector_url(collector_url)

        return cls(
            api_key=api_key,
            client_id=client_id,
            agent_id=agent_id,
            collector_url=collector_url,
            session_budget=session_budget,
            timeout=timeout,
            dry_run=dry_run,
            verbose=verbose,
            shadow_mode=shadow_mode,
            fail_mode=fail_mode,
            **{k: v for k, v in kwargs.items() if k not in [
                'api_key', 'client_id', 'agent_id', 'collector_url',
                'session_budget', 'timeout', 'dry_run', 'verbose', 'shadow_mode', 'fail_mode'
            ]}
        )

    def with_agent_id(self, agent_id: str) -> "VantinelConfig":
        """Set the agent ID."""
        self.agent_id = agent_id
        return self

    def with_session_budget(self, budget: float) -> "VantinelConfig":
        """Set the session budget in USD."""
        self.session_budget = budget
        return self

    def with_collector_url(self, url: str) -> "VantinelConfig":
        """Set the collector URL."""
        self.collector_url = validate_collector_url(url)
        return self

    def with_timeout(self, timeout: float) -> "VantinelConfig":
        """Set the request timeout in seconds."""
        self.timeout = timeout
        return self

    def with_batching(self, batch_size: int, batch_interval: float) -> "VantinelConfig":
        """Enable event batching."""
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        return self

    def with_sampling_rate(self, rate: float) -> "VantinelConfig":
        """Set sampling rate (0.0 to 1.0)."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")
        self.sampling_rate = rate
        return self

    def with_dry_run(self) -> "VantinelConfig":
        """Enable dry-run mode (no network calls)."""
        self.dry_run = True
        return self

    def with_verbose(self) -> "VantinelConfig":
        """Enable verbose logging."""
        self.verbose = True
        return self

    def with_circuit_breaker(
        self, threshold: int, reset_timeout: float
    ) -> "VantinelConfig":
        """Configure circuit breaker."""
        self.circuit_breaker_threshold = threshold
        self.circuit_breaker_reset = reset_timeout
        return self
