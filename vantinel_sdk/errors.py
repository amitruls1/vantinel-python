"""Error types for Vantinel SDK."""


class VantinelError(Exception):
    """Base exception for all Vantinel SDK errors."""

    pass


class ToolCallBlockedError(VantinelError):
    """Raised when a tool call is blocked by policy."""

    def __init__(self, message: str, reason: str = None):
        super().__init__(message)
        self.reason = reason


class CollectorUnavailableError(VantinelError):
    """Raised when the Vantinel Collector is unavailable."""

    pass


class ConfigurationError(VantinelError):
    """Raised when configuration is invalid."""

    pass


class CircuitBreakerOpenError(VantinelError):
    """Raised when circuit breaker is open (collector down)."""

    pass
