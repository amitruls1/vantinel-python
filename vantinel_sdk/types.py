"""Type definitions for Vantinel SDK."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime


class Decision(str, Enum):
    """Policy enforcement decision from the Collector."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    WARN = "warn"


@dataclass
class VantinelEvent:
    """Telemetry event sent to the Vantinel Collector."""

    event_type: str
    client_id: str
    session_id: str
    agent_id: str
    tool_name: str
    tool_args_hash: str
    timestamp: int  # Unix timestamp in milliseconds
    latency_ms: Optional[float] = None
    estimated_cost: Optional[float] = None
    status: str = "pending"  # pending, success, error
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class VantinelResponse:
    """Response from the Vantinel Collector."""

    decision: Decision
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VantinelResponse":
        """Create from dictionary."""
        return cls(
            decision=Decision(data.get("decision", "allow")),
            message=data.get("message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModelPricing:
    """Pricing information for LLM models."""

    model_name: str
    input_cost_per_1k: float  # USD per 1k tokens
    output_cost_per_1k: float  # USD per 1k tokens
    cache_read_cost_per_1k: Optional[float] = None

    def calculate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Calculate total cost for token usage."""
        regular_input = max(0, input_tokens - cached_tokens)
        cache_price = self.cache_read_cost_per_1k if self.cache_read_cost_per_1k is not None else self.input_cost_per_1k * 0.5
        input_cost = (regular_input / 1000.0) * self.input_cost_per_1k + (cached_tokens / 1000.0) * cache_price
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k
        return input_cost + output_cost


# Standard model pricing (as of 2026)
MODEL_PRICING = {
    # OpenAI 2026 Models
    "gpt-5.2": ModelPricing("gpt-5.2", 0.00175, 0.014),
    "gpt-5.2-pro": ModelPricing("gpt-5.2-pro", 0.021, 0.168),
    "gpt-5-mini": ModelPricing("gpt-5-mini", 0.00025, 0.002),
    
    # OpenAI Legacy
    "gpt-4o": ModelPricing("gpt-4o", 0.0025, 0.010),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.00015, 0.0006),
    "o1": ModelPricing("o1", 0.015, 0.060),
    "o3-mini": ModelPricing("o3-mini", 0.0011, 0.0044),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03),
    "gpt-4": ModelPricing("gpt-4", 0.03, 0.06),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),

    # Anthropic 2026 Models
    "claude-4.6-opus": ModelPricing("claude-4.6-opus", 0.005, 0.025),
    "claude-4.6-sonnet": ModelPricing("claude-4.6-sonnet", 0.003, 0.015),
    "claude-4.5-opus": ModelPricing("claude-4.5-opus", 0.005, 0.025),
    "claude-4.5-sonnet": ModelPricing("claude-4.5-sonnet", 0.003, 0.015),
    "claude-4.5-haiku": ModelPricing("claude-4.5-haiku", 0.001, 0.005),

    # Anthropic Legacy
    "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", 0.003, 0.015),
    "claude-3-5-haiku-20241022": ModelPricing("claude-3-5-haiku-20241022", 0.0008, 0.004),
    "claude-3-opus": ModelPricing("claude-3-opus", 0.015, 0.075),

    # Anthropic Future/Pre-release names keeping alias for backwards compat
    "claude-opus-4-6": ModelPricing("claude-opus-4-6", 0.005, 0.025),
    "claude-sonnet-4-6": ModelPricing("claude-sonnet-4-6", 0.003, 0.015),
    "claude-haiku-4-5": ModelPricing("claude-haiku-4-5", 0.001, 0.005),
    "claude-haiku-4-5-20251001": ModelPricing("claude-haiku-4-5-20251001", 0.001, 0.005),
    "claude-sonnet-4-5": ModelPricing("claude-sonnet-4-5", 0.003, 0.015),

    # Google Models 2026
    "gemini-3.1-pro": ModelPricing("gemini-3.1-pro", 0.002, 0.012),
    "gemini-3.0-pro": ModelPricing("gemini-3.0-pro", 0.002, 0.012),
    "gemini-3-flash": ModelPricing("gemini-3-flash", 0.0005, 0.003),
    "gemini-2.5-flash": ModelPricing("gemini-2.5-flash", 0.000075, 0.0003),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", 0.0001, 0.0004),
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", 0.00125, 0.005),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """Get pricing for a model by name."""
    # Try exact match first
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try partial match (e.g., "gpt-4-0613" -> "gpt-4")
    for key in MODEL_PRICING:
        if model_name.startswith(key):
            return MODEL_PRICING[key]

    return None
