"""Cost estimation utilities."""

import hashlib
from typing import Optional

from .types import get_model_pricing


def estimate_cost(
    model_name: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    text: Optional[str] = None,
    cached_tokens: int = 0,
) -> float:
    """Estimate the cost of an LLM call.

    Args:
        model_name: Name of the model (e.g., "gpt-4", "claude-3-opus")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        text: Text to estimate tokens from (rough approximation)
        cached_tokens: Number of cached input tokens

    Returns:
        Estimated cost in USD
    """
    if model_name and input_tokens is not None and output_tokens is not None:
        pricing = get_model_pricing(model_name)
        if pricing:
            return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)

    # Fallback: rough estimation based on text length
    if text:
        # Very rough: ~4 chars per token
        estimated_tokens = len(text) // 4
        # Use a default rate of $0.01 per 1k tokens
        return (estimated_tokens / 1000.0) * 0.01

    return 0.0


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer

    Returns:
        Number of tokens

    Note:
        This requires the tiktoken package. If not available,
        falls back to rough estimation (4 chars per token).
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimation
        return len(text) // 4
    except Exception:
        # If tiktoken fails for any reason, use fallback
        return len(text) // 4


def hash_tool_args(tool_name: str, args: str) -> str:
    """Hash tool arguments for zombie detection.

    Args:
        tool_name: Name of the tool
        args: JSON string of arguments

    Returns:
        SHA-256 hash (first 32 hex chars) as hex string
    """
    combined = f"{tool_name}:{args}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]
