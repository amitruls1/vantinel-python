"""Tests for cost estimation utilities."""

import pytest
from vantinel_sdk.cost import estimate_cost, count_tokens, hash_tool_args
from vantinel_sdk.types import MODEL_PRICING


def test_estimate_cost_with_tokens():
    """Test cost estimation with token counts."""
    cost = estimate_cost(
        model_name="gpt-4", input_tokens=1000, output_tokens=500
    )
    # gpt-4: $0.03 per 1k input, $0.06 per 1k output
    # 1000 tokens input = $0.03, 500 tokens output = $0.03
    assert cost == pytest.approx(0.06)


def test_estimate_cost_claude():
    """Test cost estimation for Claude models."""
    cost = estimate_cost(
        model_name="claude-3-opus", input_tokens=2000, output_tokens=1000
    )
    # claude-3-opus: $0.015 per 1k input, $0.075 per 1k output
    # 2000 tokens input = $0.03, 1000 tokens output = $0.075
    assert cost == pytest.approx(0.105)


def test_estimate_cost_text_fallback():
    """Test cost estimation fallback with text."""
    text = "a" * 400  # 400 chars ~= 100 tokens
    cost = estimate_cost(text=text)
    # Should be roughly (100 / 1000) * $0.01 = $0.001
    assert cost > 0


def test_count_tokens_fallback():
    """Test token counting fallback (without tiktoken)."""
    text = "This is a test sentence."
    tokens = count_tokens(text)
    # Fallback: ~4 chars per token
    # 25 chars / 4 = ~6 tokens
    assert tokens > 0


def test_hash_tool_args():
    """Test tool argument hashing."""
    hash1 = hash_tool_args("search", '{"query": "test"}')
    hash2 = hash_tool_args("search", '{"query": "test"}')
    hash3 = hash_tool_args("search", '{"query": "other"}')

    # Same inputs should produce same hash
    assert hash1 == hash2
    # Different inputs should produce different hash
    assert hash1 != hash3
    # Should be MD5 hex (32 chars)
    assert len(hash1) == 32


def test_model_pricing_lookup():
    """Test model pricing database."""
    assert "gpt-4" in MODEL_PRICING
    assert "claude-3-opus" in MODEL_PRICING

    gpt4 = MODEL_PRICING["gpt-4"]
    assert gpt4.input_cost_per_1k == 0.03
    assert gpt4.output_cost_per_1k == 0.06
