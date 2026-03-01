.PHONY: install test test-unit test-integration format lint type-check clean build publish-test publish check audit help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	uv sync --extra dev

test:  ## Run all tests
	uv run pytest tests/ -v

test-unit:  ## Run unit tests only (skip integration tests)
	uv run pytest tests/ -v -m "not integration"

test-integration:  ## Run integration tests only (requires collector running)
	uv run pytest tests/ -v -m "integration"

format:  ## Format code with ruff
	uv run ruff format vantinel_sdk/ tests/ examples/

lint:  ## Lint code with ruff
	uv run ruff check vantinel_sdk/ tests/ examples/

type-check:  ## Run mypy type checker
	uv run mypy vantinel_sdk/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	uv build

publish-test:  ## Publish to Test PyPI
	uv publish --publish-url https://test.pypi.org/legacy/

publish:  ## Publish to PyPI
	uv publish

audit:  ## Run security audit on dependencies
	uv run pip-audit

check: format lint type-check test-unit  ## Run all checks (format, lint, type-check, tests)
	@echo "All checks passed!"
