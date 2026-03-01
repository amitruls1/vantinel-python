.PHONY: install test format lint clean build help build-secure audit

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e ".[dev]"

test:  ## Run all tests
	pytest tests/ -v

test-unit:  ## Run unit tests only (skip integration tests)
	pytest tests/ -v -m "not integration"

test-integration:  ## Run integration tests only (requires collector running)
	pytest tests/ -v -m "integration"

format:  ## Format code with black
	black vantinel_sdk/ tests/ examples/

lint:  ## Lint code with ruff
	ruff check vantinel_sdk/ tests/ examples/

type-check:  ## Run mypy type checker
	mypy vantinel_sdk/

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
	python -m build

publish-test:  ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

example-basic:  ## Run basic usage example
	python examples/basic_usage.py

example-decorator:  ## Run decorator example
	python examples/decorator_example.py

example-langchain:  ## Run LangChain integration example
	python examples/langchain_integration.py

example-sampling:  ## Run high-volume sampling example
	python examples/high_volume_sampling.py

run-examples:  ## Run all examples
	@echo "Running basic usage..."
	@python examples/basic_usage.py
	@echo "\n\nRunning decorator example..."
	@python examples/decorator_example.py
	@echo "\n\nRunning high-volume sampling..."
	@python examples/high_volume_sampling.py

check: format lint type-check test-unit  ## Run all checks (format, lint, type-check, tests)
	@echo "All checks passed!"

build-secure:  ## Build Cython-compiled secure wheel
	pip install cython
	python setup_cython.py bdist_wheel

audit:  ## Run security audit on dependencies
	pip install pip-audit
	pip-audit
