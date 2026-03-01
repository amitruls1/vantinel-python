# Changelog

All notable changes to `vantinel-sdk` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.0-beta] - 2025-02-01

### Added
- `VantinelMonitor` with `@watch_tool()` decorator for automatic instrumentation
- Async, fire-and-forget telemetry (no blocking of agent execution)
- Automatic session ID generation (UUID v4)
- Token counting via `tiktoken` integration
- Cost estimation with per-model pricing lookup
- Sampling support (e.g., monitor 10% of high-volume traffic)
- Circuit breaker: fails gracefully if Collector is unreachable
- Security module with hash-based call deduplication
- LangChain integration example

## [0.3.0-beta] - 2025-01-15

### Added
- Initial beta release
- Core monitoring functionality
- HTTP client for Collector API
