"""Setup configuration for vantinel-sdk."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vantinel-sdk",
    version="0.1.0-beta",
    author="Vantinel Team",
    author_email="team@vantinel.dev",
    description="Lightweight observability and guardrails SDK for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vantinel/vantinel-sdk",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.24.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    keywords="observability monitoring ai agents guardrails llm",
    project_urls={
        "Bug Reports": "https://github.com/vantinel/vantinel-sdk/issues",
        "Documentation": "https://docs.vantinel.dev",
        "Source": "https://github.com/vantinel/vantinel-sdk",
    },
)
