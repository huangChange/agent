# Sandbox image for agent-generated code (agent.sandbox.backend: docker).
# Runs with --network none, read-only root fs, non-root user, cpu/memory/pids limits (see agent/sandbox.py).
# Build from the repo root:  docker build -f docker/sandbox.Dockerfile -t gsforge-sandbox:0.1.0 .
FROM python:3.11-slim

RUN useradd --uid 1000 --create-home sandbox
WORKDIR /opt/gsforge
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# geometry stack only: no API client credentials, no cloud SDKs inside the sandbox
RUN pip install --no-cache-dir ".[geo]" && pip uninstall -y anthropic

ENV GSFORGE_CONFIG_DIR=/opt/gsforge/configs \
    PYTHONDONTWRITEBYTECODE=1
USER sandbox
WORKDIR /work
