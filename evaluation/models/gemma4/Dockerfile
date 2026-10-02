# =====================================================================
# Dockerfile for Gemma 4 evaluation container
#
# Model server: ai/gemma4:<tag> (separate Docker service, not this image)
# This container runs only the CDB evaluation wrapper:
#   - dataloader.py  — builds prompts.jsonl from the benchmark dataset
#   - inference.py   — calls the Gemma 4 OpenAI-compatible API
#   - postprocess.py — parses answers, computes metrics, generates reports
#
# No GPU required here — the ai/gemma4 service handles all GPU work.
#
# Build context: neurips-dataset/ (repo root)
# Build via docker-compose from evaluation/models/gemma4/
# =====================================================================

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

LABEL model.name="CausalDriveBench-Gemma4" \
      model.description="Autonomous Driving Causal Reasoning Evaluation (Gemma 4 API)" \
      maintainer="Renesas Electronics"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0 are required by Pillow for JPEG decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no model weights, just API client + image utils)
COPY evaluation/models/gemma4/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create a non-root user whose UID/GID match the host user.
# This ensures files written to bind-mounted output volumes are owned by the
# host user, not root.  Pass the matching IDs at build time:
#   docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
# (docker-compose.yml already wires ${UID} and ${GID} from the shell.)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser && \
    mkdir -p /workspace && chown -R ${UID}:${GID} /workspace

# Copy the evaluation framework (common modules + model-specific scripts).
# Build context is the repo root (neurips-dataset/), so paths are relative to it.
COPY --chown=appuser:appuser evaluation/ /workspace/repo/evaluation/

USER appuser

WORKDIR /workspace/repo
