# =====================================================================
# Dockerfile for Alpamayo 1.5
# Repo: https://github.com/NVlabs/alpamayo1.5
# Model: nvidia/Alpamayo-1.5-10B (~22 GB, gated on HuggingFace)
# Requires: NVIDIA GPU with >=24 GB VRAM (>=40 GB for multi-trajectory)
#
# Build context: neurips-dataset/ (repo root)
# Build via docker-compose from evaluation/models/alpamayo_1_5/
# =====================================================================

ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

LABEL model.name="CausalDriveBench-Alpamayo1_5" \
      model.version="1.5" \
      model.description="Autonomous Driving Causal Reasoning Evaluation" \
      maintainer="Renesas Electronics"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace weights cache → mounted volume so weights persist across containers
    HF_HOME=/workspace/weights \
    # uv: don't prompt for Python install confirmation
    UV_PYTHON_PREFERENCE=managed

# build-essential + ninja are required for flash-attn compilation from source
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv to /usr/local/bin so it is accessible by all users
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSfo /opt/uv-installer.sh https://astral.sh/uv/install.sh \
    && bash /opt/uv-installer.sh \
    && rm /opt/uv-installer.sh

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

USER appuser

# Clone Alpamayo 1.5 and install its dependencies
# NOTE: physical-ai-av==0.2.0 (listed in pyproject.toml) is an NVIDIA proprietary
# package. If it is not on PyPI, uv sync will fail. Contact NVIDIA for access or
# comment out the physical-ai-av line from pyproject.toml before building.
RUN git clone --depth 1 https://github.com/NVlabs/alpamayo1.5 /workspace/model

WORKDIR /workspace/model

# Create virtual env and install all project deps from pyproject.toml / uv.lock.
# flash-attn is built from source (requires the cuda-devel layer above).
ENV VIRTUAL_ENV=/workspace/py_venv
RUN uv venv ${VIRTUAL_ENV} --python 3.12
RUN uv sync --active

# Install evaluation-only extras (e.g. jupyter)
COPY --chown=appuser:appuser evaluation/models/alpamayo_1_5/requirements.txt /workspace/model/requirements.txt
RUN uv pip install -r /workspace/model/requirements.txt --python ${VIRTUAL_ENV}

# Copy the evaluation framework (common modules + model-specific scripts).
# Build context is the repo root (neurips-dataset/), so paths are relative to it.
COPY --chown=appuser:appuser evaluation/ /workspace/repo/evaluation/

# Weights are mounted at runtime (see docker-compose.yml).
# Run the weight-downloader service (--profile setup) to populate WEIGHTS_DIR
# before running inference.
