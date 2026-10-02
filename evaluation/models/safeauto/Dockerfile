# =====================================================================
# Dockerfile for <YOUR MODEL NAME>
# Replace all <PLACEHOLDER> values before use.
#
# This template uses `uv` for fast, reproducible Python environment
# management. If your model ships a pyproject.toml, `uv sync` installs
# its deps. Otherwise, fall back to `uv pip install -r requirements.txt`.
# =====================================================================
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

# ======================================================================
# Stage 1 — builder: compile pinned requirements
# ======================================================================

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

LABEL model.name="<MODEL_NAME>" \
      model.version="<MODEL_VERSION>" \
      model.description="<SHORT_DESCRIPTION>" \
      maintainer="<YOUR_NAME_OR_TEAM>"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # uv settings
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=managed \
    # HuggingFace cache (weights mounted at runtime)
    HF_HOME=/workspace/weights

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (fast Python package manager) ───────────────────────
# uv is the default for models that ship a pyproject.toml / uv.lock.
# If the model uses conda, replace the block below with miniconda:
#
#   RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#           -O /tmp/miniconda.sh \
#    && bash /tmp/miniconda.sh -b -p /opt/conda \
#    && rm /tmp/miniconda.sh && conda clean -afy
#   ENV PATH="/opt/conda/bin:${PATH}"
#   RUN conda create -n model python=3.10 -y
#   SHELL ["conda", "run", "-n", "model", "/bin/bash", "-c"]
#   # Then install deps with pip inside the conda env:
#   RUN pip install torch==<version> ...
#   ENV PATH="/opt/conda/envs/model/bin:${PATH}"
#
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ── (Optional) Non-root user ──────────────────────────────────────
# Uncomment the block below to run as a non-root user.
# Some models require root for CUDA; leave commented if unsure.
#
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g ${GID} appuser && \
#     useradd -u ${UID} -g ${GID} -m appuser
# USER appuser
# ENV PATH="/home/appuser/.local/bin:${PATH}"

# ── Clone model repository ─────────────────────────────────────────
# TODO: Replace with your model's repo URL.
# Use --depth 1 to minimise image size.
RUN git clone --depth 1 https://github.com/<ORG>/<REPO> /workspace/model

# ── Python virtual environment ─────────────────────────────────────

ENV VIRTUAL_ENV=/workspace/py_venv

RUN uv venv ${VIRTUAL_ENV} --python ${PYTHON_VERSION}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ── Install model dependencies ─────────────────────────────────────
# Option A (recommended): model ships pyproject.toml — install via uv sync
WORKDIR /workspace/model
# TIP: Use BuildKit cache mounts to avoid re-downloading packages on every build:
#   RUN --mount=type=cache,target=/root/.cache/uv uv sync --active
# Without BuildKit (default):
RUN uv sync --active

# Option B: model has a requirements.txt — install with uv pip
# RUN uv pip install -r /workspace/model/requirements.txt

# Option C: model requires editable install (has setup.py / CUDA extensions)
# RUN pip install -v -e /workspace/model
# RUN pip install -r /workspace/model/requirements.txt

# ── Install evaluation framework dependencies ──────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install -r /workspace/requirements.txt

# ── Copy evaluation framework source ──────────────────────────────
# Build context must be the repo root (../../..) so the entire
# evaluation/ directory is available.
COPY evaluation/ /workspace/repo/evaluation/

# ── (Optional) Run one-time setup at build time ───────────────────
# Uncomment to bake weights into the image (large images).
# Recommended: keep commented and mount weights at runtime.
# RUN bash /workspace/repo/evaluation/models/<model_folder>/setup.sh

# WORKDIR /workspace/repo/evaluation/models/<model_folder>
