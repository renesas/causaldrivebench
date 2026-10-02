# =====================================================================
# Dockerfile for Doe-1
# wzzheng/Doe — World model with multimodal token generation
#
# Python 3.10 | CUDA 11.8 | torch 2.4.0
# Weights are NOT on HuggingFace — hosted on Tsinghua Cloud.
# VQVAE weights require Facebook's Chameleon checkpoint.
# Dependencies installed via pip.
# =====================================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL model.name="Doe-1" \
      model.version="1.0" \
      model.description="World model for autonomous driving (multimodal discrete token generation)" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights \
    WEIGHTS_PATH=/workspace/weights

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

# ── PyTorch (CUDA 11.8) ────────────────────────────────────────────
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu118

# ── Clone Doe ─────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/wzzheng/Doe.git /workspace/model

# ── Install xllmx model package ──────────────────────────────────
RUN cd /workspace/model/model && touch README.md && pip install . --no-build-isolation

# ── Main requirements ─────────────────────────────────────────────
# NOTE: flash_attn wheel bundled in repo is for torch 2.0/cu118; install
# separately for torch 2.4 compatibility.
RUN pip install \
        transformers==4.44.2 \
        accelerate==0.33.0 \
        peft==0.12.0 \
        deepspeed==0.14.4 \
        xformers==0.0.27.post2 \
        einops \
        sentencepiece \
        tqdm \
        jsonlines \
        numpy \
        pillow

RUN pip install flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn build failed; inference will fall back to standard attention"

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/doe/requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ── Create non-root user whose UID/GID match the host user ───────
# Files written to bind-mounted output volumes are then owned by the host
# user rather than root.  Pass IDs at build time:
#   docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser && \
    mkdir -p /workspace && chown -R ${UID}:${GID} /workspace

# ── Evaluation framework source ──────────────────────────────────
COPY --chown=appuser:appuser evaluation/ /workspace/repo/evaluation/

USER appuser

WORKDIR /workspace/repo/evaluation/models/doe
