# =====================================================================
# Dockerfile for Orion
# xiaomi-mlab/Orion — End-to-end AD with LLM (EVA-ViT + LLM backbone)
#
# Python 3.8 | CUDA 11.8 | torch 2.4.1+cu118
# Installed via: pip install -e . && pip install -r requirements.txt
# Weights on HuggingFace (poleyzdk/Orion).
# Requires 32 GB GPU (FP32) or 17 GB GPU (FP16).
# =====================================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL model.name="Orion" \
      model.version="1.0" \
      model.description="End-to-end autonomous driving with LLM (Xiaomi MLAB)" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    CPATH=/usr/local/cuda/include:${CPATH}

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        python3.8 \
        python3.8-dev \
        python3-pip \
        python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
 && python -m pip install --upgrade pip

# ── PyTorch (CUDA 11.8) ────────────────────────────────────────────
RUN pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu118

# ── Clone Orion ───────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/xiaomi-mlab/Orion.git /workspace/model

# ── Install package in editable mode (builds CUDA extensions) ────
WORKDIR /workspace/model
RUN pip install setuptools wheel packaging
RUN python setup.py develop || python setup.py install || pip install . --no-build-isolation

# ── Install remaining requirements ───────────────────────────────
# Skip torch lines (already installed) and install rest
RUN grep -v "^torch" /workspace/model/requirements.txt | pip install -r /dev/stdin || \
    pip install -r /workspace/model/requirements.txt

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/orion/requirements.txt /workspace/requirements.txt
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

WORKDIR /workspace/repo/evaluation/models/orion
