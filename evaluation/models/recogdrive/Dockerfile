# =====================================================================
# Dockerfile for ReCogDrive
# xiaomi-research/recogdrive — InternVL3 + Diffusion Planner
#
# Python 3.10 | CUDA 12.1 | torch >=2.0 (SM >= 8 required for flash-attn)
# Installed via: pip install -e . (NAVSIM planning package)
# InternVL pretraining deps from internvl_chat/internvl_chat.txt
# Weights on HuggingFace (owl10 collection).
# =====================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL model.name="ReCogDrive" \
      model.version="1.0" \
      model.description="Reinforced Cognitive Driving — InternVL3 + diffusion planner (NAVSIM)" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

# ── PyTorch (CUDA 12.1) ────────────────────────────────────────────
RUN pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

# ── Clone ReCogDrive ──────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/xiaomi-research/recogdrive.git /workspace/model

# ── InternVL pretraining dependencies ────────────────────────────
RUN pip install -r /workspace/model/internvl_chat/internvl_chat.txt || true

# ── NAVSIM / ReCogDrive package (editable install) ───────────────
WORKDIR /workspace/model
RUN pip install setuptools wheel
RUN pip install -e . --no-build-isolation

# ── Remaining requirements (skip conflicting lines) ───────────────
RUN pip install \
        numpy==1.26.4 \
        pytorch-lightning \
        tensorboard==2.16.2 \
        diffusers \
        "opencv-python==4.9.0.80" \
        huggingface_hub \
        einops \
        tqdm \
        jsonlines \
        pillow

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/recogdrive/requirements.txt /workspace/requirements.txt
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

WORKDIR /workspace/repo/evaluation/models/recogdrive
