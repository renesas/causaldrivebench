# =====================================================================
# Dockerfile for OpenDriveVLA
# DriveVLA/OpenDriveVLA — Qwen2.5-VL based driving VLA
#
# Python 3.10 | CUDA 11.8 | torch 2.1.2
# Requires MMCV compiled from source (MMCV_WITH_OPS=1).
# MMDet3D v1.0.0rc6 from source (bundled as third_party).
# Weights on HuggingFace (OpenDriveVLA/OpenDriveVLA-0.5B).
# =====================================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL model.name="OpenDriveVLA" \
      model.version="0.5B" \
      model.description="Qwen2.5-VL based Vision-Language-Action model for autonomous driving" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        libgfortran5 \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

# ── PyTorch ────────────────────────────────────────────────────────
RUN pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu118

# ── Clone OpenDriveVLA ────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/DriveVLA/OpenDriveVLA.git /workspace/model

# ── MMCV from bundled source (requires CUDA_HOME) ────────────────
RUN cd /workspace/model/third_party/mmcv_1_7_2 \
 && pip install -r requirements/optional.txt \
 && MMCV_WITH_OPS=1 pip install . --no-build-isolation

# ── MM ecosystem packages ─────────────────────────────────────────
RUN pip install \
        mmdet==2.26.0 \
        mmsegmentation==0.29.1 \
        mmengine==0.9.0 \
        motmetrics==1.4.0

# ── MMDet3D from bundled source ───────────────────────────────────
RUN cd /workspace/model/third_party/mmdetection3d_1_0_0rc6 \
 && pip install scipy==1.10.1 scikit-image==0.19.3 fsspec \
 && pip install . --no-build-isolation

# ── Core dependencies ─────────────────────────────────────────────
RUN pip install \
        deepspeed==0.14.2 \
        accelerate==0.29.3 \
        peft==0.4.0 \
        einops \
        sentencepiece \
        numpy \
        pillow \
        tqdm \
        jsonlines

# ── transformers (pinned commit for compatibility) ────────────────
RUN pip install git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4

# ── flash-attn ───────────────────────────────────────────────────
RUN pip install flash-attn==2.5.7 --no-build-isolation

# ── Install OpenDriveVLA package ──────────────────────────────────
RUN cd /workspace/model && pip install . --no-build-isolation 2>/dev/null || true

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/opendrivevla/requirements.txt /workspace/requirements.txt
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

WORKDIR /workspace/repo/evaluation/models/opendrivevla
