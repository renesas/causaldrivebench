# =====================================================================
# Dockerfile for OmniDrive
# NVlabs/OmniDrive — EVA-ViT + TinyLlama for autonomous driving VQA
#
# Python 3.9 | CUDA 11.7 | torch 1.13.1+cu117
# Dependencies installed via pip (mmcv-full, mmdet, mmsegmentation,
# mmdetection3d from source, OpenLane-V2 from source).
# =====================================================================

FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

LABEL model.name="OmniDrive" \
      model.version="1.0" \
      model.description="End-to-end autonomous driving with LLM (EVA-ViT + TinyLlama)" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        python3.9 \
        python3.9-dev \
        python3-pip \
        python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && python -m pip install --upgrade pip setuptools==59.5.0

# ── PyTorch (CUDA 11.7) ────────────────────────────────────────────
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# ── flash-attn ────────────────────────────────────────────────────
RUN pip install flash-attn==2.5.6 --no-build-isolation \
 || echo "WARNING: flash-attn build failed; inference will fall back to standard attention"

# ── Core ML dependencies ──────────────────────────────────────────
RUN pip install \
        transformers==4.31.0 \
        "accelerate==0.20.3" \
        einops \
        timm \
        shapely==1.8.5.post1

# ── mmcv-full (CUDA 11.7 / torch 1.13) ───────────────────────────
RUN pip install mmcv-full==1.6.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html \
        --no-build-isolation

# ── mmdet / mmsegmentation ────────────────────────────────────────
RUN pip install mmdet==2.28.2 mmsegmentation==0.30.0

# ── mmdetection3d v1.0.0rc6 from source ──────────────────────────
RUN git clone --depth 1 --branch v1.0.0rc6 \
        https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d \
 && cd /tmp/mmdetection3d && pip install . --no-build-isolation \
 && rm -rf /tmp/mmdetection3d/.git

# ── OpenLane-V2 from source ───────────────────────────────────────
# ortools==9.2.9972 doesn't exist on PyPI; pre-install a compatible version
# then skip dep resolution for the package itself.
RUN git clone --depth 1 \
        https://github.com/OpenDriveLab/OpenLane-V2.git /tmp/OpenLane-V2 \
 && pip install "ortools>=9.3.0" \
 && cd /tmp/OpenLane-V2 && pip install . --no-build-isolation --no-deps \
 && rm -rf /tmp/OpenLane-V2/.git

# ── Clone OmniDrive ───────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/NVlabs/OmniDrive.git /workspace/model

# ── OmniDrive requirements ────────────────────────────────────────
# Fix typo in repo: shapely==1.8.5.post doesn't exist, .post1 does
RUN sed -i 's/shapely==1\.8\.5\.post$/shapely==1.8.5.post1/' /workspace/model/requirements.txt \
 && pip install -r /workspace/model/requirements.txt

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/omnidrive/requirements.txt /workspace/requirements.txt
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

WORKDIR /workspace/repo/evaluation/models/omnidrive
