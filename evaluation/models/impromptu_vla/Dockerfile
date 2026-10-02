# =====================================================================
# Dockerfile for Impromptu-VLA
# ahydchh/Impromptu-VLA — Qwen2.5-VL with sglang/vLLM serving
#
# Python 3.10 | CUDA 12.6 | torch 2.7.1
# Uses sglang for inference serving; vLLM as alternative.
# Weights on HuggingFace (aaaaaap/ImpromptuVLAModel).
# =====================================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

LABEL model.name="Impromptu-VLA" \
      model.version="1.0" \
      model.description="Impromptu data augmentation for VLA models (Qwen2.5-VL, sglang)" \
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
        libgl1 \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

# ── PyTorch (CUDA 12.6) ────────────────────────────────────────────
# torch 2.7.1 is cutting-edge — use the nightly/test index if unavailable
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126 || \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# ── sglang (primary inference backend) ───────────────────────────
RUN pip install sglang==0.4.7 || pip install sglang

# ── Core LLM dependencies ─────────────────────────────────────────
RUN pip install \
        transformers==4.52.3 \
        accelerate \
        peft \
        deepspeed==0.17.1 \
        bitsandbytes \
        vllm \
        einops \
        sentencepiece \
        numpy \
        pillow \
        tqdm \
        jsonlines \
        pydantic

# ── Clone Impromptu-VLA ──────────────────────────────────────────
RUN git clone --depth 1 https://github.com/ahydchh/Impromptu-VLA.git /workspace/model

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/impromptu_vla/requirements.txt /workspace/requirements.txt
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

WORKDIR /workspace/repo/evaluation/models/impromptu_vla
