# =====================================================================
# Dockerfile for SimLingo
# RenzKa/simlingo — InternVL2-based language-driven driving agent
#
# Python 3.8 | CUDA 12.1 | torch 2.2.0
# NOTE: SimLingo's closed-loop evaluation requires CARLA 0.9.15.
# This image is scoped to language-model inference only (no CARLA).
# Dependencies installed via conda then pip (using miniconda).
# =====================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL model.name="SimLingo" \
      model.version="1.0" \
      model.description="Language-conditioned driving agent (InternVL2-based, CARLA optional)" \
      maintainer="neurips-dataset-eval"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/weights \
    CONDA_DIR=/opt/conda \
    PATH="/opt/conda/bin:${PATH}"

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniforge3 (conda-forge, no Anaconda ToS) ─────────────────────
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh \
 && conda clean -afy

# ── Create Python 3.8 environment ─────────────────────────────────
RUN conda create -n simlingo python=3.8 -y
SHELL ["conda", "run", "-n", "simlingo", "/bin/bash", "-c"]

# ── PyTorch + flash-attn ──────────────────────────────────────────
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
RUN pip install flash-attn==2.7.0.post2 --no-build-isolation

# ── Clone SimLingo ────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/RenzKa/simlingo.git /workspace/model

# ── SimLingo language-inference requirements ──────────────────────
RUN pip install -r /workspace/model/team_code/requirements.txt || true

RUN pip install \
        transformers \
        huggingface_hub \
        accelerate \
        peft \
        einops \
        timm \
        sentencepiece \
        numpy \
        pillow \
        tqdm \
        jsonlines \
        omegaconf \
        hydra-core \
        pytorch-lightning

# ── Evaluation framework dependencies ────────────────────────────
COPY evaluation/models/simlingo/requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# ── Create non-root user whose UID/GID match the host user ───────
# Files written to bind-mounted output volumes are then owned by the host
# user rather than root.  Pass IDs at build time:
#   docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
# Note: conda envs at /opt/conda remain root-owned (read-only at runtime);
# only /workspace output dirs need to be writable by appuser.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser && \
    mkdir -p /workspace && chown -R ${UID}:${GID} /workspace

# ── Evaluation framework source ──────────────────────────────────
COPY --chown=appuser:appuser evaluation/ /workspace/repo/evaluation/

# Make conda environment the default python
ENV PATH="/opt/conda/envs/simlingo/bin:${PATH}"

USER appuser

WORKDIR /workspace/repo/evaluation/models/simlingo
