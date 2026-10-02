# =====================================================================
# llamacpp-server:gemma4 — High-throughput llama.cpp inference server
#
# Extracts the llama-server binary from docker/model-runner:latest-cuda
# and exposes it as a directly-configurable ENTRYPOINT.
#
# Problem solved: the Docker model runner daemon caps n_parallel at ~4
# and does not expose --parallel configuration via its API.  This image
# runs the same llama-server binary with full --parallel/--ctx-size control.
#
# Build (from repo root):
#   docker build -t llamacpp-server:gemma4 \
#     evaluation/models/gemma4/server/
#
# Or via setup.sh (recommended — handles weights + idempotent startup):
#   bash evaluation/models/gemma4/setup.sh
# =====================================================================
FROM docker/model-runner:latest-cuda
ARG UID=1000
ARG GID=1000
USER root
RUN chmod +x /app/bin/llama-server && \
    groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH
USER appuser
ENTRYPOINT ["/app/bin/llama-server"]
