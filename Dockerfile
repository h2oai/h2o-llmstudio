FROM 353750902984.dkr.ecr.us-east-1.amazonaws.com/thirdparty-chainguard-python310:latest-fips-dev

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_MAJOR_VERSION=12
ARG CUDA_MINOR_VERSION=6

ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV NVIDIA_VISIBLE_DEVICES="all"

USER root

RUN apk update \
    && apk upgrade \
    && apk add wget \
    && wget -O /etc/apk/keys/chainguard-extras.rsa.pub https://packages.cgr.dev/extras/chainguard-extras.rsa.pub \
    && echo "https://packages.cgr.dev/extras" | tee -a /etc/apk/repositories \
    && apk update \
    && apk add --no-cache \
    nvidia-cudnn-8 \
    nvidia-cudnn-8-cuda-${CUDA_MAJOR_VERSION} \
    nvidia-cudnn-8-cuda-${CUDA_MAJOR_VERSION}-dev \
    nvidia-cuda-cudart-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
    nvidia-cuda-cudart-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}-dev \
    nvidia-cuda-nvcc-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
    nvidia-libcublas-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
    cuda-toolkit-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}-dev \
    make \
    curl \
    git

WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

RUN python -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Install uv and python dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN --mount=type=bind,src=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,src=uv.lock,target=uv.lock \
    /root/.local/bin/uv sync --frozen --no-cache

# Add the venv to the PATH
ENV PATH=/workspace/.venv/bin:$PATH

# We need to create a mount point for the user to mount their volume
# All persistent data lives in /mount
RUN mkdir -p /mount 
RUN mkdir -p /mount && chown -R nonroot:nonroot /mount
ENV H2O_LLM_STUDIO_WORKDIR=/mount

# Download the demo datasets and place in the /workspace/demo directory
# Set the environment variable for the demo datasets
ENV H2O_LLM_STUDIO_DEMO_DATASETS=/workspace/demo
COPY --chown=nonroot:nonroot ./llm_studio/download_default_datasets.py /workspace/
RUN python download_default_datasets.py

COPY --chown=nonroot:nonroot ./llm_studio /workspace/llm_studio
COPY --chown=nonroot:nonroot ./prompts /workspace/prompts
COPY --chown=nonroot:nonroot ./model_cards /workspace/model_cards
COPY --chown=nonroot:nonroot ./LICENSE /workspace/LICENSE
COPY --chown=nonroot:nonroot ./entrypoint.sh /workspace/entrypoint.sh
COPY --chown=nonroot:nonroot ./pyproject.toml /workspace/pyproject.toml

ENV HF_HOME=/mount/huggingface
ENV TRITON_CACHE_DIR=/mount/.triton/cache
ENV H2O_WAVE_DATA_DIR=/mount/wave_data
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DO_NOT_TRACK=1

# Set the environment variables for the wave server
ENV H2O_WAVE_APP_ADDRESS=http://127.0.0.1:8756
ENV H2O_WAVE_MAX_REQUEST_SIZE=25MB
ENV H2O_WAVE_NO_LOG=true
ENV H2O_WAVE_PRIVATE_DIR="/download/@/mount/output/download"

# Make the entrypoint.sh script executable
RUN chmod 755 /workspace/entrypoint.sh

EXPOSE 10101

USER nonroot

ENTRYPOINT [ "/workspace/entrypoint.sh" ]
