FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# git is needed for flash-attention
# curl is needed to download get-pip.py
# software-properties-common is needed for add-apt-repository
# We get python 3.10 from the deadsnakes PPA to have the latest version
# We install pip from the get-pip.py script to have the latest version
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt install -y \
    python3.10 \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Pick an unusual UID for the llmstudio user.
# In particular, don't pick 1000, which is the default ubuntu user number.
# Force ourselves to test with UID mismatches in the common case.
RUN adduser --uid 1999 llmstudio
USER llmstudio
ENV HOME=/home/llmstudio

# Static application code lives in /workspace/
WORKDIR /workspace

ENV PATH=/home/llmstudio/.local/bin:$PATH
RUN \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    chmod -R a+w /home/llmstudio
COPY Makefile Pipfile Pipfile.lock /workspace/

# Python virtualenv is installed in /workspace/.venv/
# give read and write permissions to the /workspace/.venv/ directory for all users to allow wave to write files
ENV PIPENV_VENV_IN_PROJECT=1
RUN make setup && chmod -R 777 /workspace/.venv

# Add the venv to the PATH
ENV PATH=/workspace/.venv/bin:$PATH

# We need to create a mount point for the user to mount their volume
# All persistent data lives in /home/llmstudio/mount
RUN mkdir -p /home/llmstudio/mount
ENV H2O_LLM_STUDIO_WORKDIR=/home/llmstudio/mount

# Download the demo datasets and place in the /workspace/demo directory
# Set the environment variable for the demo datasets
ENV H2O_LLM_STUDIO_DEMO_DATASETS=/workspace/demo
COPY llm_studio/download_default_datasets.py /workspace/
RUN python download_default_datasets.py

COPY . /workspace

# Remove unnecessary packages remove build packages again
# Prevent removal of cuda packages
USER root
RUN apt-get purge -y git curl python3.10-distutils software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER llmstudio

ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DO_NOT_TRACK=1

# Set the environment variables for the wave server
ENV H2O_WAVE_APP_ADDRESS=http://127.0.0.1:8756
ENV H2O_WAVE_MAX_REQUEST_SIZE=25MB
ENV H2O_WAVE_NO_LOG=true
ENV H2O_WAVE_PRIVATE_DIR="/download/@/home/llmstudio/mount/output/download"

USER root

# Make all of the files in the llmstudio directory read & writable for all users so that the
# application can install other (non-persisted) new packages and other things
# if it wants to. e.g. triton uses /home/llmstudio/.triton as a cache directory.
RUN chmod -R 777 /home/llmstudio

# Make the entrypoint.sh script executable by all users
RUN chmod 755 /workspace/entrypoint.sh

USER llmstudio

EXPOSE 10101

ENTRYPOINT [ "/workspace/entrypoint.sh" ]
