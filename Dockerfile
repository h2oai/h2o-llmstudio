FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && apt install -y python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
COPY . .
RUN make setup
ENV H2O_WAVE_MAX_REQUEST_SIZE=25MB
ENV H2O_WAVE_NO_LOG=True
ENV H2O_WAVE_PRIVATE_DIR="/download/@/workspace/output/download"
EXPOSE 10101
ENTRYPOINT [ "python3.10", "-m", "pipenv", "run", "wave", "run", "app" ]
