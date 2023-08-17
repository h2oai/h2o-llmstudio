FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && apt install -y python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Pick an unusual UID for the llmstudio user.
# In particular, don't pick 1000, which is the default ubuntu user number.
# Force ourselves to test with UID mismatches in the common case.
RUN adduser --uid 1999 llmstudio
USER llmstudio

# Python virtualenv is installed in /home/llmstudio/.local
# Application code and data lives in /workspace
#
# Make all of the files in the llmstudio directory writable so that the
# application can install other (non-persisted) new packages and other things
# if it wants to.  This is really not advisable, though, since it's lost when
# the container exits.
WORKDIR /workspace
RUN \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    chmod -R a+w /home/llmstudio
COPY Makefile .
COPY Pipfile .
COPY Pipfile.lock .
RUN \
    make setup && \
    mkdir -p /home/llmstudio/mount && \
    chmod -R a+w /home/llmstudio
COPY . .

ENV HOME=/home/llmstudio
ENV H2O_WAVE_MAX_REQUEST_SIZE=25MB
ENV H2O_WAVE_NO_LOG=true
ENV H2O_WAVE_PRIVATE_DIR="/download/@/workspace/output/download"
EXPOSE 10101
ENTRYPOINT [ "python3.10", "-m", "pipenv", "run", "wave", "run", "--no-reload", "app" ]
