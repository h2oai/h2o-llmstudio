#!/bin/bash

# Install core packages
sudo apt -y update && \
sudo DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
  curl \
  apt-utils \
  apache2-utils \
  wget \
  libblas-dev \
  default-jre \
  clinfo \
  vim \
  software-properties-common \
  gnupg2 \
  ca-certificates

# System installs (Python 3.10)

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo apt-get install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Clone h2o-llmstudio

git clone https://github.com/h2oai/h2o-llmstudio.git
cd h2o-llmstudio

# Create virtual environment (pipenv)

make setup
make wave