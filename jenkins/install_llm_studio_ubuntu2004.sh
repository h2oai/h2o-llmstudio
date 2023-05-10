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

sudo add-apt-repository -y  ppa:deadsnakes/ppa
sudo apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt -y install \
  python3.10 \
  python3.10-dev \
  python3-pip \
  python3-dev \
  python3-virtualenv \
  python3-distutils \
  libopenblas-dev

# Clone h2o-llmstudio

git clone https://github.com/h2oai/h2o-llmstudio.git
cd h2o-llmstudio

# Create virtual environment (pipenv)

make setup

# -----
cd /etc/systemd/system
printf """
[Unit]
Description=LLM Studio Service
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/h2o-llmstudio
ExecStart=/usr/bin/make wave
Restart=always
[Install]
WantedBy=multi-user.target
""" >> llm_studio.service

sudo systemctl daemon-reload
sudo systemctl enable llm_studio.service
sudo systemctl start llm_studio.service