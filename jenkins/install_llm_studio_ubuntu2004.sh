#!/bin/bash

# Install core packages
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install \
  curl \
  build-essential

# Verify make installation
ls /usr/bin/make

# System installs (Python 3.10)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo DEBIAN_FRONTEND=noninteractive apt -y install python3.10
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Clone h2o-llmstudio

git clone https://github.com/h2oai/h2o-llmstudio.git
cd h2o-llmstudio

# Create virtual environment (pipenv)

make setup

# -----
cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .

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