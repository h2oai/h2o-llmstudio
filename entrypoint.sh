#!/bin/bash
set -e

nvidia-smi

# Your custom commands go here
echo "Starting H2O LLM Studio..."

wave run --no-reload app
