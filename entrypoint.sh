#!/bin/bash
set -e

nvidia-smi

echo "Starting H2O LLM Studio..."

wave run --no-reload app
