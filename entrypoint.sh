#!/bin/bash
set -e

# Ensure USER is set so that getpass.getuser() works for arbitrary UIDs
# that may not exist in /etc/passwd (e.g. when running with --user <uid>).
export USER="${USER:-$(id -un 2>/dev/null)}"

nvidia-smi

echo "Starting H2O LLM Studio..."

wave run --no-reload llm_studio.app
