#!/bin/bash
NUM_PROC=$1
shift
uv run torchrun --nproc_per_node=$NUM_PROC llm_studio/train.py "$@"
