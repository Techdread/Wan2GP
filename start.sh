#!/bin/bash
# WanGP Launch Script (venv-based setup)
cd "$(dirname "$0")"
source venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
python wgp.py "$@"
