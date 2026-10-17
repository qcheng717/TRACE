#!/usr/bin/env bash
# Run with the RoboTwin simulator environment active. Policy may use a separate Python.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES="${TRACE_EVAL_GPU:-0}"
exec python tools/robotwin/evaluate.py "$@"
