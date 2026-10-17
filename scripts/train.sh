#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec torchrun --standalone --nproc_per_node="${TRACE_NPROC:-1}" scripts/train.py "$@"
