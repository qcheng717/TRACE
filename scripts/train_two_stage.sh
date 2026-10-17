#!/usr/bin/env bash
# Two-stage training: 10K -> freeze selectors -> 20K -> policy 100K.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RUN="${TRACE_RUN:-outputs/trace_$(date +%Y%m%d_%H%M%S)}"
BATCH="${TRACE_BATCH:-128}"
mkdir -p "$RUN" runtime
common=(--set "data.batch_size=$BATCH" --set "train.batch_size=$BATCH")
bash scripts/train.sh --config configs/stage1.yaml --stage stage1   --stop-after-step 10000 --set "train.output_dir=$RUN/stage1_initial" "${common[@]}" 2>&1 | tee "$RUN/stage1_initial.log"
bash scripts/train.sh --config configs/stage1_frozen.yaml --stage stage1   --stop-after-step 20000 --set "train.resume=$RUN/stage1_initial/step_010000.pt"   --set "train.output_dir=$RUN/stage1" "${common[@]}" 2>&1 | tee "$RUN/stage1.log"
bash scripts/train.sh --config configs/policy.yaml --stage policy   --set "train.stage1_checkpoint=$RUN/stage1/step_020000.pt"   --set "train.output_dir=$RUN/policy" "${common[@]}" 2>&1 | tee "$RUN/policy.log"
