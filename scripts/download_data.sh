#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_REVISION=3dc3b798668feb99ac61cc9086d84cbcc3d79186
export HF_ARCHIVE_NAME=demo_clean.zip
mapfile -t tasks < <(python -c 'from trace_robotwin.data.tasks import ROBOTWIN_TASKS; print("\n".join(ROBOTWIN_TASKS))')
bash "$ROBOTWIN_ROOT/scripts/download_xpolicylab_data.sh" "${tasks[@]}"
