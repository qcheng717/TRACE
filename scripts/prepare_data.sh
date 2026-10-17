#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
python tools/robotwin/audit.py
python tools/robotwin/prepare.py
python tools/robotwin/prepare_language.py
