#!/usr/bin/env bash
# Fetch source only. Follow upstream instructions for simulator packages/assets.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
REV=ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
if [ ! -e "$ROBOTWIN_ROOT" ]; then
  mkdir -p "$(dirname -- "$ROBOTWIN_ROOT")"
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_ROOT"
  git -C "$ROBOTWIN_ROOT" checkout "$REV"
fi
ACTUAL="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
if [ "$ACTUAL" != "$REV" ]; then
  echo "RoboTwin is at $ACTUAL; expected $REV. Existing checkout was left untouched." >&2
  exit 1
fi
git -C "$ROBOTWIN_ROOT" submodule update --init --recursive
