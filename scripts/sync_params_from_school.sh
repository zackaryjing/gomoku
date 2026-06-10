#!/usr/bin/env bash
set -Eeuo pipefail

# Pull trained model parameter files from the configured SSH server.
# Defaults match the server-side project path used for training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-school}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/projects/reinforce_learning/gomoku}"
REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-${REMOTE_ROOT}/checkpoints}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"

mkdir -p "${LOCAL_CHECKPOINT_DIR}"

echo "[gomoku] pulling checkpoints"
echo "[gomoku] remote: ${REMOTE_HOST}:${REMOTE_CHECKPOINT_DIR}/"
echo "[gomoku] local:  ${LOCAL_CHECKPOINT_DIR}/"

rsync -avh --progress \
  --include='*/' \
  --include='*.pt' \
  --include='*.pth' \
  --include='*.ckpt' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_CHECKPOINT_DIR%/}/" \
  "${LOCAL_CHECKPOINT_DIR%/}/"

echo "[gomoku] done"
