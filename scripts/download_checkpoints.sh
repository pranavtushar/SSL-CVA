#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-}"
OUT_DIR="${OUT_DIR:-all_checkpoints}"

usage() {
  cat <<'EOF'
Download SSL-CVA checkpoints from Hugging Face into a local folder.

Usage:
  bash scripts/download_checkpoints.sh --repo <user>/<repo> [--out all_checkpoints]

Env:
  REPO_ID=<user>/<repo>         (same as --repo)
  OUT_DIR=all_checkpoints       (same as --out)
  HF_TOKEN=...                  (only needed if the HF repo is private)

Notes:
  - This uses huggingface_hub's snapshot_download.
  - Optional speedup: pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_ID="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$REPO_ID" ]]; then
  echo "Missing --repo (or REPO_ID=...)" >&2
  usage
  exit 2
fi

mkdir -p "$OUT_DIR"

export REPO_ID
export OUT_DIR

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
out_dir = os.environ["OUT_DIR"]

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=out_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Downloaded checkpoints to: {out_dir}")
PY

