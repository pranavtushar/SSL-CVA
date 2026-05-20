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
import shutil
from pathlib import Path
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

out_p = Path(out_dir)

# Some checkpoint repos already contain a top-level folder called "all_checkpoints".
# If we download into OUT_DIR=all_checkpoints, we end up with:
#   all_checkpoints/all_checkpoints/<models...>
# Flatten to:
#   all_checkpoints/<models...>
inner = out_p / "all_checkpoints"
if inner.is_dir() and any(inner.glob("*/config.json")):
    for item in inner.iterdir():
        dst = out_p / item.name
        if not dst.exists():
            shutil.move(str(item), str(dst))
            continue
        # Merge directories conservatively (don't overwrite).
        if item.is_dir() and dst.is_dir():
            for child in item.rglob("*"):
                rel = child.relative_to(item)
                out_child = dst / rel
                if out_child.exists():
                    continue
                out_child.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(out_child))
            shutil.rmtree(item, ignore_errors=True)
    shutil.rmtree(inner, ignore_errors=True)

print(f"Downloaded checkpoints to: {out_p.resolve()}")
PY

