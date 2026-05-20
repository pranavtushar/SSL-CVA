#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_DIR="${CONDA_DIR:-$root_dir/miniconda}"
ENV_NAME="${ENV_NAME:-ssl-cva}"
REQ_FILE="${REQ_FILE:-$root_dir/requirements.txt}"
CONDA_URL="${CONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

recreate=0
while [ $# -gt 0 ]; do
  case "$1" inp
    --conda-dir) CONDA_DIR="$2"; shift 2 ;;
    --name) ENV_NAME="$2"; shift 2 ;;
    --req) REQ_FILE="$2"; shift 2 ;;
    --recreate) recreate=1; shift ;;
    *) echo "Usage: bash scripts/install.sh [--conda-dir <path>] [--name <env>] [--req <requirements.txt>] [--recreate]"; exit 2 ;;
  esac
done

case "$CONDA_DIR" in
  /*) ;;
  *) CONDA_DIR="$root_dir/$CONDA_DIR" ;;
esac

if [ ! -f "$REQ_FILE" ]; then
  echo "Missing requirements file: $REQ_FILE"
  exit 1
fi

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "Bootstrapping Miniconda into: $CONDA_DIR"
  installer="$root_dir/$(basename "$CONDA_URL")"
  if [ ! -f "$installer" ]; then
    wget -O "$installer" "$CONDA_URL"
  fi
  bash "$installer" -b -p "$CONDA_DIR"
fi

# shellcheck disable=SC1090
source "$CONDA_DIR/etc/profile.d/conda.sh"

if [ "$recreate" -eq 1 ]; then
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
fi

if conda run -n "$ENV_NAME" python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
  echo "Conda env exists: $ENV_NAME"
else
  echo "Creating conda env: $ENV_NAME (python=3.9)"
  conda create -y -n "$ENV_NAME" python=3.9 pip
fi

echo "Installing requirements into: $ENV_NAME"
# fairseq==0.12.2 depends on omegaconf<2.1 whose older wheels have metadata
# rejected by pip>=24.1. Keep pip below 24.1 for compatibility.
conda run -n "$ENV_NAME" python -m pip install "pip<24.1"
conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"

cat > "$root_dir/env.sh" <<EOF
#!/usr/bin/env bash
set -eo pipefail

# SSL-CVA local Miniconda + env activator
# Usage:
#   source env.sh

# Resolve CONDA_DIR at source-time (so env.sh remains portable).
_ssl_cva_root="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="\${CONDA_DIR:-$_ssl_cva_root/miniconda}"

# shellcheck disable=SC1090
source "\$CONDA_DIR/etc/profile.d/conda.sh"

# Avoid nounset issues in conda activation scripts.
_nounset_was_on=0
case "\$-" in *u*) _nounset_was_on=1 ;; esac
set +u
conda activate "$ENV_NAME"
if [ "\$_nounset_was_on" -eq 1 ]; then set -u; fi
EOF

chmod +x "$root_dir/env.sh"

echo
echo "Done."
echo "Next:"
echo "  cd \"$root_dir\""
echo "  source env.sh"
