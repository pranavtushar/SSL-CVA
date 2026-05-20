#!/usr/bin/env bash
set -eo pipefail

# SSL-CVA local Miniconda + env activator
# Usage:
#   source env.sh

# Resolve CONDA_DIR at source-time (so env.sh remains portable).
_ssl_cva_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$_ssl_cva_root/miniconda}"

# shellcheck disable=SC1090
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Avoid nounset issues in conda activation scripts.
_nounset_was_on=0
case "$-" in *u*) _nounset_was_on=1 ;; esac
set +u
conda activate "ssl-cva"
if [ "$_nounset_was_on" -eq 1 ]; then set -u; fi
