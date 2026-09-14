#!/usr/bin/env bash
# HTTP part-splitting service. AutoDL custom-service port is 6006.
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
# shellcheck disable=SC1091
source "$ROOT/env.sh"
PY=/root/autodl-tmp/envs/trellis2/bin/python
exec "$PY" "$ROOT/serve_api.py" "$@"
