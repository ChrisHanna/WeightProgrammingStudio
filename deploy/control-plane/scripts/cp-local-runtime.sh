#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$ROOT"
exec sh deploy/control-plane/scripts/cp.sh \
  -f deploy/control-plane/compose.local-runtime.yaml "$@"
