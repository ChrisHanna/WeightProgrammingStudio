#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
# Explicit --env-file prevents Compose from reading an unrelated root .env.
exec docker compose --project-name nf-control-plane-dev \
  --project-directory "$ROOT/deploy/control-plane" \
  --env-file "$ROOT/deploy/control-plane/env.example" \
  -f "$ROOT/deploy/control-plane/compose.yaml" "$@"
