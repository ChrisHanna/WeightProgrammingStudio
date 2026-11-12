#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
CP="$ROOT/deploy/control-plane/scripts/cp.sh"
sh "$CP" -f "$ROOT/deploy/control-plane/compose.test.yaml" up -d --wait postgres
EXISTS=$(sh "$CP" exec -T postgres psql --username=control_plane --dbname=postgres \
  --no-psqlrc --tuples-only --no-align --set=ON_ERROR_STOP=1 \
  --command="SELECT 1 FROM pg_database WHERE datname='control_plane_test'")
if test "$EXISTS" != 1; then
  sh "$CP" exec -T postgres createdb --username=control_plane control_plane_test
fi
printf 'Integration DB ready: host=127.0.0.1 port=55432 user=control_plane database=control_plane_test\n'
printf 'Credential file: artifacts/control-plane-local/secrets/postgres_password (read in-process only)\n'
