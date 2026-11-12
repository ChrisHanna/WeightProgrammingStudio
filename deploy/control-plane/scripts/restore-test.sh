#!/bin/sh
# Always restores to a NEW database; never overwrites the live database.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
test "$#" -ge 1 && test "$#" -le 2 || { echo "Usage: sh restore-test.sh BACKUP.dump [ARTIFACTS.tar.gz]" >&2; exit 2; }
test -f "$1"
if test "$#" -eq 2; then
  test -f "$2"
fi
DB="restore_check_$(date -u +%Y%m%d%H%M%S)_$$"
CP="$ROOT/deploy/control-plane/scripts/cp.sh"
sh "$CP" exec -T postgres createdb --username=control_plane "$DB"
sh "$CP" exec -T postgres pg_restore --username=control_plane --dbname="$DB" --exit-on-error < "$1"
sh "$CP" exec -T postgres psql --username=control_plane --dbname="$DB" \
  --no-psqlrc --set=ON_ERROR_STOP=1 \
  --command="SELECT version AS restored_schema FROM cp_schema; SELECT count(*) AS restored_runs FROM cp_runs;"
printf 'Restore verified in separate database %s (retained for inspection).\n' "$DB"
if test "$#" -eq 2; then
  VOLUME="nf-control-plane-dev_${DB}_artifacts"
  docker volume create "$VOLUME" > /dev/null
  docker run --rm --interactive --network none --entrypoint tar \
    --mount "type=volume,source=$VOLUME,target=/restore" nf-control-plane:local \
    --extract --gzip --directory /restore --file - < "$2"
  docker run --rm --interactive --network none --entrypoint tar \
    --mount "type=volume,source=$VOLUME,target=/restore,readonly" nf-control-plane:local \
    --compare --gzip --directory /restore --file - < "$2"
  docker run --rm --network none --entrypoint python \
    --mount "type=volume,source=$VOLUME,target=/restore,readonly" nf-control-plane:local \
    -c "from pathlib import Path; print('Restored artifact files:', sum(p.is_file() for p in Path('/restore').rglob('*')))"
  printf 'Artifacts restored to fresh volume %s (retained for inspection).\n' "$VOLUME"
fi
