#!/bin/sh
set -eu
umask 077
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
DIR="$ROOT/artifacts/control-plane-local/backups"
mkdir -p "$DIR"
NAME="control-plane-$(date -u +%Y%m%dT%H%M%SZ).dump"
test ! -e "$DIR/$NAME"
sh "$ROOT/deploy/control-plane/scripts/cp.sh" exec -T postgres \
  pg_dump --username=control_plane --dbname=control_plane --format=custom > "$DIR/$NAME.partial"
mv "$DIR/$NAME.partial" "$DIR/$NAME"
sh "$ROOT/deploy/control-plane/scripts/cp.sh" exec -T api \
  tar --create --gzip --numeric-owner --directory /var/lib/control-plane --file - . \
  > "$DIR/$NAME.artifacts.tar.gz.partial"
mv "$DIR/$NAME.artifacts.tar.gz.partial" "$DIR/$NAME.artifacts.tar.gz"
printf 'Database backup: artifacts/control-plane-local/backups/%s\n' "$NAME"
printf 'Artifact backup: artifacts/control-plane-local/backups/%s.artifacts.tar.gz\n' "$NAME"
