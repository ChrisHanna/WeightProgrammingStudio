# Private single-host control plane

This deployment is for **one trusted operator, one host, and CPU fixtures only**.
It is not a public multi-tenant service. The API/coordinator serves the thin UI;
one worker claims durable PostgreSQL jobs. No provider credentials, models,
external datasets, existing root `.env`, or unrelated application directories
are mounted. PostgreSQL is the only supported initial database.

## Requirements and initialization

Use Docker Engine with Compose v2 or newer, Linux containers, and Python 3.11+.
On Windows run the following commands in WSL from the repository root.
The image builds the real Rust extension with locked Cargo dependencies and
maturin, then installs the wheel and `control-plane` extra. Both stages use
Python 3.12/Debian bookworm; the Rust toolchain image is also bookworm.
The image selects its installed Rust 1.93.1 rather than floating the repository's
`stable` toolchain channel during builds (the workspace minimum remains 1.80).
CPU PyTorch is explicitly installed as `2.9.1+cpu` from the official CPU index.
No native-extension stubs or host-built wheels are used.

```sh
python3 deploy/control-plane/scripts/bootstrap.py
sh deploy/control-plane/scripts/cp.sh up --build -d --wait
sh deploy/control-plane/scripts/cp.sh ps
python3 deploy/control-plane/scripts/verify-image.py
python3 deploy/control-plane/scripts/smoke.py
```

Bootstrap generates two random secrets without printing their contents:
`artifacts/control-plane-local/secrets/postgres_password` and `operator_token`.
The directory is ignored by Git; files are mode 0600 and the directory 0700.
Existing nonempty secrets are preserved, not rotated. Never copy secrets into
`env.example`, a command argument, shell history, screenshots, logs, or Git.

**WSL/DrvFS:** bootstrap refuses filesystems that cannot enforce mode 0600.
Enable DrvFS `metadata` before initializing secrets, or use an ext4 checkout.
For this local provision, only the ignored `artifacts/control-plane-local`
directory was mounted with DrvFS metadata (the global C: mount was unchanged).
That narrow mount must be recreated after a WSL shutdown unless your WSL
configuration already enables metadata. A Windows administrator can mount the
specific existing directory from PowerShell, replacing the absolute paths for
their checkout:

```powershell
wsl -u root -- mount -t drvfs "C:\path\to\repo\artifacts\control-plane-local" /mnt/c/path/to/repo/artifacts/control-plane-local -o metadata,uid=1000,gid=1000
```

Use your WSL user's actual UID/GID. Do not initialize on a shared filesystem
until its Windows ACLs also exclude untrusted users.

`cp.sh` always selects project `nf-control-plane-dev` and the explicit
non-secret `deploy/control-plane/env.example`; Compose never implicitly loads
the repository root `.env`. The Dockerfile-specific ignore file allowlists
build inputs, excluding `.env`, artifacts, apps, model files outside the
Python package, and host environments.

## Networking, auth, and readiness

Only `127.0.0.1:8787` is published. PostgreSQL has no published port in the
normal deployment. Its private network is internal; the API additionally has
an ingress bridge so Docker can publish its loopback port. The worker has no
ingress bridge. Never change host bindings to `0.0.0.0`.

Operator API actions require the operator bearer token. Read the token file directly
when logging in; never put the token into a URL or command argument. Health
probes read mounted secret files and send bearer auth in process memory.
The entrypoint reads the PostgreSQL password, URL-encodes it, and constructs
`CP_DATABASE_URL` only in the child process environment. Secrets mount
read-only. A short privileged entrypoint initializes artifact-volume ownership,
reads secrets, then drops supplementary groups/GID/UID to 10001 before launching
the API, worker, or migration. Do not grant Docker access to untrusted users:
daemon administrators can inspect container environments and secrets.

The one-shot migration must succeed before API/worker startup. PostgreSQL
health uses `pg_isready`; application readiness checks database/schema health.
The API probe uses `/ready`, not the shallow `/health` liveness route.
The worker probe checks the database/schema, not a guarantee of worker forward
progress; inspect worker heartbeat/run status through the API for that.

The smoke script checks unauthenticated rejection, database readiness, the
authenticated installed-wheel catalog (six fixture templates, repository-only
proof registry explicitly unavailable, and live/proof operations disabled), and ten
bounded software fixture operations (including actual CPU Torch training and
weight editing), then reads every result artifact and the completion audit
event. It reports a non-secret run ID. It never loads an external model or
claims model quality or release authorization. Recheck that ID after restarting
API/worker/PostgreSQL to verify persistence:

```sh
python3 deploy/control-plane/scripts/smoke.py --verify-run RUN_ID
```

For **explicit integration tests only**, publish PostgreSQL on loopback:

```sh
sh deploy/control-plane/scripts/cp.sh \
  -f deploy/control-plane/compose.test.yaml up -d postgres
# Or also create the separate control_plane_test database, idempotently:
sh deploy/control-plane/scripts/init-test-db.sh
```

The WSL host endpoint is `127.0.0.1:55432`, user `control_plane`.
Use database `control_plane_test` for tests, not the live `control_plane`.
Tests should read the password file and URL-encode it in process, never
print/serialize a credential-bearing DSN. Reuse the same override on subsequent
`up` invocations while host tests are needed. Running normal `up` without the
override removes the host publication.

## Persistence, restart, and operator lifecycle

PostgreSQL 18 persists its cluster beneath `/var/lib/postgresql` in named volume
`nf-control-plane-dev_postgres`; artifact volume is
`nf-control-plane-dev_artifacts`. Containers are disposable; these volumes are
not. API/worker/PostgreSQL use `restart: unless-stopped`. A failed migration
stays failed and blocks application startup rather than silently serving an
old schema.

```sh
sh deploy/control-plane/scripts/cp.sh restart api worker
sh deploy/control-plane/scripts/cp.sh stop
sh deploy/control-plane/scripts/cp.sh up -d
```

`down` preserves named volumes, but removes project containers/networks.
**Never use `down -v`, broad Docker prune commands, or delete artifact/secret
paths** as an operational reset. Do not rotate only the PostgreSQL password
file: the initialized cluster keeps its original password. Plan an explicit
database password change plus coordinated secret update and application restart.

## Backup and non-destructive restore rehearsal

Stop new submissions and stop the worker before a consistent application
backup. `pg_dump` itself uses a consistent database snapshot, but artifact files
and database references require operator coordination.

```sh
sh deploy/control-plane/scripts/cp.sh stop worker
sh deploy/control-plane/scripts/backup.sh
# Copy both backup files AND the secret files to protected storage.
sh deploy/control-plane/scripts/cp.sh start worker
sh deploy/control-plane/scripts/restore-test.sh \
  artifacts/control-plane-local/backups/control-plane-YYYYMMDDTHHMMSSZ.dump \
  artifacts/control-plane-local/backups/control-plane-YYYYMMDDTHHMMSSZ.dump.artifacts.tar.gz
```

Backup files are mode 0600 under ignored local artifacts. The restore script
creates a **new uniquely named database**, restores with `--exit-on-error`,
queries the restored schema version/run count, and retains that database for inspection. It never
overwrites `control_plane`. When given the companion artifact archive it also
restores to a new uniquely named Docker volume, never to the live artifact
volume and byte-compares the restored files against the archive. Both scratch
resources are retained for inspection. Only restore trusted
archives generated by the backup script. A database dump alone does not include
artifact volume bytes or secrets. Protect/retain all three, and validate actual artifact
retrieval during recovery. Production replacement/restoration of the live
cluster is deliberately not automated by this rehearsal script.

## Image pins

Official image manifests were pulled and their `RepoDigests` inspected before
pinning. Compose and Dockerfile contain both a recognizable version tag and
the immutable digest; the digest is authoritative:

| Component | Tag | Manifest digest |
|---|---|---|
| PostgreSQL | `postgres:18-alpine` | `sha256:9ba53c7d41ea093baf5c1ba635ba5434c4f979880527d49bb7d9237d3193085b` |
| Python builder/runtime | `python:3.12-slim-bookworm` | `sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df` |
| Rust build toolchain | `rust:1.93-slim-bookworm` | `sha256:5b9332190bb3b9ece73b810cd1f1e9f06343b294ce184bcb067f0747d7d333ea` |

Review security updates deliberately: pull a supported official tag, inspect
its digest, update the pin, rebuild, and rerun auth/readiness, CPU fixture,
restart-persistence, and backup/restore checks. Never replace pins with `latest`.

## Exercised local checks

The local Docker/WSL provision exercised real PostgreSQL migrations (including
v1→v2), readiness, bearer-auth rejection, all ten bounded CPU software
operations, artifact/audit retrieval, and persistence after restarting
PostgreSQL/API/worker. Backup rehearsal restored one completed run into a
separate fresh database and ten artifact files into a separate fresh volume;
`tar --compare` verified restored bytes. Long-running API/worker processes were
verified as UID/GID 10001, and secret/backup files as mode 0600. These are real
deployment checks, not claims derived from mocked unit tests. Browser/UI
validation is a separate operator check.

Do not restart the shared PostgreSQL container while integration suites use
another database in that same container: separate databases do not isolate
tests from server restarts. Coordinate restart rehearsals before/after tests.
