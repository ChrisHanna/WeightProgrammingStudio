"""Dedicated worker and transactional migration entrypoint."""

from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path

from .models import ControlPlaneError
from .service import ControlPlaneService
from .store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    dsn = os.environ.get("CP_DATABASE_URL", "")
    try:
        if args.migrate_only:
            Store(dsn).migrate()
            print("Control-plane PostgreSQL schema ready.")
            return 0
        service = ControlPlaneService(dsn, Path(os.environ.get("CP_ARTIFACT_ROOT", ".control-plane-artifacts")))
        service.health()
        worker_id = os.environ.get("CP_WORKER_ID", "singlehost-worker")
        stopping = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda signum, frame: stopping.set())
        try:
            while not stopping.is_set():
                result = service.worker_tick(worker_id)
                if args.once:
                    break
                if result is None:
                    stopping.wait(1)
        finally:
            service.abandon_owned(worker_id)
        return 0
    except ControlPlaneError as exc:
        print(f"{exc.code}: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
