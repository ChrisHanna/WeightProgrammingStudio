"""Readiness probes intentionally omit credential-bearing diagnostics."""

import os
import sys
import urllib.request
from pathlib import Path

from entrypoint import database_url


def main() -> int:
    try:
        os.environ["CP_DATABASE_URL"] = database_url()
        if "--database" in sys.argv:
            from neural_foundry.control_plane.service import ControlPlaneService

            ControlPlaneService(
                os.environ["CP_DATABASE_URL"], Path(os.environ["CP_ARTIFACT_ROOT"])
            ).health()
        else:
            token = Path(os.environ["CP_OPERATOR_TOKEN_FILE"]).read_text().strip()
            request = urllib.request.Request(
                "http://127.0.0.1:8000/ready",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status != 200:
                    return 1
        return 0
    except Exception:  # noqa: BLE001 -- never leak credential-bearing probe diagnostics
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
