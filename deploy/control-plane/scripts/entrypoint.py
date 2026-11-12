"""Construct a process-local DSN from the mounted secret without logging it."""

import os
import sys
from pathlib import Path
from urllib.parse import quote


def database_url() -> str:
    password = Path(os.environ["CP_DATABASE_PASSWORD_FILE"]).read_text().strip()
    return (
        "postgresql://control_plane:"
        + quote(password, safe="")
        + "@postgres:5432/control_plane"
    )


if __name__ == "__main__":
    os.environ["CP_DATABASE_URL"] = database_url()
    token_file = os.environ.pop("CP_OPERATOR_TOKEN_FILE")
    os.environ["CP_OPERATOR_TOKEN"] = Path(token_file).read_text().strip()
    if os.getuid() == 0:
        os.chown(os.environ["CP_ARTIFACT_ROOT"], 10001, 10001)
        os.setgroups([])
        os.setgid(10001)
        os.setuid(10001)
    os.execvp(sys.argv[1], sys.argv[1:])
