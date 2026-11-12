"""Explicit, non-destructive local secret initialization; never prints secret values."""

import os
import secrets
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    directory = root / "artifacts" / "control-plane-local" / "secrets"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    for name in ("postgres_password", "operator_token"):
        path = directory / name
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.stat().st_size < 32:
                raise SystemExit(f"Refusing unsafe existing secret file: {path.name}")
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(48) + "\n")
        path.chmod(0o600)
        if os.name == "posix" and path.stat().st_mode & 0o077:
            raise SystemExit("Secret permissions are not private; enable WSL DrvFS metadata first.")
    print("Local secret files initialized (values intentionally hidden).")


if __name__ == "__main__":
    main()
