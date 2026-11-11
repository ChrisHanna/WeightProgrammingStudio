"""Operator-only Stage 1 account/deployment bootstrap; no packages or holdout data are installed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

ACCOUNT = "nf-custodian"
HOME = Path("/var/lib/neural-foundry-custodian")
DEPLOYMENT = Path("/opt/neural-foundry-custodian")
LAUNCHER = Path("/usr/local/bin/nf-custodian-stage1")


def _inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("custodian deployment source must not contain symlinks")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--runtime-site-packages", type=Path, required=True,
                        help="Existing CPU runtime to snapshot read-only; nothing is installed")
    parser.add_argument("--runtime-package", action="append", required=True,
                        help="Explicit cached package/file name required by software fixtures")
    args = parser.parse_args()
    if os.name != "posix" or os.geteuid() != 0:
        parser.error("run explicitly as the Linux administrator; no privilege escalation is attempted")
    import pwd

    source = args.package_root.resolve(strict=True)
    if source.name != "neural_foundry" or not (source / "study" / "custodian.py").is_file():
        parser.error("expected the neural_foundry package directory")
    try:
        account = pwd.getpwnam(ACCOUNT)
    except KeyError:
        subprocess.run(["useradd", "--system", "--create-home", "--home-dir", str(HOME),
                        "--shell", "/usr/sbin/nologin", "--no-log-init", ACCOUNT], check=True)
        account = pwd.getpwnam(ACCOUNT)
    if (account.pw_uid == 0 or account.pw_dir != str(HOME)
        or account.pw_shell != "/usr/sbin/nologin"):
        raise ValueError("existing account differs; refusing to alter another account")
    HOME.mkdir(mode=0o700, exist_ok=True)
    if HOME.is_symlink():
        raise ValueError("account home must not be a symlink")
    os.chown(HOME, account.pw_uid, account.pw_gid)
    os.chmod(HOME, 0o700)
    DEPLOYMENT.mkdir(mode=0o755, exist_ok=True)
    info = DEPLOYMENT.stat()
    if DEPLOYMENT.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError("deployment parent must be protected by the administrator")
    releases = DEPLOYMENT / "releases"
    releases.mkdir(mode=0o755, exist_ok=True)
    runtime_source = args.runtime_site_packages.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix=".runtime-", dir=releases) as temporary:
        runtime_staging = Path(temporary) / "runtime"
        runtime_staging.mkdir(mode=0o755)
        names = set(args.runtime_package)
        for name in tuple(names):
            if (runtime_source / f"{name}.libs").is_dir():
                names.add(f"{name}.libs")
        for name in sorted(names):
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("runtime package must be a basename")
            source_entry = runtime_source / name
            if source_entry.is_symlink() or not source_entry.exists():
                raise ValueError("explicit runtime package is absent or symlinked")
            target = runtime_staging / name
            if source_entry.is_dir():
                shutil.copytree(source_entry, target, symlinks=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pth", "*.egg-link"))
            else:
                shutil.copyfile(source_entry, target)
        for metadata in runtime_source.glob("*.dist-info"):
            shutil.copytree(metadata, runtime_staging / metadata.name, symlinks=True)
        runtime_inventory = _inventory(runtime_staging)
        runtime_digest = hashlib.sha256(json.dumps(runtime_inventory, sort_keys=True).encode()).hexdigest()
        runtime_release = releases / ("runtime-" + runtime_digest)
        if not runtime_release.exists():
            for path in runtime_staging.rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            os.rename(runtime_staging, runtime_release)
        elif _inventory(runtime_release) != runtime_inventory:
            raise ValueError("existing runtime snapshot differs")
    inventory = _inventory(source)
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    release = releases / digest
    if not release.exists():
        with tempfile.TemporaryDirectory(prefix=".stage1-", dir=releases) as temporary:
            staging = Path(temporary) / "release"
            package = staging / "python" / "neural_foundry"
            shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if _inventory(package) != inventory:
                raise ValueError("package changed during deployment; no release was published")
            for path in staging.rglob("*"):
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            os.chmod(staging, 0o755)
            os.rename(staging, release)
    if _inventory(release / "python" / "neural_foundry") != inventory:
        raise ValueError("existing deployment inventory differs")
    code = (
        f"import sys; sys.path[:0]=[{str(release / 'python')!r},{str(runtime_release)!r}]; "
        "from neural_foundry.study.custodian import main; raise SystemExit(main())"
    )
    command = ["/usr/sbin/runuser", "-u", ACCOUNT, "--", "/usr/bin/env", "-i",
               f"HOME={HOME}", "PATH=/usr/bin:/bin", "/usr/bin/python3", "-I", "-B", "-c", code,
               "--required-uid", str(account.pw_uid), "--software-fixtures-only"]
    launcher = "#!/bin/sh\nexec " + " ".join(shlex.quote(value) for value in command) + ' "$@"\n'
    if LAUNCHER.is_symlink():
        raise ValueError("refusing a symlinked launcher")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-custodian-", dir=LAUNCHER.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(launcher)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o755)
        os.replace(temporary, LAUNCHER)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"kind": "nf_custodian_stage1_deployment_v1", "account": ACCOUNT,
               "uid": account.pw_uid, "package_inventory_digest": "sha256:" + digest,
               "runtime_inventory_digest": "sha256:" + runtime_digest,
               "runtime_release": str(runtime_release),
               "launcher": str(LAUNCHER), "release": str(release),
               "state_root": str(HOME), "software_fixtures_only": True,
               "native_ledger_bridge": False, "signed": False, "container_deployment": False}
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
