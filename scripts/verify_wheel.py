#!/usr/bin/env python3
"""Verify a real wheel offline in a fresh environment and run the packaged demo."""

import argparse
import json
import os
import stat
import subprocess
import venv
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("work_dir", type=Path, help="explicit empty verification directory")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit("Wheel verification requires an empty directory; nothing was deleted")
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for required in [
            "neural_foundry/demo.py",
            "neural_foundry/data/m1_fixture.json",
            "neural_foundry/data/THIRD_PARTY_NOTICES.txt",
        ]:
            if required not in names:
                raise SystemExit(f"Missing wheel payload: {required}")
        if not any(
            name.startswith("neural_foundry/_native") and name.endswith(".so") for name in names
        ):
            raise SystemExit("Wheel lacks the actual native extension")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names) or not any(
            name.endswith(".dist-info/licenses/NOTICE") for name in names
        ):
            raise SystemExit("Wheel lacks project license/notice")
        metadata = archive.read(
            next(name for name in names if name.endswith(".dist-info/METADATA"))
        ).decode()
        if "License-Expression: Apache-2.0" not in metadata:
            raise SystemExit("Wheel license expression differs")
        if any(
            line.startswith("Requires-Dist:") and "extra ==" not in line
            for line in metadata.splitlines()
        ):
            raise SystemExit("Unexpected mandatory Python runtime dependency")
        if any(name.split("/", 1)[0].endswith(".data") for name in names):
            raise SystemExit("Wheel .data schemes are unsupported by the stdlib-only verifier")
    environment = work / "environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site_packages = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()).resolve()
    site_packages.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            destination = (site_packages / member.filename).resolve()
            if not destination.is_relative_to(site_packages):
                raise SystemExit("Wheel member escapes the clean environment")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise SystemExit("Wheel symlinks are unsupported by the stdlib-only verifier")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    code = """
import json,sys
from pathlib import Path
import neural_foundry
from neural_foundry.demo import run
result=run()
assert result['values']==[2.0,4.0,6.0,8.0]
assert result['source_preserved'] and result['snapshot_parity']
assert result['candidate']==result['replayed_candidate']
assert result['promotion_status']=='not_evaluated'
assert Path(neural_foundry.__file__).is_relative_to(Path(sys.prefix))
assert not any(name in sys.modules for name in ('torch','numpy','transformers','requests'))
print(json.dumps({'module':neural_foundry.__file__,'version':neural_foundry.__version__,'result':result}))
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", code], cwd=work, check=True, capture_output=True, env=env
    )
    proof = json.loads(completed.stdout)
    module = subprocess.run(
        [str(python), "-I", "-m", "neural_foundry.demo"],
        cwd=work,
        check=True,
        capture_output=True,
        env=env,
    )
    assert json.loads(module.stdout)["candidate"] == proof["result"]["candidate"]
    print(
        json.dumps(
            {
                "wheel": wheel.name,
                "isolated_environment": str(environment),
                "proof": proof,
                "module_entrypoint_passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
