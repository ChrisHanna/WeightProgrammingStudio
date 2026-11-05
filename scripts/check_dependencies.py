#!/usr/bin/env python3
"""Check locked Rust license scope and retain actual upstream notice texts."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = {
    "Apache-2.0",
    "Zlib",
    "MIT",
    "MIT OR Apache-2.0",
    "Apache-2.0 OR MIT",
    "Unlicense OR MIT",
    "MIT/Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
    "Apache-2.0 OR BSL-1.0",
    "(MIT OR Apache-2.0) AND Unicode-3.0",
}


def rust_inventory() -> tuple[dict, str]:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--locked"], cwd=ROOT
        )
    )
    if len(metadata["packages"]) > 128:
        raise SystemExit("Dependency count exceeds reviewed M1 bound")
    records = []
    notices = [
        "Neural Programming Foundry: upstream Rust licenses and notices\n",
        "Generated from locked package sources; these texts keep their original terms.\n",
    ]
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        expression = package["license"]
        if expression not in ALLOW:
            raise SystemExit(f"Unreviewed Rust license: {package['name']} {expression}")
        if package["source"] is None:
            if not package["name"].startswith("nf-"):
                raise SystemExit("Unreviewed path dependency")
            continue
        if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
            raise SystemExit("Unreviewed dependency source")
        directory = Path(package["manifest_path"]).parent
        paths = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        )
        if not paths or len(paths) > 16:
            raise SystemExit(f"Missing/unbounded license texts: {package['name']}")
        licenses = []
        for path in paths:
            data = path.read_bytes()
            if len(data) > 524288:
                raise SystemExit("Upstream notice exceeds review bound")
            licenses.append({"file": path.name, "sha256": hashlib.sha256(data).hexdigest()})
            notices.append(f"\n=== {package['name']} {package['version']} / {path.name} ===\n")
            notices.append(data.decode("utf-8"))
            notices.append("\n")
        if package["name"] == "libsqlite3-sys":
            with (directory / "sqlite3/sqlite3.c").open("rb") as source:
                header = b"".join(source.read(8192).splitlines(keepends=True)[:49])
            if b"The author disclaims copyright" not in header:
                raise SystemExit("SQLite upstream public-domain notice requires review")
            licenses.append(
                {"file": "sqlite3/sqlite3.c#header", "sha256": hashlib.sha256(header).hexdigest()}
            )
            notices.append(
                f"\n=== {package['name']} {package['version']} / bundled SQLite public-domain header ===\n"
            )
            notices.append(header.decode("utf-8"))
        records.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license_expression": expression,
                "source": "crates.io",
                "license_files": licenses,
            }
        )
    text = "".join(notices)
    if len(text.encode()) > 16777216:
        raise SystemExit("Combined notice bound exceeded")
    return {
        "schema_version": 1,
        "scope": "locked_rust_dependency_superset_including_build_and_target_specific",
        "cargo_lock_sha256": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest(),
        "packages": records,
    }, text


def python_inventory() -> list[dict]:
    reviewed = {
        "Jinja2": "BSD-3-Clause",
        "pathspec": "MPL-2.0 (development tool only)",
        "setuptools": "MIT",
        "mpmath": "BSD-3-Clause",
        "sympy": "BSD-3-Clause",
    }
    result = []
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        name = dist.metadata["Name"]
        if name == "neural-foundry":
            continue
        license_name = dist.metadata.get("License-Expression") or dist.metadata.get("License")
        if name in reviewed:
            candidates = [
                f
                for f in dist.files or []
                if ".dist-info/" in str(f) and "license" in str(f).lower()
            ]
            if not candidates:
                raise SystemExit(f"Missing tool license text: {name}")
            license_name = reviewed[name]
        if not license_name or license_name.splitlines()[0] not in {
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "MIT OR Apache-2.0",
            "Apache-2.0 OR BSD-2-Clause",
            "PSF-2.0",
            "MPL-2.0 (development tool only)",
        }:
            raise SystemExit(f"Unreviewed Python tool license: {name} {license_name}")
        result.append(
            {
                "name": name,
                "version": dist.version,
                "license": license_name.splitlines()[0],
                "scope": "development_or_optional_cpu_reference_not_wheel_runtime",
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate reviewed inventory and bundled notices"
    )
    parser.add_argument(
        "--python", action="store_true", help="also record this development environment"
    )
    args = parser.parse_args()
    inventory, notices = rust_inventory()
    documents = {
        ROOT / "licenses/dependencies.json": json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        ROOT / "python/neural_foundry/data/THIRD_PARTY_NOTICES.txt": notices,
    }
    if args.python:
        packages = python_inventory()
        documents[ROOT / "licenses/python-tools.json"] = (
            json.dumps(packages, indent=2, sort_keys=True) + "\n"
        )
        documents[ROOT / "requirements-dev.lock"] = (
            "# Development/optional CPU tools, not wheel runtime dependencies.\n--extra-index-url https://download.pytorch.org/whl/cpu\n"
            + "".join(f"{p['name']}=={p['version']}\n" for p in packages)
        )
    for path, content in documents.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        elif not path.is_file() or path.read_bytes() != content.encode("utf-8"):
            raise SystemExit(f"Dependency/notice drift: {path.relative_to(ROOT)}")
    print(f"Reviewed {len(inventory['packages'])} locked Rust packages; notices match")


if __name__ == "__main__":
    main()
