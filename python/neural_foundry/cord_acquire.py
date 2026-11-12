"""Allowlisted, pinned CORD/BIPIA acquisition; native tests are never opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

CORD_REV = "7f0115a4b758a71d6473b8d085751692da2fef98"
BIPIA_REV = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"
CORD_CODE_REV = "327310ce58c1623255821d062b3a759ff3789e3c"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(root: Path, metadata: Path) -> dict:
    review = json.loads(metadata.read_bytes())
    if (review["cord"]["dataset_revision"] != CORD_REV
            or review["bipia"]["revision"] != BIPIA_REV):
        raise ValueError("unapproved source revision")
    shards = [s for s in review["cord"]["shards"]
              if s["path"].startswith(("data/train-", "data/validation-"))]
    if len(shards) != 5 or sum(s["bytes"] for s in shards) != 2_073_081_477:
        raise ValueError("approved train/validation allowlist differs")
    sources = []
    for shard in shards:
        sources.append((f"cord-v2-{CORD_REV}/{shard['path']}",
                        f"https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{CORD_REV}/{shard['path']}",
                        shard["bytes"], shard["sha256"], None))
    for name in ("LICENSE-CC-BY", "README.md"):
        sources.append((f"cord-v2-{CORD_REV}/notices/{name}",
                        f"https://raw.githubusercontent.com/clovaai/cord/{CORD_CODE_REV}/{name}",
                        None, None, None))
    for name in ("LICENSE", "NOTICE.md", "README.md", "benchmark/README.md",
                 "benchmark/text_attack_train.json"):
        sources.append((f"bipia-{BIPIA_REV}/{name}",
                        f"https://raw.githubusercontent.com/microsoft/BIPIA/{BIPIA_REV}/{name}",
                        5999 if name.endswith("text_attack_train.json") else None, None,
                        "190918308f4ef25d5f43d7c1b118bf3fe5dc61d2"
                        if name.endswith("text_attack_train.json") else None))
    records = []
    for relative, url, size, expected, git_blob in sources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as stream:
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
            partial.rename(path)
        actual = file_hash(path)
        if (size is not None and path.stat().st_size != size) or (expected and actual != expected):
            raise ValueError(f"source size/hash differs: {relative}")
        if git_blob:
            data = path.read_bytes()
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != git_blob:
                raise ValueError("BIPIA pinned Git blob differs")
        records.append({"path": relative, "url": url, "bytes": path.stat().st_size,
                        "sha256": actual, "git_blob_sha": git_blob})
        print(f"verified {relative} {actual}", flush=True)
    manifest = {
        "kind": "nf_cord_bipia_acquisition_v1", "cord_revision": CORD_REV,
        "bipia_revision": BIPIA_REV, "files": records,
        "cord_attribution": "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing; "
                            "Clova AI Research, NAVER Corp.; CC BY 4.0; "
                            "https://github.com/clovaai/cord; "
                            "https://creativecommons.org/licenses/by/4.0/",
        "bipia_use": "Internal evaluation only. MIT notices retained. Research-use wording "
                     "remains an OPEN question; no production redistribution clearance.",
        "sealed": {"cord_test_bytes": 234202795, "cord_test_accessed": False,
                   "bipia_test_accessed": False},
    }
    (root / "cord-bipia-acquisition.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--metadata", type=Path, default=Path(
        "artifacts/cord-bipia-license-review-20261111/metadata.json"))
    args = parser.parse_args()
    acquire(args.root, args.metadata)


if __name__ == "__main__":
    main()
