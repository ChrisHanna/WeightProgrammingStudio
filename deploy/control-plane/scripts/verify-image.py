"""Verify the image contains the exact current control-plane source and UI."""

import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    package = root / "python" / "neural_foundry" / "control_plane"
    expected = {
        str(path.relative_to(package)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    code = (
        "import hashlib,json; from pathlib import Path; "
        "import neural_foundry.control_plane as cp; "
        "root=Path(cp.__file__).parent; "
        "print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}))"
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
         "nf-control-plane:local", "-c", code],
        capture_output=True, text=True, check=True,
    )
    actual = json.loads(result.stdout)
    if expected != actual:
        mismatches = sorted(key for key in expected.keys() | actual.keys()
                            if expected.get(key) != actual.get(key))
        raise SystemExit("Image/source mismatch; rebuild before startup: " + ", ".join(mismatches))
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"source_files": len(expected), "source_tree_sha256": digest, "matches_image": True}))


if __name__ == "__main__":
    main()
