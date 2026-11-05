"""Actual SDK/native/CAS intake, never coordinator import execution."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import NativeError, content_digest, gyms

IMAGE = "sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"


def package(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "import-side-effect"
    (source / "gym.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('BAD')\n"
        "raise RuntimeError('coordinator must not import this module')\n"
        "def evaluate(x): return x\ndef score(x,y): return float(x==y)\n"
        "def oracle(x): return x\nclass Environment: pass\n"
    )
    (source / "LICENSE").write_text("Fixture license: local test use only.\n")
    (source / "lock.json").write_text(
        json.dumps({"schema_version": 1, "image": IMAGE, "packages": {}})
    )
    (source / "data.jsonl").write_text('{"task":"a","group":"g","x":2}\n')
    (source / "manifest.json").write_text('{"name":"fixture","version":"1"}')
    assets = {
        "gym.py": "code",
        "LICENSE": "license",
        "lock.json": "dependency_lock",
        "data.jsonl": "data",
        "manifest.json": "manifest",
    }

    def ep(symbol: str) -> dict[str, str]:
        return {"file": "gym.py", "symbol": symbol}

    draft: dict[str, Any] = {
        "schema_version": 1,
        "name": "fixture",
        "version": "1",
        "usage_rights": "local test only",
        "confidentiality": "confidential",
        "assets": {name: {"kind": kind, "expected": None} for name, kind in assets.items()},
        "dependency_lock": "lock.json",
        "licenses": ["LICENSE"],
        "interface": {
            "kind": "single_shot",
            "evaluate": ep("evaluate"),
            "score": ep("score"),
            "oracle": ep("oracle"),
            "input": {"type": "integer", "min": 0, "max": 100},
            "output": {"type": "number", "min": 0, "max": 100},
        },
        "dataset": {
            "kind": "frozen",
            "files": ["data.jsonl"],
            "task_key": "task",
            "group_key": "group",
        },
    }
    return source, tmp_path / "store", draft


def view(handle: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(handle.manifest_bytes())
    assert result["status"] == "snapshotted_not_validated"
    assert result["trust"] == "untrusted_local_source"
    assert "approval" not in result and "validation" not in result
    return result


def test_custom_gym_import_real_snapshot_no_side_effects_or_extra_files(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    (source / "not-selected.py").write_text("raise RuntimeError('hidden')")
    handle = gyms.import_local(source, draft, store)
    metadata = view(handle)
    assert set(metadata["assets"]) == set(draft["assets"])
    assert handle.snapshot_id == content_digest(handle.manifest_bytes())
    assert handle.asset_bytes("gym.py") == (source / "gym.py").read_bytes()
    assert not (tmp_path / "import-side-effect").exists()
    assert not (source / "__pycache__").exists()
    with pytest.raises(NativeError, match="not in snapshot scope"):
        handle.asset_bytes("not-selected.py")
    with pytest.raises(NativeError):
        handle.asset_bytes("../data.jsonl")
    immutable: Any = handle
    with pytest.raises(AttributeError):
        immutable.snapshot_id = "forged"
    with pytest.raises(TypeError):
        type(handle)()


def test_custom_gym_import_nested_package_is_selected_without_discovery(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    folder = source / "pkg"
    folder.mkdir()
    (source / "gym.py").rename(folder / "gym.py")
    (folder / "__init__.py").write_text("raise RuntimeError('package import forbidden')")
    draft["assets"]["pkg/gym.py"] = draft["assets"].pop("gym.py")
    draft["assets"]["pkg/__init__.py"] = {"kind": "code", "expected": None}
    for key in ["evaluate", "score", "oracle"]:
        draft["interface"][key]["file"] = "pkg/gym.py"
    handle = gyms.import_local(source, draft, store)
    assert handle.asset_bytes("pkg/gym.py") == (folder / "gym.py").read_bytes()
    assert not (tmp_path / "import-side-effect").exists()
    (folder / "gym.py").unlink()
    alias_target = tmp_path / "other"
    alias_target.mkdir()
    (alias_target / "gym.py").write_text("secret")
    (folder / "__init__.py").unlink()
    folder.rmdir()
    folder.symlink_to(alias_target, target_is_directory=True)
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)


def test_custom_gym_import_single_file_scope_and_persisted_restore(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    first = gyms.import_file(source / "gym.py", draft, store)
    expected = first.manifest_bytes()
    code = first.asset_bytes("gym.py")
    (source / "gym.py").unlink()
    restored = gyms.restore(first.snapshot_id, store)
    assert restored.manifest_bytes() == expected
    assert restored.asset_bytes("gym.py") == code
    with pytest.raises(NativeError, match="declared main"):
        gyms.import_file(source / "other.py", draft, store)


def test_custom_gym_import_changed_code_data_lock_license_and_manifest_are_new_versions(
    tmp_path: Path,
) -> None:
    source, store, draft = package(tmp_path)
    first = gyms.import_local(source, draft, store)
    identities = {first.snapshot_id}
    for name in ["gym.py", "data.jsonl", "LICENSE", "lock.json", "manifest.json"]:
        old = view(gyms.import_local(source, draft, store))
        path = source / name
        if name == "lock.json":
            value = json.loads(path.read_bytes())
            value["image"] = "sha256:" + "1" * 64
            path.write_text(json.dumps(value))
        elif name == "manifest.json":
            path.write_text('{"name":"fixture","version":"2"}')
        else:
            path.write_bytes(path.read_bytes() + b"\n#new identity\n")
        handle = gyms.import_local(source, draft, store)
        assert handle.snapshot_id not in identities
        identities.add(handle.snapshot_id)
        new = view(handle)
        if name == "gym.py":
            assert new["code_identity"] != old["code_identity"]
        if name == "data.jsonl":
            assert new["data_identity"] != old["data_identity"]
    assert len(identities) == 6
    assert gyms.restore(first.snapshot_id, store).manifest_bytes() == first.manifest_bytes()


def test_custom_gym_import_episode_generator_schemas_and_seeds(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    (source / "config.json").write_text('{"count":3}')
    draft["assets"]["config.json"] = {"kind": "config", "expected": None}
    ep = {"file": "gym.py", "symbol": "Environment"}
    simple = {"type": "boolean"}
    draft["interface"] = {
        "kind": "episode",
        "factory": ep,
        "score": {**ep, "symbol": "score"},
        "oracle": {**ep, "symbol": "oracle"},
        "reset": {"type": "object", "properties": {}, "required": []},
        "observation": simple,
        "action": simple,
        "result": simple,
        "max_steps": 10,
        "max_episode_ms": 1000,
    }
    draft["dataset"] = {
        "kind": "generator",
        "factory": ep,
        "config": "config.json",
        "seeds": [1, 2],
        "task_key": "task",
        "group_key": "group",
    }
    first = gyms.import_local(source, draft, store)
    assert view(first)["draft"]["interface"]["kind"] == "episode"
    draft["dataset"]["seeds"] = [2, 3]
    second = gyms.import_local(source, draft, store)
    assert view(first)["data_identity"] != view(second)["data_identity"]
    assert not (tmp_path / "import-side-effect").exists()
    (source / "config.json").write_text('{"count":4}')
    third = gyms.import_local(source, draft, store)
    assert view(third)["data_identity"] != view(second)["data_identity"]
    (source / "gym.py").write_bytes((source / "gym.py").read_bytes() + b"\n# generator revision\n")
    fourth = gyms.import_local(source, draft, store)
    assert view(fourth)["data_identity"] != view(third)["data_identity"]
    draft["interface"]["max_steps"] = 0
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)


@pytest.mark.parametrize(
    "name", ["../escape.py", "/etc/passwd", "pkg/../gym.py", "pkg\\gym.py", "a//b", "x:y", "."]
)
def test_custom_gym_import_path_traversal_rejected(tmp_path: Path, name: str) -> None:
    source, store, draft = package(tmp_path)
    draft["assets"][name] = {"kind": "code", "expected": None}
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    assert not store.exists()


def test_custom_gym_import_symlink_hardlink_and_nonregular_sources_rejected(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("secret")
    (source / "gym.py").unlink()
    (source / "gym.py").symlink_to(outside)
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    (source / "gym.py").unlink()
    os.link(outside, source / "gym.py")
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    (source / "gym.py").unlink()
    os.mkfifo(source / "gym.py")
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(NativeError):
        gyms.import_local(alias, draft, store)


def test_custom_gym_import_expected_digests_dependencies_licenses_and_duplicate_json(
    tmp_path: Path,
) -> None:
    source, store, draft = package(tmp_path)
    draft["assets"]["gym.py"]["expected"] = "sha256:" + "0" * 64
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    draft["assets"]["gym.py"]["expected"] = content_digest((source / "gym.py").read_bytes())
    lock: dict[str, Any] = {
        "schema_version": 1,
        "image": IMAGE,
        "packages": {
            "fixture-package": {
                "version": "1.2.3",
                "integrity": content_digest(b"declared dependency"),
                "license": "LICENSE",
            }
        },
    }
    (source / "lock.json").write_text(json.dumps(lock))
    assert (
        view(gyms.import_local(source, draft, store))["dependency"]["packages"]["fixture-package"][
            "version"
        ]
        == "1.2.3"
    )
    for version in [">=1.2", "latest", "main"]:
        lock["packages"]["fixture-package"]["version"] = version
        (source / "lock.json").write_text(json.dumps(lock))
        with pytest.raises(NativeError):
            gyms.import_local(source, draft, store)
    (source / "lock.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
    draft["licenses"] = []
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)


def test_custom_gym_import_schema_roles_bounds_and_cas_tamper(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    bad = copy.deepcopy(draft)
    bad["interface"]["input"] = {"type": "object", "properties": {}, "required": ["absent"]}
    with pytest.raises(NativeError):
        gyms.import_local(source, bad, store)
    bad = copy.deepcopy(draft)
    bad["interface"]["input"] = {"type": "boolean", "surprise": True}
    with pytest.raises(NativeError):
        gyms.import_local(source, bad, store)
    bad = copy.deepcopy(draft)
    bad["assets"]["gym.py"]["kind"] = "data"
    with pytest.raises(NativeError):
        gyms.import_local(source, bad, store)
    bad = copy.deepcopy(draft)
    bad["surprise"] = True
    with pytest.raises(NativeError):
        gyms.import_local(source, bad, store)
    with pytest.raises(NativeError):
        gyms.import_local(source, b'{"schema_version":1,"schema_version":1}', store)
    handle = gyms.import_local(source, draft, store)
    digest = view(handle)["assets"]["data.jsonl"]["digest"].split(":")[1]
    blob = store / "blobs" / digest
    # An owner-side corruption fixture is not a trusted worker edit.
    raw = blob.read_bytes()
    blob.unlink()
    blob.write_bytes(b"!" + raw[1:])
    with pytest.raises(NativeError):
        gyms.restore(handle.snapshot_id, store)
    with pytest.raises(NativeError):
        handle.asset_bytes("data.jsonl")


def test_custom_gym_import_oversized_file_and_unselected_changes(tmp_path: Path) -> None:
    source, store, draft = package(tmp_path)
    first = gyms.import_local(source, draft, store)
    (source / "unselected.txt").write_text("not part of this package")
    assert gyms.import_local(source, draft, store).snapshot_id == first.snapshot_id
    (source / "data.jsonl").write_bytes(b"x" * (1_048_576 + 1))
    with pytest.raises(NativeError):
        gyms.import_local(source, draft, store)
