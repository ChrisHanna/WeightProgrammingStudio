"""Actual Rust/SQLite/CAS scopes and qualified namespace data isolation."""

from __future__ import annotations

import copy
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

BINARY = Path(__file__).resolve().parents[1] / "target/debug/examples/fixture_splits"


class Controller:
    def __init__(self, process: subprocess.Popen[str], ready: dict[str, Any]) -> None:
        self.process = process
        self.ready = ready

    def send(self, op: str, **fields: Any) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"op": op, **fields}) + "\n")
        self.process.stdin.flush()
        reply: dict[str, Any] = json.loads(self.process.stdout.readline())
        return reply

    def ok(self, op: str, **fields: Any) -> Any:
        reply = self.send(op, **fields)
        assert "error" not in reply, reply
        return reply["ok"]

    def grant(self, recipient: str, **fields: Any) -> dict[str, Any]:
        result: dict[str, Any] = self.ok("grant", recipient=recipient, **fields)
        return result


@contextmanager
def server(tmp_path: Path) -> Iterator[Controller]:
    root = tmp_path / "controller"
    root.mkdir(mode=0o700)
    process = subprocess.Popen(
        [str(BINARY), str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write("{}\n")
    process.stdin.flush()
    ready = json.loads(process.stdout.readline())
    assert ready.get("ready") is True, ready
    try:
        yield Controller(process, ready)
    finally:
        process.stdin.close()
        process.wait(timeout=15)
        assert process.stderr is not None
        errors = process.stderr.read()
        assert process.returncode == 0, errors


def request(welcome: dict[str, Any], labels: bool = False) -> dict[str, Any]:
    return {
        key: copy.deepcopy(welcome[key])
        for key in [
            "schema_version",
            "contract",
            "revision",
            "binding",
            "worker",
            "lease_id",
            "split",
            "panel",
        ]
    } | {"item": welcome["items"][0], "labels": labels}


def test_split_broker_proposer_only_development_inputs_no_hidden_metadata(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        welcome = controller.grant("proposer")
        assert welcome["split"] == "development" and welcome["labels_allowed"] is False
        payload = controller.ok("read", request=request(welcome))
        assert payload["input"]["role"] == "development" and payload["label"] is None
        assert "PRIVATE_LABEL" not in json.dumps(payload)
        assert (
            controller.send("read", request=request(welcome, labels=True))["error"]["code"]
            == "PERMISSION_DENIED"
        )
        for split in ["selection", "holdout"]:
            forged = request(welcome)
            forged["split"] = split
            forged["item"] = controller.ready["bindings"][0]["items"][split]
            assert "error" in controller.send("read", request=forged)
        assert controller.ok("panel") is None


@pytest.mark.parametrize(
    "recipient,role",
    [("development_evaluator", "development"), ("selection_evaluator", "selection")],
)
def test_split_broker_evaluator_exact_split_and_labels(
    tmp_path: Path, recipient: str, role: str
) -> None:
    with server(tmp_path) as controller:
        welcome = controller.grant(recipient)
        payload = controller.ok("read", request=request(welcome, labels=True))
        assert payload["classification"] == role
        assert payload["label"]["secret"] == f"PRIVATE_LABEL_0_{role}"
        forged = request(welcome, labels=True)
        forged["item"] = controller.ready["bindings"][0]["items"]["holdout"]
        assert "error" in controller.send("read", request=forged)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lease_id", "forged"),
        ("revision", 2),
        ("contract", "sha256:" + "0" * 64),
        ("binding", "sha256:" + "1" * 64),
        ("panel", "sha256:" + "2" * 64),
    ],
)
def test_split_broker_forged_ids_revision_panel_and_worker_lease_denied(
    tmp_path: Path, field: str, value: Any
) -> None:
    with server(tmp_path) as controller:
        welcome = controller.grant("proposer")
        forged = request(welcome)
        forged[field] = value
        assert "error" in controller.send("read", request=forged)
        assert controller.ok("read", request=request(welcome))["label"] is None


def test_split_broker_freezes_panel_before_any_holdout_then_closes_search(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        dev = controller.grant("proposer")
        prepared = controller.ok("prepare", request=request(dev))
        assert "error" in controller.send("grant", recipient="final_evaluator")
        controller.ok("final_phase")
        assert "error" in controller.send("grant", recipient="final_evaluator")
        first = controller.ok("freeze")["panel"]
        assert controller.ok("freeze")["panel"] == first
        assert "error" in controller.send("freeze", candidate_salt="different-finalist")
        final = controller.grant("final_evaluator")
        assert final["panel"] == first
        payload = controller.ok("read", request=request(final, labels=True))
        assert payload["input"]["role"] == "holdout"
        assert payload["label"]["secret"] == "PRIVATE_LABEL_0_holdout"
        assert "error" in controller.send("read", request=request(dev))
        assert "error" in controller.send("grant", recipient="selection_evaluator")
        assert "error" in controller.send("search")
        assert "error" in controller.send("reserve_search")
        assert "error" in controller.send("dispatch_prepared", access=prepared["access"])
        controller.ok("reopen")
        assert controller.ok("panel")[0] == first
        assert (
            controller.ok("read", request=request(final, labels=True))["classification"]
            == "holdout"
        )


def test_split_broker_panel_limits_and_unknown_task_access(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        controller.ok("final_phase")
        assert "error" in controller.send("freeze", candidate_count=0)
        assert "error" in controller.send("freeze", candidate_count=2)
        assert controller.ok("panel") is None
        controller.ok("freeze")
        final = controller.grant("final_evaluator")
        forged = request(final, labels=True)
        forged["item"] = "sha256:" + "f" * 64
        assert "error" in controller.send("read", request=forged)


def test_split_broker_expiry_stop_and_actual_access_quota(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        short = controller.grant("proposer", ttl_ms=100)
        time.sleep(0.15)
        assert "error" in controller.send("read", request=request(short))
        valid = controller.grant("proposer")
        for _ in range(64):
            controller.ok("read", request=request(valid))
        assert controller.send("read", request=request(valid))["error"]["code"] == "QUOTA_EXCEEDED"
        other = controller.grant("proposer")
        controller.ok("stop")
        assert "error" in controller.send("read", request=request(other))


@pytest.mark.parametrize(
    "option",
    [
        "duplicate_inputs",
        "cross_binding_alias",
        "wrong_dataset_role",
        "wrong_identity",
        "unresolved",
    ],
)
def test_split_broker_unresolved_or_aliased_actual_datasets_fail(
    tmp_path: Path, option: str
) -> None:
    root = tmp_path / "controller"
    root.mkdir(mode=0o700)
    result = subprocess.run(
        [str(BINARY), str(root)],
        input=json.dumps({option: True}) + "\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert "error" in json.loads(result.stdout)


def test_split_broker_revision_cannot_rename_consumed_holdout_inputs(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        controller.ok("final_phase")
        controller.ok("freeze")
        final = controller.grant("final_evaluator")
        controller.ok("read", request=request(final, labels=True))
        reply = controller.send("revision_alias")
        assert reply["error"]["code"] == "SCOPE_VIOLATION", reply
        assert "history" in reply["error"]["message"]


def test_split_broker_distinct_binding_worker_tokens_and_unknown_fields(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        first = controller.grant("proposer")
        second = controller.grant("proposer", binding_index=1)
        forged = request(second)
        forged["lease_id"] = first["lease_id"]
        assert "error" in controller.send("read", request=forged)
        forged = request(first)
        forged["binding"] = second["binding"]
        forged["item"] = second["items"][0]
        assert "error" in controller.send("read", request=forged)
        forged = request(first)
        forged["unknown_permission"] = True
        assert "error" in controller.send("read", request=forged)


def test_split_broker_real_separate_oci_scope_has_no_host_store_or_other_labels(
    tmp_path: Path,
) -> None:
    with server(tmp_path) as controller:
        proposer = controller.grant("proposer")
        root = str(tmp_path / "controller")
        code = f"""import json,sys,os
from pathlib import Path
p=json.load(sys.stdin)
assert os.getuid()==65534 and p['split']=='development' and p['label'] is None
assert 'PRIVATE_LABEL' not in json.dumps(p)
assert not Path({root!r}).exists()
assert not Path('/var/run/docker.sock').exists()
Path('/out/proposer-only').write_text('private temporary data')
print(json.dumps({{'input':p['input']['value'],'split':p['split']}}))
"""
        receipt = controller.ok("domain", request=request(proposer), code=code)
        assert receipt["status"] == "completed" and receipt["cleanup_confirmed"] is True, receipt
        assert receipt["outputs_eligible"] is False
        assert json.loads(bytes.fromhex(receipt["report"]["stdout_hex"])) == {
            "input": 0,
            "split": "development",
        }
        controller.ok("final_phase")
        controller.ok("freeze")
        final = controller.grant("final_evaluator")
        code = """import json,sys
from pathlib import Path
p=json.load(sys.stdin)
assert p['split']=='holdout' and p['label']['secret']=='PRIVATE_LABEL_0_holdout'
assert not Path('/out/proposer-only').exists()
print('separate final domain')
"""
        receipt = controller.ok("domain", request=request(final, labels=True), code=code)
        assert receipt["status"] == "completed" and receipt["cleanup_confirmed"] is True, receipt
        assert bytes.fromhex(receipt["report"]["stdout_hex"]) == b"separate final domain\n"
        assert "error" in controller.send(
            "domain", request=request(proposer), code="print('forbidden')"
        )
