"""Actual private IPC clients and Rust-owned CPU fixture inference, not a sandbox claim."""

import json
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neural_foundry.worker import Channel, read_frame, write_frame

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/examples/fixture_broker"


@contextmanager
def server(
    tmp_path: Path, actor: str = "proposer", welcome: bool = True, wall_ms: int = 30000
) -> Iterator[tuple[Channel, socket.socket, subprocess.Popen[bytes], dict[str, Any]]]:
    assert BINARY.is_file(), "cargo build -p nf-experiments --example fixture_broker is required"
    host, peer = socket.socketpair()
    process = subprocess.Popen(
        [
            str(BINARY),
            str(ROOT / "python/neural_foundry/data/m1_fixture.json"),
            str(tmp_path / actor),
            actor,
            str(wall_ms),
        ],
        stdin=peer.fileno(),
        stdout=peer.fileno(),
        stderr=subprocess.PIPE,
    )
    peer.close()
    channel = host.makefile("rwb", buffering=0)
    initial = read_frame(channel) if welcome else {}
    try:
        yield channel, host, process, initial
    finally:
        channel.close()
        host.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        assert process.stderr is not None
        errors = process.stderr.read()
        process.stderr.close()
        assert process.returncode in (0, 2), errors.decode(errors="replace")


def envelope(welcome: dict[str, Any], sequence: int, action: dict[str, Any]) -> dict[str, Any]:
    result = {key: welcome[key] for key in ("contract", "revision", "worker", "lease_id", "split")}
    result.update(schema_version=1, sequence=sequence, action=action)
    return result


def call(
    channel: Channel, welcome: dict[str, Any], sequence: int, action: dict[str, Any]
) -> dict[str, Any]:
    write_frame(channel, envelope(welcome, sequence, action))
    return read_frame(channel)


def test_worker_broker_real_inference_tools_and_staged_outputs_cross_only_bytes(
    tmp_path: Path,
) -> None:
    with server(tmp_path) as (channel, _, _, welcome):
        task = welcome["tasks"][0]
        task_reply = call(channel, welcome, 0, {"type": "task_input", "task": task})
        assert task_reply["result"]["input"] == [1.0, 2.0]
        forward = call(channel, welcome, 1, {"type": "infer", "model": "target", "task": task})
        assert forward["ok"] and forward["result"]["output"] == [5.0, 11.0]
        tool = call(
            channel, welcome, 2, {"type": "tool", "name": "fixture.sum", "left": 2, "right": 3}
        )
        assert tool["result"] == {"sum": 5}
        staged = call(
            channel,
            welcome,
            3,
            {"type": "stage_output", "slot": "result", "document": {"score": 1}},
        )
        assert staged["result"]["status"] == "staged_not_promoted"
        assert staged["result"]["handle"].startswith("staged:")
        assert staged["result"]["digest"].startswith("sha256:")
        assert staged["result"]["split"] == "development"
        assert staged["result"]["classification"] == "unreviewed_worker_output"
        assert (
            welcome["access_assurance"]
            == "trusted_fixture_scope_discipline_not_adversarial_isolation"
        )
        assert not {
            "checkpoint",
            "weights",
            "store",
            "epoch",
            "coordinator_lease",
            "database",
        }.intersection(welcome)


def test_worker_broker_forged_contract_worker_lease_or_split_is_denied(tmp_path: Path) -> None:
    with server(tmp_path) as (channel, _, _, welcome):
        for field, value in [
            ("worker", "forged"),
            ("lease_id", "coordinator-fence-is-not-worker-authority"),
            ("contract", "sha256:" + "0" * 64),
            ("revision", 2),
            ("split", "holdout"),
        ]:
            request = envelope(welcome, 0, {"type": "task_input", "task": welcome["tasks"][0]})
            request[field] = value
            write_frame(channel, request)
            reply = read_frame(channel)
            assert not reply["ok"] and reply["error"]["code"] == "PERMISSION_DENIED"
        request = envelope(
            welcome, 0, {"type": "infer", "model": "target", "task": welcome["tasks"][0]}
        )
        request["coordinator_lease"] = {"owner": "forged", "epoch": 1}
        write_frame(channel, request)
        assert read_frame(channel)["error"]["code"] == "INVALID_VALUE"
        assert call(
            channel, welcome, 0, {"type": "infer", "model": "target", "task": welcome["tasks"][0]}
        )["ok"]


def test_worker_broker_hidden_paths_labels_tools_and_models_are_not_authority(
    tmp_path: Path,
) -> None:
    with server(tmp_path) as (channel, _, _, welcome):
        for sequence, handle in enumerate(
            ("/etc/passwd", "../ledger/events.sqlite3", "scorer_labels", "checkpoint")
        ):
            reply = call(channel, welcome, sequence, {"type": "artifact", "handle": handle})
            assert reply["error"]["code"] == "PERMISSION_DENIED"
        assert "scorer_labels" not in welcome["artifacts"]
        reply = call(channel, welcome, 4, {"type": "tool", "name": "shell", "left": 0, "right": 0})
        assert reply["error"]["code"] == "PERMISSION_DENIED"
        reply = call(
            channel,
            welcome,
            5,
            {"type": "infer", "model": "hidden-target", "task": welcome["tasks"][0]},
        )
        assert reply["error"]["code"] == "PERMISSION_DENIED"
        reply = call(
            channel, welcome, 6, {"type": "infer", "model": "target", "task": "sha256:" + "0" * 64}
        )
        assert reply["error"]["code"] == "PERMISSION_DENIED"


def test_worker_broker_forward_tool_output_quotas_and_replay_are_exact(tmp_path: Path) -> None:
    with server(tmp_path) as (channel, _, _, welcome):
        action = {"type": "infer", "model": "target", "task": welcome["tasks"][0]}
        first = call(channel, welcome, 0, action)
        assert call(channel, welcome, 0, action) == first
        changed = dict(action, model="other")
        assert call(channel, welcome, 0, changed)["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        assert call(channel, welcome, 1, action)["result"]["forwards_used"] == 2
        assert call(channel, welcome, 2, action)["error"]["code"] == "QUOTA_EXCEEDED"
        for sequence in (3, 4):
            assert call(
                channel,
                welcome,
                sequence,
                {"type": "tool", "name": "fixture.sum", "left": 1, "right": 1},
            )["ok"]
        assert (
            call(
                channel, welcome, 5, {"type": "tool", "name": "fixture.sum", "left": 1, "right": 1}
            )["error"]["code"]
            == "QUOTA_EXCEEDED"
        )
        assert (
            call(
                channel,
                welcome,
                6,
                {"type": "stage_output", "slot": "result", "document": {"large": "x" * 512}},
            )["error"]["code"]
            == "QUOTA_EXCEEDED"
        )


def test_worker_broker_evaluator_labels_are_scoped_to_its_own_final_or_selection_tasks(
    tmp_path: Path,
) -> None:
    observed = []
    for actor in ("selection_evaluator", "final_evaluator"):
        with server(tmp_path, actor) as (channel, _, _, welcome):
            labels = call(
                channel,
                welcome,
                0,
                {"type": "artifact", "handle": welcome["artifacts"]["scorer_labels"]},
            )
            assert labels["ok"] and welcome["tasks"][0] in labels["result"]["labels"]
            assert labels["result"]["labels"][welcome["tasks"][0]]["expected"] == [5.0, 11.0]
            observed.append(welcome["tasks"][0])
    assert observed[0] != observed[1]


def test_worker_broker_slow_partial_frame_cannot_extend_the_fixed_scope_deadline(
    tmp_path: Path,
) -> None:
    with server(tmp_path, wall_ms=300) as (channel, _, process, _):
        channel.write(b"\\x00")
        channel.flush()
        process.wait(timeout=3)
        assert process.returncode == 2


def test_worker_broker_model_forwards_are_charged_once_despite_exact_replay(tmp_path: Path) -> None:
    with server(tmp_path) as (channel, _, _, welcome):
        action = {"type": "infer", "model": "target", "task": welcome["tasks"][0]}
        assert call(channel, welcome, 0, action)["ok"]
        assert call(channel, welcome, 0, action)["ok"]
    with sqlite3.connect(tmp_path / "proposer/ledger/events.sqlite3") as connection:
        settlements = [
            json.loads(row[0]) for row in connection.execute("SELECT body FROM budget_outcomes")
        ]
    assert sum(row["settlement"]["usage"]["evaluations"] or 0 for row in settlements) == 1


def test_worker_broker_oversized_raw_frame_is_refused_before_payload_allocation(
    tmp_path: Path,
) -> None:
    with server(tmp_path) as (channel, _, process, _):
        channel.write((16385).to_bytes(4, "big"))
        channel.flush()
        process.wait(timeout=5)
        assert process.returncode == 2


def test_worker_broker_actual_python_worker_receives_only_a_private_fd_and_scope(
    tmp_path: Path,
) -> None:
    script = """
import json,os,socket,sys
from neural_foundry.worker import Client
sock=socket.socket(fileno=int(os.environ['BROKER_FD']))
channel=sock.makefile('rwb',buffering=0)
client=Client(channel)
reply=client.request({'type':'infer','model':'target','task':client.welcome['tasks'][0]})
assert reply['result']['output']==[5.0,11.0]
assert not any(name in sys.modules for name in ('torch','transformers'))
assert set(os.environ).issubset({'BROKER_FD','PATH','LC_CTYPE'})
print(json.dumps({'reply':reply,'welcome':client.welcome}))
channel.close();sock.close()
"""
    with server(tmp_path, welcome=False) as (_, sock, _, _):
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            pass_fds=(sock.fileno(),),
            env={"BROKER_FD": str(sock.fileno()), "PATH": ""},
            check=True,
            capture_output=True,
        )
        proof = json.loads(completed.stdout)
        assert proof["reply"]["result"]["output"] == [5.0, 11.0]
        assert "coordinator_lease" not in proof["welcome"]
