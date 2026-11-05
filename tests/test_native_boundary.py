"""Exercise the installed PyO3 boundary with real native model data."""

import copy
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest
from native_fixture import make_fixture, wire
from neural_foundry import _native


def test_native_boundary_real_edit_and_snapshot_keep_source_and_aliases() -> None:
    fixture = make_fixture()
    plan = _native.compile_plan(
        fixture.source, wire(fixture.program), wire(fixture.space), wire(fixture.policy), {}
    )
    summary = json.loads(plan.summary_json())
    assert summary["effective_writes"]
    candidate = _native.apply_plan(plan, wire(fixture.smoke))
    assert struct.unpack("<4d", candidate.tensor_bytes("w")) == (2, 4, 6, 8)
    assert candidate.tensor_bytes("w") == candidate.tensor_bytes("tied")
    assert struct.unpack("<4d", fixture.source.tensor_bytes("w")) == (1, 2, 3, 4)
    receipt = json.loads(candidate.receipt_json())
    assert receipt["changed_parameters"] == 4
    assert receipt["promotion_status"] == "not_evaluated"
    snapshot = candidate.snapshot()
    assert snapshot.parameter_digest == candidate.parameter_digest
    assert snapshot.tensor_bytes("w") == candidate.tensor_bytes("w")
    assert json.loads(snapshot.manifest_json())["assets"]["checkpoint"]


def test_native_boundary_records_use_rust_validation_and_typed_errors() -> None:
    fixture = make_fixture()
    record = _native.validate_record("manifest", wire(fixture.manifest))
    assert record.kind == "manifest"
    assert record.digest == fixture.source.manifest_digest
    assert json.loads(record.json()) == fixture.manifest
    with pytest.raises(_native.NativeError) as caught:
        _native.validate_record("program", b'{"ir_version":1,"ir_version":1}')
    assert caught.value.code == "DUPLICATE_KEY"
    assert isinstance(caught.value.path, str)
    assert str(caught.value) == caught.value.message
    with pytest.raises(_native.NativeError) as caught:
        _native.validate_record("executable_python", b"{}")
    assert caught.value.code == "UNSUPPORTED_CAPABILITY"
    with pytest.raises(_native.NativeError) as caught:
        _native.validate_record("manifest", b" " * 1_048_577)
    assert caught.value.code == "LIMIT_EXCEEDED"


def test_native_boundary_no_mutable_views_handles_or_object_callbacks() -> None:
    fixture = make_fixture()
    with pytest.raises(TypeError):
        cast(Any, _native.SourceHandle)()
    with pytest.raises(AttributeError):
        cast(Any, fixture.source).parameter_digest = "forged"
    with pytest.raises(TypeError):
        memoryview(cast(Any, fixture.source))
    changed = bytearray(fixture.source.tensor_bytes("w"))
    changed[0] ^= 1
    assert changed != fixture.source.tensor_bytes("w")

    class Trap:
        called = False

        def __bytes__(self) -> bytes:
            self.called = True
            return b"arbitrary code"

    trap = Trap()
    with pytest.raises(TypeError):
        _native.load_source(wire(fixture.manifest), cast(Any, trap), fixture.assets)
    assert not trap.called
    with pytest.raises(TypeError):
        _native.apply_plan(cast(Any, {"digest": "forged"}), wire(fixture.smoke))
    with pytest.raises(TypeError):
        _native.load_source(wire(fixture.manifest), bytearray(fixture.checkpoint), fixture.assets)  # type: ignore[arg-type]


def test_native_boundary_cancel_is_sticky_and_cannot_expose_a_candidate() -> None:
    fixture = make_fixture()
    token = _native.CancellationHandle()
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
    with pytest.raises(_native.CancelledError) as caught:
        _native.load_source(b"invalid", fixture.checkpoint, fixture.assets, token)
    assert caught.value.code == "TRANSACTION_ABORTED"
    plan = _native.compile_plan(
        fixture.source, wire(fixture.program), wire(fixture.space), wire(fixture.policy), {}
    )
    for _ in range(2):
        with pytest.raises(_native.CancelledError):
            _native.apply_plan(plan, wire(fixture.smoke), token)
    assert struct.unpack("<4d", fixture.source.tensor_bytes("w")) == (1, 2, 3, 4)
    assert (
        _native.apply_plan(plan, wire(fixture.smoke)).parameter_digest
        != fixture.source.parameter_digest
    )


def test_native_boundary_stale_and_denied_submissions_cannot_launder_scope() -> None:
    fixture = make_fixture()
    policy = copy.deepcopy(fixture.policy)
    policy["permissions"]["tools"] = ["inspect"]
    with pytest.raises(_native.NativeError) as caught:
        _native.compile_plan(
            fixture.source, wire(fixture.program), wire(fixture.space), wire(policy), {}
        )
    assert caught.value.code == "PERMISSION_DENIED"
    stale = copy.deepcopy(fixture.program)
    stale["base_model"]["parameter_digest"] = "sha256:" + "0" * 64
    with pytest.raises(_native.NativeError) as caught:
        _native.compile_plan(
            fixture.source, wire(stale), wire(fixture.space), wire(fixture.policy), {}
        )
    assert caught.value.code == "SOURCE_MISMATCH"
    with pytest.raises(_native.NativeError):
        _native.load_source(
            wire(fixture.manifest), fixture.checkpoint, {"config": b'{"changed":true}'}
        )
    assert struct.unpack("<4d", fixture.source.tensor_bytes("w")) == (1, 2, 3, 4)


def test_native_boundary_shared_plan_runs_as_isolated_native_candidates() -> None:
    fixture = make_fixture()
    plan = _native.compile_plan(
        fixture.source, wire(fixture.program), wire(fixture.space), wire(fixture.policy), {}
    )

    def run(_: int) -> str:
        return _native.apply_plan(plan, wire(fixture.smoke)).digest

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(4)))
    assert len(set(results)) == 1
    assert struct.unpack("<4d", fixture.source.tensor_bytes("w")) == (1, 2, 3, 4)
