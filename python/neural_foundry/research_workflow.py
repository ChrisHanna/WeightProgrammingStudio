"""Manual supplied-IR facade; Rust owns all validation, hashing and state.

This is a controller API, never a gym/worker API. Planning requires an existing
explicit local CAS and immutable source export. It never creates a store, runs a
model, executes smoke tests or qualifies a backend. Freeze persists a contract;
it does not authenticate the approval reference. Run verifies a root-issued,
store/ledger/contract/permissions-bound grant and executes only its named
operation. Explicit v2 supplied-IR inputs run through the supported linear CPU
coordinator in a dedicated bounded process, with genuine model outputs. This is
not general model support, strict publication, or promotion. Root is a trusted
operator, never a researcher process. Launch additionally consumes the grant
through the separately installed root-owned local authority service; an uncertain
response burns the grant rather than retrying. Resume is recovery-only.
"""

import json
from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from typing import Any, cast

from . import _native
from .records import RecordInput, _wire


def _request(operation: str, record: RecordInput) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_native.manual_workflow_request(operation, _wire(record))))


class ResearchWorkflow:
    """Compose supplied records, not an autonomous proposer or execution grant.

    Legacy v1 binds ``contract.intervention.method_protocol`` to the plan's
    manifest digest. Explicit v2 (with ``control_program``) keeps that field
    equal to the supplied program's protocol ID, and the operator separately
    approves the exact ``operator_binding`` returned by plan. No contract or
    program is silently rewritten. Source/store identities never use aliases.
    Caller-owned records are encoded once so later dictionary edits cannot alter
    the workflow silently. No Python implementation of native policy is present.
    """

    def __init__(
        self, *, store: str | PathLike[str], source_export: str,
        contract: Mapping[str, object], program: Mapping[str, object],
        search_space: Mapping[str, object], policy: Mapping[str, object],
        smoke_suite: Mapping[str, object],
        control_program: Mapping[str, object] | None = None,
    ) -> None:
        self._store = fspath(store)
        draft: dict[str, object] = {
            "schema_version": 2 if control_program is not None else 1,
            "store": self._store, "source_export": source_export,
            "contract": contract, "program": program, "search_space": search_space,
            "policy": policy, "smoke_suite": smoke_suite,
        }
        if control_program is not None:
            draft["control_program"] = control_program
        self._draft = _wire(draft)

    def plan(self) -> dict[str, Any]:
        """Static source/IR compilation only; no inference or persistent writes."""
        return _request("plan", self._draft)

    def validate(self) -> dict[str, Any]:
        """Offline cross-record validation, not dynamic gym validation/approval."""
        return _request("validate", self._draft)

    def freeze(self, *, ledger: str | PathLike[str], expected_plan_digest: str) -> dict[str, Any]:
        """Persist immutable inputs/contract, without claiming verified approval."""
        return _request("freeze", {"plan": json.loads(self._draft), "ledger": fspath(ledger),
                                   "expected_plan_digest": expected_plan_digest})

    def _existing(self, operation: str, *, ledger: str | PathLike[str], contract_digest: str,
                  authorization: str | None = None, workers: Sequence[str] = ()) -> dict[str, Any]:
        return _request(operation, {"schema_version": 1, "store": self._store,
            "ledger": fspath(ledger), "contract_digest": contract_digest,
            "authorization": authorization, "workers": list(workers)})

    def run(self, *, ledger: str | PathLike[str], contract_digest: str,
            authorization: str | None = None) -> dict[str, Any]:
        """Run only the exact operator-granted operation, with durable costs.

        ``authorization`` identifies an unexpired root-owned registry entry; it
        is not a caller approval flag. The operator issues it separately with
        ``nf study authorize``. A native metadata-fixture grant cannot authorize
        supplied-IR execution. V2 additionally requires an explicit grant for
        ``nf.manual.supplied_ir.linear_cpu.v2`` and the plan's exact input binding.
        The approved executable runs in its own hard-limited process; this never
        authorizes strict durable publication or promotion. The operator must
        separately install/start the immutable ``nf study serve-authority`` local
        IPC publisher. It durably consumes each grant outside researcher-owned
        SQLite before dispatch, so restoring the database cannot replay it. An
        absent/busy publisher or uncertain response never authorizes a fallback.
        """
        return self._existing("run", ledger=ledger, contract_digest=contract_digest,
                              authorization=authorization)

    def status(self, *, ledger: str | PathLike[str], contract_digest: str) -> dict[str, Any]:
        """One read-only committed snapshot; internal diagnostic, not acceptance."""
        return self._existing("status", ledger=ledger, contract_digest=contract_digest)

    def report(self, *, ledger: str | PathLike[str], contract_digest: str) -> dict[str, str]:
        """JSON/Markdown from the same existing committed snapshot renderer."""
        return self._existing("report", ledger=ledger, contract_digest=contract_digest)

    def stop(self, *, ledger: str | PathLike[str], contract_digest: str) -> dict[str, Any]:
        """Persist stop first; request acknowledgment is not cleanup confirmation."""
        return self._existing("stop", ledger=ledger, contract_digest=contract_digest)

    def resume(self, *, ledger: str | PathLike[str], contract_digest: str,
               workers: Sequence[str] = ()) -> dict[str, Any]:
        """Recover native workers with a fresh fence; never redispatch.

        With no worker list, discover retained native intents for this exact
        contract, including launches whose ID was never returned before a crash.
        Unknown costs are retained. A persisted stop remains set, and terminal
        runs cannot reopen. Native recovery is not proof of final-fixture/reload
        cleanup and does not resume execution. Interrupted supplied-IR final
        intents are reconciled by the coordinator, never silently redispatched.
        """
        return self._existing("resume", ledger=ledger, contract_digest=contract_digest,
                              workers=workers)
