# T042 storage integration and remaining owner boundaries

## Implemented

Reference Budget JSON now REQUIRES storage_bytes (positive, at most 16 GiB;
missing fields and numeric-maximum sentinels reject). Before qualification in
the Python native binding, and before any audit publication/job in Rust, reserve:

    max_executions * STORAGE_JOB_BYTES
      + (2 * max_executions + 2) * (1_048_576 + 4_096)

Every job slot charges the full sandbox ceiling, including repeated source,
input, guardian, policy, assets, directories, journal evidence and admission
preallocation. This is intentionally conservative cumulative allocation, not
peak workspace or retained bytes. All slots are charged up front, including
unused/cancelled/failed slots. No deduplication/reclamation refund is inferred.
The real reference caller now claims a slice before calling
execute_assets_with_staging_limit. Existing fixed 16 MiB job / 64 MiB root
ceilings, locking, isolation and qualification remain unchanged. The old
reserved_asset_bytes field is only a payload diagnostic, not admission.

CAS plan/report and at most one receipt/observation per slot each reserve the
canonical 1 MiB document maximum plus headroom. Claims precede publication,
including the report's own claim before serialization. Existing CAS quota and
safe staging mechanisms still enforce individual writes. Physical filesystem
metadata/allocation is NOT portably guaranteed by this logical bound.

Reports distinguish conservative charge from observations. Complete finalized
receipts supply observed workspace peaks and cumulative per-job retained
journals (not a root delta, since pruning may occur). Missing/incomplete samples
remain sticky unknown. Whole-study physical usage is null, never zero.
capacity_guaranteed=false and durable_family_accounting=false are explicit.

## Exact remaining coordinator/contract handoff (not implemented here)

This standalone exposed audit receives a trusted controller declaration, NOT
an opaque coordinator Reservation/Lease. Repeated invocations, retries and
process crashes must not be presented as a durable study-wide ledger. The
controller must reserve the formula above from the existing family/branch
storage_bytes allocation BEFORE invoking the audit, persist operation/attempt
identity, and settle the conservative full ceiling on failures/unknown usage.
It must also separately include existing/imported CAS, SQLite + rollback
journal, qualification/recovery and other study operations. Unknown usage must
remain None under coordinator budgets semantics, not become a zero refund.
There is no Python lease/reservation handoff in this intake API. A future study
entrypoint must accept native opaque authority, recheck current lease/contract
permissions, and preserve the accounting hold across crashes. Do not silently
treat storage_bytes JSON as that authority or advertise study execution enabled.
No coordinator, final fixture, splits, promotion or model-worker file changed.

## Exact sandbox-owner handoff

No API change is required for conservative per-invocation admission. A tighter
admission API would need a public pure workspace estimate using the SAME
source/input/guardian/policy/directory formula as storage::workspace_bound,
plus a separately bounded journal/probe charge. Do not duplicate its private
formula in validation or add a caller-supplied measured-usage flag.

Still required for the stronger ENOSPC guarantee: an owner-local regression at
actual execute_assets_with_staging_limit staging/cleanup boundaries which
exhausts only an isolated fixture tmpfs AFTER admission, verifies teardown is
attempted even if journal persistence fails, restores capacity, calls
recover_pending, then checks finalized accounting, reclaimed workspace and
ineligible interrupted outputs. Current ENOSPC storage test proves capacity
can disappear after the preallocation probe and finish succeeds after capacity
returns; it does NOT cover public execution + recover_pending under ENOSPC.
The new reference_storage_enospc regression separately exercises production
Accounting::put + ArtifactStore::put under actual isolated tmpfs ENOSPC: partial
CAS staging is removed; after capacity returns, publication succeeds without
resetting the charge. It does not replace the sandbox-owner regression above.
Do not weaken the untrusted profile or expose production fault-injection knobs.
Future capacity remains unsupported: capacity_guaranteed must stay false.

## Verification

- Offline accounting tests: 4 passed, including genuine CAS ENOSPC recovery.
- Real OCI Rust reference integration tests: 6 passed.
- Python suites: 12 + 74 = 86 passed (the first combined run timed out; all
  cases were rerun in bounded subsets). After sticky-unknown changes/native
  rebuild, the 7 positive-storage/rejection cases passed again.
- Existing isolated sandbox ENOSPC capacity-probe fixture: 1 passed.
- Offline nf-python check and maturin editable build: passed.
- Ruff and mypy for the four changed Python files: passed.
- Strict Clippy blocked by cross-owner final_fixture duplicate_mod and
  too_many_arguments plus promotion needless_borrow (type_complexity was also
  suppressed). Clippy with only those four lint classes suppressed passed.
- No network, downloads, models or non-fixture studies. No Git metadata.
