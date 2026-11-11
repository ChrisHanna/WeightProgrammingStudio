# Reference audit deadline v2

The approved [v2 policy](reference-audit-deadline-v2.json) separates 7,200 ms of summed guardian-receipt work from 26,768 ms of controller/container lifecycle. The maximum outer allowance is their sum, 33,968 ms. CPU, storage, per-job wall/CPU limits and external lease/reservation deadlines remain independent and unchanged.

Opt in with `"deadline_policy": 2` in the reference audit budget, retaining `"wall_ms": 7200`. Absence of the version field preserves legacy v1 aggregate-wall behavior. Unknown versions and work allowances above the approved cap are rejected before sandbox qualification. Durable coordinator reservations must admit the **sum**, not just the timed-work allowance.

The clock includes supplied request preflight/qualification time and later staging, container control, teardown and persistence. Work comes only from the trusted guardian receipt. While a job is in flight, its actual work duration is not yet available: lifecycle checking uses the conservative lower bound obtained by subtracting that job's bounded work allowance. Both actual counters are checked on receipt, before further work and before successful return; the outer deadline remains continuously enforced. This is not a claim of exact interrupt at the component boundary during an unobserved in-flight phase. Guardian millisecond quantization remains explicit.

Reports retain `wall_ms` for guardian work and add `lifecycle_wall_ms` for v2. That lifecycle field is the measurement before the report's own CAS publication; a final live check after publication prevents accepting an over-budget return. Neither a successful container exit nor passing the outer deadline alone qualifies both sub-budgets.

Three focused native tests reject excess work and excess lifecycle independently even with outer allowance remaining, reconcile the in-flight bound, and verify opt-in/legacy behavior. The four former Python known-failure cases were first executed successfully with `--runxfail`, then their markers were removed and all four passed again under ordinary pytest. Receipts:

- [Real execution before marker removal](../artifacts/proof-phase-ab-20261111/deadline-four-real-command/receipt.json).
- [Normal execution after marker removal](../artifacts/proof-phase-ab-20261111/deadline-four-unmarked-command/receipt.json).

Full Cargo acceptance uses **two test threads under the documented capability runner** in [core testing](core-testing-v1.md#running-the-nf-experiments-tests).
