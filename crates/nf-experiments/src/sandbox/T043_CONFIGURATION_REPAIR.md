# T043 OCI configuration interoperability repair

## Diagnosis and fix

Read the full review at /home/ot/.kiro/crew/subagents/98da3f4633b1e6ce/result.txt
and protocols/m2-vertical-slice-v1.md. The retained inspection named in that
review is no longer present; the review records CapAdd as KILL, SETUID, SETGID.
The verifier instead required an ordered CAP_KILL, CAP_SETGID, CAP_SETUID array.
The six failed test IDs were confirmed from pytest's retained failure cache.
The older M2 example binaries also predated the T042 launcher repair.

Changed only sandbox.rs and the new sandbox/configuration_tests.rs source.
CapAdd now has exact, duplicate-free normalized set membership. Only one optional
CAP_ prefix is ignored; order is immaterial. Case folding, whitespace trimming,
unknown capabilities, ALL, extra/missing elements, malformed values and duplicate
aliases are not accepted. The allowed set remains KILL, SETGID, SETUID. CapDrop,
image/owner/mount, guest UID/capability/seccomp/cgroup and cleanup checks remain.
No protected coordinator, promotion, manual-workflow, protocol or checklist edits.

Three unit tests pass: 48 valid order/prefix combinations, 15 invalid capability
representations, and rejection of changed host security/configuration fields.
These are comparator tests, not substitutes for the actual OCI tests below.

## Rebuilt offline and reran all six failures

Build target: /tmp/nf-t042-independent-26f76ad4. Rebuilt fixture_sandbox,
fixture_splits, Rust unit tests, and the editable Python extension from current
source, offline. NF_M2_TARGET_DIR selected those rebuilt examples. No stale M2
example or pre-existing Python extension was used for the following rerun.

Each original failed case ran by exact pytest node ID with its existing 85-second
alarm plus an outer 90-second deadline. Wall times below include pytest startup.
All six passed, with no skips:

| Test in tests/test_m2_vertical_slice.py | Seconds |
| --- | ---: |
| test_m2_vertical_slice_real_mixed_five_gyms_and_required_retention[repair] | 38.87 |
| test_m2_vertical_slice_real_mixed_five_gyms_and_required_retention[retention-damage] | 40.88 |
| test_m2_vertical_slice_frozen_holdout_real_domains_close_search_and_survive_reopen | 7.55 |
| test_m2_vertical_slice_renamed_consumed_holdout_cannot_reenter_search | 4.28 |
| test_m2_vertical_slice_actual_sandbox_cancellation_is_not_publishable | 5.65 |
| test_m2_vertical_slice_actual_sandbox_crash_recovery_retains_incomplete_costs | 10.31 |

## Earlier repairs rechecked after the launcher change

Four named Rust tests pass: complete six-control durable audit, active persisted
stop/lease revocation and cleanup recovery, barrier-controlled two-root recovery
race, and kernel controller hard-memory ceiling. The two callback-forgery Python
cases plus the positive immutable-gym case pass against the rebuilt extension
(three cases, 31.41 seconds total). No individual hard limits were changed.

This resolves the six configuration failures, NOT the entire T043 joined
arithmetic study/promotion gap. The other 16 M2 cases were not rerun in this
follow-up, and no new independent review or release acceptance is claimed.
No network downloads or nested agents were used. Both declared servers
@auto-improvement:auto-improvement and @mochi:mochi were unavailable, not retried.
