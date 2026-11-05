# Deterministic and paired group statistical analysis v1

`statistics.plan(protocol)` creates an immutable native analysis-plan witness. `statistics.analyze(plan, observations)` returns a bounded native result; neither JSON nor model/metric names can construct a plan/result handle directly. Identities, schema checks, pairing, arithmetic, RNG and verdicts belong to Rust. There are no provider calls, model downloads, fitting, optimizer steps or Python numerical dependencies.

## Frozen family and exact pairing

A plan declares its family, estimator/scope, estimand, unique repeated seeds, minimum distinct task/group support, confidence, resample count/seed, reserved maximum comparisons and exact contrasts. Every contrast freezes candidate/baseline references, original/no-op/best-known comparator, portfolio binding, metric schema/direction/unit, value statistic, context/workload/measurement protocol, task/input/group identities, hard thresholds, required status and no-op-control role. Comparator baselines and candidate control roles cannot change between family metrics. Duplicate comparison dimensions, task/input aliases, group relabeling and undercounted multiplicity reservations are rejected.

Each observation must match that exact contrast, model arm, task/input/group, seed, context, workload, protocol, units and statistic. Identical duplicate rows are counted once and reported; conflicting duplicates reject. Nonfinite, out-of-domain or absolute values above 1e12 reject. This finite numerical bound keeps aggregation admitted and meaningful; it is not silently clipping data.

The complete frozen task × seed × arm panel is expected. Failed/timed-out/missing rows, `null` values and unmatched pairs remain **incomplete**, never dropped as a favorable complete-case subset. Complete but insufficient unique task/group support is inconclusive, not a smaller claimed denominator. Seeds are repeats, not extra independent groups.

## Estimand and deterministic fixtures

The supported estimand is explicitly `equal_weight_group_mean_of_task_seed_means`: first average the frozen paired seed outcomes for each task, then average tasks within their group, then weight groups equally. Large correlated groups and repeated tasks cannot dominate by pseudoreplication. Changing the estimand requires a new plan rather than an implicit switch to task weighting.

Native means use compensated summation before division. This preserves representable constant means and avoids avoidable cancellation/rounding drift in unequal-sized groups. Deterministic exact analysis requires fixture-only scope, confidence one and no bootstrap; per-task outcomes must repeat exactly over declared seeds. Repeat variation makes the exact claim inconclusive. Its interval collapses to the fully observed finite-panel estimate, not a population/generalization claim.

## Seeded paired group bootstrap and multiplicity

Bootstrap requires at least two unique groups and at least two distinct frozen seed repeats. It resamples **groups with replacement**, retaining both arms and each group's entire task/seed aggregation together. The empirical target is the declared fixed seed panel; it does not pretend resampling task duplicates supplies new groups or generalizes to unobserved seeds.

The portable `xorshift64_rejection_index_v1` draw schedule is deterministically seeded from protocol bootstrap seed, family and contrast IDs. Rejection sampling avoids modulo index bias. Group order is canonical. Correcting a larger multiplicity reservation changes interval tails, not the underlying draw schedule for the same contrast. The implementation uses bounded compensated aggregation and deterministic sorting/quantiles; Python tests independently reproduce draw indices and interval endpoints.

Two-sided Bonferroni correction uses `(1-confidence) / maximum_planned_comparisons`. The reservation includes all frozen metrics, candidate comparisons, required baseline contrasts and confirmatory analyses; missing or optional results do not shrink it. Lower/upper empirical percentile endpoints use conservative floor/ceil indices. At least five expected replicates must occupy each corrected tail (`resamples * alpha/(2*m) >= 5`); inadequate resolution is explicitly inconclusive, not an unjustified extreme-tail interval.

Plans admit at most 64 contrasts, 256 reserved comparisons, 256 tasks per contrast, 16 seeds, 4096 expected paired-arm rows and ten million group draws. Bootstrap resamples are 1000–50000; raw rows including duplicates are bounded to 8192 and canonical byte/node limits. Cancellation is checked between contrasts and during bootstrap. The empirical percentile interval is a finite-sample estimate, **not guaranteed population coverage**; representativeness, cluster assumptions, calibration and upstream trial accounting remain required.

## Directions and verdicts

Gain is candidate-minus-baseline for higher-is-better metrics and baseline-minus-candidate for lower-is-better metrics. Candidate intervals assess floors/ceilings; paired gain intervals assess positive gains and non-inferiority/regression limits. A hard condition passes only when the entire relevant confidence bound satisfies it. A confidently opposite interval fails; an overlapping boundary is inconclusive. Point gain alone cannot pass. A no-op control or identical candidate/baseline reference cannot satisfy a strictly positive gain rule even if supplied noisy observations look favorable.

Reports retain raw support/completeness counts, duplicate accounting, means, intervals, direction, threshold, resampling count, corrected family alpha and per-contrast verdict. Missing final comparisons keep the family incomplete. `all_required_analysis_conditions_satisfied` is only an analysis condition, not a promotion decision.

## Evidence boundary

Inputs are supplied observations. Their candidate/baseline/workload/measurement IDs are exact **references**, not proof of actual model execution, independently administered holdout custody, or that a user froze a final panel before seeing results. Re-freezing an analysis record cannot reconstruct such authority. The upstream persisted contract/panel/exposure and execution receipts must supply those witnesses, and prior/adaptive trials must be accounted for there. `scientific_acceptance=false`, `outputs_eligible=false` and the analysis-only scope remain explicit. Immutable evidence acceptance is T038, not a rename of this numerical API.
