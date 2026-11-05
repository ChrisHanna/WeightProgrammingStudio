# Synthetic arithmetic model and gyms v1

## Known-target software fixture, never pretrained research evidence

`arithmetic.fixture()` builds an actual immutable float64 CPU Safetensors checkpoint in Rust, loads it through the real inventory/source verifier and returns a frozen native handle. The registered factory is `nf.arithmetic.relu` version1 and adapter `nf.arithmetic_relu.v1`. There are exactly42 parameters: `hidden.weight[6,2]`, `heads.weight[5,6]`; no biases, buffers, tokenizer or external model objects. No model is downloaded and no optimizer/backward/training procedure runs.

The native canonical config/graph/provenance assets are hashed and carried with the checkpoint. The graph records actual input→linear→ReLU→linear operators, shapes and tensor names. Runtime validation requires the exact config/graph, shapes, names and f64 dtype; it does not silently ignore unsupported extra parameters/buffers or caller graph metadata. Existing `fixture.cpu`/linear behavior remains unchanged. This adds one qualified synthetic factory, not a generic model adapter or research/study readiness claim.

## Exact designed heads

```text
h0=ReLU(x1), h1=ReLU(-x1), h2=ReLU(x2), h3=ReLU(-x2)
h4=ReLU(x1-x2), h5=ReLU(x2-x1)

sum                 h0-h1+h2-h3
difference          h0-h1-h2+h3
maximum             h2-h3+h4
minimum             h0-h1-h4
absolute_difference h4+h5
```

Modes are `corrupt_sum` (default: sum h0 coefficient0.5), `reference` (sum h0 coefficient1.0), and `empty_sum` (all six sum-row coefficients0). Other heads and hidden weights are identical. These are scripted hand constructions with known targets, direct-only provenance, zero optimizer/backprop usage and explicit synthetic/known-target labels. An empty row is a future construction starting fixture, not evidence that recipe compilation already works. Forward outputs are computed from the actual loaded parameter bytes, never hard-coded expected tables.

Native `SourceHandle.forward` dispatches only registered CPU fixtures, releases the GIL, checks byte/inventory integrity, domain, shape and work caps, and returns copied values. Arithmetic smoke suites execute the same native layer calculation against candidate bytes inside the isolated transaction; ordinary smoke success remains not scientific acceptance. The old linear adapter keeps its original digest/numerical contract.

## Independent oracles and frozen data

The supported domain is integer pairs in[-8,8], including signs, zero, equal/boundary/mirrored pairs. Nonfinite, fractional, out-of-range and wrong-shaped native inputs reject. Independent trusted Python references use direct `+`, `-`, `max`, `min`, `abs`, not the ReLU weights, native forward or construction helper. Expected outputs and all five independently scored heads use absolute tolerance1e-10. Invalid/wrong/nonfinite/bool outputs do not become positive scores.

All289 ordered pairs have native content identities and a versioned hash partition into nonempty development/selection/holdout. Mirrored inputs share one of153 unordered-pair groups across every head, so symmetry cannot put a twin in development and its counterpart in final tests. The split protocol, group/task/family IDs, exact inventory and tolerance are frozen by content identity. The same source inventory is used for corrupt/reference/empty modes. Native gym descriptors bind head/index/metric/tolerance, actual Python oracle/scorer module bytes and inventory identity; each has its own actual canonical digest.

The fixture's full descriptor/evaluate APIs are controller diagnostic tooling for deliberately public known synthetic tasks. They are NOT an authorization to expose private final corpora to a proposer. Real studies must still use the split/exposure broker and qualified evaluator domains. No held-out novelty claim follows from these known equations or supplied repair coordinates.

## User sum package and mixed retention

`arithmetic.write_user_sum_package(new_directory, model)` creates explicit new files only: a separate source module, actual Apache license, pinned stdlib runtime lock, development-only task data and a draft manifest. It never includes selection/holdout task rows in that user package and does not import it. `gyms.import_local` snapshots it through the Rust FD/CAS path; explicit `gyms.validate(..., enabled=True)` executes real known-good/wrong/invalid/replay probes only under the qualified sandbox. `gyms.approve` is a separate exact-version controller action.

The user sum implementation is standalone Python arithmetic, not a model callback or wrapper around native ReLU forward. Independent controller fixture truths still validate it. `mixed_portfolio(snapshot, approval)` proposes exactly five required bindings: the exact user sum target and the four native retention head gyms with zero allowed retention regression. These are proposed bindings, explicitly NOT a frozen StudyContract, approved model invocation, coverage/measurement result or promotion grant. Actual study execution and hard metric gating belong to the subsequent control/evaluation tasks.

Tests apply genuine validated Weight IR to the corrupt checkpoint: scaling sum[0,0] by2 repairs it. A two-cell composition also zeros difference[0,0], improving sum while measurably harming difference. Real per-head results expose that retention failure instead of hiding it in a target/weighted mean. Source bytes and hidden weights remain unchanged; candidate receipts stay `promotion_status=not_evaluated`.

## Qualification

`cargo test -p nf-experiments arithmetic_gyms` verifies the actual native grouped inventory. `python -m pytest -q -k arithmetic_gyms` checks every pair and five heads against independent Python arithmetic, exact corrupt/empty counts, actual IR repair/harmful composition/smoke/source preservation, native domain/shape/factory/graph denial, stable group-disjoint partitions, truthful synthetic method provenance, scoped development-only user package, real intake/sandbox validation/approval and mixed required retention bindings. No Torch/mock forward or skipped integration stands in for these native claims. Safe PyTorch worker qualification is the following task, not implied by this registered native fixture.
