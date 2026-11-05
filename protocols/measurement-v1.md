# Frozen CPU measurement protocol v1

`measurements.environment(runtime)` observes the live qualified trusted runtime and boot-visible CPU inventory, affinity and kernel. `measurements.run(runtime, original, candidate, protocol)` executes a new resource-bounded trusted child; a JSON sample list or controlled-clock test cannot mint its opaque `MeasurementHandle`. This initial measured profile is the actual registered f64 arithmetic model, not an arbitrary model factory or GPU runtime. No gym/agent Python, live Torch object, optimizer or callback crosses into the controller.

## Exact environment and workload

The frozen environment binds runtime/code/dependency identity, CPU inventory and affinity under the current boot, architecture/OS/kernel driver identity, pinned Torch2.9.1+cpu, CPU-only device and `CLOCK_MONOTONIC_RAW` nanosecond timer. The physical-device fingerprint is explicitly **boot-visible CPU inventory and affinity, not a unique chip serial**; virtualized host/thermal/scheduling effects are not hidden by that label. Both the actual child affinity and before/after parent environment must match. Changed/missing environments are rejected, not silently rebound.

The native workload digest binds exact finite integer-pair inputs, per-input seeds, batch size, concurrency, precision, model-cache policy and stateless reset policy. Supported settings are f64, batch one, concurrency one, cache disabled and the registered stateless forward checked before/after. Both actual source/candidate manifest/checkpoint identities and measurement code are bound to the protocol ID. Different workloads, precisions, timers, units, scopes or statistics cannot share a threshold by name alone.

Limits are explicit: 1–16 input vectors, one seed each, 0–32 warmup rounds and 3–128 measured rounds, 1–16 distinct metric definitions, plus T032 CPU/address-space/wall/output bounds. Both restored model instances are independently checked against native inventory and outputs. No-op detection uses actual execution-equivalent parameter/config identity, not merely a manifest name: changing provenance bytes or storage-group labels cannot create a timing-improvement candidate.

## Warmup, order, boundaries and statistics

The child executes paired interleaved rounds. Deterministic paired order alternates which arm runs first; randomized interleaved order is frozen by the recorded xorshift64 schedule/seed. Every round executes both arms. Native checks verify every actual sample ordinal, arm and warmup flag against that schedule. Warmup measurements remain visible but are excluded from measured statistics. Both arms must supply exactly the frozen measured sample count. Every warmup and measured result digest must match the independent native forward for that arm, and model state must remain unchanged after all measurements.

Two different measured timer boundaries are retained:

- **Forward latency:** sum of per-model-call monotonic-raw intervals across the declared workload. Seed setting, input tensor creation, output conversion and between-call loop time are excluded.
- **End-to-end latency:** the **in-process numeric request** from seed/input creation through forward and output conversion. Model load, process/interpreter/framework startup, transport, checkpoint I/O, export and whole-study cost are excluded. It is not mislabeled whole-service or whole-study latency.

Statistics are native arithmetic mean, median (average of two central samples for even support), nearest-rank p95/p99 and maximum. Raw samples are nanoseconds; thresholds/results in milliseconds or seconds use explicit conversions. The workload is the whole frozen vector sequence, not a per-token/TTFT/TPOT proxy. Empty support, nonpositive forward intervals, malformed order, nonfinite domains or a request interval shorter than its contained forward work fail verification. Controlled sample vectors test gate/statistic/unit/overflow behavior; actual CPU runs establish runtime support.

## Memory scope and unavailable metrics

RSS samples are genuine Linux `/proc/self/status` `VmRSS` observations, converted from KiB to bytes after each workload. The scope is **the whole paired co-resident process, both models, interpreter and runtime**, not an inferred per-model weight size. Child `getrusage` high-water RSS and parent kernel `wait4` peak RSS/CPU are separate observations. RLIMIT_AS remains a virtual-address hard limit, not a claimed hard RSS cap. Explicit model-transfer/staging/clone/factory and input-allocation counters retain their limited allocation scope, not an all-interpreter zero-copy claim.

An absolute process-RSS ceiling can be assessed for this exact paired-process workload. A per-model RSS gain/regression cannot be attributed from co-resident samples: its gate is explicitly inconclusive, not a supported optimization claim. Isolated-model/device memory qualification is not invented from these measurements.

Device allocator peak and device-wide peak are independently represented as unavailable (`null`) on this CPU backend. They are never substituted by process RSS or zeros. A required unavailable metric blocks measurement conditions; optional unavailable data remains visible.

## Gates and evidence boundaries

Each threshold binds its exact scope, unit, statistic and frozen workload/environment. Missing measurements are unavailable; violated ceilings/regressions are failed. Identical executed models cannot satisfy a strictly positive gain rule even when noisy clocks report a lower value. Raw gain and sample distributions are still visible. All decisions are **point-estimate measurement conditions only**: calibrated uncertainty/multiplicity and immutable scientific acceptance are later tasks. `scientific_acceptance=false`, `outputs_eligible=false` and `promotion_status=not_evaluated` remain explicit.

Actual child kernel CPU/peak RSS and process wall duration include startup and measurement work; model-request timing has the narrower boundaries above. Parent staging, reference checks, CPU inventory reads and hashing/setup outside that child remain additional unknown costs, not zero. Execution/cancellation/cleanup/digest fences use the T032 qualified trusted process. This standalone public-fixture diagnostic grants no study budget, private split custody, promotion or hostile-gym isolation.
