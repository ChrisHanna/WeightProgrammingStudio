# Neural Programming Foundry & Weight Programming Studio

Apache-2.0. Local-first, Rust-authoritative direct weight programming, verification firewall, and Python Weight Programming Studio SDK.

## Architecture Overview (M1–M5 & Weight Programming Studio)

The repository implements a 9-crate Rust workspace (`nf-core`, `nf-checkpoint`, `nf-graph`, `nf-weight-ir`, `nf-surgeon`, `nf-patches`, `nf-experiments`, `nf-cli`, `nf-python`) paired with the `neural_foundry` Python SDK and `weightctl` / `nf` CLI:

- **M1 — Trusted Execution Core**: Strict Weight IR (`replace_slice`, `add_delta`, `scale`, `mask`, `permute_axis`, `linear_map`, `subspace_projection`, `low_rank_update`), search-space validation, copy-on-write transactional execution (`nf-surgeon`), single/multi-parent patch replay (`nf-patches`), and standalone Safetensors export/restore (`nf-checkpoint`).
- **M2 — Epistemic Firewall, Sandboxed Gyms & Paired Statistics**: Three-way split broker (`development`, `validation`, `holdout`), exposure ledger, static-ELF guardian + rootless/no-network OCI container sandbox, trusted PyTorch model worker, paired no-op controls, and bootstrap/exact statistical gates.
- **M3 — Autonomous Scientist Loop & Pretrained Model Repair**: Bounded weight/activation inspector (`neural_foundry.inspection`), localization & structured hypothesis cards (`neural_foundry.diagnostics`), multi-branch candidate search & ablation (`neural_foundry.autonomous`), Qwen2 & Pythia/GPT-NeoX pretrained adapters (`neural_foundry.pretrained`, `neural_foundry.pythia`), and first-class independent evaluation engine (`neural_foundry.evals`, `weightctl eval`).
- **M4 — Multi-Modality Weight Programming, Adapters & Cross-Model Learning**: SQLite typed scientific graph (`nf_experiments::scientific_graph`), patch dependency/conflict composition (`nf_patches::composition`), evolutionary operators (`nf_patches::evolutionary_ops`), versioned recipe library (`nf_experiments::library_lifecycle`), symbolic rule-to-weights construction (`CG-09`), permutation/fitted weight transfer (`CG-10`), authorized LoRA/SFT/distillation/RL baselines (`CG-11`), `EasyEdit` (`ROME`, `MEMIT`, `AlphaEdit`) & `mergekit` (`task_arithmetic`, `ties`, `dare`) adapters (`neural_foundry.prd_adapters`), and `lm-evaluation-harness` / `Inspect AI` benchmark importers.
- **M5 — Structural Architecture Surgery, Quantization & GPU Dispatch**: Symbolic architecture dependency & shape solver (`nf_graph::shape_solver`), structural IR lowering (`nf_weight_ir::structural_lowering`), coupled MLP/attention/layer surgery (`neural_foundry.m5_surgery`), and DLPack GPU dispatch & VRAM resource leasing (`neural_foundry.gpu`).

See [Weight Programming Studio Architecture & PRD Guide](docs/weight-programming-studio.md) for the complete `WP-01`–`WP-24` requirement mapping, live `EleutherAI/pythia-14m` & `EleutherAI/pythia-31m` results, and API reference.

## Current scope: M1 trusted execution baseline

The base qualified M1 slice is Linux x86-64, dense f32/f64 CPU tensors and the registered native linear fixture. It parses strict Weight IR/search spaces, compiles bounded immutable plans, executes source-safe edits, verifies effects and smoke, replays immutable single-parent patches, and exports/reloads standalone Safetensors bundles. The CLI works without Python. The abi3 Python extension shares the same Rust APIs and identities.

Base M1 fixture execution is **software fixture qualification**, and milestone/scientific status is always reported from actual verification evidence (`neural_foundry.studio.generate_release_conformance_report`) rather than task checkbox counts. Base wheel installation (`import neural_foundry`) needs no Torch, NumPy, network client or model stack; research and GPU submodules (`gpu`, `pythia`, `prd_adapters`, `m4_modalities`, `m4_ecosystems`, `m5_surgery`, `studio`) load lazily on demand.

```sh
nf core capabilities
```

The capability manifest names the supported backend, operations, fixture factory and unavailable gates. Unknown inputs/backends fail; there is no silent fallback to a different edit or model.

## Development setup

Use the pinned rust-toolchain.toml and a Python 3.12 development interpreter/library. The project Cargo config defaults embedded-Python tests to .venv/bin/python; an explicit PYO3_PYTHON overrides it. Native-only cargo builds do not need Python.

```sh
uv venv --python 3.12 .venv
UV_CACHE_DIR="$KIROCREW_SCRATCH/uv-cache" UV_LINK_MODE=copy \
  uv pip install --python .venv/bin/python -r requirements-dev.lock
export PATH="$HOME/.cargo/bin:$PATH"
CARGO_BUILD_JOBS=2 cargo build -p nf-cli --example offline_direct
CARGO_BUILD_JOBS=2 cargo build -p nf-cli
CARGO_BUILD_JOBS=2 cargo build -p nf-experiments --example fixture_broker
CARGO_BUILD_JOBS=2 cargo build -p nf-experiments --example fixture_sandbox
CARGO_BUILD_JOBS=2 cargo build -p nf-experiments --example fixture_splits
CARGO_BUILD_JOBS=2 cargo build -p nf-experiments --example fixture_exposure
# Clear only this invocation's inherited Conda marker if it conflicts.
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" CARGO_BUILD_JOBS=2 .venv/bin/maturin develop
```

The optional real CPU PyTorch reference dependency is pinned separately from wheel runtime dependencies; the development lock supplies its CPU index. Base wheel installation needs no Torch, NumPy, network client or model stack. See [dependency policy](docs/dependency-policy.md) and [bounded test protocol](protocols/core-testing-v1.md).

## Offline examples

Native Rust example, using only the supplied finite JSON/hex fixture:

```sh
./target/debug/examples/offline_direct \
  python/neural_foundry/data/m1_fixture.json \
  "$KIROCREW_SCRATCH/m1-native-store"
```

The second argument is an explicit absolute artifact-store directory. Verified immutable blobs are written there; source/active references are never changed. The example performs a real scale edit, creates a patch record, recompiles/replays it, exports both original and edited checkpoints, reloads edited values, and restores the original by exact identity. Successful output includes exact replay IDs and `promotion_status: not_evaluated`.

Packaged Python example, also valid from an isolated installed wheel:

```sh
.venv/bin/python -m neural_foundry.demo
```

It loads the same fixture from package resources, compiles/applies the real native plan twice, compares numerical candidate identities, materializes/reloads a snapshot and checks source preservation. Its replay label explicitly says same-root plan execution; the native example exercises the persisted patch DAG/export/restore path. Neither imports Torch or claims scientific acceptance.

## CLI and SDK

```sh
nf model inspect CHECKPOINT.safetensors
nf ir validate PROGRAM.json
nf search-space validate SPACE.json
nf ir compile REQUEST.json
nf ir explain REQUEST.json
nf ir apply REQUEST.json
nf patch replay REQUEST.json
nf checkpoint restore REQUEST.json

# First-class Evaluation Engine & Studio CLI (available via both `nf` and `weightctl`)
weightctl eval import --suite SUITE.json
weightctl eval run --model MODEL.json --suite SUITE.json --compare-to BASELINE.json
weightctl eval compare BASELINE_RUN.json CANDIDATE_RUN.json
weightctl eval verify --model CANDIDATE_RUN.json --policy production
weightctl studio solve-shape GRAPH.json
weightctl studio lower-structural LOWER_REQUEST.json
weightctl studio compose-patches COMPOSE_REQUEST.json
weightctl studio scientific-graph GRAPH_REQUEST.json
weightctl studio library LIBRARY_REQUEST.json
```

Requests carry explicit source, policy, verification and resource bindings. Read commands do not execute studies. Typed JSON errors use stable reason codes. [Native CLI protocol](protocols/native-cli-v1.md), [Python SDK](protocols/python-sdk-v1.md), [control boundary](protocols/python-boundary-v1.md) and [export/restore](protocols/export-restore-v1.md) document exact formats and limitations.

```python
import neural_foundry as nf
from neural_foundry.studio import WeightProgrammingStudio

# 1. Low-level Rust transactional compilation & execution
source = nf.surgeon.load(manifest, checkpoint_bytes, assets)
plan = nf.ir.compile(source, program, search_space, policy, operands)
candidate = nf.surgeon.apply(plan, smoke_suite)

# 2. High-level Weight Programming Studio: select models, run reusable gyms, EasyEdit, mergekit & Evals
studio = WeightProgrammingStudio()
models = studio.list_selectable_models()
report = studio.run_reusable_gym_study("nf/mini-pythia-fixture", gym="algorithmic_three_skill")
```

## Verification

```sh
CARGO_BUILD_JOBS=2 cargo check --workspace --locked
CARGO_BUILD_JOBS=2 cargo test --workspace --locked
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
.venv/bin/ruff check python tests scripts
.venv/bin/mypy python tests
.venv/bin/python scripts/check_dependencies.py
.venv/bin/python -m pytest -q --basetemp="$KIROCREW_SCRATCH/pytest-full"
```

The full integration tests require a local Linux Docker daemon with cgroupv2 and seccomp. Explicitly fetch the CPU Python runtime first (no model weights):

```sh
docker pull python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
```

The native sandbox defaults off and qualifies the actual fixed profile before any untrusted calculation. Missing prerequisites fail the isolation gate; they are not skipped. This is not gym approval or a released study/model adapter. See [sandbox scope and limits](protocols/sandbox-status.md).

Custom gym source snapshots never import source into the coordinator. Explicit validation runs finite controller-supplied probes in the actual qualified CPU sandbox; approval is a separate exact-version action, not scientific acceptance or a study execution grant. See [local intake](protocols/gym-intake-v1.md) and [validation/approval scope](protocols/gym-validation-v1.md).

Three-way split data uses exact native catalogs and frozen final panels. Proposer inputs, evaluator labels and locked holdout scopes remain separate; actual OCI checks do not certify semantic independence or model execution. See [split access protocol](protocols/split-broker-v1.md).

Exposure history is retained across retries and family revisions. Two-way legacy edit/sealed data can be imported only with explicit exposed/unknown status, never silently promoted to independent three-way evidence. See [exact exposure and legacy intake](protocols/exposure-v1.md).

The registered synthetic arithmetic fixture has six ReLU units, five independently scored heads and frozen grouped three-way splits. A separate user sum package can be snapshotted and validated, then proposed alongside four built-in retention heads; no resulting descriptor accepts a model. See [arithmetic fixture scope](protocols/arithmetic-gyms-v1.md).

Pinned trusted PyTorch CPU workers load only registered numeric factories from native-verified Safetensors/config snapshots. Models remain in child processes, actual aliases/buffers and forward parity are checked, and copied receipts never promote weights. This trusted adapter is not the hostile-gym sandbox. See [model worker scope and limits](protocols/model-worker-v1.md).

The explicitly compiled `nf.trusted_torch_cpu` v1 edit profile dispatches fixed same-shape operations under inference mode. Rust checks every step's actual byte scope, aliases, native-reference parity and edited-model smoke before admitting an opaque candidate. Allocation/copy receipts and kernel measurements retain their distinct scopes. See [trusted Torch edit protocol](protocols/torch-edit-backend-v1.md).

Native portfolio planning now composes live-probed Foundry and actually approved user gym sources, freezes exact metric gates and schedules, reports missing intervention coverage and refuses weighted compensation for a required failure. Exploratory subsets remain distinct from a five-gym declaration; neither is an assurance pass or dispatch grant. See [portfolio coverage protocol](protocols/portfolio-coverage-v1.md).

Paired public-fixture controls now actually apply/export/reload a no-op, compare fresh seeded CPU processes against the original, deduplicate task/group counts and reuse source results only under an identical native evidence key. No-op controls cannot claim positive accuracy gain. See [paired controls](protocols/paired-controls-v1.md).

Frozen CPU measurement protocols now record real warmup/repeated interleaved forward/request timings and process RSS, bind units/statistics to exact workloads and environments, and keep unavailable device metrics separate. No-op timing noise cannot satisfy a positive-gain gate; co-resident RSS is not per-model memory attribution. See [measurement scope](protocols/measurement-v1.md).

Native statistical plans now analyze complete paired task/seed panels using exact finite-fixture checks or seeded group bootstrap, with frozen multiplicity reservations and confidence-bound gates. Missing pairs, insufficient support and unresolved uncertainty cannot pass; supplied measurements remain distinct from execution evidence and scientific acceptance. See [paired statistics](protocols/paired-statistics-v1.md).

For wheel isolation, see scripts/verify_wheel.py. Build a real wheel first and run the installed example outside the source tree. The CI recipe uses fresh native/wheel environments, not Python replacements or skipped integration gates. Repository workflow configuration is not evidence that a remote CI run has happened.

## License and distribution

Project terms are in LICENSE and NOTICE. Third-party Rust texts are bundled in the installed package and are not relicensed by this project. The source inventory records the locked dependency superset, including build/target-specific crates. Optional tooling remains separately licensed. Models, private gyms and user data require their own rights and are not covered by this project license. Do not redistribute private payloads or infer model/data rights from a valid checksum.
