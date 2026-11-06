# Weight Programming Studio (`neural_foundry`)

**Enterprise & Open-Source Platform for Agent-Native Weight-Space Programming, Pluggable Task Gyms, and Verifiable Model Editing**  
*License: Apache-2.0 · 9-Crate Rust Transactional Kernel + Python Studio SDK + `weightctl` CLI*

**Weight Programming Studio** lets researchers, engineers, and frontier AI agents (`Claude`, `GPT-6`, or custom LLMs) inspect, understand, modify, merge, and verify the weights of open-weight neural networks (`Pythia` / `GPT-NeoX`, `Qwen2`, `Llama`, `Mistral`, `Gemma`, or custom Safetensors models) **without conventional gradient training** — and compare those edits head-to-head against LoRA, SFT, distillation, and model-merging baselines under a single cryptographic evaluation firewall.

---

## Why Weight Programming Studio?

Most model-editing and merging scripts mutate PyTorch tensors in-place inside the same Python process that holds the benchmark answers. **Weight Programming Studio** turns weight-space programming into a verifiable engineering discipline:

1. **Pluggable Open-Weight Models (`studio.models`)** — Load `EleutherAI/pythia-14m`, `EleutherAI/pythia-31m`, `Qwen2`, `Llama`, `Mistral`, `Gemma`, or register any custom Safetensors checkpoint via architecture-agnostic site profiles (`ArchitectureSiteProfile`).
2. **Reusable Task Gyms (`studio.gyms`)** — Give agents and solvers structured experimentation environments with separated development, prefix-invariance validation, and skill-retention splits (`from_rules`, `NeMo-Gym`, `Verifiers`, `Gymnasium`).
3. **Extensible Weight Editors & Mergers (`studio.methods`)** — Run built-in methods (`rome`, `memit`, `alphaedit`, `contrastive_subspace`, `task_arithmetic`, `ties`, `dare`, `mlp_surgery`, `attention_surgery`, `lora_baseline`) or register your own algorithm with `@studio.methods.register(...)` backed by `WeightIRTransactionBuilder`.
4. **Interchangeable Frontier Agents (`studio.agents`)** — Run `claude-4-5-sonnet`, `gpt-6-astra`, or custom LLM agents over identical tool schemas with hypothesis tracking, duplicate-proposal blocking, failed-experiment memory, and singular-component ablation.
5. **Rust-Enforced Weight IR & Sealed Evaluation Firewall (`nf-weight-ir`, `nf-surgeon`, `studio.evals`)** — Every weight modification is compiled into a typed, content-addressed Weight IR plan, executed copy-on-write in Rust with snapshot reload parity, and verified against sealed holdout suites (`lm-evaluation-harness` and `Inspect AI` compatible) that agents cannot inspect or game.

---

## 5-Minute Quickstart

### 1. Run a Gym-Driven Weight Edit on an Open-Weight Model
```python
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()

# Select any registered model ("EleutherAI/pythia-14m", "EleutherAI/pythia-31m",
# "Qwen/qwen2-dense-mini", "nf/mini-pythia-fixture", or a local checkpoint path)
# and teach it a multi-skill task Gym in 0 gradient steps:
report = studio.run_reusable_gym_study(
    "nf/mini-pythia-fixture",
    gym="algorithmic_three_skill",
    target_strength=80.0,
)

print(f"Rust Plan Digest:      {report.rust_plan_digest}")
print(f"Rust Candidate Digest: {report.rust_candidate_digest}")
print(f"Reload Parity:         {report.reload_parity_verified}")
print(report.promotion_verification["markdown_dashboard"])
```

### 2. Knowledge Editing (`EasyEdit`) & Multi-Checkpoint Merging (`mergekit`)
```python
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()
model = studio.select_model("nf/mini-pythia-fixture")

# Synthesize two independent capability edits via AlphaEdit (or "ROME" / "MEMIT")
capitals_edit = studio.run_easyedit(
    model,
    algorithm="AlphaEdit",
    associations=[("The capital of France is", "ĠParis"), ("The capital of Japan is", "ĠTokyo")],
    nullspace_prompts=["Boolean True AND False gives", "Boolean True OR False gives"],
)
logic_edit = studio.run_easyedit(
    model,
    algorithm="AlphaEdit",
    associations=[("Boolean True AND False gives", "ĠFalse"), ("Boolean True OR False gives", "ĠTrue")],
    nullspace_prompts=["The capital of France is", "The capital of Japan is"],
)

# Compose both edited checkpoints using TIES-Merging (or "task_arithmetic" / "dare") in Rust Weight IR
merged = studio.run_mergekit(
    model,
    algorithm="ties",
    checkpoint_weights=[capitals_edit.edited_weight, logic_edit.edited_weight],
    scaling_weights=(1.0, 1.0),
    density=0.90,
)
assert merged.reload_parity_verified is True
```

### 3. Run Interchangeable Frontier Agents (`Claude` & `GPT-6`)
```python
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()
model = studio.select_model("nf/mini-pythia-fixture")

for agent_id in ("claude-4-5-sonnet", "gpt-6-astra"):
    agent = studio.create_frontier_agent(agent_id)
    session = agent.run_reasoning_loop_repair_experiment(model)
    print(agent_id, "-> candidate:", session.rust_candidate_digest, "loop_rate:", session.repaired_loop_rate_after)
```

---

## Building on Weight Programming Studio (Extensibility Guide)

Weight Programming Studio is designed as a **platform you can extend** without modifying core Rust or Python files:

### A. Create & Register Your Own Task Gym
```python
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()

custom_gym = studio.gyms.from_rules(
    "gym.custom.antonyms.v1",
    skills_taught=("boolean_complement",),
    base_rules=[
        ("Boolean True AND False gives", "ĠFalse"),
        ("Boolean True OR False gives", "ĠTrue"),
    ],
    register_as="custom_antonyms",
)

report = studio.run_reusable_gym_study("nf/mini-pythia-fixture", gym="custom_antonyms")
```

### B. Write & Register a Custom Weight-Editing Algorithm
Use `studio.ir_builder` (`WeightIRTransactionBuilder`) to compile any custom mathematical update (`low_rank_update`, `add_delta`, `scale`, `subspace_projection`) into a Rust-verified transaction:

```python
import torch
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()

@studio.methods.register("spectral_dampener", category="custom_repair", description="Dampens dominant singular mode")
def spectral_dampener(weight_matrix: torch.Tensor, *, damping: float = 0.25):
    u, s, vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    q_top1 = u[:, :1].contiguous()
    # Compile and execute a rank-1 subspace_projection in Rust with full precondition & smoke checks:
    return studio.ir_builder.execute_matrix_edit(
        weight_matrix,
        operation_type="subspace_projection",
        orthonormal_basis=q_top1,
        projection_side="left",
        factor=damping,
        provenance_tag="custom.spectral_dampener.v1",
    )

tx_receipt = studio.methods.run("spectral_dampener", torch.eye(8, dtype=torch.float64), damping=0.5)
assert tx_receipt.reload_parity_verified is True
```

### C. Register a New Open-Weight Model or Architecture Profile
```python
from neural_foundry.pythia import ArchitectureSiteProfile
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()

# 1. Register tensor naming conventions for any custom transformer family
studio.models.register_architecture(
    ArchitectureSiteProfile(
        family_id="custom_decoder",
        mlp_down_proj_template="decoder.blocks.{layer}.ffn.w2.weight",
        mlp_up_proj_template="decoder.blocks.{layer}.ffn.w1.weight",
        attn_out_template="decoder.blocks.{layer}.attn.wo.weight",
        embed_in_tensor="decoder.tok_embeddings.weight",
        unembed_out_tensor="decoder.output.weight",
        final_norm_weight_tensor="decoder.norm.weight",
    )
)

# 2. Register a custom model loader into the Studio catalog
studio.models.register_model(
    "my-org/custom-pythia",
    loader=lambda: studio.select_model("nf/mini-pythia-fixture"),
    architecture="GPTNeoXForCausalLM",
)
```

---

## Measured Results on Open-Weight Models (`Pythia-14m`, `Pythia-31m`, `Qwen2`)

Running the reusable `algorithmic_three_skill` Gym (teaching **Number Parity**, **Boolean Logic**, and **Inequality Comparison** simultaneously across 6 output classes) and comparing **Direct Contrastive Prefix-Nullspace Weight Programming (`0` gradient steps)** against a **25-step Gradient LoRA baseline (`CG-11`)** on identical data:

| Selected Model | Architecture | Total Params | Edited Params (`%`) | Gym Dev | Gym Prefix-Val | Unseen Holdout Eval | 25-Step LoRA Holdout | Skill Retention | Eval Gate |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`EleutherAI/pythia-14m`** | `GPTNeoXForCausalLM` | `14,067,712` | `65,536` (`0.47%`) | **`100%`** | **`100%`** | **`100.0%` (`12/12`)** | `50.0%` (`0%` retention) | **`100%`** | **`passed`** |
| **`EleutherAI/pythia-31m`** | `GPTNeoXForCausalLM` | `30,495,232` | `262,144` (`0.86%`) | **`100%`** | **`100%`** | **`91.7%` (`11/12`)** | `50.0%` (`0%` retention) | **`100%`** | **`passed`** |
| **`Qwen/qwen2-dense-mini`** | `Qwen2ForCausalLM` | `51,360` | `2,048` (`3.99%`) | **`100%`** | **`100%`** | **`75.0%` (`9/12`)** | `0.0%` | **`100%`** | **`passed`** |
| **`nf/mini-pythia-fixture`** | `GPTNeoXForCausalLM` | `49,920` | `2,048` (`4.10%`) | **`100%`** | **`100%`** | **`83.3%` (`10/12`)** | `0.0%` | **`100%`** | **`passed`** |

See [Weight Programming Studio Architecture & PRD Guide](docs/weight-programming-studio.md) for the complete `WP-01`–`WP-24` requirement mapping and Priority Experiments `A`–`D` details.

---

## CLI Reference (`weightctl` / `nf` & Studio CLI)

Both the native Rust binaries (`weightctl` and `nf`) and the Python Studio CLI (`python -m neural_foundry.studio`) are included:

```sh
# High-level Studio CLI (inspect catalogs, run Gym studies, run Experiments A-D, emit conformance)
.venv/bin/python -m neural_foundry.studio catalog
.venv/bin/python -m neural_foundry.studio run-gym --model nf/mini-pythia-fixture --gym algorithmic_three_skill
.venv/bin/python -m neural_foundry.studio run-experiments --model nf/mini-pythia-fixture
.venv/bin/python -m neural_foundry.studio conformance

# First-class Evaluation Engine & Studio Rust CLI (available via both `weightctl` and `nf`)
weightctl studio catalog
weightctl studio solve-shape GRAPH.json
weightctl studio lower-structural LOWER_REQUEST.json
weightctl studio compose-patches COMPOSE_REQUEST.json
weightctl studio scientific-graph GRAPH_REQUEST.json
weightctl studio library LIBRARY_REQUEST.json
weightctl eval import --suite SUITE.json
weightctl eval run --model MODEL.json --suite SUITE.json --compare-to BASELINE.json
weightctl eval compare BASELINE_RUN.json CANDIDATE_RUN.json
weightctl eval verify --model CANDIDATE_RUN.json --policy production

# Low-level Rust Weight IR, Search Space, Patch Diff/Sign/Audit/Rollback & Checkpoint CLI
nf core capabilities
nf model inspect CHECKPOINT.safetensors
nf ir validate PROGRAM.json
nf search-space validate SPACE.json
nf ir compile|explain|apply REQUEST.json
nf patch diff PATCH.json
nf patch sign PATCH.json --signer operator.v1 --key SECRET_KEY
nf patch verify-signature SIGNED_PATCH.json --key SECRET_KEY
nf patch audit SIGNED_PATCH.json
nf patch rollback SIGNED_PATCH.json
nf patch replay REQUEST.json
nf checkpoint restore REQUEST.json
```

---

## Rust & Python Architecture (`M1`–`M5`)

The repository implements a 9-crate Rust workspace (`nf-core`, `nf-checkpoint`, `nf-graph`, `nf-weight-ir`, `nf-surgeon`, `nf-patches`, `nf-experiments`, `nf-cli`, `nf-python`) paired with the `neural_foundry` Python SDK:

- **M1 — Trusted Execution Core**: Strict Weight IR (`replace_slice`, `add_delta`, `scale`, `mask`, `permute_axis`, `linear_map`, `subspace_projection`, `low_rank_update`), search-space validation, copy-on-write transactional execution (`nf-surgeon`), single/multi-parent patch replay (`nf-patches`), and standalone Safetensors export/restore (`nf-checkpoint`).
- **M2 — Epistemic Firewall, Sandboxed Gyms & Paired Statistics**: Three-way split broker (`development`, `validation`, `holdout`), exposure ledger, static-ELF guardian + rootless/no-network OCI container sandbox, trusted PyTorch model worker, paired no-op controls, and bootstrap/exact statistical gates.
- **M3 — Autonomous Scientist Loop & Pretrained Model Repair**: Bounded weight/activation inspector (`neural_foundry.inspection`), localization & structured hypothesis cards (`neural_foundry.diagnostics`), multi-branch candidate search & ablation (`neural_foundry.autonomous`), Qwen2 & Pythia/GPT-NeoX pretrained adapters (`neural_foundry.pretrained`, `neural_foundry.pythia`), and first-class independent evaluation engine (`neural_foundry.evals`, `weightctl eval`).
- **M4 — Multi-Modality Weight Programming, Adapters & Cross-Model Learning**: SQLite typed scientific graph (`nf_experiments::scientific_graph`), patch dependency/conflict composition (`nf_patches::composition`), evolutionary operators (`nf_patches::evolutionary_ops`), versioned recipe library (`nf_experiments::library_lifecycle`), symbolic rule-to-weights construction (`CG-09`), permutation/fitted weight transfer (`CG-10`), authorized LoRA/SFT/distillation/RL baselines (`CG-11`), `EasyEdit` (`ROME`, `MEMIT`, `AlphaEdit`) & `mergekit` (`task_arithmetic`, `ties`, `dare`) adapters (`neural_foundry.prd_adapters`), and `lm-evaluation-harness` / `Inspect AI` benchmark importers.
- **M5 — Structural Architecture Surgery, Quantization & GPU Dispatch**: Symbolic architecture dependency & shape solver (`nf_graph::shape_solver`), structural IR lowering (`nf_weight_ir::structural_lowering`), coupled MLP/attention/layer surgery (`neural_foundry.m5_surgery`), and DLPack GPU dispatch & VRAM resource leasing (`neural_foundry.gpu`).

Base wheel installation (`import neural_foundry`) requires **zero third-party Python runtime dependencies** (no Torch, NumPy, or network clients); PyTorch-backed research and GPU submodules load lazily on demand. Milestone and capability-gate statuses are always generated from live verification evidence (`generate_release_conformance_report`) rather than task checkbox counts.

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

Requests carry explicit source, policy, verification and resource bindings. Read commands do not execute studies. Typed JSON errors use stable reason codes. [Native CLI protocol](protocols/native-cli-v1.md), [Python SDK](protocols/python-sdk-v1.md), [control boundary](protocols/python-boundary-v1.md) and [export/restore](protocols/export-restore-v1.md) document exact formats and limitations.

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
