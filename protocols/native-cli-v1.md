# Native core CLI v1

The `nf` binary uses the same canonical parser, manifests, compiler, isolated transaction executor, checkpoint store and exact replay verifier as the Rust libraries. It does not import Python, call models/providers, run training, execute arbitrary code, or start studies. Core commands return JSON on stdout; errors return a versioned JSON CoreError on stderr. Version/help remain text.

## Commands

```
nf model inspect CHECKPOINT.safetensors
nf ir validate PROGRAM.json
nf search-space validate SPACE.json
nf search-space show SPACE.json
nf ir compile REQUEST.json
nf ir explain REQUEST.json
nf ir apply REQUEST.json
nf patch replay REQUEST.json
nf checkpoint restore REQUEST.json
```

Inputs are single-link regular files opened no-follow/nonblocking at the final component. Files are read into bounded owned snapshots; JSON is limited to 1 MiB and checkpoint inspection to 64 MiB. Parent paths of a user-selected input are not a sandbox boundary. Artifact-store paths are absolute and FD-rooted with every component no-follow. Requests reject duplicate/unknown fields and unsupported versions. Unknown commands are denied before opening files or creating stores.

Inspect reports actual Safetensors checkpoint/header/tensor digests, shapes, dtypes and byte ranges, without loading model code. IR validation and search-space validation/show run only static schema checks; search-space quality remains not evaluated. Compile/explain load an explicit verified source export, validate its real runtime inventory, resolve the frozen search space, load bounded actual operands, and return the shared immutable plan identity/summary. They perform no edits or smoke/model forwards and cannot create store directories or blobs.

### Compile/explain/apply request

The strict request contains exactly `schema_version: 1`, `store` (absolute existing store path), `source_export` (immutable export root digest), `program` (Weight IR), `search_space`, `policy` (CompilePolicy), and `smoke_suite`. No hidden CLI-created permission grant, default source identity, or lowered policy replaces these records. Actual operands come from verified blob digests in the selected store. Missing source/operands fail explicitly. The current apply request is root-only; parent-bearing edits require qualified ancestry and are not flattened into roots.

Apply compiles, executes once through `certify_history` with a newly minted opaque witness, materializes actual edited parameters, creates a content-addressed PatchRecord, and uses the shared native export/reload smoke gate. Output includes `patch`, complete core `export` receipt and `certificate_execution_passes: 1`. The patch can be submitted directly to replay; it is not itself an execution certificate. No accepted/source pointer advances. Output says `promotion_status: not_evaluated` even when all core gates pass.

### Replay request

The strict request contains `schema_version: 1`, `store`, `source_export` (exact original root), `records` (PatchRecord list), and `target` (patch ID). The native DAG validates metadata, loads bounded real operands, and recompiles/executes the reachable ancestry. Every replayed candidate is bound to the recorded expected ID; optional expected IDs are used only when initially creating a new witness, never to skip persisted replay comparison. Output includes the result model, full final receipt, candidate identity and number of nodes replayed. Replay reads do execute the explicitly requested numerical replay, but write no artifacts and start no studies.

### Restore request

The strict request contains `schema_version: 1`, `store`, and `export_root`. Restore freshly loads and verifies explicit checkpoint/model/assets/aliases and returns their identities with `active_reference: unchanged`. It never chooses a latest target or edits an accepted reference. Workflow restore events and active-pointer promotion belong to the later experiment ledger.

## Exit codes and scope

- 0: command succeeded within its declared core stage.
- 2: invalid/unsupported command or validation/capability error.
- 3: denied core permission.
- 4: I/O failure, including unavailable artifacts.
- 5: source/integrity/scope/alias/smoke failure.
- 6: resource, artifact quota or parameter budget exceeded.

Requests and all core budgets are finite. The M1 store cap is 512 MiB/4096 entries with 64 MiB per blob; smaller compiler/execution/export/replay limits still apply. Read-only store handles cannot publish and open existing roots/locks/directories only. Compilation receipts are static plans, core export receipts verify the registered CPU fixture only, and neither is scientific acceptance. Unsupported model adapters/backends do not fall back silently. Subprocess tests run the binary with an empty PATH/environment to establish no Python dependency and exercise actual inspect/validate/compile/apply/export/replay/restore flows.
