# Local custom gym intake v1

## This is a snapshot, not an import hook or approval

`neural_foundry.gyms.import_local(directory, manifest, store)` snapshots exactly the explicitly selected local files. `import_file(source, manifest, store)` additionally requires the selected file to equal the declared primary single-shot/episode entry file. Both use the installed PyO3 extension and authoritative Rust `nf_experiments::intake`; Python performs wire serialization only. Importing the SDK does not scan directories, import source, load a framework, install dependencies, run factories/scorers, call tools or start training. There is no coordinator `exec`, importlib discovery or Python callback.

The native frozen `GymSnapshotHandle` exposes only `snapshot_id`, canonical `manifest_bytes()` and bounded `asset_bytes(selected_name)`. It cannot be constructed, edited or used to read arbitrary store keys. `gyms.restore(id, store)` opens an existing store read-only and verifies the root plus every selected blob. It never follows the now-live source path. Linux FD-rooted intake is the current supported capability; optional intake absence does not break base SDK import.

Snapshot status is always `snapshotted_not_validated` and trust is `untrusted_local_source`. There is no validation/approval receipt in this record and no execution/promotion permission. It is intentionally not the approved `GymManifest` used in frozen studies: later validation/approval/binding must resolve exact snapshot, policy, schema, scorer/oracle and image identities. Arbitrary names, licenses and oracle declarations do not establish origin, legal permission, correctness, independence or confidentiality compliance.

## Explicit draft declaration

The strict version1 `Draft` has these fields:

- `schema_version`, `name`, `version`, `usage_rights`, `confidentiality` (the native AccessClass).
- `assets`: normalized relative names mapped to `{kind, expected}`. Kinds are `code`, `data`, `config`, `manifest`, `dependency_lock`, `license`; expected is an optional exact SHA256 assertion. No directory glob/discovery is performed.
- `dependency_lock`: an explicitly selected dependency-lock asset. `licenses`: explicitly selected, nonempty UTF8 license assets. Selected config/manifest files must be bounded valid JSON; arbitrary dataset bytes are retained without pretending they are scientifically valid tasks.
- `interface`: tagged `single_shot` or `episode`. Single-shot declares `evaluate`, `score`, `oracle` file/symbol entrypoints plus input/output schemas. Episode declares `factory`, `score`, `oracle`, reset/observation/action/result schemas, positive finite maximum steps/time. Symbols are syntactically checked identifiers, not imported/discovered methods.
- `dataset`: tagged `frozen` with selected data files and distinct task/group fields, or `generator` with a selected code factory/config, unique bounded seed list and task/group fields. Generators do not execute during intake.

The bounded declarative schema language supports closed objects (`properties`, `required`), arrays (`items`, `max_items`), strings (`max_bytes`), finite ordered numbers, ordered integers and booleans. Unknown fields/types, impossible required fields and unbounded/deep schemas fail. Object schemas are closed: missing optional fields may be absent, but no undeclared property is implied. Runtime instance/scorer/oracle/lifecycle validation belongs to the next task, not to source snapshotting.

The version1 lock has `{schema_version, image, packages}`. `image` is an immutable SHA256 reference. Each package declares `{version, integrity, license}` with an exact numeric/v-prefixed version (no mutable aliases/ranges/commands), immutable content assertion and a retained selected license. The lock itself is snapshotted. This records a supplied dependency specification, NOT evidence that packages were installed, resolved, audited or present in a runtime. The fixed T026 sandbox may reject unsupported images/dependencies during subsequent authorized validation; intake never installs them into the coordinator.

## Bounds and source fences

Rust requires an explicit absolute non-root source directory and opens every root/child component with no-follow directory FDs. Asset names are relative, bounded to8 components/256bytes; empty/dot/parent/absolute/backslash/colon/control paths are rejected. Only explicitly named regular, single-link files are read, using no-follow/nonblocking opens (FIFOs/devices/directories and symlink/hard-link aliases fail).

Limits are64 files,1MiB/file and8MiB total. Schemas are limited to16 recursive levels/256 nodes, arrays4096 and strings64KiB; parser bytes/depth/node limits apply before typed deserialization. The SDK's content store additionally bounds128MiB/1024 entries/1MiB per blob. There is no unbounded tree walk or source execution to discover assets.

The read bytes, lengths, roles and actual hashes determine the snapshot. Before root publication, Rust reopens each selected path through the original directory FD, verifies inode identity and rehashes its bounded content. It also checks the physical source root before/after this fence. Same-size rapid rewrites cannot be authorized by timestamps alone. A changed/replaced source rejects that intake; already written unreferenced CAS blobs are not a published snapshot. As with other bounded local snapshots, this is a content snapshot with a final consistency fence, not a filesystem-wide transaction or an assurance against a hostile host/root changing files after the fence.

## Identities, persistence and disclosure

Actual selected bytes are stored in the native no-follow immutable CAS. The canonical root manifest is published last, after all checks/fences. It binds the full typed draft, selected relative paths/roles/lengths/digests, parsed lock and aggregate identities; absolute host source paths are not embedded. Source/data/dependency/license/manifest/schema/entrypoint/seed changes create new snapshot identities, even under an unchanged human version label. Unselected files do not silently enter the snapshot or alter it.

`code_identity` binds selected code blobs. `data_identity` binds the dataset declaration and data/config bytes; generators additionally bind all selected code and dependency-lock bytes, not just a filename. These are exact snapshot-definition identities, not semantic independence certificates. Renaming/copying/relabeling an exposed dataset cannot establish fresh holdout evidence; later split/exposure checks remain mandatory.

Restore rechecks every asset's hash/size/role/expected identity, lock/license references, aggregate identities and canonical typed root identity. Corrupt/missing/replaced blobs fail; a frozen handle never switches to changed source files. A caller who owns the store can supply untrusted bytes, but cannot obtain validation, approval, model provenance or an execution grant from this handle.

The controller explicitly selects paths and controls the private store. It must not pass its handle/store credentials to a gym worker or mount hidden inputs into proposal scopes. Only approved scope bytes may later be materialized into the qualifying sandbox; this intake layer adds no arbitrary host mount or artifact-path capability. Confidentiality declarations do not authorize exporting captured source/data/licenses or private hashes.

## Verification

`cargo test -p nf-experiments custom_gym_import` runs the real content fence against same-size rewrites and inode replacement. `python -m pytest -q -k custom_gym_import` uses the real installed extension and CAS: side-effectful imports remain inert, nested package/single-file declarations work without discovery, source changes make new immutable versions, episode generators bind seeds/config/code, and traversal/symlink/hard-link/FIFO/role/schema/bounds/dependency/license/duplicate-JSON/CAS-corruption/undeclared-asset attempts reject. No mock/import interception or skipped runtime qualifies these checks.
