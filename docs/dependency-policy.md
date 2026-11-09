# M1 dependency and license policy

Project code is Apache-2.0. Licensing is separate from permission to download, use or redistribute a model, gym or dataset. No external weights/data are included. A dependency's metadata is a declared license, not a legal warranty or a vulnerability audit.

## Rust distribution

- Cargo.lock pins all resolved crates and registry checksums; direct versions/toolchain are pinned in workspace manifests. Builds use --locked. No unreviewed Git/path dependency or build-script/network execution is authorized by license validation.
- scripts/check_dependencies.py queries actual locked cargo metadata, rejects unknown license expressions/sources and missing notice texts, checks the reviewed inventory, and verifies bundled text digests. It does not execute studies. The allow-list explicitly covers observed MIT/Apache alternatives, Apache-2.0 with LLVM exception, BSD software license alternatives, and the MIT/Apache plus Unicode-3.0 conjunction. Slash legacy notation is retained as declared, not rewritten silently.
- licenses/dependencies.json records all 64 third-party packages in the current resolved superset, including build tools and Windows-target crates. This is intentionally broader than the linked Linux runtime. It records full upstream expressions, not a claim that every optional license branch was selected.
- SQLite coordinator additions use rusqlite 0.37.0/libsqlite3-sys 0.35.0 with bundled SQLite 3.50.2. The SQLite public-domain copyright-disclaimer header is retained alongside the MIT Rust binding license. Foldhash's Zlib terms were reviewed and its full attribution retained; this does not relabel it MIT/Apache.
- Actual upstream LICENSE/COPYING/NOTICE files are retained without stripping copyright text in python/neural_foundry/data/THIRD_PARTY_NOTICES.txt and included in wheels with the project LICENSE/NOTICE. Regenerate only with explicit --write after reviewing dependency/notice changes. Unknown/copyleft runtime terms stop the check for human review; adding a word to an allow-list is not a substitute for checking obligations.

## Python environments

The wheel has no mandatory third-party Python runtime dependencies. The extension includes Rust code under the upstream texts above. Python development/reference tools are not vendored into the wheel. Their distributions retain their own licenses and any bundled library notices.

requirements-dev.lock pins the actual development/CPU-reference environment, including transitive tool packages. It names the explicit PyTorch CPU index; no CUDA/model package or model weights are required. The locked CPU smoke environment includes Transformers 4.57.3 and Tokenizers 0.22.1 for the tiny, randomly initialized Qwen fixture. licenses/python-tools.json records versions, declared/reviewed license labels and the development/optional-reference scope. Tool-license metadata may use classifiers rather than SPDX. Known missing-expression cases were checked against packaged license files: Jinja2, mpmath and SymPy BSD terms, setuptools MIT, pathspec and certifi MPL-2.0, plus the classifier-only Tokenizers Apache-2.0 declaration. These are unmodified development/reference dependencies, not linked or shipped project runtime components; upstream file-level obligations still apply if redistributed/modified. The license labels remain tool-specific: MPL-2.0 dependencies are not relabeled Apache-2.0, and Tokenizers' Apache-2.0 label is explicitly based on its classifier because its wheel omits the license file.

Review and record the full development environment with:

```
.venv/bin/python scripts/check_dependencies.py --python
```

When pytest itself runs in the optional CUDA interpreter, trusted native model
workers still need the CPU-reference Torch build under their 2 GiB address-space
cap. Set `NF_TRUSTED_TORCH_SITE_PACKAGES` to the CPU `.venv`'s
`lib/python3.12/site-packages`; the pytest parent remains in the GPU interpreter.
Run `nf-experiments` integration tests inside the documented empty-capability
user namespace. Docker-backed tests additionally require the pinned local Docker
runtime; they are not qualified on a host without Docker control.

This compares it to the checked-in environment inventory/lock. It is optional in clean wheel environments, which intentionally lack development/reference packages. CPU PyTorch has its own extensive LICENSE/NOTICE bundle; optional reference installation does not certify third-party subcomponents or supply target model rights. No dependency check is represented as a security advisory/CVE scan.

The T024 native-fixture supervisor invokes the existing host `/usr/bin/prlimit` as a separate verified system utility. It is not vendored or relicensed in the wheel; the host distribution retains its terms/notices. Qualification fails if the actual utility or its resource probe is unavailable. This is separate from the Rust-linked dependency inventory and is not a general sandbox qualification.

## Updates and CI

A new direct/transitive version, changed registry checksum, license expression or upstream text changes the review artifacts. Update Cargo.lock/tool manifests, inspect diffs, then regenerate explicitly with --write (and --python when reviewing Python tools), retain notices, and rerun build/tests and isolated wheel verification. Never edit a policy to evade an actual permission denial. Automatic release, registry publication, cloud jobs and paid provider calls remain outside the approved local implementation.

An optional GPU runtime is intentionally outside the locked CPU reference environment. `torch 2.9.1+cu128` is installed through the explicit PyTorch CUDA index into a separate local interpreter and selected by kernel preflight, not by the wheel metadata alone. It is not part of the base wheel dependency graph or the CPU worker's trusted environment.

The repository CI file is a reproducible recipe, not a claim of remote execution. Local acceptance records actual native example and isolated wheel test results separately. Distribution qualification is Linux x86-64/Python 3.12 for the current host; abi3 packaging metadata is not evidence that every OS/interpreter/backend was tested.

The optional generated-response evidence runtime additionally installs
`transformers 4.57.3`, `huggingface-hub 0.36.0`, and their upstream dependencies
only in the separate CUDA interpreter. They retain upstream distributions and
licenses and are not vendored into the wheel or added to the locked CPU reference
inventory. Public checkpoint downloads are opt-in and pinned to immutable
revisions; the Qwen3/SmolLM2 examples declare Apache-2.0 on their model cards.
Local execution evidence is not a redistribution permission review or release
qualification. The manual GPU workflow requires an already qualified local
runtime/checkpoint configuration and does not publish a package or deployment.
