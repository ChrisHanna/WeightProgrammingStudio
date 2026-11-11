# Public proof runner (Phase B)

The user authorized Phase B independently of proof/custody acceptance. This is
public-only development evidence, not final proof or release authorization.
The retired synthetic holdout is not read or rehabilitated by this work.

## Contract

`neural_foundry.proof_pipeline` accepts a digest-verified
`nf_public_proof_run`, version 1. The spec explicitly binds:

- Base model ID/revision; candidate roles and methods as data, not a fixed roster.
- Exact version/digest plugin references for Gym, training model, `TrainerAdapter`,
  training-coverage audit, serving export, augmenter and public audit. Each
  candidate selects its server/provider and evidence verifier independently.
- Ordered train-only rows, public-set row digests and integer success-count gates.
- Versioned decoding/output cap, identity tolerance/count, API attempt/cost
  ceilings, report-only decisions and prohibition on adaptive retraining.

The runner resolves the Gym's own scorer through `GymComponentRegistry`.
No invoice scorer, provider name, candidate method/role branch or custody import
is present in the generic runner. Local identity uses the existing 16-probe,
`1e-6` same-runtime policy. The first provider wraps the existing OpenAI client,
preserving receipts, reservations and decoding identity; it never reads `.env`.
The current approved spec forbids new API calls. A separate recorded-response
plugin verifies original receipt/configuration/interaction/response digests and
preserves v2-to-identical-input-v3 lineage without issuing requests.

Training coverage is derived from the verified committed-step log and frozen
batch-one, unshuffled, unpacked schedule, with an explicit per-step scenario-ID
roster. It is **not direct per-batch ID telemetry**. A short historical schedule
is labelled incomplete rather than reclassified as complete.

SFT full artifacts pass verified serving export. LoRA export verifies adapter
reload, merges, saves, reloads the full model and binds source/result/state/file
lineage. Other artifact kinds require another exporter plugin. Serving identity
is frozen before scoring, rechecked on scoring and checked at shutdown.
Public score stages have exclusive started claims and bounded failure receipts;
they cannot silently repeat an API budget after a failed/previous attempt.
An explicit `--retry-of-failure` can link an incomplete local-server
infrastructure attempt to its preserved failed receipt. Provider retries are
not allowed by this path, and completed public scorecards cannot be replaced.

`scripts/run_json_validation.py --spec ...` delegates to this runner. Calls
without `--spec` retain historical behavior in `json_validation_compat.py`.
JSON training execution in `run_gym_lifecycle_real.py` delegates after its old
preparation checks; non-JSON behavior is unchanged.

## Initial 448-row contract (superseded before training)

The proposed spec is
`artifacts/phase-b-balanced-authority-20261111/frozen-final/run-spec.json`.
It retains the original ordered 384 rows and interleaves 64 fresh authority
examples: **448 rows / 448 steps**, within 512. The 64 rows balance trusted
body/footer placement, trusted/decoy JSON versus plain encoding, full versus
single-field disagreement, and exact target/decoy vocabulary counts. JSON is
sometimes authoritative and footer text is sometimes authoritative.

Before any training, the frozen report-only thresholds are **16/16 required,
at least 91/96 injection, at least 150/192 total**. Public64 is reported too.
These supersede the earlier proposed competitive gate, not specialist-v2 policy.

**The initial contract was blocked by the approved public overlap audit.** Required16 passes
zero exact and zero shared five-grams. The new 64 examples also pass these checks
against panel192. However, the retained 384 rows already share **30 distinct
five-grams** with panel192. The policy is not weakened, old rows are not silently
deleted/rewritten, and the runner refuses before loading a model. The persisted
spec is inspectable/executable but is not an approved-to-dispatch preparation.
The user subsequently approved the replacement contract below.
Full custodial audit remains unpassed independently.

The frozen spec digest is
`sha256:2717bb6670478254662440ef56915c0016b68181a9dda21346cf6ac7a3a818ec`;
the failing public-audit digest is
`sha256:3e6f5723ac17e12aeffeb06380f28567491279b5d7b2929ffbd7da3bba063809`.
All-training/panel shared five-grams are 30; injection-only/panel are 20.
Neither audit scope has any exact-input matches.

Actual persisted results under that `frozen-final` directory (earlier
development freezes and failed test attempts are retained in the parent):

- `focused-tests-command/receipt.json`: **60 passed, 2 live tests
  skipped**, return code 0. The earlier failed test receipt is retained.
- `frontier-reuse-command/receipt.json`: return code 0, **zero new API calls**.
  `round-one/public-score-frontier-reused.json` preserves the old manifest/model
  receipts, reproduces **160/192 overall and 96/96 injection**, and explicitly
  marks the full gate incomplete because required16 was not re-scored.
- `freeze-command/receipt.json`: expected return code 2 for the blocked public
  audit; `public-audit.json` is not a custodial-pass receipt.

## Approved clean training contract v2

The user explicitly approved pretraining data rejection sampling, without
changing thresholds, public templates or labels. Attribution of the original
30 shared five-grams is **10 from base320, 20 from old64, zero from new64**.
The old64 injection augmentation is removed. Of base320, 308 inputs/references
are retained unchanged; 12 colliding rows are replaced with freshly sampled
train values using their original train templates. Fourteen candidates total
were rejected, including two replacement candidates. Each replacement reference
is generated with its input, never relabelled after evaluation.

The deterministic seed is 20261114. The excluded-candidate roster digest is
`sha256:4aaeabc9c61386368c7515cbb9c71cdfcb23b20f0f3309b3313d8f30024910ff`.
The receipt binds both public row digests and before/after counts. This is
**public-panel-aware pretraining filtering**, not evidence of independence from
that public panel, unseen phrasing, or private-data decontamination.

The unchanged balanced new64 preserve competing-authority and extra-field
defenses. Five base rows then one new authority row yields **384 rows / 384
ordered steps**, with no outcome-based retraining. All-training and
injection-only audits against **both required16 and panel192** have zero exact
matches and zero distinct shared five-grams; required phrasing families remain
disjoint. The frozen score thresholds remain **16, 91, 150**.

The new spec is
`artifacts/phase-b-clean-contract-v2-20261111/run-spec.json`,
digest `sha256:2fda94400e06c6d758117e4b781b150739b6b775fb3ee708603b375a185fb1c1`.
Its passing **public-only** audit digest is
`sha256:011bec072c6d7daa451363c7d91102d3cb7cf8c75cf71d7c50542c27e2c8e146`.
Full custodial audit remains independently unpassed. Initial blocked artifacts
and source preparations are retained unchanged.

The actual SFT and LoRA training commands completed once each, with 384 committed
steps and complete 384-row schedule-derived coverage. Durations were 279.049 s
and 140.051 s. Generic SFT serving export and LoRA merge/full-model reload passed.
Training-result digests:

- SFT: `sha256:d3d64c3833a170e8ec1580e477535f08e0a2f28d2939ef4010af2a588e51e900`.
- LoRA: `sha256:ceb53ebdf6b101c2ec02bfb943144e8a0b9455fbc37c3ae508d1e3e6dc8c8063`.

The first SFT identity worker passed, but a relative-path advisory status rename
hit `EXDEV`; its successful worker receipt was retained and status repaired
without repeating probes. The first SFT scoring server failed **before any
inference**, while Transformers resolved a relative path after Docker/WSL
detached the process's working-directory mount. The fix reanchors an invalid
cwd after Docker returns and uses absolute run/status paths. A direct regression
test simulates an unlinked cwd. The failed startup claim/receipt is retained,
and its explicit local retry does not change weights, recipes, decoding,
identity tolerance, public rows or thresholds.

### Completed fixed-round public results

| Method | Public64 | Required16 | Panel192 | Injection96 | Frozen gate |
| --- | ---: | ---: | ---: | ---: | --- |
| SFT | 64/64 | 13/16 | 73/192 | 1/96 | failed |
| LoRA | 64/64 | 13/16 | 93/192 | 9/96 | failed |

Both serving identities are verified, all 272 responses per method satisfy the
output schema, and owned serving snapshots were removed. The output mode is
the existing guided JSON schema mode, not unconstrained generation. No
quality-driven retries or additional training rounds were performed.

Results and command receipts are under
`artifacts/phase-b-clean-contract-v2-20261111`:

- `round-one/public-score-sft.json`:
  `sha256:d7aac3c61869d1a6aa2bb0b8f2b5de56207f58a301c2ccd417149a5dd84a4f22`.
- `round-one/public-score-lora.json`:
  `sha256:e7bc484e2ad76a0f1be635136ecb73c3ce62b31b4c209dc93265ca3b19df426d`.
- Successful stage receipts: `sft-train-command`, `lora-train-command`,
  `sft-export-command`, `lora-export-command`, `sft-identity-command`,
  `lora-identity-command`, `sft-score-startup-fix-command`, `lora-score-command`,
  each containing `receipt.json`. The original `sft-score-command` failure and
  started claim are retained, not relabelled as successful.
- `final-public-tests-command/receipt.json`: **43 passed**, using explicit
  public-only selectors. `final-lint-command/receipt.json` passes scoped lint.

The LoRA tokenizer loader emitted a regex-pattern warning. No tokenizer setting
was silently changed or warning suppressed. A no-forward CPU audit compares
the pinned base and merged-export prompt token IDs: **zero mismatches across
all 272 public inputs**. Its receipt is `tokenizer-lineage-command/receipt.json`;
`lora-tokenizer-lineage-audit.json` has digest
`sha256:70291a504d2480ec11e5652c233673174b236ea67bc4cf36f03c230f7dbcf0b8`.
This check does not resolve or certify tokenization on other inputs.

Earlier broad compatibility-test invocations exercised the existing
deterministic **seed31 synthetic-holdout unit fixture** with stub providers.
That is development-fixture generation, not a read of the retired seed17 panel,
an independent custodial draw, or eligible holdout evidence. This exposure is
retained explicitly; final selectors exclude those legacy tests. The generic
public operational runner has no holdout stage or fallback.

## Exact commands (single approved round)

Run the following Python in WSL from the repository. It uses cached runtimes,
fresh per-stage logs, the bounded-job runner and no downloads. Coordinate with
the native/deadline owner before running GPU stages. This is one round per
method, with no score-driven retraining.

The wrapper writes a per-job `*-command-status.json` beside its log directory;
it never updates the main owner's `artifacts/STATUS.json`.

```python
import os
import subprocess
from pathlib import Path

project = Path.cwd()
home = Path.home()
gpu = home / ".venvs" / "neural-foundry-gpu" / "bin" / "python"
automodel = home / ".venvs" / "nf-automodel-0.7.0" / "bin" / "python"
reload = home / ".venvs" / "nf-peft-qlora" / "bin" / "python"
spec = Path("artifacts") / "phase-b-clean-contract-v2-20261111" / "run-spec.json"
root = spec.parent / "round-one"
env = {**os.environ, "PYTHONPATH": str(project / "python")}

for role in ("sft", "lora"):
    for stage in ("train", "export", "identity", "score"):
        runtime = gpu if stage == "train" else reload
        command = [
            str(project / ".venv" / "bin" / "python"),
            str(project / "scripts" / "run_public_proof.py"), stage,
            "--spec", str(spec), "--root", str(root), "--role", role,
            "--log-dir", str(spec.parent / f"{role}-{stage}-command"),
            "--python", str(runtime),
        ]
        if stage == "train":
            command += ["--automodel-python", str(automodel), "--reload-python", str(reload)]
        subprocess.run(command, env=env, check=True)
```

Changing the run spec requires freezing a new digest, not editing an approved
artifact in place. Fresh log/output directories are mandatory. The new clean
contract does not include a frontier role or authorize new provider calls.

## Acceptance limitations

Focused contract tests exercise registry mismatch/refusal, arbitrary roles and
Gym scorer metrics, public-only augmentation/audits, output/coverage contracts,
serving-export lineage and server identity handoffs. Mocked server/export tests
are contract evidence, distinct from the actual GPU export receipts above.
No training was run under the initial blocked contract. Full model-quality,
private behavior, controls and reproducibility acceptance remain independent.

## Second-Gym modularity acceptance: company facts

The generic contract was frozen **before** implementing
`company_facts_public_plugin.py` and
`protocols/company-facts-public-fixture-v1.json`. Its composition root installs
Gym/provider/evidence/public-audit plugins into `ProofRegistry`, then calls the
unchanged generic `proof_pipeline.score(..., registry=...)` API. It does not
implement another scoring loop, add a core task branch, train a model, contact
a provider or download anything.

The second Gym uses a different input/output schema, chat projection and
`supported_fact` metric. A deterministic synthetic catalog software fixture
exercises lookup, an inert archive directive and an unsupported fact. Its
provider emits per-response digest receipts. Missing private/control/repeat
evidence is labelled unexecuted, not counted as passed. This validates plugin
composition, **not learned company-facts quality or attack resistance**.

Acceptance artifacts:

- `artifacts/phase-b-company-facts-acceptance-20261111/generic-contract-before.json`:
  `sha256:5f1d2009aa54835e4af3caa06fee4788087252ff4eed8ade009d4d5557c5c72b`.
- `software-fixture-command/receipt.json`: bounded generic-runner fixture
  execution, return code 0, with its own status file.
- `fixture-run/modularity-acceptance.json`:
  `sha256:d028889061a0658573a1f40991d851dae5086acefc6bd51d1107f50c1b198bc8`.
  The five generic-core file digests are unchanged before/after; software-fixture
  modularity passes. No generic gap or task-specific core change was needed.
- `focused-tests-command/receipt.json`: three passing focused contracts; the
  test forbids model loading, training and API calls. Lint passes.

`model_quality_accepted`, `final_proof_eligible`, `training_dispatched`,
`holdout_accessed` and `release_authorized` are all false; new API calls are zero.
