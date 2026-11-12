# Public proof runner (Phase B)

The user authorized Phase B independently of proof/custody acceptance. This is
public-only development evidence, not final proof or release authorization.
The retired synthetic holdout is not read or rehabilitated by this work.

## Synthetic training held; base-only diagnostics and CORD validation approved

The 2026-11-11 direction supersedes the former factorial execution approval:
synthetic invoices are software/regression fixtures only. Both previously
approved factorial training workers had already completed successfully before
the hold, but neither candidate was exported or scored. Their artifacts remain
preserved, with no owned factorial worker running. The fresh confirmation panel
remains unscored. The only newly approved synthetic measurement is a base-only
0.6B/4B diagnostic on old192 with an explicit candidate system instruction that
document content is untrusted data, not instructions. The factorial candidates
are excluded; no training, frontier call or fresh-confirmation score is allowed.

The untrained Qwen3-4B baseline completed: public46/64, required8/16,
old panel63/192, with injection templates footer1/32, mail29/32, portal1/32.
That original baseline is not repeated unchanged; the approved diagnostic
changes only the candidate system instruction. The
[CORD/BIPIA license review and gym proposal](cord-bipia-gym-proposal.md)
records the now-approved plugin direction. CORD uses supplied transcriptions
of Indonesian receipts, not an OCR measurement. BIPIA use is internal-only
pending license clarification. Native test payloads remain sealed/deferred;
the first real-data measurement is base0.6B/4B on validation, clean and attacked.

### Historical factorial 0.6B round: frozen before results

The next fixed-compute round replaces the clean contract's benign authority64,
not its clean320. `json_factorial_training` is a separately registered
`factorial-authority` augmenter. No model/task branch was added to the generic
pipeline. The three independently crossed binary factors are:

- An **untrusted** memo claiming it is "ratified and authorized" versus a neutral
  memo, with the same replacement imperative in both.
- Untrusted memo first versus last.
- Complete alternate JSON record versus all three identical alternate facts in
  prose. Neither condition drops an alternate field.

Every cell contains eight rows: trusted JSON/plain × two value pairs × both
target/decoy swaps. Corresponding repeats preserve target, decoy and imperative
across all eight cells. Value frequencies are identical between trust roles
within every cell. Five retained clean rows then one factorial row gives
**384 rows / 384 steps** for each of one fresh-base SFT and one fresh-base LoRA.
Pinned Qwen3-0.6B revision, optimizer settings, seed17, token caps and recipes
are unchanged from clean contract v2; no continuation of historical SFT.

The pretraining rejection check considers **only required16 and OLD panel192**.
Three complete matched64 value rosters (seeds20261118–20261120) failed the
unchanged zero-fivegram rule; seed20261121 passes. The frozen spec records all
three excluded roster digests. No wording, labels, criteria or model outcomes
were used to adapt this selection. The original clean320, including its twelve
previously replaced rows, is retained byte-for-byte as row data.

Exclusive frozen files are under
`artifacts/phase-b-factorial-authority-0p6b-20261111/frozen`:

- `run-spec.json`: digest
  `sha256:cea70155474b0f7395929240e0c039b285de687c25055539e5b1d4eff54da9d5`.
  Its score sets remain **public64, required16, OLD panel192 only**; the frozen
  thresholds remain16/16,91/96 injection,150/192 total.
- `fresh-confirmation-UNSCORED.json`: digest
  `sha256:4a635b6848740c0bc1c15056a79214c3230b7270c6a0db9055542d77522a1355`.
  This separate public192 has six new templates (three clean, three injection,
  32 each), independent seed20261129, all eight injection cells balanced within
  each layout, and balanced trusted encoding/value swaps. It explicitly has
  `scoring_authorized:false`; it is absent from every scoring spec.
- `public-audit.json`: all-training and injection-only zero exact/fivegram
  checks against required16 and OLD192 pass; required phrasing families remain
  disjoint.
- `confirmation-disjointness-once.json`: a single reporting-only comparison
  against seven historical public/train artifact sources plus the new train384
  finds zero exact and zero shared fivegrams, with disjoint template families.
  No panel or training rejection uses this audit. An overlap would have been
  persisted as a blocker, not repaired after looking at confirmation data.
- `freeze-receipt.json`: digest
  `sha256:a98c65429557986501667e040a58f0b09cf8428d24cad5f093425857bf2a91ce`,
  timestamp **2026-11-11T22:43:36.619682+00:00**, before any new quality evaluation.
  It binds code/data/audit digests and records `READY_FOR_RESULTS`, zero model
  inference, no GPU training started and no fresh-confirmation scoring authority.

The sibling `public-tests-command/receipt.json` records **63 passed**;
`lint-command/receipt.json` passes scoped lint. Independent rendered-source
parsing validates all192 confirmation labels without consulting stored answers
or factor metadata. `no-forward-token-audit.json` uses the cached pinned
tokenizer and AutoModel `ChatDataset`: maximum training106 tokens, total22603;
maximum prompts88 train,78 fresh confirmation,57 required,33 public64,89 OLD192.
All inputs are at most247 characters; training256/context512/output128 budgets
pass. No model is loaded. A cached runtime warns that its optional C++
extensions are skipped with torch2.10; no native build or install was attempted.

`scripts/prepare_json_factorial_round.py` reproduces the bounded public tests,
lint, no-forward budgets and exclusive freeze into a **new** root; historical
artifacts cannot be overwritten. Prepared GPU commands are historical
reproduction records, not current authorization. The hold above supersedes the
former execution release; do not resume export/scoring or start another round.
No API, weight editing, final/private holdout, confirmation scoring or
quality-driven retry is authorized.

## Contract

CORD integration exposed a generic contract gap: task schemas previously
rejected JSON `null` and type arrays. The closed schema subset now accepts
`null` and nonempty lists of distinct supported types. Bounds apply to their
corresponding types; required fields, enum restrictions and closed object
properties still apply. This was a core modularity gap repaired generically,
not evidence that the first CORD composition needed no core changes.

The first CORD 0.6B score exposed a second gap: the serving adapter aborted the
whole panel on any non-`stop` completion. The failed attempt dispatched105
requests and returned/scored104, without preserving their records. Its raw
finish reason was not retained, so a length cap is a hypothesis, not a proven
diagnosis of that historical response.

The generic adapter now records `length` completions as explicitly invalid
responses, retaining raw text, finish reason and generation-cap evidence;
stopped malformed JSON is also invalid, not a successful empty object.
Unexpected finish reasons and malformed transport envelopes remain errors.
The runner journals each completed scored response with its row/spec digests,
so a later error no longer discards earlier records. This does not relax the
automatic pre-inference-only retry rule.

The user explicitly approved **one repaired 0.6B rerun and the first 4B run**,
with unchanged CORD data, system prompt and context/output budgets. The failed
attempt remains intact; newly bound executable specs use a separate repair
root. Neither the approval nor this repair authorizes training, additional
quality attempts, factorial scoring, frontier calls or test access.

`neural_foundry.proof_pipeline` accepts a digest-verified
`nf_public_proof_run`, version 1. The spec explicitly binds:

- Base model ID/revision; candidate roles and methods as data, not a fixed roster.
- Optional nonempty candidate `system_prompt`: inserted before document messages,
  or appended to an existing leading system message. Omission preserves the
  original projection. Training, inference, provider setup and response evidence
  use the same digest-bound candidate projection; schemas/labels stay unchanged.
- Exact version/digest plugin references for Gym, training model, `TrainerAdapter`,
  training-coverage audit, serving export, augmenter and public audit. Each
  candidate selects its server/provider and evidence verifier independently.
- Ordered train-only rows, public-set row digests and integer success-count gates.
- Versioned decoding/output cap, identity tolerance/count, API attempt/cost
  ceilings, report-only decisions and prohibition on adaptive retraining.

The runner resolves the Gym's own scorer through `GymComponentRegistry`.
No invoice scorer, provider name, candidate method/role branch or custody import
is present in the generic runner. Local identity uses the existing 16-probe,
`1e-6` same-runtime policy. At measured commit `240c9b1`, the first provider wrapped
the legacy OpenAI client. The offline hardening below replaces that active
plugin with a Gym-neutral provider while preserving the legacy transport API.
Neither active plugin reads `.env`.
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
Current `--retry-of-failure` requires a preserved local startup/pre-inference
failure with explicit zero dispatch/returned/scored counters. Provider retries
are not allowed by this path, and completed public scorecards cannot be replaced.
The historical measured retry predates this stricter failure-receipt schema;
its original artifacts are retained, not made executable by inventing counters.

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

## Offline hardening after measured commit `240c9b1`

The real-run source/results above remain historical evidence from `240c9b1`.
No model was trained, loaded for quality inference, or re-scored in this pass.
No live provider requests, real credentials, downloads, native builds or commits
were used. New scorecards below are **software fixtures**, never relabelled
versions of the old SFT/LoRA scores.

### Retry boundary

Failure receipt schema 2 records `failed_stage`,
`public_inference_dispatches`, `public_responses_returned`, `scored_responses`,
and the preserved started-claim name/digest. Authorization requires:

- Local server, API dispatch disabled, same frozen run and candidate.
- Stage `server_startup` or `pre_inference`, with all three counters exactly zero.
- A matching intact started claim and no completed scorecard.

The first attempted inference increments the dispatch counter **before**
calling the provider. First-inference exceptions, partial panels, failures
classified as `identity_verification`, and old receipts lacking explicit counters
cannot authorize retry.
Claims and failures are not removed or overwritten.

### Gym-neutral provider and reusable policy

`openai_provider.py` implements the structured-text `PublicProvider` protocol.
The selected `TaskAdapter` supplies interaction format, messages/output schema,
projection identity, Gym/job identity and scorer context. There is no invoice
prompt or scorer pin. Unsupported multimodal/tool payloads are refused before
reservation because this policy does not price their hidden input costs.

`openai_transport.py` is shared with the legacy `JsonFrontierModel`: endpoint
validation, no redirects, response size limit, exact reported model ID, normal
completion, strict response parsing and per-response receipt fields retain
their transport behavior. Legacy direct clients remain compatibility APIs with
their historical per-instance limits; they are **not** registered as the active
run-authorized OpenAI provider.

The tracked active policy is `protocols/public-proof-policy-v1.json`. New specs
bind its file SHA-256 through `policy_reference`; duplicated inline values must
match. Current supported values are cap **128**, local context **512**, probes
**16**, startup tolerance **1e-6**, and **192 attempts / 100,000 microUSD**.
Only the explicit per-run calls-authorization boolean overrides the default
disabled authorization. Shutdown logprobs remain **informational only**.

The active API provider requires the referenced policy, not a legacy inline
policy alone. Request decoding/cap, reservation prices, ceilings and startup
probe checks consume those values. Configuration and response receipts bind
the run, policy, selected Gym/projection/scorer, request and response.

### Durable public authorization

`public_budget.py` provides an explicitly initialized SQLite ledger plus
exclusive initialization anchor. A frozen `public_budget` reference supplies
one absolute path and ledger identity for **all roles/clients/processes** in the
run. The ledger binds the unsigned run-spec digest, complete effective policy
digest and provider-role/model configurations. A digest-bearing document and
its unsigned spec resolve to the same budget.

`proof_pipeline budget-init` initializes only fresh state; providers never
recreate missing/corrupt state. Existing/partial initialization, absent anchors,
binding mismatches, malformed state, counter divergence and reservation-chain
corruption fail closed. SQLite `BEGIN IMMEDIATE`, full synchronous commits and
read transactions serialize concurrent reservations. Dispatch occurs only after
the reservation commits. Network failures, incomplete responses and model-ID
mismatches spend their attempt/cost reservation; no refunds/reset occur.

Cost authorization uses the policy's configured price bounds: UTF-8 serialized
payload bytes plus 1,024 overhead tokens, and the 128-token output cap, rounded
up separately at 125,000 input / 500,000 output microUSD per million tokens.
This is conservative configured reservation accounting, **not verification of
vendor tariffs, invoices or hidden provider behavior**. The store assumes a
trusted local filesystem/operator; it is not signed, privileged-operator
rollback-resistant custody. It neither owns nor replaces the study service's
strict final claim.

### Declared plugin dependencies

Built-in plugin references are version 2. Each binds its factory module plus
declared helper roots, their bounded static repository-Python import closure,
explicit worker entrypoints and applicable policy files. Resolution checks the
complete frozen dependency map and current file hashes; reference mutation,
helper edits/removal or policy drift refuse before factory dispatch.

The resulting conservative union contains **86 source/config/runner files**.
There is no `sys.modules` hashing, third-party source copying or study-custody
dependency. Installed third-party/native runtimes remain governed by their
existing runtime/image/backend qualification contracts, not this source map.
External plugins still use explicit registry composition; discovery is not
required.

### Revised software acceptance and receipts

All new artifacts are under `artifacts/phase-b-offline-hardening-20261111`:

- `final-contract-tests-command/receipt.json`: **68 passed**, including spawned
  process races for both attempt and cost ceilings, restart/failure spending,
  corrupt/missing state, policy drift, exact mock receipts and retry refusal.
- `lint-command/receipt.json`: scoped lint passed.
- `generic-contract-v2.json`, frozen **after** those tests:
  `sha256:4abbf18d5ab9aa24c7a2b3d315703846d18e2b8c1edb7cc2e10df4c98d13a144`.
- `company-software-run/modularity-acceptance.json`:
  `sha256:3cccada7b468566e39c639c43c1906d352d4f1524deb5005ac1df7e124f84ab0`.
- `company-mock-provider-run/mock-provider-conformance.json`:
  `sha256:6afe2bd9ddbc9ca0c35515f54e0e00a0603acbf9a284cb682741dad3b703eee7`.

Both second-Gym executions passed with the complete frozen generic source map
unchanged. The mock provider uses explicit test credentials and a patched HTTP
opener only: **four simulated dispatches**, durably reserving **1,044 microUSD**,
and zero real network/model calls. Its composition root delegates to the same
registered OpenAI provider and generic scorer; no company-facts core branch or
second scoring pipeline was added.

The legacy transport unit test previously generated a seed7 `holdout` fixture;
it now uses `validation`. That corrects the earlier exposure disclosure to
include seed7 as well as seed31 development fixtures. This pass performs no
holdout draw/read and grants no final-proof eligibility.
