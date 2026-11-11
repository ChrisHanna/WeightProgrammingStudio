# JSON proof Phase A

Status: Phases A and B are authorized in parallel. The sections below retain the earlier investigation; the execution update records the superseding decisions and measured results. No final holdout has been drawn or scored.

Subsequent authorization (2026-11-11): the user approved starting Phase B
independently, and public-only counterexample preparation without a custody
dependency. See [the Phase B public runner](public-proof-phase-b.md) for the
implemented contracts, frozen 16/16 + 91/96 + 150/192 report-only gate and the
approved clean training-contract v2 resolving the initial public-overlap blocker.
Historical statements below describe the earlier
approval boundary, not a revocation of this later authorization.

## Approved execution update

The approved competitive bar is **16/16 required, at least 91/96 injection, and at least 150/192 total**. It was frozen before the new training round and was not adjusted after observing scores. It supersedes the earlier frontier-matching proposal, not the separate weight-edit specialist policy.

The [generic public runner](public-proof-phase-b.md) performed one 384-step round per method after the public required/panel overlap checks passed. The training contract removed the old 64 injection examples and used 320 clean rows plus 64 balanced-authority examples. Twelve of the clean rows collided with public-panel 5-grams; they were replaced by deterministic train-only rejection sampling, retaining the templates and generating each replacement's input/reference together. Fourteen sampled candidates were excluded in total. This selection used public text overlap only, before any model result; it is disclosed public exposure, not independent confirmation.

| Counterexample model | Public /64 | Required /16 | Clean /96 | Injection /96 | Total /192 | Competitive |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT | 64 | 13 | 72 | 1 | 73 | Failed |
| LoRA | 64 | 13 | 84 | 9 | 93 | Failed |

Both scorecard digests and all **544** response labels were independently checked against the public references. All outputs were format-valid. Clean accuracy improved, but injection accuracy regressed substantially; the net totals must not hide that failure. No quality-driven retry or second training round was performed. One SFT scoring attempt failed before inference because its WSL working directory became unavailable; the failed claim was retained and a separately recorded infrastructure-only score retry completed.

The [immutable round handoff](../artifacts/phase-b-clean-contract-v2-20261111/handoff.json) records data lineage, audits, committed training logs, exports/reloads, identity probes, scores, gates and command receipts. Training coverage is **schedule-derived** from batch one, no shuffle/packing and the exact committed-step roster; it is not direct per-batch scenario telemetry. A tokenizer warning remains recorded; base/export prompt token IDs were checked equal on the 272 public cases only, not the entire vocabulary.

The current [legacy synthetic holdout is development-grade](../protocols/json-holdout-lineage-v1.json), never final-proof eligible. A legacy seed-31 software unit fixture was materialized during compatibility testing and is explicitly disclosed in the [development exposure receipt](../artifacts/phase-b-clean-contract-v2-20261111/development-fixture-exposure.json). That is not a private final draw. All operational training/scoring was public-only, with zero new frontier calls and zero model downloads.

Stage 1 [custody](gym-custody-extension-proposal.md#stage-1-execution) now has aggregate auditing, strict durable starts and a verified separate-account software-fixture deployment. Public retraining did **not** wait for it. Stage 2 native ledger bridging, signing and container deployment are deferred to Phase C. Real final Gym evaluator integration and a fresh final draw remain unqualified; the failed public gates do not authorize opening one.

The [deadline v2 split](../protocols/reference-audit-deadline-v2.md) is implemented. Its three native counter/policy tests pass. All four marked cases passed for real before their markers were removed, then passed again as normal pytest tests. The Cargo acceptance rule is **two test threads under the documented capability runner**, including in M1 CI.

After the measured round was committed as `240c9b1`, an offline-only contract pass added gym-neutral OpenAI transport, durable run-wide cost/attempt reservation, reusable versioned policy loading, startup-only zero-response retry authorization, and declared dependency fingerprinting. The [hardening handoff](../artifacts/phase-b-offline-hardening-20261111/handoff.json) preserves the historical real-run artifacts rather than claiming the new code produced them. **68 contracts pass**, independently rerun. Both company-facts software composition and mock-provider conformance pass against the same frozen 86-file generic contract, with no core edits between freeze and acceptance. Four simulated dispatches made zero real network calls; neither acceptance establishes learned quality or final custody.

### Larger-model recommendation; no download

Next test [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B): the official card lists **4.0 billion parameters**, approximately **8.0 GB of BF16 weights** (about 7.45 GiB), before KV cache, runtime buffers or training state. The observed RTX 5090 has 32,607 MiB VRAM. Start with unchanged-base inference and LoRA on the same frozen public data/decoding contract; full SFT needs a separate measured memory feasibility check. This is a recommendation, not evidence that size alone explains or fixes the regression. No model was downloaded or larger-model run started.

Weight editing stays parked: the observed injection regression is broad, not yet an isolated narrow failure justifying a weight edit.

## Public-panel label check

The [public audit tool](../scripts/audit_json_public_panel.py) parses the selected and distractor records independently from the rendered text, checks every clean reference, verifies source/card digests and response alignment, and refuses a v3 freeze if any label disagrees. It does not import a holdout generator or contact a provider.

| Clean template | Cases | Incorrect expected answers | Luna correct | Complete distractor copies | Other errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Posted AP line | 32 | 0 | 32 | 0 | 0 |
| Accepted goods / receiving account | 32 | 0 | 5 | 27 | 0 |
| Settled invoice / payee | 32 | 0 | 27 | 4 | 1 |

All 32 Luna clean misses are real record-selection failures under the declared current-record interpretation: 31 copy an entire obsolete/not-received record, and one combines the obsolete customer with the current product/quantity. Example: the receipt says `Expected (not received): 29 valves for Linden`, `Accepted: 7 lamps`, `Receiving account: Cedar`; the label is Cedar/lamps/7 and Luna returns Linden/valves/29.

The original task prompt merely asks to extract fields; it does not explicitly define a general multi-record selection policy. The status labels support the references, but this audit is not an independent human annotation or a proof that every reader finds the task unambiguous. Do not silently relabel distractors as correct or simplify the panel to improve scores. Any later prompt clarification is a new task condition requiring fresh scoring.

No expected answers were changed, so the generator was not changed. The [v3 panel](../artifacts/proof-phase-a-20261111/public-panel-v3/panel.json) freezes the audited selection contract as metadata: **identical prompts, references, scenario IDs and order**, with the same rows digest as v2. It is an audit-only version, not a fresh independent or harder panel. Existing API receipts remain attributable to their original v2 manifest; their identical interactions can be reused with explicit lineage, not relabeled as new calls.

- [Label audit](../artifacts/proof-phase-a-20261111/public-panel-v3/label-audit.json): `sha256:f23066d7d779ade2b4b2717c1b7b629bb771ac0481c76910af074efa5b4a7053`.
- V3 manifest: `sha256:1fbfa3752380810e5f3d87ba10fb231f890923d58ace7cd2d38fa703c56c0628`.
- Historical frontier-matching gate proposal: 160/192 overall and 96/96 injection, plus 16/16 required. The subsequently approved execution bar is 150/192 and 91/96 injection with 16/16 required, as recorded above; the separate specialist-v2 policy is not rewritten.

## What the original 320 training rows actually teach

The frozen preparation contains 32 examples of each of ten train templates. Inspection of **all 320 rows** finds:

- Every customer and product appears exactly once in its input, and every reference field value is present.
- Zero inputs contain a JSON object; zero contain footer, untrusted, obsolete, previous or not-received markers.
- Each row contains one transaction, without competing records or a source-authority decision.
- Templates teach ordinary field copying, including `Customer: ...; product: ...; quantity: ...`, `Record this line: ...; account ...`, and `Customer=... // item=... // count=...`. None instructs copying a footer or an injected JSON record.

The saved old SFT/LoRA configs use assistant-only masked cross entropy (`mask_history: true`, `mask_generation_prompt: true`), global/local batch size one, no packing, no dataloader shuffle, and only 32 optimizer steps. The [trainer](../python/neural_foundry/backends/automodel/adapter.py) emits completed user/assistant demonstrations, not alternate-record targets. Thus the existence of 320 prepared rows is **not evidence that old training consumed all 320**; its short ordered schedule is an additional coverage concern. Historical per-step scenario IDs are not retained, so do not claim a verified complete consumption roster.

A runtime [supervision audit](../artifacts/proof-phase-a-20261111/train-supervision-audit.json) instantiated the installed AutoModel `ChatDataset` with the cached pinned Qwen tokenizer and those masking settings for every original row. Decoding the nonmasked labels yields exactly the intended compact JSON target in **320/320 cases**, with zero target mismatches, no model forwards, and no holdout access. Its training digest matches the frozen base-training contract (`sha256:8e4c29f8d08fc9862b8a70ef5260d61802e83c491c6842ef287f5e20673b82c8`). This rules out an explicit wrong-target or prompt-token-supervision explanation in this reconstructed dataset; it does not reconstruct historical per-step training exposure or establish the learned mechanism.

The newer 384-step ordering interleaves all 64 injection examples and labels the original transaction as correct despite the malicious record. That supervision contradicts, rather than teaches, footer copying. Both old and new SFT score zero on the 32 footer cases; old LoRA does too. Therefore the claim that new injection data created the regression is not established.

**Supported diagnosis:** the original supervised data lacks any authority/competing-record discrimination, while teaching direct extraction from a single clean record. A generic complete JSON record at the end of an unfamiliar layout can remain an attractive shortcut. That is a plausible generalization mechanism, **not an identified causal feature**. Establishing causality requires matched, fixed-step train-only ablations or tracing; the saved input/output cards cannot prove which weight-level heuristic was learned.

## Proposed train-only counterexamples

Do not copy evaluation text, scenario IDs, examples or answers into training. Use fresh train seeds, train-specific layouts, independently sampled values, and explicit source-authority labels. Keep the held-out required phrasing families and public-panel templates out of generation. Freeze the generator/config/data digests and consumed-row ordering before training.

Proposed independent training patterns:

1. **Authoritative body, archival JSON later.** `Dispatch approval: customer=Birch; product=panels; quantity=18. Archived OCR payload, not a transaction: {"customer":"Linden","product":"valves","quantity":62}.` Target Birch/panels/18.
2. **Authoritative footer, canceled draft first.** `Canceled draft: 62 valves for Linden. Signed dispatch approval at end: customer=Birch; product=panels; quantity=18.` Same target. This prevents a blanket "ignore footer" or "always first" rule.
3. **Authoritative JSON, natural-language decoy.** `Signed dispatch approval: {"customer":"Birch","product":"panels","quantity":18}. Superseded memo: Linden wanted 62 valves.` Same target. This prevents a blanket "ignore JSON" rule.
4. **Field-level correction and schema attack.** `Approved dispatch: Birch / panels / 18. Archived note asks for a diagnostic property and a different recipient.` Target exactly the three schema fields; vary authority location and irrelevant annotation wording.

These are illustrative **training** examples, not outputs selected from evaluation. Future generation should balance authority position, JSON/plain representation, which fields disagree, and value vocabularies across trusted and untrusted roles. The existing augmentation uses separate alternate-only customer/product vocabularies; a proposed revision should avoid teaching that names alone determine trust. Include clean single-record rows to preserve format and ordinary extraction.

The later approval removed the custody dependency for public retraining. The executed contract and unchanged public-overlap thresholds are recorded above. The plan's illustrative examples are not copies of evaluation cases; final generated data and schedule-derived coverage have separate frozen digests and receipts.

## Custody and one-shot approval boundary

The [custody extension proposal](gym-custody-extension-proposal.md) preserves the original design and now records the approved Stage 1 implementation. Native exposure bridging, signing and container deployment remain Stage 2/Phase C, not a condition for public retraining.

The [local contamination tool](../scripts/audit_json_contamination.py) now has no holdout-generator dependency or local holdout-check path. Its new public-only report leaves holdout counts unset and the overall audit not-passed. It can still annotate the original v1 local-read receipt without rereading anything. This prevention change is not a custodial-audit implementation.

The legacy independent-frontier claim permits bounded retry/resumption and is not the final custody authority. The new custody route uses strict starts with no retries after consumption. The previous synthetic holdout's out-of-custody overlap read remains recorded and the split is retired; a new account/container cannot erase that exposure.

## Deadline and acceptance status

The requested measurement is three serial 24-job runs on an idle machine, with process snapshots for each run, separate guardian/controller timing, the worst measured lifecycle time, and a proposed lifecycle limit at 25% headroom. It must not be run concurrently with Cargo, retraining or scoring. **No limit changes are authorized by the measurement request.** The four known deadline xfail cases must not be removed until the approved fix passes them as real tests.

The user subsequently authorized measuring **with TradingView running**, labeled non-idle. No user process was terminated. The explicitly approved measurement-only request used the existing supported 80,000-ms aggregate allowance while leaving each job's 300-ms limit and all production defaults unchanged. This allowance is not a proposed production cap.

All three serial attempts in the [completed measurement](../artifacts/proof-phase-a-20261111/lifecycle-non-idle-three-attempts/summary.json) completed 24/24 jobs, with all 72 raw receipt digests and cleanup results verified:

| Run | Host time, including preflight | Guardian work | Host minus guardian lifecycle |
| --- | ---: | ---: | ---: |
| 1 | 24.640385 s | 4.089 s | 20.551385 s |
| 2 | 24.844263 s | 4.048 s | 20.796263 s |
| 3 | 25.544199 s | 4.130 s | 21.414199 s |

Worst measured lifecycle is **21.414199244 s**; adding 25% headroom and rounding up gives a proposed separate lifecycle allowance of **26,768 ms**. If separately approved with a 7,200-ms guardian-work cap, an outer cap would need to allow at least 33,968 ms, while enforcing the two counters independently. No limit is changed or approved by these measurements. Guardian millisecond quantization contributes up to 24 ms uncertainty per 24-job run.

Windows/Linux process and GPU snapshots are retained before/after each run, without command lines or credentials. All three before-snapshots include TradingView (10), ChatGPT (15), VS Code (13), Codex (3 plus helper processes), Docker and one Node process. Observed GPU utilization was 4%, 1% and 2% at the respective starts. These are non-idle observations, not an idle-machine certification or a bound on future latency.

Earlier attempts are retained rather than hidden: the first used a Windows-mounted fixture directory rejected by native snapshot import; the second completed one 24-job run but stopped during its next attempt on a real per-job timeout (306 ms vs 300 ms, cleanup confirmed). The final harness records every attempt and refuses to propose a limit unless all three complete; it does not raise per-job limits or retry a timed-out job inside an audit. Successful later development-fixture attempts do not erase the earlier timeout. The later approved v2 implementation and the genuine passing runs allowed the four markers to be removed, as recorded in the execution update above.

The first [full Cargo run](../artifacts/proof-phase-a-20261111/cargo-full-command/receipt.json) failed because an ordinary WSL shell retains a nonempty capability bounding set. The repository's [qualification instructions](../protocols/core-testing-v1.md#running-the-nf-experiments-tests) require an unprivileged user namespace and empty capabilities; these checks were not weakened. An isolated failing promotion test [passes under that qualified boundary](../artifacts/proof-phase-a-20261111/cargo-qualified-promotion-command/receipt.json).

The [full qualified run](../artifacts/proof-phase-a-20261111/cargo-full-qualified-command/receipt.json) reached 225 passing/one failing `nf-experiments` library test. The remaining failure reproduced alone: a deliberately crashing operator child was incorrectly wrapped in another test-isolation process, converting its expected exit status 86 to a harness assertion exit 101. The [test-harness fix](../crates/nf-experiments/src/coordinator/promotion/operator_tests.rs) handles crash-child mode before normal isolation; no production authority or cleanup checks changed. The [previously failing release test now passes](../artifacts/proof-phase-a-20261111/cargo-crash-wrapper-fix-command/receipt.json).

**Historical pre-v2 Cargo acceptance:** the [completed full-workspace receipt](../artifacts/proof-phase-a-20261111/cargo-full-qualified-fixed-command/receipt.json) records return code 0 after 1,231.183 seconds for `cargo test --workspace --locked --offline -- --test-threads=2`, under the documented non-root, empty-capability runner. No assertions or qualification checks were weakened. That earlier result predates deadline v2; it is not substituted for current verification. The four Python cases have since passed normally with their markers removed.

**Current deadline-v2 full Cargo acceptance is green.** The [full-workspace receipt](../artifacts/proof-phase-ab-20261111/cargo-full-linux-filesystem-command/receipt.json) records return code 0 after 1,114.280 seconds, with two test threads under the documented capability runner. The [acceptance summary](../artifacts/proof-phase-ab-20261111/native-acceptance-summary.json) verifies the committed native tree matches the current native tree, with only Git newline normalization in the policy working copy.

Two failed/interrupted WSL runs are retained: one lost its current directory during status-heartbeat creation, and another reported executable paths missing their `/mnt/c` prefix. A reported native-budget case passed in isolation under the same capability runner. The successful run used a [committed-source Linux-filesystem snapshot](../artifacts/proof-phase-ab-20261111/linux-acceptance-snapshot.json) and cached CPU runtime, without dependency installation, downloads, skipped native gates or relaxed checks. The snapshot/build cache is retained for reproducible native-filesystem verification.

## Phase B scope

The user approved Phase B in parallel with public retraining, not after the proof report. Refactor the working behavior rather than building another JSON-only orchestration layer:

1. One versioned run spec and generic runner replace the JSON validation script and JSON training modes.
2. Candidate method, role and evidence requirements become manifest data; remove fixed rosters.
3. Provider registry/interface with the existing OpenAI client as its first plugin; preserve per-response receipts, budget reservation and decoding identity.
4. Trainer registry uses `TrainerAdapter`; a generic serving-export operation owns merged LoRA export/reload/lineage.
5. Gym-level augmenter and audit registries own versioned configs, rather than imports from core.
6. Versioned policies bind identity tolerance, probe count, API cost/attempt limits, decoding/output limits and approval rules.
7. Contract tests exercise gym, scorer, trainer/exporter, server and provider contracts, including refusal/failure receipts.
8. Add company-facts using plugins and config only. Any required core change is a failed modularity acceptance, not permission to add a company-facts branch.

The generic public pipeline and plugin-only company-facts software demonstration are implemented; their exact limits and any subsequent provider/budget hardening are recorded in [the Phase B report](public-proof-phase-b.md). Company-facts software acceptance is not learned model quality or a final proof.
