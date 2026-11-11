# JSON proof Phase A

Status: in progress; no holdout scoring, custody implementation, deadline change or Phase B implementation authorized by this report.

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
- Frozen competitive gate approved by the user: at least 160/192 overall and 96/96 injection on this identical-input panel, plus 16/16 required-set success. The existing specialist-v2 gates remain independently applicable; this does not rewrite that policy.

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

Before retraining: approve the actual new data contract and a fixed-step comparison roster; audit it against public required rows under the [approved contamination policy](../protocols/json-contamination-v1.json), then obtain the custody overlap receipt through the approved extension. Record per-step scenario coverage rather than inferring it from available dataset size. No new examples have been generated into an executable training preparation by this proposal.

## Custody and one-shot approval boundary

The [custody extension proposal](gym-custody-extension-proposal.md) describes the existing protocol, generic registry manifest, pretraining `audit` operation, native exposure bridge, vLLM/API evaluation and one authoritative strict-one-shot claim. It is **interface design only**, as requested. No custodian code or holdout material was touched.

The [local contamination tool](../scripts/audit_json_contamination.py) now has no holdout-generator dependency or local holdout-check path. Its new public-only report leaves holdout counts unset and the overall audit not-passed. It can still annotate the original v1 local-read receipt without rereading anything. This prevention change is not a custodial-audit implementation.

The current frontier claim permits bounded retry/resumption; it is not equivalent to the custodian's permanently consumed started query. Unification and crash-safety implementation wait for the design approval. The previous synthetic holdout's out-of-custody overlap read remains recorded; a new account/container cannot erase that exposure.

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

Earlier attempts are retained rather than hidden: the first used a Windows-mounted fixture directory rejected by native snapshot import; the second completed one 24-job run but stopped during its next attempt on a real per-job timeout (306 ms vs 300 ms, cleanup confirmed). The final harness records every attempt and refuses to propose a limit unless all three complete; it does not raise per-job limits or retry a timed-out job inside an audit. Successful later development-fixture attempts do not erase the earlier timeout. Production qualification and removal of the four xfail markers remain blocked on the approved deadline implementation and real passing tests.

The first [full Cargo run](../artifacts/proof-phase-a-20261111/cargo-full-command/receipt.json) failed because an ordinary WSL shell retains a nonempty capability bounding set. The repository's [qualification instructions](../protocols/core-testing-v1.md#running-the-nf-experiments-tests) require an unprivileged user namespace and empty capabilities; these checks were not weakened. An isolated failing promotion test [passes under that qualified boundary](../artifacts/proof-phase-a-20261111/cargo-qualified-promotion-command/receipt.json).

The [full qualified run](../artifacts/proof-phase-a-20261111/cargo-full-qualified-command/receipt.json) reached 225 passing/one failing `nf-experiments` library test. The remaining failure reproduced alone: a deliberately crashing operator child was incorrectly wrapped in another test-isolation process, converting its expected exit status 86 to a harness assertion exit 101. The [test-harness fix](../crates/nf-experiments/src/coordinator/promotion/operator_tests.rs) handles crash-child mode before normal isolation; no production authority or cleanup checks changed. The [previously failing release test now passes](../artifacts/proof-phase-a-20261111/cargo-crash-wrapper-fix-command/receipt.json). A full qualified rerun follows separately; no green full-suite result is claimed until it finishes.

## Phase B boundary

After Phase A proof gates and custody approval, refactor the working behavior, not a second JSON-specific pipeline:

1. One versioned run spec and generic runner replace the JSON validation script and JSON training modes.
2. Candidate method, role and evidence requirements become manifest data; remove fixed rosters.
3. Provider registry/interface with the existing OpenAI client as its first plugin; preserve per-response receipts, budget reservation and decoding identity.
4. Trainer registry uses `TrainerAdapter`; a generic serving-export operation owns merged LoRA export/reload/lineage.
5. Gym-level augmenter and audit registries own versioned configs, rather than imports from core.
6. Versioned policies bind identity tolerance, probe count, API cost/attempt limits, decoding/output limits and approval rules.
7. Contract tests exercise gym, scorer, trainer/exporter, server and provider contracts, including refusal/failure receipts.
8. Add company-facts using plugins and config only. Any required core change is a failed modularity acceptance, not permission to add a company-facts branch.

No Phase B implementation has begun while Phase A and custody design remain open.
