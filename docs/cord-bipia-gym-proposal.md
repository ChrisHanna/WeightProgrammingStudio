# CORD + BIPIA: license review and real-receipt gym proposal

**Subsequent decision:** the user approved the CORD plugin build/acquisition and
base-only validation measurements. BIPIA is approved for **internal evaluation
only**: no public performance claims, marketing use or product distribution
until the research-use wording is clarified. Training remains unauthorized.
The zero-download statements below describe the initial review, not the later
approved acquisition. Implementation/provenance is documented in
[the plugin documentation](cord-receipt-gym.md).

**Checked 2026-11-11. Proposal only.** No CORD archive, Parquet shard, receipt image or BIPIA attack JSON has been downloaded. Research retrieved license/readme pages and file metadata only. The official Hugging Face dataset-card page displayed its automatic **training** preview; those visible examples are not treated as unseen. No test viewer was opened.

## 1. Immediate execution boundary

- Synthetic invoices are now a **software/regression fixture**, not the next model-quality training program.
- The formerly approved factorial SFT and LoRA training stages had **already completed before the hold**, with successful worker receipts. Their artifacts are retained. Neither model has been exported, served or scored; no owned factorial training process remains. No further synthetic stage will start without a new instruction.
- The fresh synthetic confirmation panel remains frozen and **unscored**. It is not a real-data confirmation set.
- The untrained-by-this-project Qwen3-4B baseline already completed through the generic runner, so it is not repeated:

| Model | Public validation /64 | Required /16 | Footer /32 | Forwarded mail /32 | Portal /32 | Old panel /192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3-4B, new pinned run | 46 | 8 | 1 | 29 | 1 | 63 |

All **65 failing 4B injection cases copy the complete injected record**. The larger checkpoint does not, by itself, repair footer/portal behavior under the existing prompt and guided decoder. This does not distinguish training-data design from task framing or establish a causal effect of parameter count.

Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
The [download record](../artifacts/factorial-round-20261111/qwen3-4b-download.json) contains all file SHA-256 values: 8,060,925,056 bytes total, including 8,044,982,000 bytes of BF16 weight shards. The [completed base scorecard](../artifacts/factorial-round-20261111/base4b/public-score-base4b.json) has digest `sha256:9fc95a303acd84beb33e4a7c512ee14dd33a49a90e859e2b4f27b71753ebc9ee`; all 272 saved labels and input digests were independently verified. No 4B training occurred.

## 2. License findings

These are findings from the published terms, **not blanket legal clearance** for a commercial product.

### CORD: commercial reuse permitted by CC BY 4.0

The [official CORD repository](https://github.com/clovaai/cord/blob/327310ce58c1623255821d062b3a759ff3789e3c/README.md) and its [LICENSE-CC-BY](https://github.com/clovaai/cord/blob/327310ce58c1623255821d062b3a759ff3789e3c/LICENSE-CC-BY) state **Creative Commons Attribution 4.0 International**. The publisher-linked CORD-v2 [dataset metadata](https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/7f0115a4b758a71d6473b8d085751692da2fef98/dataset_infos.json) independently gives the same license.

**Yes, CC BY 4.0 permits commercial use and adaptation.** It is not CC BY-NC. The [official CC deed](https://creativecommons.org/licenses/by/4.0/) makes commercial permission explicit.

For distribution/sharing, retain appropriate attribution, source and license links, notices supplied by the licensor, and identify modifications such as transcription normalization, redaction and attack insertion. Do not imply endorsement or impose prohibited additional restrictions on recipients' licensed material. Keep derived-corpus rights separate from our application-code license.

Privacy, publicity, trademarks and other third-party rights are not comprehensively licensed or warranted by CC BY. CORD's README says some target classes, including `store_info`, `payment_info` and `etc`, were removed because of Indonesian legal issues. Do not reconstruct those as new targets or assume all text visible in images is cleared for public logging.

CORD-v2 is the **same 1,000-receipt release with corrected labels and hierarchy information**, not an expanded synthetic dataset. The much larger original collection described in the paper is not the downloadable 1,000-item package proposed here.

### BIPIA: MIT for code and non-excepted attack material; not one license for every context

The [pinned repository LICENSE](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/LICENSE) is **MIT** and explicitly applies to repository parts except its listed dataset components.

**The MIT grant permits commercial use of code and the non-excepted attack-template files**, subject to retaining copyright/license notices. The four attack JSON files are not identified as one of the enumerated third-party exceptions. This is the published scope, not an independent rights audit of every payload.

The exceptions matter:

| Material | Published terms / limitation | Proposed use |
| --- | --- | --- |
| BIPIA-authored code and text attack files | MIT grant, retain notices | Text attack subset only, subject to approval below |
| WikiTableQuestions contexts | CC BY-SA 4.0 | Excluded |
| Stack Exchange/Stack Overflow contexts | CC BY-SA 4.0 with source/author attribution | Excluded |
| OpenAI Evals invoice contexts | MIT per BIPIA's notice; verify upstream if used | Excluded |
| WebQA / Summarization contexts | Must obtain NewsQA/XSum separately and follow their terms | Excluded |

CC BY-SA is not a non-commercial license, but its attribution/share-alike obligations require their own analysis. We avoid that issue rather than treating all BIPIA context data as MIT. See the [benchmark README](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/README.md) and [NOTICE](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/NOTICE.md).

**Production-use caution:** BIPIA's [main README](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/README.md) expresses an intended research-only use of the code. It also says its documentation is informational and does not supersede applicable license terms. The permissive MIT grant and that intended-use statement should both be disclosed; obtain legal/maintainer clarification for a commercial production use rather than claiming unconditional clearance. Training/evaluation provenance and any embedded third-party material still need review after an approved acquisition.

The repository is archived. Pin the reviewed revision rather than assume future maintenance or install the complete upstream project and its dependencies.

## 3. Verified download sizes

Decimal MB/GB below mean powers of 1,000. These are repository file-size metadata, not measurements from downloaded payloads.

### CORD-v2: 2,307,284,272 bytes (2.307 GB / approximately 2.149 GiB)

Pinned dataset revision: `7f0115a4b758a71d6473b8d085751692da2fef98`.

| Native split | Receipts | Parquet shards | Download bytes |
| --- | ---: | ---: | ---: |
| Train | 800 | 4 | 1,831,000,677 |
| Validation | 100 | 1 | 242,080,800 |
| Test | 100 | 1 | 234,202,795 |
| **Total** | **1,000** | **6** | **2,307,284,272** |

The [pinned file tree](https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/tree/7f0115a4b758a71d6473b8d085751692da2fef98?recursive=true&expand=false) exposes byte sizes and LFS SHA-256 identities. The publisher's `dataset_infos.json` reports the same `download_size`, about 2.318 GB logical dataset size, and about 4.626 GB combined download/cache size. Reserve at least about 5 GB before extracted-image, OCR, tokenization and run-artifact overhead; actual materialization depends on the loader.

These Parquet files include images and ground truth. A public text-only official package/size has **not** been verified. Do not quote the annotation-only size as the full download. The original Google Drive v0 archive size is unverified and is not the proposed acquisition.

### BIPIA: receipt-relevant text attack files total 12,321 bytes

Pinned repository revision: `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`.

| File | Bytes |
| --- | ---: |
| `benchmark/text_attack_train.json` | 5,999 |
| `benchmark/text_attack_test.json` | 6,322 |
| **Proposed text-only attack payload total** | **12,321** |
| `benchmark/code_attack_train.json` (not needed) | 16,343 |
| `benchmark/code_attack_test.json` (not needed) | 16,356 |
| All four attack files | 45,020 |

The [pinned Git tree](https://api.github.com/repos/microsoft/BIPIA/git/trees/a004b69ec0dd446e0afd461d98cb5e96e120a5d0?recursive=1) reports 2,176,959 bytes of tracked `benchmark/` blobs, and 2,463,701 bytes of tracked blobs for the whole repository. These sums are **not ZIP transfer sizes**, do not include Git history/dependencies, and do not include externally sourced NewsQA/XSum corpora.

The minimum proposed acquisition is CORD-v2 plus the two text attack files and relevant license/notice/readme material, not the complete BIPIA task corpora or model/judge packages. Per-file metadata is retained in the [size/source record](../artifacts/cord-bipia-license-review-20261111/metadata.json). No data payload was fetched to obtain these counts.

## 4. Proposed gym: `cord_text_bipia_v1`

**A new plugin and config, not an implemented gym or an official CORD/BIPIA leaderboard submission.** CORD supplies real receipts. BIPIA supplies benchmark-authored adversarial instructions; the resulting overlays remain a constructed attack evaluation, not captured real-world attacks.

Use the existing generic runner and trainer/server/provider contracts. Application code and dependency defaults are not changed by this proposal.

| Component | Proposed registry plugin / responsibility |
| --- | --- |
| Dataset source | `cord_v2_source`: pinned local shards, upstream IDs, native split, byte checksums and license notices |
| Text projection | `cord_receipt_text`: deterministic line/word ordering, text only, no target labels in prompts |
| Gym/scorer | `cord_fields_v1`: frozen JSON schema, exact structured match and field/item metrics |
| Attack augmenter | `bipia_text_overlay`: pinned attack IDs/text, controlled insertion and explicit lineage |
| Audit | Source/split/near-duplicate/attack-family disclosure and license checks; no automatic split repair |
| Trainer/export/server | Existing registered SFT, LoRA, serving export and pinned local server |
| Provider | Local models by default; no external provider transfer without separate authorization |

Configs select registry names, versions, implementation/config digests and local manifests. Do not allow arbitrary import/command strings from receipt content. If integration needs a core business-logic branch for CORD, record a modularity failure and fix the generic contract before claiming plugin-only acceptance.

## 5. Receipt text without leaking ground truth

There are two distinct input tracks; do not conflate them:

1. **Provided-transcription pilot:** take text/coordinates from the supplied annotation transcription, strip category labels, `is_key`, semantic group/subgroup IDs and `gt_parse` from the model input. Do not filter text using the target categories. Preserve original OCR-style strings, line structure and case. This tests post-transcription parsing, **not OCR accuracy**. Verify on train whether `valid_line` contains only curated/labelled regions; if so, disclose the oracle-selection limitation and do not claim full-document noise coverage.
2. **Local-OCR text track:** run an independently selected, pinned local OCR plugin on the images, without label-guided region selection. This is the preferable test of production noise, but its software/model acquisition, privacy handling and resource budget require separate approval. No OCR engine, weights or dataset is installed/downloaded now.

Image/store/address privacy needs review before public logging. Do not restore the legally removed target classes through OCR. Persist raw-to-normalized provenance privately and use the same frozen preprocessing on every split/model.

## 6. JSON output and deterministic scoring

Do **not** reuse synthetic invoice `customer/product/quantity`, the quantity bound 1-99, or a single-record assumption. A pilot schema would contain:

```text
line_items: array of {
  name: string,
  quantity_text: string | null,
  unit_price_text: string | null,
  line_total_text: string | null,
  parent_item_index: integer | null
}
totals: {
  subtotal_text: string | null,
  discount_text: string | null,
  tax_text: string | null,
  service_text: string | null,
  grand_total_text: string | null
}
```

Bind these fields to explicit released `gt_parse` paths, including menu/submenu hierarchy. Normalize dict-versus-list forms to a consistent array, preserve void/cancelled status where relevant, and record which released fields are outside the pilot scope. Missing annotated fields become null, never guessed values. `additionalProperties` is false in the eventual schema.

Freeze Unicode/whitespace and locale-aware amount rules using train data only. Indonesian price strings and quantity strings such as `1 x` must not be blindly coerced into the synthetic task's integers. Report structured exact match, field precision/recall/F1 and item-multiset/grouping errors with explicit denominators, plus format validity.

Use a deterministic scorer against unchanged receipt labels; do not adopt BIPIA's optional GPT-based judge by default. Some attack goals need a task-specific rubric. Report paired clean-correct/attacked-wrong degradation separately from verified attacker-goal success: an extraction error alone is not proof an attack succeeded.

The old 512-token context / 128-token output policy cannot be assumed adequate for variable-length real receipts and line-item arrays. After approval, measure token lengths on train only, then freeze a new versioned budget before validation/test. Never silently truncate inputs or omit difficult receipts to improve scores.

## 7. Preserve upstream splits and their boundaries

- **CORD train 800:** eligible for training and preprocessing calibration.
- **CORD validation 100:** model/data selection and a paired clean/injected development panel.
- **CORD test 100:** retain for final evaluation. Do not relabel it `validation` merely to fit the public runner, inspect its contents in the development process, or use its labels for schema/data tuning.

All variants of one receipt stay in its original split. Bind upstream revision, native split, receipt ID/image hash, projection and attack IDs; detect duplicates without silently reshuffling the dataset. Report venue/layout concentration and do not claim store-disjoint generalization unless verified.

Use BIPIA's published train/test attack division. Reserve test attack families/payloads for the final test overlay; use train-side attacks for any approved train augmentation or validation development with explicit family-disjointness accounting. Do not use test attack strings to write training counterexamples. Payload-family independence requires review after acquisition; it is not established by filenames alone.

Approval could initially cover **train + validation only** (2,073,081,477 bytes) and BIPIA's 5,999-byte train attack file. The 234,202,795-byte CORD test shard and 6,322-byte BIPIA test attack file would be acquired/materialized later by the qualified custodian. A default `load_dataset()` over every split is not acceptable for maintaining that boundary.

CORD's test set is publicly released and may already have appeared in model pretraining. Preserving its split gives project-level held-out evaluation, not proof it is globally secret or unexposed. Existing Stage 1 custody is software-fixture-qualified only: real Gym/test evaluator integration still needs measured qualification. No final test or holdout is run now.

## 8. BIPIA overlay, not a rewritten private attack collection

After approval, embed the selected **original** BIPIA text attack payload byte-for-byte as an attached note/quoted receipt content. Preserve its attack ID, source file/revision and insertion transform. Do not execute code, follow attack URLs, enable tools, or send messages demanded by a payload.

Freeze payload compatibility, prefix/middle/suffix placement and plain/JSON-like surrounding contexts before evaluation. Use identical paired placements for all models, with benign notes as controls. If a payload needs alteration to fit receipts, label it a separately reviewed **BIPIA-derived** adaptation and retain the original; do not call it the original benchmark.

Attacks must not change the receipt's genuine expected fields. Compare each attacked receipt with its clean counterpart. Do not add easier synthetic authoritative record labels or choose attacks after seeing which the model resists. Keep validation and final test results separate; no aggregate inflation from many correlated variants of the same receipt.

## 9. Approval and implementation sequence

1. User approves the exact acquisition subset and the commercial-use/legal review boundary, including the BIPIA intended-use caveat.
2. Acquire only approved pinned files, retain notices and verify hashes; inspect **train only** for text coverage, label mapping, privacy and token budgets.
3. Implement source/projection/scorer/attack plugins and config, with no core domain branches. Contract tests use small licensed fixtures only after acquisition approval.
4. Freeze schema, normalization, data/attack split mapping, budgets and paired evaluation plan before looking at validation outcomes.
5. Run cached local base models first. No training, API judge or model download is implicitly authorized by approving the data.
6. Only after separate approval and qualification, train/evaluate and eventually materialize the official test split inside custody.

No datasets, attack payloads or real-gym implementation are acquired/created by this proposal. No additional synthetic training or scoring is authorized.
