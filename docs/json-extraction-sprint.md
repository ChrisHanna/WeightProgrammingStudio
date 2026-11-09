# JSON Extraction Proof Sprint

## Current Sprint

1. Commit and push independently tested work.
2. Finish backlog cleanup, reproduce the 13 review findings, and get green full Python/native acceptance.
3. Provide the synthetic JSON-extraction schema, generator, and scorer. Keep its unseen-template holdout sealed.
4. Compile public train and validation Gym Job data into a NeMo Gym package; reject holdout compilation.
5. Bind the NeMo verifier to the digest-checked scorer and run a basic scorer audit.
6. Commit the vLLM adapter, then serve the base and an AutoModel SFT/LoRA candidate with the pinned vLLM container, with file checks and logging disabled.
7. Compare base and SFT on public validation only. Do not invoke the current workflow's holdout path during this step.
8. After the user selects a frontier provider, compare it on validation, freeze the candidates, then perform one-shot holdout scoring. Do not use holdout to tune or retry.
9. Keep generic weight editing and NeMo RL/GRPO paused until the foundation is complete and explicitly resumed.

The reference fixture, compiler, verifier, audit, vLLM adapter, and workflow are implemented. The verifier has been exercised through the real NeMo Gym HTTP route. The original workflow was exercised with real GPU SFT and real vLLM base/SFT servers, but the frontier was a labeled test stub. The generic weight-edit trainer and new three-way wiring have numerical and mocked integration coverage; a real three-way run is the pending acceptance gate. The old one-step, two-example smoke run is not evidence of model quality.

## Run

Run in Linux with Docker GPU access, the cached pinned vLLM image, pinned Qwen checkpoint, and a matching passed compatibility report. The default report is `docs/evidence/weight-edit-qwen3-bf16.json`; `--compatibility-probe` selects another explicitly. Defaults are 32 rows per template (64 training and 64 holdout cases) and 32 SFT steps. Preservation uses separate public-train cases and does not consume selection validation or holdout.

```bash
./.venv/bin/python -m neural_foundry.json_proof_workflow /path/to/new-output-directory \
  --gpu-python /path/to/gpu-python \
  --automodel-python /path/to/automodel-python
```

The output directory must not already exist. Public packages are under `nemo_train` and `nemo_validation`; the report contains only comparison aggregates, not individual holdout responses. For the generated NeMo verifier, set `NF_NEMO_SCORER_PYTHON` to the trusted scorer interpreter and `NF_NEMO_SCORER_PYTHONPATH` to this repository's Python package root. The current scorer worker supports the synthetic reference job only. The workflow can score holdout, so do not run it until the provider is selected and the candidates are frozen; the current approved comparison is validation-only.

Add `--frontier` only after configuring `NF_FRONTIER_URL`, `NF_FRONTIER_MODEL`, and `NF_FRONTIER_API_KEY` directly in the environment. Frontier uses an OpenAI-compatible chat-completions endpoint over HTTPS (literal loopback HTTP is permitted for local testing) and requires the returned model ID to match the selected provider model. Without that flag no frontier request is made and no provider credentials are required.

The reference fixture, compiler, verifier, audit, vLLM adapter, and workflow are implemented. The verifier has been exercised through the real NeMo Gym HTTP route. The original workflow was exercised with real GPU SFT and real vLLM base/SFT servers, but the frontier was a labeled test stub. The generic weight-edit trainer and new three-way wiring have numerical and mocked integration coverage; a real three-way run is the pending acceptance gate. The old one-step, two-example smoke run is not evidence of model quality.

## On Hold

After the validation-only SFT comparison: select a frontier provider, freeze candidates, and schedule the single holdout evaluation. Weight editing and NeMo RL remain paused until the foundation is complete and explicitly resumed. Independent custody and aggregate-only release; vLLM weight-digest and log-probability identity probes; Docker placement, artifact store, reload images; tool and multi-turn compilation/training; and dashboard foundations remain later work. Full Python and Cargo acceptance gates remain unresolved.