# JSON Extraction Proof Sprint

## Current Sprint

1. Commit and push independently tested work.
2. Provide the synthetic JSON-extraction schema, generator, scorer, and unseen-template holdout.
3. Compile public train and validation Gym Job data into a NeMo Gym package; reject holdout compilation.
4. Bind the NeMo verifier to the digest-checked scorer and run a basic scorer audit.
5. Serve full base and SFT CandidateRefs with the pinned vLLM container, with file checks and logging disabled.
6. Compile, train AutoModel SFT and generic weight edit on the same approved data, and compare base, SFT, and weight edit on the synthetic holdout; frontier is an optional fourth arm.

The reference fixture, compiler, verifier, audit, vLLM adapter, and workflow are implemented. The verifier has been exercised through the real NeMo Gym HTTP route. The original workflow was exercised with real GPU SFT and real vLLM base/SFT servers, but the frontier was a labeled test stub. The generic weight-edit trainer and new three-way wiring have numerical and mocked integration coverage; a real three-way run is the pending acceptance gate. The old one-step, two-example smoke run is not evidence of model quality.

## Run

Run in Linux with Docker GPU access, the cached pinned vLLM image, pinned Qwen checkpoint, and a matching passed compatibility report. The default report is `docs/evidence/weight-edit-qwen3-bf16.json`; `--compatibility-probe` selects another explicitly. Defaults are 32 rows per template (64 training and 64 holdout cases) and 32 SFT steps. Preservation uses separate public-train cases and does not consume selection validation or holdout.

```bash
./.venv/bin/python -m neural_foundry.json_proof_workflow /path/to/new-output-directory \
  --gpu-python /path/to/gpu-python \
  --automodel-python /path/to/automodel-python
```

The output directory must not already exist. Public packages are under `nemo_train` and `nemo_validation`; the report contains only comparison aggregates, not individual holdout responses. For the generated NeMo verifier, set `NF_NEMO_SCORER_PYTHON` to the trusted scorer interpreter and `NF_NEMO_SCORER_PYTHONPATH` to this repository's Python package root. The current scorer worker supports the synthetic reference job only.

Add `--frontier` only after configuring `NF_FRONTIER_URL`, `NF_FRONTIER_MODEL`, and `NF_FRONTIER_API_KEY` directly in the environment. Frontier uses an OpenAI-compatible chat-completions endpoint over HTTPS (literal loopback HTTP is permitted for local testing) and requires the returned model ID to match the selected provider model. Without that flag no frontier request is made and no provider credentials are required.

The synthetic holdout is not independently held in custody. Weight-edit candidate serving verifies file lineage and recorded vLLM identity probes; base and SFT arms without creation-time probes remain identity-unverified. Frontier identity is provider-reported. The workflow does not authorize promotion, and incomplete comparisons have no scorecards. Preservation score and newly broken cases are reported separately. The edit uses a recorded token-loss gradient surrogate, not a claim of optimizing nondifferentiable exact-match reward directly.

## On Hold

After the proof loop runs: independent one-shot custody and aggregate-only release; vLLM weight-digest and log-probability identity probes; Docker placement, artifact store, and reload images; NeMo RL with reward guard; tool and multi-turn compilation and training. Dashboard foundations can proceed alongside those later items. Full Python and Cargo acceptance gates remain unresolved.