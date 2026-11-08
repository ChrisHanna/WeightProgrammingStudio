# JSON Extraction Proof Sprint

## Current Sprint

1. Commit and push independently tested work.
2. Provide the synthetic JSON-extraction schema, generator, scorer, and unseen-template holdout.
3. Compile public train and validation Gym Job data into a NeMo Gym package; reject holdout compilation.
4. Bind the NeMo verifier to the digest-checked scorer and run a basic scorer audit.
5. Serve full base and SFT CandidateRefs with the pinned vLLM container, with file checks and logging disabled.
6. Compile, train AutoModel SFT, and compare base, SFT, and an explicitly configured frontier model on the synthetic holdout.

The reference fixture, compiler, verifier, audit, vLLM adapter, and workflow are implemented. The verifier has been exercised through the real NeMo Gym HTTP route. The workflow has been exercised with real GPU SFT and real vLLM base and SFT servers, but the frontier was a labeled test stub. A real three-model proof remains blocked until a frontier provider is configured. The one-step, two-example smoke run is not evidence of model quality.

## Run

In Linux with Docker GPU access, the cached pinned vLLM image, and the pinned Qwen checkpoint, configure `NF_FRONTIER_URL`, `NF_FRONTIER_MODEL`, and `NF_FRONTIER_API_KEY` in the environment. The URL must be an OpenAI-compatible chat-completions endpoint over HTTPS, except that literal loopback HTTP is allowed for local testing. Use a provider model ID that is returned unchanged in the response.

```bash
./.venv/bin/python -m neural_foundry.json_proof_workflow /path/to/new-output-directory \
  --gpu-python /path/to/gpu-python \
  --automodel-python /path/to/automodel-python
```

The output directory must not already exist. Public packages are under `nemo_train` and `nemo_validation`; the report contains only comparison aggregates, not individual holdout responses. For the generated NeMo verifier, set `NF_NEMO_SCORER_PYTHON` to the trusted scorer interpreter and `NF_NEMO_SCORER_PYTHONPATH` to this repository's Python package root. The current scorer worker supports the synthetic reference job only.

The synthetic holdout is not independently held in custody. The vLLM adapter verifies candidate files and lineage but does not attest the endpoint's loaded weights. Frontier model identity is provider-reported. The workflow does not authorize promotion, and incomplete comparisons have no scorecards.

## On Hold

After the proof loop runs: independent one-shot custody and aggregate-only release; vLLM weight-digest and log-probability identity probes; Docker placement, artifact store, and reload images; NeMo RL with reward guard; tool and multi-turn compilation and training. Dashboard foundations can proceed alongside those later items. Full Python and Cargo acceptance gates remain unresolved.