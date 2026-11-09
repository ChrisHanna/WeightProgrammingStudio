# Text-task Gym contract v1

This contract is the shared data boundary for prompt-target Gym tasks consumed by
weight-space search and LoRA/SFT baselines. It is not an execution format for
arbitrary Python environments and does not grant holdout access or promotion
authority.

For higher-level business job inputs/outputs, versioned scenario-provider,
scorer/verifier, candidate-method, and gate references, see
[Gym Job and Studio API v1](gym-job-v1.md). That contract composes with this
text-task Gym format without changing its v1 schema.

## Spec file

`nf-studio run-gym` and `nf-studio goal-search` accept `--gym-spec PATH`. The
loader accepts bounded UTF-8 JSON with duplicate-key rejection and no imports,
callbacks, or code execution. The root has `schema_version: 1`,
`kind: "nf_text_task_gym"`, a stable `gym_id`, `skills_taught`, explicit
`train_rules`, `validation_rules`, optional `holdout_rules`, and split prefixes.
Each rule is exactly `{ "prompt": string, "target": string }`.

Training, validation, and holdout rule prompts must be pairwise distinct.
Validation is never synthesized by changing only the prefix of a training
example. Train and validation prefixes may provide prompt variation within
their own split; holdout prefixes are assigned deterministically one per frozen
holdout rule. The selected model tokenizer must encode every public prompt and
each target as exactly one token before a run starts.

A spec may omit `holdout_rules` for training-only use. In that case independent
goal-search evaluation is recorded as incomplete, and Gym study/seal commands
refuse to substitute training rules for a holdout.

## Shared use and identity

`materialize_public_task_data` is the single source of train/validation examples
for the WSP `Lab`, baseline tuning, and the historical Gym study. The materialized
record binds the schema version, full Gym digest, and train/validation/holdout
panel digests. LoRA/SFT worker jobs and archived candidates carry the same Gym and
dataset identities. Holdout items are materialized only through the separate
holdout function; training workers receive no holdout examples.

The recipe remains explicit in each result: LoRA/SFT kind, sites, rank, optimizer
settings, steps, seed, and checkpoints are recorded with recipe schema version 1.
This establishes matched inputs and reproducibility identity, not matched compute
or scientific superiority. Experiments still need frozen hypotheses, sufficient
independent task groups, sealed evaluation, and multiplicity-aware analysis.

## NeMo Gym boundary

`neural_foundry.nemo_gym` separates a task-adapter registry from the generic
NeMo evaluation runner. Each registered adapter declares whether it can supply
WSP, LoRA, SFT, and/or evaluation inputs. Adding an environment-specific adapter
does not change the runner or model-training code; a capability is refused
unless that adapter supplies the required data contract.

The MCQA adapters support importing prepared panels or generating a bounded
synthetic arithmetic Gym without a labeled dataset:

- `load_mcqa_panels` imports three prepared MCQA JSONL panels into this
	contract. Every row must have a stable ID, one text-only user prompt, an
	`expected_answer` letter, and an options list. The chosen model tokenizer must
	represent the answer as exactly one token. Training, validation, and holdout
	IDs and prompts must be disjoint. The result is a normal `ReusableTaskGym`, so
	the same materialized train/validation data drives WSP and LoRA/SFT. The prompt
	must end at the exact next-token answer position, and the selected NeMo verifier
	config must accept that output format; otherwise the training objective and reward
	disagree.
- `simulated_arithmetic_mcqa` uses an explicit non-negative seed to generate
	unique single-digit addition MCQA tasks, then sends the three JSONL panels
	through the same loader and tokenizer checks. It is a deterministic synthetic
	fixture for exercising the train/evaluate workflow, not a general job or
	environment simulator and not evidence of real-task performance. The
	`simulate-nemo-gym` command writes the train/validation/holdout panels, a
	portable `gym.json` spec, and `gym.sources.json` with generator identity,
	seed, panel digests, and row counts. The output directory must be new or empty.
- `NeMoGymRunner` invokes the pinned `gym eval run --no-serve` CLI in a separate
	NeMo Python environment (Python 3.13.14+). It targets an already-running
	OpenAI-compatible endpoint, joins NeMo rollouts to materialized inputs by
	`_ng_task_index` and `_ng_rollout_index`, checks masked/failed/missing rows,
	and records source/config/result hashes. It never imports NeMo into the Studio
	process or puts endpoint credentials in command arguments.
- `train_mcqa_rollouts` accepts only the exact, complete, health-checked NeMo
	run artifacts, joins them by stable task/rollout indices, verifies every input
	belongs to the source train panel, and filters to a declared minimum reward.
	`fit_sequence_baseline` then trains LoRA or SFT on response tokens only and
	scores the separately declared validation panel. The imported holdout is not
	sent to the trainer. Tool calls, multi-turn responses, masked outcomes, missing
	rewards, and partial runs are refused rather than flattened into examples.
	SFT/DPO imports also require the NeMo runner receipt and verify that its artifact
	digests match the exact bytes used for training; the receipt digest is carried
	into the training dataset and training receipt. This detects accidental artifact
	drift but is not a signature, endpoint identity proof, or sealed custody.

Import panels with `nf-studio import-nemo-panels --adapter mcqa_single_token`
(`import-nemo-mcqa` remains an alias), then pass the emitted `--gym-spec` to
`goal-search` or `run-gym`. `WeightProgrammingStudio.nemo_gym_adapters` lists
registered adapters and their declared method capabilities. Evaluation-only
adapters can be registered without falsely advertising trainability.

After collecting a complete successful training rollout run, invoke the
registered sequence trainer:

```sh
nf-studio nemo-sft --model /models/pythia \
	--train-jsonl train.jsonl --validation-jsonl validation.jsonl \
	--holdout-jsonl holdout.jsonl \
	--materialized-inputs runs/train/rollouts_materialized_inputs.jsonl \
	--rollouts runs/train/rollouts.jsonl --output-dir runs/nemo-sft \
	--kind lora --targets all_down_proj --rank 4 --steps 100
```

The result contains `merged_weights.pt` and `training_receipt.json`. Reload a
candidate with `WeightProgrammingStudio.load_nemo_sft_candidate`; it checks the
base digest, safe tensor map, weight digest, and resulting candidate digest.
That candidate can then be exported with
`WeightProgrammingStudio.export_model_for_nemo_gym` for another external Gym
evaluation.

NeMo Gym is an external optional environment, not a dependency of the Studio
package. The runner requires `nemo-gym==0.7.0`, its `gym` executable, an existing
NeMo Gym checkout/configuration, and a user-managed model endpoint. It does not
start vLLM or the NeMo servers, install packages, download benchmark data, or
claim that the external endpoint serves a particular checkpoint digest.

Example panel import and shared WSP/LoRA search:

```sh
nf-studio import-nemo-mcqa --model /models/pythia \
	--train-jsonl train.jsonl --validation-jsonl validation.jsonl \
	--holdout-jsonl holdout.jsonl --output-spec nemo-mcqa.json
nf-studio goal-search --model /models/pythia --gym-spec nemo-mcqa.json \
	--output-dir runs/nemo-mcqa
```

For evaluation, start the NeMo resources/agent servers and the candidate model's
OpenAI-compatible endpoint using the pinned NeMo config, then run:

```sh
nf-studio nemo-eval --gym-python /opt/nemo-gym/.venv/bin/python \
	--gym-root /opt/nemo-gym --config eval.yaml --agent mcqa_simple_agent \
	--input-jsonl benchmark.jsonl --output-dir runs/candidate \
	--model-url http://127.0.0.1:8000/v1 --model-name candidate
```

This initial adapter deliberately does not train on arbitrary multi-turn/tool
trajectories or reinterpret sub-threshold rewards as labels. It supports
SFT/direct edits on one-token MCQA panels, sequence SFT on complete successful
single-turn responses, and offline DPO from same-task repeated responses whose
reward difference exceeds a declared margin. GRPO/on-policy training and a
trajectory-aware WSP objective are not implemented. The runner's reward receipt
is external evaluation evidence, not native sealed-data
custody, a native family ledger entry, or promotion authority. Stateful
observation/action tasks need a separate rollout adapter, and multi-token/tool
trajectories need a sequence-aware WSP/SFT trainer before they can train weights.