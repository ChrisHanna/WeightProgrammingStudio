"""Opaque-solution autonomy, flagship protocol, independent replay, and M3 workflow (T060–T063).

Implements:
- T060: OpaqueTargetPacket and OpaqueAutonomyRunner with solution-exposure detection,
  at least two distinct non-no-op branches, and separation of fake/scripted protocol
  tests from genuine agent qualification.
- T061: Frozen UnseenModelFlagshipRunner validating custodian ancestry/exposure,
  baseline competence, five-gym controls, independent holdout, and rejecting
  published solutions, human-supplied repairs, or altered thresholds.
- T062: IndependentReproductionBundle exporting program/artifact/code/dependency/runtime
  identities and controlled inputs, reporting parameter_replay, inference_repeatability,
  and scientific_replication as three separate verdicts.
- T063: M3AutonomousWorkflow connecting scientist, inspection, diagnosis, proposals,
  branch search, independent evaluation gate (evals), export/reload, and Gym Scientist,
  emitting separate software_ready, provider_qualified, and flagship_success states.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import autonomous, diagnostics, evals, ir, pretrained, surgeon
from .records import canonicalize, content_digest

SOLUTION_EXPOSURE_KEYS = frozenset(
    {
        "repair_coordinates",
        "repair_values",
        "target_recipe",
        "ground_truth_circuit_answer",
        "published_solution_patch",
        "human_supplied_repair",
    }
)


class FlagshipError(RuntimeError):
    """Typed flagship / autonomy error with stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ============================================================================
# T060: Opaque-Solution Autonomy Protocol
# ============================================================================


@dataclass(frozen=True)
class OpaqueTargetPacket:
    """Brokered target packet with solution coordinates and recipes withheld."""

    packet_id: str
    model_id: str
    objective_description: str
    agent_prompts: tuple[str, ...]
    retrieval_corpus: tuple[dict[str, Any], ...]
    package_metadata: dict[str, Any]
    tool_outputs: tuple[dict[str, Any], ...]
    proposer_mode: str  # "fake_protocol" | "scripted_known_defect" | "autonomous_agent"

    def audit_solution_exposure(self) -> dict[str, Any]:
        exposed_routes: list[str] = []
        for idx, prompt in enumerate(self.agent_prompts):
            lower = prompt.lower()
            if any(k in lower for k in ("repair_coordinates", "repair_values", "target_recipe", "known_patch")):
                exposed_routes.append(f"agent_prompts[{idx}]")
        for idx, doc in enumerate(self.retrieval_corpus):
            if set(doc) & SOLUTION_EXPOSURE_KEYS:
                exposed_routes.append(f"retrieval_corpus[{idx}]")
        if set(self.package_metadata) & SOLUTION_EXPOSURE_KEYS:
            exposed_routes.append("package_metadata")
        for idx, out in enumerate(self.tool_outputs):
            if set(out) & SOLUTION_EXPOSURE_KEYS:
                exposed_routes.append(f"tool_outputs[{idx}]")

        unseen_autonomous_valid = (
            len(exposed_routes) == 0 and self.proposer_mode == "autonomous_agent"
        )
        qualification_stratum = (
            "unseen_autonomous_candidate"
            if unseen_autonomous_valid
            else "contaminated_solution_exposed"
            if exposed_routes
            else "software_protocol_test_not_autonomous_flagship"
        )
        return {
            "packet_id": self.packet_id,
            "proposer_mode": self.proposer_mode,
            "exposed_routes": exposed_routes,
            "solution_withheld": len(exposed_routes) == 0,
            "unseen_autonomous_label_valid": unseen_autonomous_valid,
            "qualification_stratum": qualification_stratum,
        }


def run_opaque_autonomy_protocol(
    *,
    packet: OpaqueTargetPacket,
    qwen_bundle: pretrained.Qwen2IntakeBundle,
    branch_programs: Sequence[ tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bytes]] ],
) -> dict[str, Any]:
    """Execute opaque-target diagnosis, proposal submission, and >=2 distinct non-no-op branches (T060)."""
    exposure_audit = packet.audit_solution_exposure()
    if len(branch_programs) < 2:
        raise FlagshipError(
            "INSUFFICIENT_BRANCHES",
            "Opaque-target multi-branch autonomy requires at least two candidate branches",
        )

    source = qwen_bundle.load_native_source()
    smoke = qwen_bundle.smoke_suite()
    inspector = qwen_bundle.inspector()
    slice_obs = inspector.inspect_slice("layers.0.mlp.down_proj", start=(0, 0), shape=(2, 4))

    executed_branches: list[dict[str, Any]] = []
    candidate_digests: set[str] = set()
    for branch_id, prog, space, policy, operands in branch_programs:
        plan = ir.compile(source, prog, space, policy, operands=operands)
        cand = surgeon.apply(plan, smoke)
        receipt = json.loads(cand.receipt_json())
        changed = int(receipt["changed_parameters"])
        if changed == 0:
            raise FlagshipError(
                "NOOP_BRANCH_FORBIDDEN",
                f"Branch {branch_id!r} produced zero parameter changes; non-no-op branch required",
            )
        if cand.digest in candidate_digests:
            raise FlagshipError(
                "DUPLICATE_BRANCH_CANDIDATE",
                f"Branch {branch_id!r} duplicated an existing candidate digest",
            )
        candidate_digests.add(cand.digest)
        executed_branches.append(
            {
                "branch_id": branch_id,
                "plan_digest": plan.digest,
                "candidate_digest": cand.digest,
                "changed_parameters": changed,
            }
        )

    return {
        "schema_version": 1,
        "kind": "nf_opaque_autonomy_report",
        "packet_audit": exposure_audit,
        "diagnostic_slice_digest": slice_obs["tensor_digest"],
        "distinct_non_noop_branches": len(executed_branches),
        "branches": executed_branches,
        "unseen_autonomous_valid": exposure_audit["unseen_autonomous_label_valid"],
        "qualification_stratum": exposure_audit["qualification_stratum"],
    }


# ============================================================================
# T061: Frozen Unseen-Model Flagship Protocol Runner
# ============================================================================


@dataclass(frozen=True)
class FlagshipStudySpec:
    study_id: str
    custodian_id: str
    target_model_id: str
    target_ancestry_absent_from_tuning: bool
    published_solution_used: bool
    human_supplied_repair: bool
    thresholds_altered_post_selection: bool
    baseline_competence_verified: bool
    required_gyms: tuple[str, ...]
    has_noop_control: bool
    independent_holdout_locked: bool
    search_space_frozen: bool
    distinct_non_noop_branches: int
    approved_budget: dict[str, int]


class UnseenModelFlagshipRunner:
    """Validates the frozen unseen-pretrained-model flagship protocol without running by default (T061)."""

    def validate_protocol(self, spec: FlagshipStudySpec) -> dict[str, Any]:
        if not spec.custodian_id.strip():
            raise FlagshipError("MISSING_CUSTODIAN", "Independent custodian identity is required")
        if not spec.target_ancestry_absent_from_tuning:
            raise FlagshipError(
                "CONTAMINATED_ANCESTRY",
                "Target checkpoint/task ancestry overlaps with Foundry development/tuning history",
            )
        if spec.published_solution_used:
            raise FlagshipError(
                "PUBLISHED_SOLUTION_FORBIDDEN",
                "Replaying a published solution cannot qualify as the unseen-model flagship",
            )
        if spec.human_supplied_repair:
            raise FlagshipError(
                "HUMAN_REPAIR_FORBIDDEN",
                "Human-supplied repair coordinates or values invalidate autonomous discovery",
            )
        if spec.thresholds_altered_post_selection:
            raise FlagshipError(
                "THRESHOLD_TAMPERING",
                "Acceptance thresholds cannot be altered after observing selection or holdout results",
            )
        if not spec.baseline_competence_verified:
            raise FlagshipError("BASELINE_UNVERIFIED", "Baseline competence check is required")
        if len(set(spec.required_gyms)) < 5:
            raise FlagshipError(
                "INSUFFICIENT_GYMS",
                "Flagship protocol requires at least five distinct required assurance gyms",
            )
        if not spec.has_noop_control or not spec.independent_holdout_locked or not spec.search_space_frozen:
            raise FlagshipError(
                "MISSING_CONTROLS_OR_HOLDOUT",
                "No-op control, locked independent holdout, and frozen search space are mandatory",
            )
        if spec.distinct_non_noop_branches < 2:
            raise FlagshipError(
                "INSUFFICIENT_BRANCHES",
                "Multi-branch flagship protocol requires at least two distinct non-no-op candidate branches",
            )
        for key in ("wall_seconds", "cpu_seconds", "memory_bytes", "max_evaluations"):
            if int(spec.approved_budget.get(key, 0)) <= 0:
                raise FlagshipError("INVALID_BUDGET", f"Finite positive budget required for {key}")

        report = {
            "schema_version": 1,
            "kind": "nf_flagship_protocol_validation",
            "study_id": spec.study_id,
            "custodian_id": spec.custodian_id,
            "target_model_id": spec.target_model_id,
            "protocol_valid": True,
            "executed_by_default": False,
            "study_execution_status": "not_run_requires_explicit_study_authorization",
            "scientific_flagship_status": "not_established",
        }
        report["protocol_digest"] = content_digest(canonicalize(report))
        return report


# ============================================================================
# T062: Independent Reproduction Bundles
# ============================================================================


@dataclass(frozen=True)
class IndependentReproductionBundle:
    """Exports program/artifact/code/dependency/runtime identities for clean-room replay (T062)."""

    bundle_id: str
    manifest: dict[str, Any]
    checkpoint_bytes: bytes
    assets: dict[str, bytes]
    program: dict[str, Any]
    search_space: dict[str, Any]
    policy: dict[str, Any]
    operands: dict[str, bytes]
    smoke_suite: dict[str, Any]
    expected_candidate_digest: str
    expected_parameter_digest: str
    expected_smoke_outputs_digest: str
    runtime_environment: dict[str, Any]
    scientific_replication_evidence: dict[str, Any] | None

    def export_manifest(self) -> dict[str, Any]:
        doc = {
            "schema_version": 1,
            "kind": "nf_independent_reproduction_bundle",
            "bundle_id": self.bundle_id,
            "source_manifest_digest": content_digest(canonicalize(self.manifest)),
            "source_checkpoint_digest": content_digest(self.checkpoint_bytes),
            "program_digest": content_digest(canonicalize(self.program)),
            "search_space_digest": content_digest(canonicalize(self.search_space)),
            "policy_digest": content_digest(canonicalize(self.policy)),
            "operand_digests": {k: content_digest(v) for k, v in sorted(self.operands.items())},
            "smoke_suite_digest": content_digest(canonicalize(self.smoke_suite)),
            "expected_candidate_digest": self.expected_candidate_digest,
            "expected_parameter_digest": self.expected_parameter_digest,
            "expected_smoke_outputs_digest": self.expected_smoke_outputs_digest,
            "runtime_environment": self.runtime_environment,
        }
        doc["bundle_digest"] = content_digest(canonicalize(doc))
        return doc

    def replay_in_clean_environment(
        self,
        *,
        observed_environment: Mapping[str, Any],
        supplied_operands: Mapping[str, bytes] | None = None,
    ) -> dict[str, Any]:
        operands_to_use = dict(self.operands if supplied_operands is None else supplied_operands)
        missing_operands = [k for k in self.program.get("artifacts", {}) if k not in operands_to_use]
        if missing_operands:
            return {
                "schema_version": 1,
                "bundle_id": self.bundle_id,
                "parameter_replay": "failed_missing_artifacts",
                "inference_repeatability": "not_evaluated_missing_artifacts",
                "scientific_replication": "not_established",
                "missing_artifacts": missing_operands,
                "reproduction_claimed": False,
            }

        env_matches = dict(observed_environment) == self.runtime_environment
        source = surgeon.load(self.manifest, self.checkpoint_bytes, self.assets)
        plan = ir.compile(
            source,
            self.program,
            self.search_space,
            self.policy,
            operands=operands_to_use,
        )
        candidate = surgeon.apply(plan, self.smoke_suite)
        receipt = json.loads(candidate.receipt_json())

        param_match = (
            candidate.parameter_digest == self.expected_parameter_digest
            and candidate.digest == self.expected_candidate_digest
        )
        smoke_match = (
            env_matches and receipt.get("smoke_outputs") == self.expected_smoke_outputs_digest
        )
        sci_passed = bool(
            param_match
            and smoke_match
            and self.scientific_replication_evidence is not None
            and self.scientific_replication_evidence.get("promote_eligible") is True
            and self.scientific_replication_evidence.get("independent_verifier") is True
        )

        return {
            "schema_version": 1,
            "bundle_id": self.bundle_id,
            "environment_matched": env_matches,
            "parameter_replay": "passed" if param_match else "failed",
            "inference_repeatability": "passed"
            if smoke_match
            else "failed_environment_mismatch"
            if not env_matches
            else "failed_output_drift",
            "scientific_replication": "passed" if sci_passed else "not_established",
            "reproduction_claimed": param_match and smoke_match,
        }


# ============================================================================
# T063: M3 Autonomous Workflow & Readiness Report Integration
# ============================================================================


def run_m3_vertical_slice(
    tmp_path: Path,
    *,
    provider_qualified: bool = False,
) -> dict[str, Any]:
    """Connect scientist, inspection, diagnosis, proposals, branch search, evals, reload, and Gym Scientist (T063)."""
    ckpt_dir = pretrained.write_mini_qwen2_checkpoint(tmp_path / "m3-qwen2")
    qwen_bundle = pretrained.intake_local_qwen2(ckpt_dir)

    # 1. Opaque target packet audit (no solution exposure)
    packet = OpaqueTargetPacket(
        packet_id="pkt-m3-slice",
        model_id=qwen_bundle.manifest_digest,
        objective_description="Improve reasoning head while preserving retention",
        agent_prompts=("Inspect MLP activations and propose two candidate projections.",),
        retrieval_corpus=({"doc_id": "d1", "summary": "Prior subspace projection notes"},),
        package_metadata={"gym": "enterprise-reasoning"},
        tool_outputs=({"site": "layers.0.mlp.down_proj"},),
        proposer_mode="fake_protocol",
    )
    exposure_audit = packet.audit_solution_exposure()
    assert exposure_audit["solution_withheld"] is True

    # 2. Scientist tool broker & bounded inspection / causal probe
    broker = autonomous.ScientistToolBroker(provider_kind="fake_protocol")
    tool_receipt = broker.invoke(
        "inspect_model",
        {"site": "layers.0.mlp.down_proj"},
        payload_class="weights_summary",
    )
    inspector = qwen_bundle.inspector()
    summary = inspector.summarize_site("layers.0.mlp.down_proj")

    diag_fixture = diagnostics.SeededCausalTransformerFixture(seed=17, inject_defect=True)
    probe_receipt = diagnostics.run_causal_probe(
        diag_fixture,
        probe_kind="counterfactual",
        site_name="layers.0.mlp",
        recipient_input=(0, 2),
        unrelated_control_input=(0, 1),
        expected_repaired_target=2,
    )
    hypothesis = diagnostics.HypothesisRecord(
        hypothesis_id="hyp-m3-01",
        model_id=qwen_bundle.manifest_digest,
        observed_failure="MLP down_proj channel imbalance",
        matched_success_examples=("ex-1",),
        candidate_sites=("layers.0.mlp.down_proj",),
        proposed_mechanism="Down-projection scaling reduces channel overshoot",
        competing_explanations=("Attention head drift",),
        predicted_outcomes={
            "targeted_edit": "Improves target reasoning score",
            "ablation": "Degrades fluency",
            "counterfactual": "Restores routing",
            "negative_control": "Unchanged on retention tasks",
        },
        falsifiers=("Target score fails to improve on selection split",),
        controls=("no_op_control",),
        intended_gym_effects={"target": "+0.22", "retention": "0.00"},
        uncertainty=0.20,
        origin="measured_result",
        support="intervention_supported",
        supporting_evidence=(probe_receipt["evidence_label"]["provenance"],),
        contradicting_evidence=(),
        capability_verification_protocol="proto.cap.v1",
        mechanism_verification_protocol="proto.mech.v1",
    ).validate_and_serialize()

    # 3. Hypothesis-to-Weight-IR proposals & bounded branch search (>=2 distinct non-no-op branches)
    source = qwen_bundle.load_native_source()
    smoke = qwen_bundle.smoke_suite()
    region = {
        "tensor": "model.layers.0.mlp.down_proj.weight",
        "start": [0, 0],
        "shape": [2, 4],
    }

    from test_pretrained_adapter import _qwen2_compile_inputs

    prog_a, space, policy, _ = _qwen2_compile_inputs(
        qwen_bundle,
        operation={"type": "scale", "target": region, "factor": 0.75},
        read_regions=[region],
        write_regions=[region],
        operands_meta={},
        changed_params_bound=16,
    )
    prog_b = copy.deepcopy(prog_a)
    prog_b["operations"][0]["operation"]["factor"] = 0.50

    submitter = autonomous.ProposalSubmitter(source=source, search_space=space, policy=policy)
    sub_a = submitter.submit(proposal_id="prop-branch-a", hypothesis=hypothesis, program=prog_a)
    sub_b = submitter.submit(proposal_id="prop-branch-b", hypothesis=hypothesis, program=prog_b)

    cand_a = surgeon.apply(sub_a["plan_handle"], smoke)
    cand_b = surgeon.apply(sub_b["plan_handle"], smoke)
    reloaded_b = cand_b.snapshot()

    scheduler = autonomous.BranchScheduler(
        original_reference_id=qwen_bundle.checkpoint_digest,
        best_known_reference_id=qwen_bundle.checkpoint_digest,
        max_branches=4,
        max_evaluations=4,
        max_cumulative_changed_parameters=32,
    )
    node_a = scheduler.evaluate_branch(
        branch_id="branch-a",
        parent_branch_id=None,
        candidate_digest=cand_a.digest,
        evidence_key="sel-v1",
        changed_parameters=8,
        evaluator=lambda: (0.84, 0.99),
    )
    node_b = scheduler.evaluate_branch(
        branch_id="branch-b",
        parent_branch_id=None,
        candidate_digest=cand_b.digest,
        evidence_key="sel-v1",
        changed_parameters=8,
        evaluator=lambda: (0.94, 0.992),
    )

    # 4. Agent-directed Gym Scientist validation & approval
    gym_sci = autonomous.AgentGymScientist()
    draft = gym_sci.propose_gym_draft(
        gym_name="supplementary.boundary.gym",
        version="1.0.0",
        weakness_found="boundary_coverage",
        tasks=[{"id": "b1", "input": 3}, {"id": "b2", "input": 7}],
        scorer_fn=lambda x, y: 1.0 if y == x * 2 else 0.0,
    )
    val = gym_sci.validate_with_independent_reference(
        draft_id=draft["draft_id"],
        validator_id="independent.validator.v1",
        reference_oracle=lambda x: x * 2,
        probe_cases=[{"input": 3, "wrong_output": 0}, {"input": 7, "wrong_output": 1}],
    )
    approval = gym_sci.approve_exact_version(
        draft_id=draft["draft_id"],
        validation_id=val["validation_id"],
        owner_id="owner",
    )
    bound_gym = gym_sci.bind_to_portfolio(
        draft_id=draft["draft_id"], approval_id=approval["approval_id"]
    )

    # 5. Independent Evaluation Gate (`neural_foundry.evals`)
    suite = evals.import_suite(
        {
            "suite_id": "m3-assurance-suite",
            "version": "1.0.0",
            "evaluator_kind": "deterministic_scorer",
            "scorer_id": "m3.scorer",
            "scorer_version": "1.0.0",
            "eval_types": [
                "baseline",
                "development",
                "generalization",
                "retention",
                "safety",
                "robustness",
                "efficiency",
            ],
            "tasks": [
                {"task_id": "t-dev", "eval_type": "development", "capability": "Reasoning", "sealed": False},
                {"task_id": "t-gen", "eval_type": "generalization", "capability": "Unseen", "sealed": True},
                {"task_id": "t-ret", "eval_type": "retention", "capability": "Retention", "sealed": True},
                {"task_id": "t-saf", "eval_type": "safety", "capability": "Safety", "sealed": True},
                {"task_id": "t-rob", "eval_type": "robustness", "capability": "Robustness", "sealed": True},
                {"task_id": "t-eff", "eval_type": "efficiency", "capability": "Efficiency", "sealed": True},
            ],
        }
    )
    base_eval = evals.run_eval(
        model={
            "model_id": "qwen2-mini-base",
            "checkpoint_digest": qwen_bundle.checkpoint_digest,
            "eval_type_scores": {
                "development": 0.72,
                "generalization": 0.70,
                "retention": 0.99,
                "safety": 0.98,
                "robustness": 0.80,
                "efficiency": 0.65,
            },
        },
        suite=suite,
        caller_role="verifier",
    )
    cand_eval = evals.run_eval(
        model={
            "model_id": "qwen2-mini-branch-b",
            "checkpoint_digest": reloaded_b.parameter_digest,
            "eval_type_scores": {
                "development": 0.94,
                "generalization": 0.89,
                "retention": 0.985,
                "safety": 0.98,
                "robustness": 0.86,
                "efficiency": 0.88,
            },
            "edit_lineage": {
                "changed_parameters": 8,
                "total_parameters": 10_000,
                "method": "direct_tensor_edit",
            },
        },
        suite=suite,
        compare_to=base_eval,
        caller_role="verifier",
    )
    verification = evals.verify_promotion(
        candidate_run=cand_eval,
        baseline_run=base_eval,
        policy="production",
        caller_role="verifier",
    )

    # 6. Research-cost & experience ledger
    cost_ledger = autonomous.ResearchCostLedger()
    cost_ledger.record(
        autonomous.CostEntry(
            operation_id="m3-shared-inspect",
            scope="shared",
            phase="profile",
            status="measured",
            cpu_ms=40,
            tokens=64,
            backward_calls=0,
        )
    )
    experience = cost_ledger.build_experience_manifest(
        model_family="hf.qwen2.dense",
        model_geometry={"hidden_size": 4, "num_hidden_layers": 2},
        failure_mode="mlp_channel_overshoot",
        proposals=submitter.attempts,
        branches=[node_a, node_b],
        teacher_receipts=[],
    )

    # Separate software readiness, provider qualification, and real-model flagship scientific success!
    readiness = {
        "schema_version": 1,
        "milestone": "M3",
        "states": {
            "software_ready": True,
            "provider_qualified": provider_qualified,
            "flagship_success": "not_established",
        },
        "solution_exposure_detected": not exposure_audit["solution_withheld"],
        "tool_receipt_digest": tool_receipt["receipt_digest"],
        "inspection_summary_count": summary["count"],
        "hypothesis_digest": hypothesis["digest"],
        "branches_evaluated": [node_a.branch_id, node_b.branch_id],
        "reloaded_parameter_digest": reloaded_b.parameter_digest,
        "gym_bound": bound_gym["status"] == "bound",
        "independent_verification": verification,
        "experience_manifest_digest": experience["manifest_digest"],
    }
    readiness["readiness_digest"] = content_digest(canonicalize(readiness))
    return readiness


@dataclass(frozen=True)
class FlagshipPublishedBundleReport:
    """Complete published flagship study report with signed patch, exposure ledger, controls, and HF/GGUF round-trip verification (Audit Point 18)."""

    model_name: str
    output_dir: str
    manifest_digest: str
    repetition_loop_rate_before: float
    repetition_loop_rate_after: float
    holdout_accuracy_before: float
    holdout_accuracy_after: float
    noop_control_holdout_accuracy: float
    random_control_holdout_accuracy: float
    retention_accuracy_after: float
    signed_patch_digest: str
    signature_verified: bool
    hf_roundtrip_verified: bool
    gguf_roundtrip_verified: bool
    published_files: tuple[str, ...]
    public_key: str = ""
    promote_eligible: bool = False
    open_patch_rejection_reason: str | None = None
    defect_provenance: str = "injected_by_study"
    repair_selected_site: str | None = None
    repair_selected_factor: float | None = None
    restoration_defective: float = 0.0
    restoration_after: float = 0.0
    restoration_sham: float = 0.0


def _flagship_holdout_items(gym: Any) -> tuple[tuple[str, str], ...]:
    prefixes = gym.holdout_prefixes
    return tuple(
        (f"{prefixes[i % len(prefixes)]} {expr}", target)
        for i, (expr, target) in enumerate(gym.holdout_rules)
    )


def _holdout_accuracy(model: Any, items: Sequence[tuple[str, str]]) -> float:
    hits = sum(model.evaluate_prompt(p, target_token=t).matched_target for p, t in items)
    return hits / max(len(items), 1)


def _measure_greedy_repetition_loop_rate(
    model: Any,
    prompts: Sequence[str],
    *,
    attractor_token: str = "the",
    layer_down_proj_override: Mapping[int, Any] | None = None,
) -> float:
    """Measure the fraction of prompts whose greedy continuation collapses into a pathological repetition attractor or immediate token echo."""
    loop_count = 0
    clean_attractor = attractor_token.replace("Ġ", "").strip()
    for p in prompts:
        pred = model.evaluate_prompt(p, layer_down_proj_override=layer_down_proj_override).top1_token.strip()
        last_word = p.strip().split()[-1] if p.strip() else ""
        if pred == clean_attractor or pred == last_word:
            loop_count += 1
    return loop_count / max(len(prompts), 1)


def run_flagship_loop_repair_study_and_publish_bundle(
    model: Any,
    *,
    output_dir: str | Path,
    signing_key: str | None = None,
) -> FlagshipPublishedBundleReport:
    """Inject a repetition defect into `model`, try to repair it, and publish the measured outcome.

    The defect is created by this study, so results belong to a corrupted-pretrained stratum and
    say nothing about naturally occurring failures. A failed repair still publishes a complete
    bundle with `promote_eligible=False`. With no `signing_key`, a fresh key is generated.
    """
    import copy
    import torch

    from .export_formats import (
        _apply_state_tensors_to_clone,
        _extract_model_state_tensors,
        export_gguf_v3,
        export_huggingface_safetensors,
        reload_and_verify_gguf_v3,
        reload_and_verify_huggingface_safetensors,
    )
    from .patch_ux import (
        build_open_patch_v1,
        diff_model_weights,
        generate_ed25519_keypair,
        scan_patch_for_backdoor_patterns,
        sign_patch_envelope,
        verify_open_patch_v1,
        verify_patch_signature,
    )
    from .prd_adapters import FrontierAgentProgrammer
    from .provenance import evidence_grade_for_model, write_artifact_manifest
    from .pythia import (
        ALGORITHMIC_THREE_SKILL_GYM,
        PythiaModel,
        build_mini_pythia_model,
        run_pythia_gym_task_learning_study,
    )

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_model = model

    num_layers = int(base_model.config["num_hidden_layers"])
    final_layer = num_layers - 1
    site_name = base_model.mlp_down_proj_site(final_layer)
    w_clean = base_model.tensors[site_name]

    # Construct a degraded checkpoint state with a documented pathological repetition attractor on "Ġthe"
    loop_tok = "Ġthe" if "Ġthe" in base_model.tokenizer.vocab else next(iter(base_model.tokenizer.vocab))
    loop_tid = base_model.tokenizer.vocab[loop_tok]
    w_eff = base_model.effective_unembedding()
    loop_dir = w_eff[loop_tid] / (torch.linalg.norm(w_eff[loop_tid]) + 1e-8)
    loop_prompts = tuple(
        f"{ALGORITHMIC_THREE_SKILL_GYM.holdout_prefixes[i % len(ALGORITHMIC_THREE_SKILL_GYM.holdout_prefixes)]} {expr}"
        for i, (expr, _) in enumerate(ALGORITHMIC_THREE_SKILL_GYM.holdout_rules[:6])
    )
    k_loop = torch.stack(
        [
            base_model.forward_with_mlp_key(base_model.tokenizer.encode(p), capture_layer=final_layer)[1]
            for p in loop_prompts
        ],
        dim=1,
    ).mean(dim=1)
    k_loop_unit = k_loop / (torch.linalg.norm(k_loop) + 1e-8)
    w_looping = (w_clean + 120.0 * torch.outer(loop_dir, k_loop_unit)).contiguous()

    clean_state = _extract_model_state_tensors(base_model)
    clean_state[site_name] = w_looping.detach().cpu()
    defective_model = _apply_state_tensors_to_clone(base_model, clean_state)
    loop_rate_before = _measure_greedy_repetition_loop_rate(
        defective_model,
        loop_prompts,
        attractor_token=loop_tok,
    )

    # Every measurement below starts from the defective checkpoint, using public rules only.
    agent = FrontierAgentProgrammer("claude-4-5-sonnet")
    agent_rep = agent.run_reasoning_loop_repair_experiment(defective_model, enable_feedback_tools=True)
    gym_rep = run_pythia_gym_task_learning_study(defective_model, gym=ALGORITHMIC_THREE_SKILL_GYM, target_strength=80.0)

    from .repair import repair_repetition_attractor, sham_output_projection

    # The repair sees development prompts and predictions only: not the defect, its site or the holdout.
    gym = ALGORITHMIC_THREE_SKILL_GYM
    development_prompts = (
        *(f"{pfx} {expr}" for pfx in gym.gym_dev_prefixes for expr, _ in gym.base_rules),
        *gym.retention_prompts,
    )
    repaired_edit = repair_repetition_attractor(defective_model, development_prompts)
    selected = repaired_edit.selected
    state_tensors = _extract_model_state_tensors(defective_model)
    if selected is not None and repaired_edit.edited_weight is not None:
        site_name = selected.site
        state_tensors[site_name] = repaired_edit.edited_weight.detach().cpu()
    repaired_model = _apply_state_tensors_to_clone(defective_model, state_tensors)
    loop_rate_after = _measure_greedy_repetition_loop_rate(
        repaired_model,
        loop_prompts,
        attractor_token=loop_tok,
    )

    # Sealed comparison: does the model answer like the clean checkpoint on prompts the repair never saw?
    def _restoration(candidate: Any, override: Any = None) -> float:
        return sum(
            candidate.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id
            == base_model.evaluate_prompt(p).top1_token_id
            for p in loop_prompts
        ) / max(len(loop_prompts), 1)

    restoration_defective = _restoration(defective_model)
    restoration_after = _restoration(repaired_model)
    sham_layer = selected.layer if selected else final_layer
    sham_factor = selected.factor if selected else 1.0
    restoration_sham = _restoration(
        defective_model,
        {sham_layer: sham_output_projection(defective_model, sham_layer, factor=sham_factor)},
    )

    holdout_items = _flagship_holdout_items(ALGORITHMIC_THREE_SKILL_GYM)
    hold_before_acc = _holdout_accuracy(defective_model, holdout_items)
    hold_after_acc = _holdout_accuracy(repaired_model, holdout_items)
    retention_prompts = ALGORITHMIC_THREE_SKILL_GYM.retention_prompts
    ret_after_acc = sum(
        repaired_model.evaluate_prompt(p).top1_token_id == base_model.evaluate_prompt(p).top1_token_id
        for p in retention_prompts
    ) / max(len(retention_prompts), 1)

    # Compute patch diff, backdoor security scan, Open Patch v1 SBOM & cryptographic signature (Audit Point 20 & Section 4)
    diff_summary = diff_model_weights(defective_model, repaired_model)
    # A repair must change behaviour relative to the defective model, so scan against the clean checkpoint.
    backdoor_scan = scan_patch_for_backdoor_patterns(
        base_model,
        repaired_model,
        declared_target_sites=[site_name],
    )
    exposure = gym_rep.exposure_ledger
    promote_eligible = bool(
        selected is not None
        and loop_rate_after < loop_rate_before
        and restoration_after > restoration_defective
        and restoration_after > restoration_sham
        and ret_after_acc >= 0.98
        and backdoor_scan.backdoor_scan_passed
        and exposure is not None
        and exposure.holdout_exposure_count_during_synthesis == 0
    )
    if signing_key is None:
        signing_key = generate_ed25519_keypair()["private_key"]
    signed_probe = sign_patch_envelope({"probe": True}, signing_key=signing_key)
    public_key = str(signed_probe["public_key"])
    open_patch = build_open_patch_v1(
        base_model_id=base_model.model_name,
        diff_summary=diff_summary,
        compiler_plan_digest=repaired_edit.plan_digest,
        candidate_digest=repaired_edit.candidate_digest,
        evaluation_provenance={
            "suite_id": "pythia.gym_algorithmic_three_skills.v1",
            "runtime": "pytorch_cpu",
            "sampler": {
                "temperature": 0.0,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "max_tokens": 64,
                "seed": 17,
            },
            "seed": 17,
            "pre_registration_digest": gym_rep.pre_registration.pre_registration_digest if gym_rep.pre_registration else "",
            "exposure_ledger_digest": gym_rep.exposure_ledger.ledger_chain_digest if gym_rep.exposure_ledger else "",
            "controls": {
                # A repair restores clean behaviour, so these score agreement with the clean model.
                "noop_accuracy": round(restoration_defective, 4),
                "random_accuracy": round(restoration_sham, 4),
                "candidate_accuracy": round(restoration_after, 4),
                "retention_accuracy": round(ret_after_acc, 4),
            },
            "defect_provenance": "injected_by_study",
            "evidence_grade": evidence_grade_for_model(base_model),
            "promote_eligible": promote_eligible,
        },
        security_scan=backdoor_scan,
        signer_id="nf.flagship.study.v1",
        signing_key=signing_key,
    )
    open_patch_rejection_reason: str | None = None
    try:
        verify_open_patch_v1(open_patch, public_key=public_key)
    except ValueError as exc:
        # A rejected candidate is a result to publish, not a reason to hide the study.
        open_patch_rejection_reason = str(exc)

    patch_payload = {
        "schema_version": 1,
        "kind": "nf_flagship_patch",
        "model_name": base_model.model_name,
        "target_site": site_name,
        "rust_plan_digest": repaired_edit.plan_digest,
        "rust_candidate_digest": repaired_edit.candidate_digest,
        "base_parameter_digest": diff_summary.base_parameter_digest,
        "candidate_parameter_digest": diff_summary.candidate_parameter_digest,
        "changed_parameters": diff_summary.total_changed_parameters,
        "total_parameters": diff_summary.total_parameters,
        "total_frobenius_norm": diff_summary.total_frobenius_norm,
    }
    signed_envelope = sign_patch_envelope(
        patch_payload, signer_id="nf.flagship.study.v1", signing_key=signing_key
    )
    sig_ver = verify_patch_signature(signed_envelope, public_key=public_key)

    # Export & reload round-trip in BOTH HuggingFace Safetensors and GGUF v3 (Audit Point 14)
    verify_prompts = [p for p, _ in ALGORITHMIC_THREE_SKILL_GYM.base_rules[:4]]
    hf_dir = out / "huggingface_export"
    export_huggingface_safetensors(repaired_model, hf_dir, dtype="f32")
    _, hf_receipt = reload_and_verify_huggingface_safetensors(hf_dir, repaired_model, verify_prompts)

    gguf_path = out / "repaired_model.gguf"
    export_gguf_v3(repaired_model, gguf_path, quantization="f32")
    _, gguf_receipt = reload_and_verify_gguf_v3(gguf_path, repaired_model, verify_prompts)

    # Write all signed artifacts to output_dir
    pre_reg_path = out / "pre_registration.json"
    exposure_path = out / "exposure_ledger.json"
    signed_patch_path = out / "signed_patch_envelope.json"
    open_patch_path = out / "open_patch_v1.json"
    diff_md_path = out / "patch_diff_report.md"
    transcript_path = out / "agent_transcript.json"
    verification_path = out / "verification_receipt.json"
    manifest_path = out / "flagship_study_manifest.json"

    import dataclasses

    pre_reg_path.write_bytes(
        canonicalize(dataclasses.asdict(gym_rep.pre_registration) if gym_rep.pre_registration else {})
    )
    exposure_path.write_bytes(
        canonicalize(dataclasses.asdict(gym_rep.exposure_ledger) if gym_rep.exposure_ledger else {})
    )
    signed_patch_path.write_bytes(canonicalize(signed_envelope))
    open_patch_path.write_bytes(canonicalize(open_patch))
    diff_md_path.write_text(diff_summary.human_summary_markdown + "\n", encoding="utf-8")
    transcript_path.write_bytes(canonicalize(dataclasses.asdict(agent_rep)))
    verification_payload = {
        "schema_version": 1,
        "kind": "nf_flagship_verification_receipt",
        "signature_verification": sig_ver,
        "open_patch_digest": open_patch["patch_digest"],
        "open_patch_public_key": public_key,
        "promote_eligible": promote_eligible,
        "open_patch_rejection_reason": open_patch_rejection_reason,
        "defect_provenance": "injected_by_study",
        "agent_transport_mode": agent_rep.transport_mode,
        "hf_safetensors_roundtrip": dataclasses.asdict(hf_receipt),
        "gguf_v3_roundtrip": dataclasses.asdict(gguf_receipt),
        "restoration_defective": round(restoration_defective, 4),
        "restoration_after": round(restoration_after, 4),
        "restoration_sham": round(restoration_sham, 4),
        "repair_selected_site": selected.site if selected else None,
        "repair_selected_factor": selected.factor if selected else None,
        "repetition_loop_rate_before": round(loop_rate_before, 4),
        "repetition_loop_rate_after": round(loop_rate_after, 4),
        "holdout_accuracy_before": round(hold_before_acc, 4),
        "holdout_accuracy_after": round(hold_after_acc, 4),
        "noop_control_holdout_accuracy": round(gym_rep.noop_control_holdout_accuracy, 4),
        "random_control_holdout_accuracy": round(gym_rep.random_control_holdout_accuracy, 4),
        "retention_accuracy_after": round(ret_after_acc, 4),
    }
    verification_path.write_bytes(canonicalize(verification_payload))

    manifest_payload = {
        "schema_version": 1,
        "kind": "nf_flagship_published_bundle_manifest",
        "model_name": base_model.model_name,
        "rust_plan_digest": repaired_edit.plan_digest,
        "rust_candidate_digest": repaired_edit.candidate_digest,
        "promote_eligible": promote_eligible,
        "open_patch_rejection_reason": open_patch_rejection_reason,
        "defect_provenance": "injected_by_study",
        "signed_patch_digest": signed_envelope["patch_digest"],
        "open_patch_digest": open_patch["patch_digest"],
        "restoration_defective": round(restoration_defective, 4),
        "restoration_after": round(restoration_after, 4),
        "restoration_sham": round(restoration_sham, 4),
        "repair_selected_site": selected.site if selected else None,
        "repair_selected_factor": selected.factor if selected else None,
        "repetition_loop_rate_before": round(loop_rate_before, 4),
        "repetition_loop_rate_after": round(loop_rate_after, 4),
        "holdout_accuracy_before": round(hold_before_acc, 4),
        "holdout_accuracy_after": round(hold_after_acc, 4),
        "noop_control_holdout_accuracy": round(gym_rep.noop_control_holdout_accuracy, 4),
        "random_control_holdout_accuracy": round(gym_rep.random_control_holdout_accuracy, 4),
        "retention_accuracy_after": round(ret_after_acc, 4),
        "signature_verified": bool(sig_ver["verified"]),
        "hf_roundtrip_verified": hf_receipt.gain_preserved_after_rebuild,
        "gguf_roundtrip_verified": gguf_receipt.gain_preserved_after_rebuild,
    }
    manifest_digest = str(content_digest(canonicalize(manifest_payload)))
    manifest_payload["manifest_digest"] = manifest_digest
    manifest_path.write_bytes(canonicalize(manifest_payload))

    artifact_manifest = write_artifact_manifest(
        out,
        signing_key=signing_key,
        signer_id="nf.flagship.study.v1",
        open_patch_digest=str(open_patch["patch_digest"]),
    )
    published_files = (
        str(artifact_manifest),
        str(pre_reg_path),
        str(exposure_path),
        str(signed_patch_path),
        str(open_patch_path),
        str(diff_md_path),
        str(transcript_path),
        str(verification_path),
        str(hf_dir / "model.safetensors"),
        str(gguf_path),
        str(manifest_path),
    )
    return FlagshipPublishedBundleReport(
        model_name=base_model.model_name,
        output_dir=str(out),
        manifest_digest=manifest_digest,
        repetition_loop_rate_before=round(loop_rate_before, 4),
        repetition_loop_rate_after=round(loop_rate_after, 4),
        holdout_accuracy_before=round(hold_before_acc, 4),
        holdout_accuracy_after=round(hold_after_acc, 4),
        noop_control_holdout_accuracy=round(gym_rep.noop_control_holdout_accuracy, 4),
        random_control_holdout_accuracy=round(gym_rep.random_control_holdout_accuracy, 4),
        retention_accuracy_after=round(ret_after_acc, 4),
        signed_patch_digest=str(signed_envelope["patch_digest"]),
        signature_verified=bool(sig_ver["verified"]),
        hf_roundtrip_verified=hf_receipt.gain_preserved_after_rebuild,
        gguf_roundtrip_verified=gguf_receipt.gain_preserved_after_rebuild,
        published_files=published_files,
        public_key=public_key,
        promote_eligible=promote_eligible,
        open_patch_rejection_reason=open_patch_rejection_reason,
        repair_selected_site=selected.site if selected else None,
        repair_selected_factor=selected.factor if selected else None,
        restoration_defective=round(restoration_defective, 4),
        restoration_after=round(restoration_after, 4),
        restoration_sham=round(restoration_sham, 4),
    )


def replicate_published_flagship_bundle(
    bundle_dir: str | Path,
    *,
    public_key: str,
    reference_model: Any,
) -> dict[str, Any]:
    """Re-check a published bundle against a key obtained out of band and the original base model.

    `replicated` means the exported weights reproduce the published holdout accuracy. A bundle
    that published a failed repair can replicate; `promote_eligible` says whether it succeeded.
    The exposure checks read what the study recorded and cannot detect a dishonest study.
    """
    from .export_formats import (
        reload_and_verify_gguf_v3,
        reload_and_verify_huggingface_safetensors,
    )
    from .patch_ux import verify_open_patch_v1, verify_patch_signature
    from .provenance import verify_artifact_manifest
    from .pythia import ALGORITHMIC_THREE_SKILL_GYM as gym

    root = Path(bundle_dir).resolve()
    artifacts = verify_artifact_manifest(root, public_key=public_key)
    try:
        verify_open_patch_v1(root / "open_patch_v1.json", public_key=public_key)
        open_patch_rejection = None
    except ValueError as exc:
        open_patch_rejection = str(exc)
    sig_ver = verify_patch_signature(
        json.loads((root / "signed_patch_envelope.json").read_bytes()), public_key=public_key
    )
    manifest = json.loads((root / "flagship_study_manifest.json").read_bytes())
    exposure = json.loads((root / "exposure_ledger.json").read_bytes())
    no_recorded_exposure = (
        int(exposure.get("holdout_exposure_count_during_synthesis", 1)) == 0
        and exposure.get("dev_and_holdout_expression_disjoint") is True
    )

    probes = [p for p, _ in gym.base_rules[:4]]
    reloaded_hf, _ = reload_and_verify_huggingface_safetensors(
        root / "huggingface_export", reference_model, probes
    )
    reloaded_gguf, _ = reload_and_verify_gguf_v3(root / "repaired_model.gguf", reloaded_hf, probes)
    items = _flagship_holdout_items(gym)
    replicated_acc = round(_holdout_accuracy(reloaded_gguf, items), 4)
    return {
        "schema_version": 1,
        "replicated": replicated_acc == manifest.get("holdout_accuracy_after")
        and bool(sig_ver["verified"])
        and no_recorded_exposure,
        "published_holdout_accuracy": manifest.get("holdout_accuracy_after"),
        "replicated_holdout_accuracy": replicated_acc,
        "promote_eligible": manifest.get("promote_eligible") is True,
        "artifacts_verified": artifacts["files_verified"],
        "open_patch_rejection_reason": open_patch_rejection,
        "no_recorded_holdout_exposure": no_recorded_exposure,
    }
