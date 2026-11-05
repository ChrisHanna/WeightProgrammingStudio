"""Autonomous Weight Programmer components (T052–T056, T058).

Implements:
- T052: Restricted scientist/provider tools (ScientistToolBroker) with prompt-injection,
  shell, and hidden-data rejection.
- T053: Authorized TeacherGateway with behavioral vs. internal access separation,
  request reservations, token/cost receipts, and uncertain completion handling.
- T054: Hypothesis-to-Weight-IR proposal submission (ProposalSubmitter) with
  search-space site enforcement, attribution lineage, and failed-attempt accounting.
- T055: Bounded sequential and branch search scheduler (BranchScheduler) with
  cumulative ancestor effect tracking, duplicate detection, and budget cancellation.
- T056: Agent-directed Gym Scientist proposals (AgentGymScientist) with independent
  reference validation and exact-version approval gates.
- T058: Complete research-cost and InterventionExperience manifests (ResearchCostLedger)
  with zero/estimated/reserved/unknown cost distinction and single shared-cost allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import ir
from .records import CancellationHandle, NativeError, canonicalize, content_digest

FORBIDDEN_TOOL_NAMES = frozenset(
    {
        "shell",
        "bash",
        "exec",
        "read_holdout",
        "read_hidden_labels",
        "override_permissions",
        "replace_checkpoint",
        "sql_write",
    }
)

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "grant_permission",
    "override_budget",
    "read_holdout_labels",
    "/etc/passwd",
    "rm -rf",
)


class AutonomousError(RuntimeError):
    """Typed autonomous scientist error with stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ============================================================================
# T052: Restricted Scientist / Provider Tools
# ============================================================================


@dataclass
class ScientistBudget:
    max_tool_calls: int = 32
    max_reasoning_tokens: int = 8192
    max_spend_usd: float = 5.0
    used_tool_calls: int = 0
    used_reasoning_tokens: int = 0
    used_spend_usd: float = 0.0


class ScientistToolBroker:
    """Routes scientist/provider actions through allowed tools, budgets, and payload checks (T052)."""

    def __init__(
        self,
        *,
        provider_kind: str = "local",
        provider_id: str = "nf.scientist.rule_v1",
        allowed_tools: Sequence[str] = ("inspect_model", "diagnose_pair", "propose_ir", "query_memory"),
        allowed_payload_classes: Sequence[str] = ("weights_summary", "activations_summary", "tasks_dev"),
        allow_external_requests: bool = False,
        budget: ScientistBudget | None = None,
        handler: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if provider_kind not in {"local", "remote", "fake_protocol"}:
            raise AutonomousError("INVALID_PROVIDER", f"Unsupported provider_kind: {provider_kind!r}")
        if provider_kind == "remote" and not allow_external_requests:
            raise AutonomousError(
                "EXTERNAL_CONTACT_DENIED",
                "Remote reasoning provider requires explicit external-request authorization",
            )
        self.provider_kind = provider_kind
        self.provider_id = provider_id
        self.allowed_tools = frozenset(allowed_tools)
        if self.allowed_tools & FORBIDDEN_TOOL_NAMES:
            raise AutonomousError("FORBIDDEN_TOOL", "Shell or hidden-data tools cannot be enabled")
        self.allowed_payload_classes = frozenset(allowed_payload_classes)
        self.allow_external_requests = allow_external_requests
        self.budget = budget or ScientistBudget()
        self.handler = handler
        self.audit_log: list[dict[str, Any]] = []

    def invoke(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        *,
        payload_class: str = "tasks_dev",
        estimated_tokens: int = 64,
        estimated_spend_usd: float = 0.0,
    ) -> dict[str, Any]:
        raw_text = canonicalize(dict(payload)).decode("utf-8").lower()
        if any(pat in raw_text for pat in INJECTION_PATTERNS):
            self.audit_log.append({"tool": tool_name, "status": "rejected_prompt_injection"})
            raise AutonomousError(
                "PROMPT_INJECTION_REJECTED",
                "Untrusted instructions or permission-escalation patterns detected in tool payload",
            )
        if tool_name in FORBIDDEN_TOOL_NAMES or tool_name not in self.allowed_tools:
            self.audit_log.append({"tool": tool_name, "status": "denied_tool"})
            raise AutonomousError(
                "TOOL_PERMISSION_DENIED",
                f"Tool {tool_name!r} is not in the contract's allowed tool set",
            )
        if payload_class in {"holdout", "labels_holdout"} or payload_class not in self.allowed_payload_classes:
            self.audit_log.append({"tool": tool_name, "status": "denied_payload_class"})
            raise AutonomousError(
                "PAYLOAD_CLASS_DENIED",
                f"Payload class {payload_class!r} is not permitted for scientist tool invocation",
            )
        if (
            self.budget.used_tool_calls + 1 > self.budget.max_tool_calls
            or self.budget.used_reasoning_tokens + estimated_tokens > self.budget.max_reasoning_tokens
            or self.budget.used_spend_usd + estimated_spend_usd > self.budget.max_spend_usd
        ):
            self.audit_log.append({"tool": tool_name, "status": "budget_exhausted"})
            raise AutonomousError("BUDGET_EXHAUSTED", "Scientist tool/token/spend budget exceeded")

        self.budget.used_tool_calls += 1
        self.budget.used_reasoning_tokens += estimated_tokens
        self.budget.used_spend_usd = round(self.budget.used_spend_usd + estimated_spend_usd, 6)

        result = (
            self.handler(tool_name, payload)
            if self.handler is not None
            else {"tool": tool_name, "echo": dict(payload), "status": "ok"}
        )
        receipt = {
            "schema_version": 1,
            "provider_kind": self.provider_kind,
            "provider_id": self.provider_id,
            "tool": tool_name,
            "payload_class": payload_class,
            "tokens_charged": estimated_tokens,
            "spend_usd_charged": estimated_spend_usd,
            "result": result,
        }
        receipt["receipt_digest"] = content_digest(canonicalize(receipt))
        self.audit_log.append(receipt)
        return receipt


# ============================================================================
# T053: Authorized Teacher Gateway
# ============================================================================


@dataclass(frozen=True)
class TeacherProfile:
    teacher_id: str
    access_mode: str  # "behavioral_output_only" | "internal_weights_and_activations"
    usage_rights: str
    destination: str
    allowed_payload_classes: frozenset[str]


class TeacherGateway:
    """Authorized teacher gateway with strict behavioral vs. internal separation (T053)."""

    def __init__(
        self,
        *,
        teachers: Sequence[TeacherProfile],
        permitted_destinations: Mapping[str, Sequence[str]],
        max_teacher_tokens: int = 4096,
        max_teacher_spend_usd: float = 2.0,
        allow_external_contact: bool = False,
    ) -> None:
        self.teachers = {t.teacher_id: t for t in teachers}
        self.permitted_destinations = {
            dest: frozenset(classes) for dest, classes in permitted_destinations.items()
        }
        self.max_teacher_tokens = max_teacher_tokens
        self.max_teacher_spend_usd = max_teacher_spend_usd
        self.allow_external_contact = allow_external_contact
        self.reserved_tokens = 0
        self.spent_tokens = 0
        self.reserved_spend_usd = 0.0
        self.spent_spend_usd = 0.0
        self.receipts: dict[str, dict[str, Any]] = {}

    def request_guidance(
        self,
        *,
        request_id: str,
        teacher_id: str,
        requested_access: str,
        payload_class: str,
        prompt_or_site: str,
        reserve_tokens: int = 128,
        reserve_spend_usd: float = 0.0,
        simulate_uncertain_completion: bool = False,
    ) -> dict[str, Any]:
        if request_id in self.receipts:
            existing = self.receipts[request_id]
            if existing["status"] == "uncertain_completion_retained":
                raise AutonomousError(
                    "UNCERTAIN_COMPLETION_NO_RETRY",
                    "Uncertain teacher request cannot be automatically retried without idempotency clearance",
                )
            return existing

        teacher = self.teachers.get(teacher_id)
        if teacher is None:
            raise AutonomousError("UNKNOWN_TEACHER", f"Teacher {teacher_id!r} is not registered")
        if not teacher.usage_rights.strip():
            raise AutonomousError("MISSING_USAGE_RIGHTS", "Teacher usage rights are required")
        if teacher.destination != "local" and not self.allow_external_contact:
            raise AutonomousError(
                "EXTERNAL_CONTACT_DENIED",
                "External teacher contact is disabled by default",
            )
        if (
            teacher.destination not in self.permitted_destinations
            or payload_class not in self.permitted_destinations[teacher.destination]
            or payload_class not in teacher.allowed_payload_classes
            or payload_class == "holdout"
        ):
            raise AutonomousError(
                "DESTINATION_OR_PAYLOAD_DENIED",
                f"Destination {teacher.destination!r} or payload_class {payload_class!r} not authorized",
            )
        if (
            teacher.access_mode == "behavioral_output_only"
            and requested_access in {"internal_weights", "internal_activations"}
        ):
            raise AutonomousError(
                "WEIGHT_ACCESS_DENIED",
                "Output-only teacher cannot expose or copy internal weights or activations",
            )
        if (
            self.spent_tokens + self.reserved_tokens + reserve_tokens > self.max_teacher_tokens
            or self.spent_spend_usd + self.reserved_spend_usd + reserve_spend_usd
            > self.max_teacher_spend_usd
        ):
            raise AutonomousError("TEACHER_BUDGET_EXCEEDED", "Teacher token/spend reservation exceeds limit")

        self.reserved_tokens += reserve_tokens
        self.reserved_spend_usd = round(self.reserved_spend_usd + reserve_spend_usd, 6)

        if simulate_uncertain_completion:
            # Uncertain completion retains the full reservation and never fabricates outputs.
            receipt = {
                "schema_version": 1,
                "request_id": request_id,
                "teacher_id": teacher_id,
                "access_mode": teacher.access_mode,
                "requested_access": requested_access,
                "destination": teacher.destination,
                "payload_class": payload_class,
                "status": "uncertain_completion_retained",
                "reserved_tokens": reserve_tokens,
                "reserved_spend_usd": reserve_spend_usd,
                "actual_tokens": None,
                "actual_spend_usd": None,
                "guidance_label": "unavailable_uncertain_completion",
            }
            receipt["receipt_digest"] = content_digest(canonicalize(receipt))
            self.receipts[request_id] = receipt
            return receipt

        self.reserved_tokens -= reserve_tokens
        self.reserved_spend_usd = round(self.reserved_spend_usd - reserve_spend_usd, 6)
        self.spent_tokens += reserve_tokens
        self.spent_spend_usd = round(self.spent_spend_usd + reserve_spend_usd, 6)

        guidance_label = (
            "behavioral_guidance_only_no_internal_weight_inspection"
            if teacher.access_mode == "behavioral_output_only"
            else "internal_teacher_representation_access"
        )
        receipt = {
            "schema_version": 1,
            "request_id": request_id,
            "teacher_id": teacher_id,
            "usage_rights": teacher.usage_rights,
            "access_mode": teacher.access_mode,
            "requested_access": requested_access,
            "destination": teacher.destination,
            "payload_class": payload_class,
            "status": "completed",
            "reserved_tokens": 0,
            "actual_tokens": reserve_tokens,
            "actual_spend_usd": reserve_spend_usd,
            "guidance_label": guidance_label,
            "response": {"prompt_or_site": prompt_or_site, "teacher_output": "verified_guidance"},
        }
        receipt["receipt_digest"] = content_digest(canonicalize(receipt))
        self.receipts[request_id] = receipt
        return receipt


# ============================================================================
# T054: Hypothesis-to-Weight-IR Proposal Submission
# ============================================================================


class ProposalSubmitter:
    """Validates hypothesis-to-Weight-IR proposals and counts all attempts/rejections (T054)."""

    def __init__(
        self,
        *,
        source: Any,
        search_space: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> None:
        self.source = source
        self.search_space = dict(search_space)
        self.policy = dict(policy)
        self.attempts: list[dict[str, Any]] = []
        self.total_proposal_cost_tokens: int = 0

    def submit(
        self,
        *,
        proposal_id: str,
        hypothesis: Mapping[str, Any],
        program: Mapping[str, Any],
        operands: Mapping[str, bytes] | None = None,
        operand_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        proposal_tokens: int = 100,
    ) -> dict[str, Any]:
        self.total_proposal_cost_tokens += proposal_tokens
        operand_metadata = dict(operand_metadata or {})

        # Check for hidden gradient-derived deltas masquerading as direct-only
        for op_name, meta in operand_metadata.items():
            if meta.get("gradient_derived", False) or meta.get("fitted_on_target", False):
                method = program.get("lineage", {}).get("method", {})
                if method.get("production_history") == "direct_only" or not method.get(
                    "gradient_derived_update_values", False
                ):
                    record = {
                        "proposal_id": proposal_id,
                        "hypothesis_id": hypothesis.get("hypothesis_id"),
                        "status": "rejected",
                        "error_code": "HIDDEN_GRADIENT_DELTA",
                        "tokens_charged": proposal_tokens,
                    }
                    self.attempts.append(record)
                    raise AutonomousError(
                        "HIDDEN_GRADIENT_DELTA",
                        f"Operand {op_name!r} is gradient-derived/fitted and cannot enter a direct-only proposal",
                    )

        try:
            plan = ir.compile(
                self.source,
                program,
                self.search_space,
                self.policy,
                operands=operands,
            )
        except NativeError as exc:
            record = {
                "proposal_id": proposal_id,
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "status": "rejected",
                "error_code": exc.code,
                "error_message": exc.message,
                "tokens_charged": proposal_tokens,
            }
            self.attempts.append(record)
            raise AutonomousError(exc.code, exc.message) from exc

        summary = {
            "proposal_id": proposal_id,
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "status": "accepted_for_execution",
            "plan_digest": plan.digest,
            "plan_handle": plan,
            "tokens_charged": proposal_tokens,
            "expected_gym_effects": hypothesis.get("intended_gym_effects", {}),
            "risks": ["retention_regression_if_projection_too_strong"],
        }
        self.attempts.append(
            {k: v for k, v in summary.items() if k != "plan_handle"}
        )
        return summary


# ============================================================================
# T055: Bounded Sequential and Branch Search
# ============================================================================


@dataclass
class BranchNode:
    branch_id: str
    parent_branch_id: str | None
    candidate_digest: str
    changed_parameters: int
    cumulative_changed_parameters: int
    target_score: float
    retention_score: float
    feasible: bool
    cached_reused: bool = False


class BranchScheduler:
    """Bounded sequential and branch search without requiring evolutionary/merge engines (T055)."""

    def __init__(
        self,
        *,
        original_reference_id: str,
        best_known_reference_id: str,
        max_branches: int = 8,
        max_evaluations: int = 16,
        max_cumulative_changed_parameters: int = 64,
    ) -> None:
        self.original_reference_id = original_reference_id
        self.best_known_reference_id = best_known_reference_id
        self.max_branches = max_branches
        self.max_evaluations = max_evaluations
        self.max_cumulative_changed_parameters = max_cumulative_changed_parameters
        self.branches: dict[str, BranchNode] = {}
        self.seen_candidates: dict[tuple[str, str], tuple[float, float]] = {}
        self.evaluations_used = 0

    def evaluate_branch(
        self,
        *,
        branch_id: str,
        parent_branch_id: str | None,
        candidate_digest: str,
        evidence_key: str,
        changed_parameters: int,
        evaluator: Callable[[], tuple[float, float]],
        retention_floor: float = 1.0,
        cancellation: CancellationHandle | None = None,
    ) -> BranchNode:
        if cancellation is not None and cancellation.cancelled:
            raise AutonomousError("SEARCH_CANCELLED", "Branch search cancelled before evaluation")
        if len(self.branches) >= self.max_branches:
            raise AutonomousError("BRANCH_LIMIT_EXCEEDED", "Maximum branch count reached")

        parent_cum = 0
        if parent_branch_id is not None:
            if parent_branch_id not in self.branches:
                raise AutonomousError("MISSING_PARENT_BRANCH", f"Unknown parent branch {parent_branch_id!r}")
            parent_cum = self.branches[parent_branch_id].cumulative_changed_parameters

        cumulative_changed = parent_cum + changed_parameters
        if cumulative_changed > self.max_cumulative_changed_parameters:
            raise AutonomousError(
                "CUMULATIVE_WRITE_BUDGET_EXCEEDED",
                f"Cumulative changed parameters ({cumulative_changed}) exceed search-space bound ({self.max_cumulative_changed_parameters})",
            )

        cache_key = (candidate_digest, evidence_key)
        if cache_key in self.seen_candidates:
            target_score, retention_score = self.seen_candidates[cache_key]
            reused = True
        else:
            if self.evaluations_used + 1 > self.max_evaluations:
                raise AutonomousError("EVALUATION_BUDGET_EXCEEDED", "Search evaluation limit reached")
            self.evaluations_used += 1
            target_score, retention_score = evaluator()
            self.seen_candidates[cache_key] = (target_score, retention_score)
            reused = False

        feasible = retention_score >= retention_floor
        node = BranchNode(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            candidate_digest=candidate_digest,
            changed_parameters=changed_parameters,
            cumulative_changed_parameters=cumulative_changed,
            target_score=target_score,
            retention_score=retention_score,
            feasible=feasible,
            cached_reused=reused,
        )
        self.branches[branch_id] = node
        return node

    def pareto_finalists(self) -> list[BranchNode]:
        feasible_nodes = [b for b in self.branches.values() if b.feasible]
        finalists: list[BranchNode] = []
        for node in feasible_nodes:
            dominated = any(
                (other.target_score >= node.target_score and other.cumulative_changed_parameters <= node.cumulative_changed_parameters)
                and (other.target_score > node.target_score or other.cumulative_changed_parameters < node.cumulative_changed_parameters)
                for other in feasible_nodes
                if other.branch_id != node.branch_id
            )
            if not dominated:
                finalists.append(node)
        return finalists


# ============================================================================
# T056: Agent-Directed Gym Scientist Proposals
# ============================================================================


class AgentGymScientist:
    """Agent-directed Gym Scientist with independent reference validation and exact approval (T056)."""

    def __init__(self) -> None:
        self.drafts: dict[str, dict[str, Any]] = {}
        self.validations: dict[str, dict[str, Any]] = {}
        self.approvals: dict[str, dict[str, Any]] = {}

    def propose_gym_draft(
        self,
        *,
        gym_name: str,
        version: str,
        weakness_found: str,
        tasks: Sequence[Mapping[str, Any]],
        scorer_fn: Callable[[Any, Any], float],
        proposer_id: str = "agent.gym_scientist.v1",
    ) -> dict[str, Any]:
        if not tasks:
            raise AutonomousError("EMPTY_GYM_DRAFT", "Gym draft requires tasks")
        draft_manifest = {
            "schema_version": 1,
            "gym_name": gym_name,
            "version": version,
            "proposer_id": proposer_id,
            "weakness_found": weakness_found,
            "task_count": len(tasks),
            "tasks": [dict(t) for t in tasks],
        }
        draft_id = content_digest(canonicalize(draft_manifest))
        self.drafts[draft_id] = {
            "draft_id": draft_id,
            "manifest": draft_manifest,
            "scorer_fn": scorer_fn,
        }
        return {"draft_id": draft_id, "manifest": draft_manifest, "status": "draft_unvalidated"}

    def validate_with_independent_reference(
        self,
        *,
        draft_id: str,
        validator_id: str,
        reference_oracle: Callable[[Any], Any],
        probe_cases: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if draft_id not in self.drafts:
            raise AutonomousError("UNKNOWN_DRAFT", f"Unknown gym draft {draft_id!r}")
        draft = self.drafts[draft_id]
        if validator_id == draft["manifest"]["proposer_id"]:
            raise AutonomousError(
                "SELF_VALIDATION_FORBIDDEN",
                "Gym proposal cannot be validated by the same proposer/generator identity",
            )
        scorer_fn = draft["scorer_fn"]
        good_scores = []
        wrong_scores = []
        for case in probe_cases:
            inp = case["input"]
            truth = reference_oracle(inp)
            good_scores.append(float(scorer_fn(inp, truth)))
            wrong_scores.append(float(scorer_fn(inp, case["wrong_output"])))

        # Task count alone cannot pass quality targets if discrimination or oracle agreement fails!
        discriminates = all(g >= 0.99 for g in good_scores) and all(w <= 0.01 for w in wrong_scores)
        status = "validated" if discriminates else "rejected_weak_discrimination"
        receipt = {
            "schema_version": 1,
            "draft_id": draft_id,
            "validator_id": validator_id,
            "good_scores": good_scores,
            "wrong_scores": wrong_scores,
            "discriminates_valid_from_invalid": discriminates,
            "status": status,
        }
        receipt["validation_id"] = content_digest(canonicalize(receipt))
        self.validations[draft_id] = receipt
        return receipt

    def approve_exact_version(
        self,
        *,
        draft_id: str,
        validation_id: str,
        owner_id: str,
    ) -> dict[str, Any]:
        val = self.validations.get(draft_id)
        if val is None or val["validation_id"] != validation_id or val["status"] != "validated":
            raise AutonomousError(
                "UNVALIDATED_GYM_APPROVAL_DENIED",
                "Only an independently validated gym version can be approved",
            )
        approval = {
            "schema_version": 1,
            "draft_id": draft_id,
            "validation_id": validation_id,
            "owner_id": owner_id,
            "status": "approved_exact_version",
        }
        approval["approval_id"] = content_digest(canonicalize(approval))
        self.approvals[draft_id] = approval
        return approval

    def bind_to_portfolio(self, *, draft_id: str, approval_id: str) -> dict[str, Any]:
        app = self.approvals.get(draft_id)
        if app is None or app["approval_id"] != approval_id:
            raise AutonomousError(
                "UNAPPROVED_GYM_BINDING",
                "Cannot bind an unapproved or modified gym version to a portfolio",
            )
        return {
            "bound_draft_id": draft_id,
            "approval_id": approval_id,
            "status": "bound",
        }


# ============================================================================
# T058: Complete Research-Cost and Experience Manifests
# ============================================================================


@dataclass
class CostEntry:
    operation_id: str
    scope: str  # "shared" or branch_id
    phase: str
    status: str  # "zero" | "measured" | "estimated" | "reserved" | "unknown"
    cpu_ms: int | None
    tokens: int | None
    backward_calls: int | None
    reason: str | None = None


@dataclass
class ResearchCostLedger:
    """Joins proposals, inspections, failures, teachers, and branch costs (T058)."""

    entries: list[CostEntry] = field(default_factory=list)
    seen_operations: set[str] = field(default_factory=set)

    def record(self, entry: CostEntry) -> None:
        if entry.operation_id in self.seen_operations:
            raise AutonomousError(
                "DUPLICATE_COST_OPERATION",
                f"Operation {entry.operation_id!r} cannot be double-counted",
            )
        if entry.status == "unknown" and (entry.cpu_ms is not None or entry.tokens is not None):
            raise AutonomousError(
                "UNKNOWN_COST_CONVERSION_FORBIDDEN",
                "Unknown costs must remain None with an explicit reason, never converted to zero",
            )
        self.seen_operations.add(entry.operation_id)
        self.entries.append(entry)

    def build_experience_manifest(
        self,
        *,
        model_family: str,
        model_geometry: Mapping[str, Any],
        failure_mode: str,
        proposals: Sequence[Mapping[str, Any]],
        branches: Sequence[BranchNode],
        teacher_receipts: Sequence[Mapping[str, Any]],
        alignment_fitted: bool = False,
    ) -> dict[str, Any]:
        shared_cpu_ms = sum(
            e.cpu_ms for e in self.entries if e.scope == "shared" and e.cpu_ms is not None
        )
        branch_cpu_ms = sum(
            e.cpu_ms for e in self.entries if e.scope != "shared" and e.cpu_ms is not None
        )
        unknown_ops = [e.operation_id for e in self.entries if e.status == "unknown"]
        reserved_ops = [e.operation_id for e in self.entries if e.status == "reserved"]
        estimated_ops = [e.operation_id for e in self.entries if e.status == "estimated"]
        zero_ops = [e.operation_id for e in self.entries if e.status == "zero"]
        backward_total = sum(
            e.backward_calls for e in self.entries if e.backward_calls is not None
        )
        failed_proposals = [p for p in proposals if p.get("status") == "rejected"]

        manifest = {
            "schema_version": 1,
            "kind": "nf_intervention_experience_manifest",
            "model_family": model_family,
            "model_geometry": dict(model_geometry),
            "failure_mode": failure_mode,
            "proposal_count": len(proposals),
            "failed_proposal_count": len(failed_proposals),
            "failed_proposals": failed_proposals,
            "branch_count": len(branches),
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "parent_branch_id": b.parent_branch_id,
                    "candidate_digest": b.candidate_digest,
                    "changed_parameters": b.changed_parameters,
                    "cumulative_changed_parameters": b.cumulative_changed_parameters,
                    "target_score": b.target_score,
                    "retention_score": b.retention_score,
                    "feasible": b.feasible,
                    "cached_reused": b.cached_reused,
                }
                for b in branches
            ],
            "teacher_receipts": [dict(t) for t in teacher_receipts],
            "alignment_fitted": alignment_fitted,
            "backward_Diagnostic_calls": backward_total,
            "cost_summary": {
                "shared_cpu_ms_counted_once": shared_cpu_ms,
                "branch_cpu_ms": branch_cpu_ms,
                "total_known_cpu_ms": shared_cpu_ms + branch_cpu_ms,
                "zero_cost_operations": zero_ops,
                "estimated_cost_operations": estimated_ops,
                "reserved_cost_operations": reserved_ops,
                "unknown_cost_operations": unknown_ops,
                "total_efficiency_claim_eligible": len(unknown_ops) == 0 and len(reserved_ops) == 0,
            },
        }
        manifest["manifest_digest"] = content_digest(canonicalize(manifest))
        return manifest
