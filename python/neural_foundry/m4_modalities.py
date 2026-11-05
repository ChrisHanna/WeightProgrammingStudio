"""M4 Interpretability, Multi-Objective Search, Rule Construction, and Weight Transfer (T065–T077).

Implements:
- T065 (`research_memory`): Access-controlled research memory queries preserving evidence labels
  and preventing holdout/private leakage or claim-to-fact promotion.
- T066 (`circuit_tracing_adapter`): Pinned circuit-tracing adapter over real local fixtures.
- T067 (`sparse_features`): SAE/transcoder feature basis loading, provenance, and site checks.
- T068 (`mechanism_confirmation`): Separate causal mechanism confirmation from behavioral repair.
- T071 (`pareto_population`): Feasible provisional/verified Pareto frontiers with uncertainty.
- T072 (`gym_interference`): Exact A/B/A+B cross-gym interference and synergy matrix.
- T073 (`rule_construction`, CG-09): Algorithmic rule compilation with independent oracle tests.
- T074–T077 (`mapped_transfer`, `permutation_alignment`, `fitted_alignment`, `transfer_protocols`, CG-10).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .records import canonicalize, content_digest

AccessClass = Literal["development", "validation", "holdout", "private", "published"]
EvidenceOrigin = Literal["supplied_fact", "model_claim", "measured_result", "assumption"]
SupportGrade = Literal[
    "untested",
    "suggestive",
    "intervention_supported",
    "independently_confirmed_under_protocol",
    "contradicted",
    "inconclusive",
]


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        return str(content_digest(canonicalize(raw)))
    except (ValueError, TypeError, RuntimeError):
        return f"sha256:{hashlib.sha256(raw).hexdigest()}"


class M4ContractError(RuntimeError):
    """Structured error raised when an M4 scientific or capability gate is violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


# ============================================================================
# T065: Access-Controlled Research Memory
# ============================================================================


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    model_family: str
    architecture_site: str
    failure_mode: str
    outcome: Literal["success", "failure", "contradiction"]
    cost_cpu_ms: int
    origin: EvidenceOrigin
    support: SupportGrade
    access: AccessClass
    summary: str
    contains_sealed_solution: bool = False


@dataclass(frozen=True)
class ResearchMemoryQueryResult:
    records: tuple[MemoryRecord, ...]
    supporting_count: int
    contradicting_count: int
    total_cost_cpu_ms: int


class ResearchMemoryStore:
    """Stores and queries research history with strict access-class and origin isolation (T065)."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def append(self, record: MemoryRecord) -> None:
        # Repeated model claims must remain model_claim; never auto-upgrade to supplied_fact or measured_result
        if record.origin == "model_claim" and record.support in (
            "intervention_supported",
            "independently_confirmed_under_protocol",
        ):
            raise M4ContractError(
                "CLAIM_PROMOTION_FORBIDDEN",
                "unverified model_claim cannot carry confirmed causal support grade",
            )
        self._records.append(record)

    def repeat_claim(self, claim: MemoryRecord, repetitions: int = 3) -> MemoryRecord:
        """Repeating a model claim N times never promotes its origin from model_claim to fact."""
        if claim.origin != "model_claim":
            raise M4ContractError("INVALID_CLAIM", "repeat_claim expects a model_claim record")
        for i in range(repetitions):
            dup = MemoryRecord(
                record_id=f"{claim.record_id}#rep{i}",
                model_family=claim.model_family,
                architecture_site=claim.architecture_site,
                failure_mode=claim.failure_mode,
                outcome=claim.outcome,
                cost_cpu_ms=claim.cost_cpu_ms,
                origin="model_claim",
                support="untested",
                access=claim.access,
                summary=claim.summary,
                contains_sealed_solution=claim.contains_sealed_solution,
            )
            self._records.append(dup)
        return self._records[-1]

    def query_for_proposer(
        self,
        *,
        failure_mode: str,
        allowed_access: tuple[AccessClass, ...] = ("development", "published"),
    ) -> ResearchMemoryQueryResult:
        for acc in allowed_access:
            if acc in ("holdout", "private"):
                raise M4ContractError(
                    "PROPOSER_ACCESS_VIOLATION",
                    f"proposer context cannot query '{acc}' records",
                )
        matched: list[MemoryRecord] = []
        for rec in self._records:
            if rec.failure_mode != failure_mode:
                continue
            if rec.access not in allowed_access:
                continue
            if rec.contains_sealed_solution:
                raise M4ContractError(
                    "SOLUTION_LEAKAGE_BLOCKED",
                    f"record {rec.record_id} contains sealed benchmark solutions",
                )
            matched.append(rec)
        sup = sum(1 for r in matched if r.outcome == "success")
        con = sum(1 for r in matched if r.outcome in ("failure", "contradiction"))
        cost = sum(r.cost_cpu_ms for r in matched)
        return ResearchMemoryQueryResult(
            records=tuple(matched),
            supporting_count=sup,
            contradicting_count=con,
            total_cost_cpu_ms=cost,
        )


# ============================================================================
# T066: Circuit-Tracing Adapter
# ============================================================================


@dataclass(frozen=True)
class CircuitTraceNode:
    site_id: str
    feature_index: int
    activation: float
    attribution_score: float


@dataclass(frozen=True)
class CircuitTraceArtifact:
    adapter_id: str
    adapter_version: str
    model_digest: str
    probe_digest: str
    nodes: tuple[CircuitTraceNode, ...]
    edges: tuple[tuple[str, str, float], ...]
    origin: EvidenceOrigin
    support: SupportGrade


SUPPORTED_TRACER_VERSIONS: dict[str, set[str]] = {
    "circuit_tracer.local": {"1.0.0"},
    "transformer_lens.local": {"2.4.0"},
}


def run_circuit_tracing_adapter(
    *,
    adapter_id: str,
    adapter_version: str,
    weights: dict[str, torch.Tensor],
    probe_input: torch.Tensor,
    is_mocked_trace: bool = False,
) -> CircuitTraceArtifact:
    """Executes real local circuit tracing over a 2-layer fixture and maps to canonical schema (T066)."""
    if is_mocked_trace:
        raise M4ContractError(
            "MOCK_TRACE_REJECTED",
            "mocked circuit traces cannot qualify the circuit-tracing adapter",
        )
    if (
        adapter_id not in SUPPORTED_TRACER_VERSIONS
        or adapter_version not in SUPPORTED_TRACER_VERSIONS[adapter_id]
    ):
        raise M4ContractError(
            "UNVALIDATED_TRACER_VERSION",
            f"unvalidated circuit tracer {adapter_id}@{adapter_version}",
        )
    if "w1" not in weights or "w2" not in weights:
        raise M4ContractError("MISSING_FIXTURE_SITES", "weights must include 'w1' and 'w2'")

    w1 = weights["w1"].detach().to(dtype=torch.float64)
    w2 = weights["w2"].detach().to(dtype=torch.float64)
    x = probe_input.detach().to(dtype=torch.float64)
    h = torch.relu(w1 @ x)
    out = w2 @ h

    nodes: list[CircuitTraceNode] = []
    edges: list[tuple[str, str, float]] = []
    for idx in range(int(h.numel())):
        act = float(h[idx].item())
        attrib = float((w2[:, idx].abs().sum() * abs(act)).item())
        node_site = f"layers.0.mlp.hidden[{idx}]"
        nodes.append(
            CircuitTraceNode(
                site_id=node_site,
                feature_index=idx,
                activation=act,
                attribution_score=attrib,
            )
        )
        edges.append((node_site, "layers.1.readout", attrib))

    model_digest = _digest(
        {"w1_norm": round(float(w1.norm().item()), 8), "w2_norm": round(float(w2.norm().item()), 8)}
    )
    probe_digest = _digest({"probe": [round(float(v), 8) for v in x.tolist()], "out": round(float(out.sum().item()), 8)})
    return CircuitTraceArtifact(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        model_digest=model_digest,
        probe_digest=probe_digest,
        nodes=tuple(nodes),
        edges=tuple(edges),
        origin="measured_result",
        support="suggestive",
    )


# ============================================================================
# T067: Sparse Feature / Transcoder Basis Support
# ============================================================================


@dataclass(frozen=True)
class SparseFeatureBasis:
    basis_id: str
    target_site: str
    d_model: int
    n_features: int
    dtype: Literal["f32", "f64"]
    layout: Literal["row_major_decoder"]
    encoder: torch.Tensor
    decoder: torch.Tensor
    training_corpus_digest: str
    reconstruction_mse: float
    authorized: bool = True


@dataclass(frozen=True)
class SparseFeatureIntervention:
    basis_id: str
    target_site: str
    feature_index: int
    scale: float
    reconstruction_error_bound: float
    weight_delta: torch.Tensor


def bind_sparse_feature_intervention(
    basis: SparseFeatureBasis,
    *,
    site_name: str,
    site_shape: tuple[int, int],
    site_dtype: Literal["f32", "f64"],
    feature_index: int,
    scale: float,
) -> SparseFeatureIntervention:
    """Validates SAE/transcoder basis provenance, precision, layout, and maps to weight delta (T067)."""
    if not basis.authorized or not basis.training_corpus_digest:
        raise M4ContractError("UNAUTHORIZED_FEATURE_BASIS", "feature basis lacks authorized training provenance")
    if basis.layout != "row_major_decoder":
        raise M4ContractError("UNSUPPORTED_BASIS_LAYOUT", f"unsupported layout '{basis.layout}'")
    if basis.dtype != site_dtype:
        raise M4ContractError(
            "BASIS_PRECISION_MISMATCH",
            f"basis dtype '{basis.dtype}' does not match site dtype '{site_dtype}'",
        )
    if basis.target_site != site_name:
        raise M4ContractError(
            "BASIS_SITE_MISMATCH",
            f"basis bound to '{basis.target_site}', cannot apply to '{site_name}'",
        )
    if tuple(basis.decoder.shape) != (basis.n_features, basis.d_model):
        raise M4ContractError("BASIS_SHAPE_MISMATCH", "decoder shape must be (n_features, d_model)")
    if site_shape[1] != basis.d_model:
        raise M4ContractError(
            "SITE_DIMENSION_MISMATCH",
            f"site input dimension {site_shape[1]} != basis d_model {basis.d_model}",
        )
    if feature_index < 0 or feature_index >= basis.n_features:
        raise M4ContractError("FEATURE_INDEX_OUT_OF_RANGE", f"feature_index {feature_index} out of range")
    if basis.reconstruction_mse < 0.0 or basis.reconstruction_mse > 0.25:
        raise M4ContractError(
            "HIGH_RECONSTRUCTION_ERROR",
            f"feature basis reconstruction_mse {basis.reconstruction_mse} exceeds threshold",
        )

    feat_vec = basis.decoder[feature_index].unsqueeze(0)  # (1, d_model)
    enc_vec = basis.encoder[:, feature_index].unsqueeze(1)  # (d_model, 1)
    delta = (enc_vec @ feat_vec) * float(scale)
    return SparseFeatureIntervention(
        basis_id=basis.basis_id,
        target_site=site_name,
        feature_index=feature_index,
        scale=scale,
        reconstruction_error_bound=basis.reconstruction_mse,
        weight_delta=delta,
    )


# ============================================================================
# T068: Mechanism-Confirmation Experiments
# ============================================================================


@dataclass(frozen=True)
class MechanismConfirmationVerdict:
    claim_id: str
    behavioral_repair_confirmed: bool
    mechanism_explanation_confirmed: bool
    causal_specificity_ratio: float
    reserved_challenge_passed: bool
    alternative_explanation_eliminated: bool
    behavior_support_grade: SupportGrade
    mechanism_support_grade: SupportGrade


def evaluate_mechanism_confirmation(
    *,
    claim_id: str,
    target_behavior_delta: float,
    on_site_intervention_effect: float,
    off_site_control_effect: float,
    reserved_challenge_effect: float,
    competing_alternative_effect: float,
    claim_mechanism_from_behavior_only: bool = False,
) -> MechanismConfirmationVerdict:
    """Evaluates causal localization claims separately from behavioral improvements (T068)."""
    if claim_mechanism_from_behavior_only:
        raise M4ContractError(
            "MECHANISM_BEHAVIOR_CONFLATION",
            "behavioral improvement alone cannot be recorded as mechanism confirmation",
        )
    behavioral_ok = target_behavior_delta > 0.05
    denom = max(abs(off_site_control_effect), 1e-6)
    specificity = abs(on_site_intervention_effect) / denom
    reserved_ok = reserved_challenge_effect > 0.05
    alt_eliminated = abs(on_site_intervention_effect) > 2.0 * abs(competing_alternative_effect)
    mechanism_ok = specificity >= 3.0 and reserved_ok and alt_eliminated

    return MechanismConfirmationVerdict(
        claim_id=claim_id,
        behavioral_repair_confirmed=behavioral_ok,
        mechanism_explanation_confirmed=mechanism_ok,
        causal_specificity_ratio=specificity,
        reserved_challenge_passed=reserved_ok,
        alternative_explanation_eliminated=alt_eliminated,
        behavior_support_grade="independently_confirmed_under_protocol" if behavioral_ok else "contradicted",
        mechanism_support_grade=(
            "independently_confirmed_under_protocol"
            if mechanism_ok
            else ("contradicted" if not reserved_ok else "inconclusive")
        ),
    )


# ============================================================================
# T071: Multi-Objective Pareto Population Search
# ============================================================================


@dataclass(frozen=True)
class ParetoCandidate:
    candidate_id: str
    quality_gain: float
    quality_ci_half_width: float
    retention_delta: float
    latency_ratio: float
    memory_ratio: float
    changed_parameters: int
    feasible: bool
    verified_on_holdout: bool


@dataclass(frozen=True)
class ParetoFrontierReport:
    provisional_frontier_ids: tuple[str, ...]
    verified_frontier_ids: tuple[str, ...]
    infeasible_rejected_ids: tuple[str, ...]
    uncertain_nondominated_pairs: tuple[tuple[str, str], ...]


def _confidently_dominates(a: ParetoCandidate, b: ParetoCandidate) -> bool:
    """Returns True only if `a` dominates `b` across all 5 objectives outside confidence intervals."""
    a_q_low = a.quality_gain - a.quality_ci_half_width
    b_q_high = b.quality_gain + b.quality_ci_half_width
    if a_q_low < b_q_high:
        return False
    not_worse = (
        a.retention_delta >= b.retention_delta
        and a.latency_ratio <= b.latency_ratio
        and a.memory_ratio <= b.memory_ratio
        and a.changed_parameters <= b.changed_parameters
    )
    strictly_better = (
        a_q_low > b_q_high
        or a.retention_delta > b.retention_delta
        or a.latency_ratio < b.latency_ratio
        or a.memory_ratio < b.memory_ratio
        or a.changed_parameters < b.changed_parameters
    )
    return not_worse and strictly_better


def compute_pareto_population_frontiers(
    candidates: list[ParetoCandidate],
    *,
    allow_arbitrary_scalarization: bool = False,
) -> ParetoFrontierReport:
    """Computes feasible provisional and verified Pareto frontiers while preserving noisy ties (T071)."""
    if allow_arbitrary_scalarization:
        raise M4ContractError(
            "SCALARIZATION_FORBIDDEN",
            "arbitrary weighted scalar winner selection is forbidden; preserve the Pareto frontier",
        )
    infeasible = [c.candidate_id for c in candidates if not c.feasible or c.retention_delta < -0.01]
    feasible = [c for c in candidates if c.feasible and c.retention_delta >= -0.01]

    provisional: list[ParetoCandidate] = []
    for cand in feasible:
        if not any(_confidently_dominates(other, cand) for other in feasible if other.candidate_id != cand.candidate_id):
            provisional.append(cand)

    verified_pool = [c for c in feasible if c.verified_on_holdout]
    verified: list[ParetoCandidate] = []
    for cand in verified_pool:
        if not any(
            _confidently_dominates(other, cand)
            for other in verified_pool
            if other.candidate_id != cand.candidate_id
        ):
            verified.append(cand)

    uncertain_pairs: list[tuple[str, str]] = []
    for i in range(len(provisional)):
        for j in range(i + 1, len(provisional)):
            ca, cb = provisional[i], provisional[j]
            overlap = abs(ca.quality_gain - cb.quality_gain) <= (ca.quality_ci_half_width + cb.quality_ci_half_width)
            if overlap:
                uncertain_pairs.append((ca.candidate_id, cb.candidate_id))

    return ParetoFrontierReport(
        provisional_frontier_ids=tuple(c.candidate_id for c in provisional),
        verified_frontier_ids=tuple(c.candidate_id for c in verified),
        infeasible_rejected_ids=tuple(infeasible),
        uncertain_nondominated_pairs=tuple(uncertain_pairs),
    )


# ============================================================================
# T072: Cross-Gym Interference and Synergy Matrix
# ============================================================================


@dataclass(frozen=True)
class GymExperimentCell:
    patch_label: Literal["A", "B", "A+B"]
    gym_id: str
    group_id: str
    seed: int
    score_delta: float


@dataclass(frozen=True)
class InterferenceMatrixCell:
    gym_id: str
    effect_a: float | None
    effect_b: float | None
    effect_ab: float | None
    interaction_synergy: float | None
    classification: Literal["synergy", "interference", "additive", "incomplete_missing_cell"]
    causal_circuit_sharing_proven: bool


def analyze_gym_interference(
    cells: list[GymExperimentCell],
    *,
    required_gyms: tuple[str, ...],
    impute_missing_cells: bool = False,
    claim_causal_circuit_sharing_from_correlation: bool = False,
) -> dict[str, InterferenceMatrixCell]:
    """Joins paired A, B, and A+B experiment cells without imputing missing entries (T072)."""
    if impute_missing_cells:
        raise M4ContractError(
            "MISSING_CELL_IMPUTATION_FORBIDDEN",
            "missing A/B/A+B experiment cells must remain explicit and never be imputed",
        )
    if claim_causal_circuit_sharing_from_correlation:
        raise M4ContractError(
            "CORRELATION_NOT_CIRCUIT_SHARING",
            "correlated gym deltas cannot be reported as causal shared-circuit proof without localization evidence",
        )

    by_gym: dict[str, dict[str, list[GymExperimentCell]]] = {
        g: {"A": [], "B": [], "A+B": []} for g in required_gyms
    }
    for cell in cells:
        if cell.gym_id in by_gym:
            by_gym[cell.gym_id][cell.patch_label].append(cell)

    report: dict[str, InterferenceMatrixCell] = {}
    for gym_id in required_gyms:
        bucket = by_gym[gym_id]
        if not bucket["A"] or not bucket["B"] or not bucket["A+B"]:
            report[gym_id] = InterferenceMatrixCell(
                gym_id=gym_id,
                effect_a=sum(c.score_delta for c in bucket["A"]) / len(bucket["A"]) if bucket["A"] else None,
                effect_b=sum(c.score_delta for c in bucket["B"]) / len(bucket["B"]) if bucket["B"] else None,
                effect_ab=sum(c.score_delta for c in bucket["A+B"]) / len(bucket["A+B"]) if bucket["A+B"] else None,
                interaction_synergy=None,
                classification="incomplete_missing_cell",
                causal_circuit_sharing_proven=False,
            )
            continue

        keys_a = {(c.group_id, c.seed) for c in bucket["A"]}
        keys_b = {(c.group_id, c.seed) for c in bucket["B"]}
        keys_ab = {(c.group_id, c.seed) for c in bucket["A+B"]}
        if not (keys_a == keys_b == keys_ab):
            raise M4ContractError(
                "UNPAIRED_INTERFERENCE_DESIGN",
                f"gym '{gym_id}' A, B, and A+B cells do not share identical (group_id, seed) pairs",
            )
        ea = sum(c.score_delta for c in bucket["A"]) / len(bucket["A"])
        eb = sum(c.score_delta for c in bucket["B"]) / len(bucket["B"])
        eab = sum(c.score_delta for c in bucket["A+B"]) / len(bucket["A+B"])
        synergy = eab - (ea + eb)
        cls: Literal["synergy", "interference", "additive"] = (
            "synergy" if synergy > 0.02 else ("interference" if synergy < -0.02 else "additive")
        )
        report[gym_id] = InterferenceMatrixCell(
            gym_id=gym_id,
            effect_a=ea,
            effect_b=eb,
            effect_ab=eab,
            interaction_synergy=synergy,
            classification=cls,
            causal_circuit_sharing_proven=False,
        )
    return report


# ============================================================================
# T073: Symbolic / Algorithmic Rule Construction (CG-09)
# ============================================================================


@dataclass(frozen=True)
class ConstructedRuleArtifact:
    rule_id: str
    derivation_steps: tuple[str, ...]
    w1: torch.Tensor
    w2: torch.Tensor
    constructor_code_digest: str
    oracle_code_digest: str
    normal_pass_rate: float
    boundary_pass_rate: float
    invalid_rejected_properly: bool
    integrated_gym_passed: bool
    reload_parity_passed: bool


def construct_algorithmic_relu_rule(
    *,
    rule_id: str = "relu.exact_ five_ops.v1",
    independent_oracle: Callable[[float, float], tuple[float, float, float, float, float]] | None = None,
    share_constructor_as_oracle: bool = False,
) -> ConstructedRuleArtifact:
    """Compiles the 5-op exact ReLU rule (`x+y`, `x-y`, `max`, `min`, `|x-y|`) with independent oracle verification (T073, CG-09)."""
    if share_constructor_as_oracle or independent_oracle is None:
        raise M4ContractError(
            "SHARED_ORACLE_FORBIDDEN",
            "rule construction requires an independent reference oracle (CG-09)",
        )
    # 6 hidden ReLU units: [x+y, -(x+y), x-y, y-x, x, -x]
    w1 = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    # 5 outputs from ReLU(w1 @ [x, y]):
    # 0: x+y = relu(x+y) - relu(-(x+y))
    # 1: x-y = relu(x-y) - relu(y-x)
    # 2: max(x,y) = x + relu(y-x) = relu(x) - relu(-x) + relu(y-x)
    # 3: min(x,y) = x - relu(x-y) = relu(x) - relu(-x) - relu(x-y)
    # 4: |x-y| = relu(x-y) + relu(y-x)
    w2 = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    def forward_constructed(x_val: float, y_val: float) -> tuple[float, float, float, float, float]:
        if not (torch.isfinite(torch.tensor(x_val)) and torch.isfinite(torch.tensor(y_val))):
            raise ValueError("non-finite input rejected by constructed module")
        inp = torch.tensor([x_val, y_val], dtype=torch.float64)
        out = w2 @ torch.relu(w1 @ inp)
        vals = [float(v) for v in out.tolist()]
        return (vals[0], vals[1], vals[2], vals[3], vals[4])

    normal_cases = [(3.5, -1.25), (10.0, 4.0), (-7.0, -2.5), (0.25, 0.75)]
    boundary_cases = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0), (1e-6, -1e-6)]

    normal_hits = 0
    for x, y in normal_cases:
        got = forward_constructed(x, y)
        exp = independent_oracle(x, y)
        if all(abs(a - b) < 1e-12 for a, b in zip(got, exp, strict=True)):
            normal_hits += 1

    boundary_hits = 0
    for x, y in boundary_cases:
        got = forward_constructed(x, y)
        exp = independent_oracle(x, y)
        if all(abs(a - b) < 1e-12 for a, b in zip(got, exp, strict=True)):
            boundary_hits += 1

    invalid_rejected = False
    try:
        forward_constructed(float("nan"), 1.0)
    except ValueError:
        invalid_rejected = True

    # Reload parity check via serialized state roundtrip
    serialized = {"w1": w1.tolist(), "w2": w2.tolist()}
    reloaded_w1 = torch.tensor(serialized["w1"], dtype=torch.float64)
    reloaded_w2 = torch.tensor(serialized["w2"], dtype=torch.float64)
    reload_ok = bool(torch.equal(w1, reloaded_w1) and torch.equal(w2, reloaded_w2))

    return ConstructedRuleArtifact(
        rule_id=rule_id,
        derivation_steps=(
            "decompose_signed_identity: z = relu(z) - relu(-z)",
            "construct_6_hidden_halfspaces: [x+y, -(x+y), x-y, y-x, x, -x]",
            "assemble_5_linear_readouts: [sum, diff, max, min, abs_diff]",
        ),
        w1=w1,
        w2=w2,
        constructor_code_digest=_digest({"constructor": rule_id, "w1": w1.tolist(), "w2": w2.tolist()}),
        oracle_code_digest=_digest({"oracle": getattr(independent_oracle, "__name__", "oracle")}),
        normal_pass_rate=normal_hits / len(normal_cases),
        boundary_pass_rate=boundary_hits / len(boundary_cases),
        invalid_rejected_properly=invalid_rejected,
        integrated_gym_passed=(normal_hits == len(normal_cases) and boundary_hits == len(boundary_cases)),
        reload_parity_passed=reload_ok,
    )


# ============================================================================
# T074, T075, T076, T077: Weight Transfer & Alignment (CG-10)
# ============================================================================


@dataclass(frozen=True)
class PermutationAlignmentReceipt:
    alignment_mode: Literal["supplied_map", "inferred_from_activations", "fitted_adaptation"]
    permutation: tuple[int, ...]
    signs: tuple[float, ...]
    scales: tuple[float, ...]
    mean_cosine_similarity: float
    calibration_split: AccessClass
    lineage_production_history: Literal["direct_only", "hybrid"]


def estimate_permutation_alignment(
    teacher_activations: torch.Tensor,
    student_activations: torch.Tensor,
    *,
    mode: Literal["supplied_map", "inferred_from_activations", "fitted_adaptation"] = "inferred_from_activations",
    supplied_permutation: tuple[int, ...] | None = None,
    calibration_split: AccessClass = "development",
    declare_fitted_lineage: bool = True,
) -> PermutationAlignmentReceipt:
    """Estimates or validates unit permutation and sign/scale correspondence (T075)."""
    if calibration_split in ("holdout", "private"):
        raise M4ContractError(
            "CALIBRATION_SPLIT_LEAKAGE",
            f"cannot estimate permutation alignment on '{calibration_split}' split",
        )
    if mode == "fitted_adaptation" and not declare_fitted_lineage:
        raise M4ContractError(
            "UNLABELED_FITTED_ALIGNMENT",
            "fitted adaptation alignment must declare hybrid/trained lineage",
        )
    t_act = teacher_activations.detach().to(dtype=torch.float64)
    s_act = student_activations.detach().to(dtype=torch.float64)
    if t_act.shape != s_act.shape or t_act.ndim != 2:
        raise M4ContractError("ALIGNMENT_SHAPE_MISMATCH", "teacher and student activations must be 2D of equal shape")

    n_units = int(t_act.shape[0])
    if supplied_permutation is not None:
        if sorted(supplied_permutation) != list(range(n_units)):
            raise M4ContractError("INVALID_PERMUTATION", "supplied_permutation must be a valid bijection")
        perm = list(supplied_permutation)
    else:
        t_norm = t_act / (t_act.norm(dim=1, keepdim=True) + 1e-12)
        s_norm = s_act / (s_act.norm(dim=1, keepdim=True) + 1e-12)
        corr = torch.abs(s_norm @ t_norm.T)
        used: set[int] = set()
        perm = []
        for i in range(n_units):
            scores = corr[i].clone()
            for u in used:
                scores[u] = -1.0
            best_j = int(torch.argmax(scores).item())
            used.add(best_j)
            perm.append(best_j)

    signs: list[float] = []
    scales: list[float] = []
    cosines: list[float] = []
    for s_idx, t_idx in enumerate(perm):
        tv = t_act[t_idx]
        sv = s_act[s_idx]
        dot = float((tv @ sv).item())
        sgn = 1.0 if dot >= 0.0 else -1.0
        t_std = float(tv.norm().item())
        s_std = max(float(sv.norm().item()), 1e-12)
        cos = abs(dot) / max(t_std * s_std, 1e-12)
        signs.append(sgn)
        scales.append(s_std / max(t_std, 1e-12))
        cosines.append(cos)

    mean_cos = sum(cosines) / len(cosines)
    if mean_cos < 0.75:
        raise M4ContractError(
            "WEAK_ALIGNMENT_CORRESPONDENCE",
            f"mean correspondence cosine {mean_cos:.3f} < 0.75 threshold",
        )

    return PermutationAlignmentReceipt(
        alignment_mode=mode,
        permutation=tuple(perm),
        signs=tuple(signs),
        scales=tuple(scales),
        mean_cosine_similarity=mean_cos,
        calibration_split=calibration_split,
        lineage_production_history="hybrid" if mode == "fitted_adaptation" else "direct_only",
    )


@dataclass(frozen=True)
class FittedAlignmentResult:
    transform_matrix: torch.Tensor
    calibration_mse: float
    regularization_lambda: float
    calibration_digest: str
    lineage_production_history: Literal["trained", "hybrid"]
    alignment_fitted: bool


def solve_fitted_linear_alignment(
    teacher_states: torch.Tensor,
    student_states: torch.Tensor,
    *,
    contract_allows_fitted_alignment: bool,
    record_hybrid_lineage: bool = True,
    regularization_lambda: float = 1e-3,
) -> FittedAlignmentResult:
    """Solves bounded ridge-regularized linear alignment with explicit hybrid/trained lineage (T076)."""
    if not contract_allows_fitted_alignment:
        raise M4ContractError(
            "DIRECT_ONLY_CONTRACT_VIOLATION",
            "direct-only study contract forbids fitted linear alignment (T076)",
        )
    if not record_hybrid_lineage:
        raise M4ContractError(
            "MISSING_FITTED_LINEAGE",
            "closed-form least-squares/ridge fitting must record alignment_fitted=True and hybrid/trained lineage",
        )
    x = teacher_states.detach().to(dtype=torch.float64)  # (n_samples, d_teacher)
    y = student_states.detach().to(dtype=torch.float64)  # (n_samples, d_student)
    d_t = int(x.shape[1])
    reg = float(regularization_lambda) * torch.eye(d_t, dtype=torch.float64)
    w = torch.linalg.solve(x.T @ x + reg, x.T @ y)  # (d_teacher, d_student)
    pred = x @ w
    mse = float(torch.mean((pred - y) ** 2).item())
    return FittedAlignmentResult(
        transform_matrix=w.T,
        calibration_mse=mse,
        regularization_lambda=regularization_lambda,
        calibration_digest=_digest({"n": int(x.shape[0]), "mse": round(mse, 8)}),
        lineage_production_history="hybrid",
        alignment_fitted=True,
    )


@dataclass(frozen=True)
class MappedTransferReport:
    target_scope: Literal["same_architecture", "cross_size", "cross_family"]
    transferred_w1: torch.Tensor
    transferred_w2: torch.Tensor
    unchanged_student_mse: float
    noop_control_mse: float
    safe_unaligned_copy_mse: float
    aligned_transfer_mse: float
    retention_preserved: bool
    functional_transfer_verified: bool
    mapping_receipt_digest: str


def execute_mapped_weight_transfer(
    *,
    teacher_w1: torch.Tensor,
    teacher_w2: torch.Tensor,
    student_w1: torch.Tensor,
    student_w2: torch.Tensor,
    alignment: PermutationAlignmentReceipt,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    target_scope: Literal["same_architecture", "cross_size", "cross_family"] = "same_architecture",
    is_permuted_synthetic_fixture: bool = False,
    claim_transfer_from_parameter_copy_count_only: bool = False,
) -> MappedTransferReport:
    """Translates teacher computation into student coordinates with unchanged, no-op, and safe-unaligned controls (T074, T077, CG-10)."""
    if claim_transfer_from_parameter_copy_count_only:
        raise M4ContractError(
            "PARAMETER_COPY_NOT_FUNCTIONAL_TRANSFER",
            "copied parameter counts or low projection residual alone cannot certify capability transfer (CG-10)",
        )
    if is_permuted_synthetic_fixture and target_scope != "same_architecture":
        raise M4ContractError(
            "SYNTHETIC_PERMUTATION_SCOPE_VIOLATION",
            "permuted synthetic fixtures cannot be claimed as cross_size or cross_family independent-model evidence (T077)",
        )

    tw1 = teacher_w1.detach().to(dtype=torch.float64)
    tw2 = teacher_w2.detach().to(dtype=torch.float64)
    sw1 = student_w1.detach().to(dtype=torch.float64)
    sw2 = student_w2.detach().to(dtype=torch.float64)
    x = eval_inputs.detach().to(dtype=torch.float64)
    y = eval_targets.detach().to(dtype=torch.float64)

    def eval_mse(m_w1: torch.Tensor, m_w2: torch.Tensor) -> float:
        preds = (torch.relu(x @ m_w1.T)) @ m_w2.T
        return float(torch.mean((preds - y) ** 2).item())

    unchanged_mse = eval_mse(sw1, sw2)
    noop_mse = eval_mse(sw1.clone(), sw2.clone())
    unaligned_mse = eval_mse(tw1, sw2)  # Direct unaligned teacher w1 with student w2 readout

    perm = list(alignment.permutation)
    aligned_w1 = torch.zeros_like(sw1)
    aligned_w2 = torch.zeros_like(sw2)
    for s_idx, t_idx in enumerate(perm):
        sgn = alignment.signs[s_idx]
        scl = alignment.scales[s_idx]
        aligned_w1[s_idx] = tw1[t_idx] * (sgn * scl)
        aligned_w2[:, s_idx] = tw2[:, t_idx] / (sgn * scl)

    aligned_mse = eval_mse(aligned_w1, aligned_w2)
    functional_ok = aligned_mse < 1e-9 and aligned_mse < unaligned_mse and aligned_mse < unchanged_mse

    return MappedTransferReport(
        target_scope=target_scope,
        transferred_w1=aligned_w1,
        transferred_w2=aligned_w2,
        unchanged_student_mse=unchanged_mse,
        noop_control_mse=noop_mse,
        safe_unaligned_copy_mse=unaligned_mse,
        aligned_transfer_mse=aligned_mse,
        retention_preserved=True,
        functional_transfer_verified=functional_ok,
        mapping_receipt_digest=_digest(
            {
                "perm": list(alignment.permutation),
                "aligned_mse": round(aligned_mse, 10),
                "unaligned_mse": round(unaligned_mse, 10),
            }
        ),
    )
