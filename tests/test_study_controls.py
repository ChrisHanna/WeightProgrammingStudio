"""Control arms in the study harness: manifest rules, paired decisions, placebo and tuning isolation.

Custodian tests use the synthetic evaluator, so they prove harness behaviour and say nothing about any method.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
import torch
from test_study_harness import make_plan

import neural_foundry
from neural_foundry import baselines as bl
from neural_foundry.pythia import build_mini_pythia_model
from neural_foundry.study import arms, decision, generators
from neural_foundry.study.client import CustodianClient, CustodianRefused, declare_exposure, write_base_arm, write_candidate
from neural_foundry.study.custodian import Custodian, Refused, audit_sealed_state
from neural_foundry.study.evaluators import decode_candidate_artifact, encode_candidate_artifact
from neural_foundry.study.plan import CONTROL_ARMS, PlanError, StudyPlan, freeze_plan

COMPARATORS = ("lora", "fine_tune", "placebo")
TUNED = ("lora", "fine_tune")


def synthetic(p: float, salt: str, **fields: Any) -> bytes:
    return json.dumps({"p": dict.fromkeys(("parity", "boolean", "comparison"), p), "salt": salt, "agreement": 0.99, "kl": 0.01, **fields}).encode()


def development(plan: StudyPlan) -> arms.DevelopmentData:
    return arms.development_data(plan, 1, rules_per_task=10, retention_prompts=8)


def tuning_record(plan: StudyPlan, data: arms.DevelopmentData) -> dict[str, Any]:
    score = bl.CheckpointScore(10, 0.5, 1.0, 0.9, 0.5, 1.0)
    config = bl.BaselineConfig(kind="sft", lr=1e-3, steps=10)

    def trial(_model: Any, _config: bl.BaselineConfig, _data: arms.DevelopmentData) -> Any:
        cost = bl.FitCost(10, 1.0, 1, 1, 1, None, 0.0)
        return [score], bl.BaselineFit(config, {}, cost, 0.0)

    return arms.tune_arm(None, plan, [config], data, trial=trial)[1]


def make_controlled(
    tmp_path: Path,
    *,
    candidate_p: float = 0.95,
    arm_p: dict[str, float] | None = None,
    plan: StudyPlan | None = None,
) -> dict[str, Any]:
    plan = plan or make_plan(control_arms=CONTROL_ARMS)
    arm_p = {"lora": 0.5, "fine_tune": 0.5, "placebo": 0.5, **(arm_p or {})}
    plan_path = tmp_path / "plan.json"
    digest = freeze_plan(plan, plan_path)
    data = development(plan)
    exposed = declare_exposure(data.clusters, data.retention_items, plan.tasks)
    entry = lambda name, seed, body: {  # noqa: E731
        "synthesis_seed": seed,
        "candidate_digest": write_candidate(tmp_path / name, body),
        "artifact_path": str(tmp_path / name),
    }
    candidates = [entry(f"candidate_{s}", s, synthetic(candidate_p, f"c{s}")) for s in plan.synthesis_seeds]
    controls: list[dict[str, Any]] = [write_base_arm(tmp_path / "base", digest)]
    for arm in COMPARATORS:
        controls += [
            {"arm": arm, **entry(f"{arm}_{s}", s, synthetic(arm_p[arm], f"{arm}{s}"))} for s in plan.synthesis_seeds
        ]
    manifest = {
        "plan_digest": digest,
        "candidates": candidates,
        "controls": controls,
        "exposed_operands": exposed,
        "tuning": {arm: tuning_record(plan, data) for arm in TUNED},
    }
    return {"plan": plan, "plan_path": plan_path, "digest": digest, "manifest": manifest, "state": tmp_path / "state", "data": data}


def client_for(study: dict[str, Any]) -> CustodianClient:
    return CustodianClient(study["plan_path"], study["digest"], study["state"])


def run_all(client: CustodianClient, manifest: dict[str, Any]) -> dict[str, Any]:
    client.freeze(manifest)
    for entry in (*manifest["candidates"], *manifest["controls"]):
        client.query(entry["candidate_digest"])
    return client.decide()


def codes(result: dict[str, Any]) -> set[str]:
    return {r["code"] for r in result["rejection_reasons"]}


def refusal(call: Any, *args: Any) -> str:
    with pytest.raises(CustodianRefused) as refused:
        call(*args)
    return refused.value.code


# The whole path ------------------------------------------------------------------------------


def test_strong_candidate_beats_every_control_and_only_the_label_stops_it(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    with client_for(study) as client:
        result = run_all(client, study["manifest"])
    assert codes(result) == {"NON_CONFIRMATORY_LABEL"}
    assert {(c["arm"], c["task"]) for c in result["controls"]} == {
        (arm, task) for arm in COMPARATORS for task in study["plan"].tasks
    }


def test_every_arm_gets_one_query_including_base(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    manifest = study["manifest"]
    with client_for(study) as client:
        client.freeze(manifest)
        base = next(c for c in manifest["controls"] if c["arm"] == "base")
        placebo = next(c for c in manifest["controls"] if c["arm"] == "placebo")
        for entry, arm in ((base, "base"), (placebo, "placebo")):
            assert client.query(entry["candidate_digest"])["arm"] == arm
            assert refusal(client.query, entry["candidate_digest"]) == "ALREADY_QUERIED"


def test_decision_waits_for_every_control(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    manifest = study["manifest"]
    with client_for(study) as client:
        client.freeze(manifest)
        for entry in manifest["candidates"]:
            client.query(entry["candidate_digest"])
        assert refusal(client.decide) == "INCOMPLETE"


def test_holm_family_and_ci_level_cover_every_comparison() -> None:
    plan = make_plan(control_arms=CONTROL_ARMS)
    assert plan.hypotheses == 3 * 4
    assert plan.corrected_ci_level == pytest.approx(1 - 0.05 / 12)
    assert make_plan().corrected_ci_level == pytest.approx(1 - 0.05 / 3)


def test_family_size_changes_the_adjusted_p_value(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    with client_for(study) as client:
        result = run_all(client, study["manifest"])
    raw = [e["p_value"] for e in result["effects"]] + [c["p_value"] for c in result["controls"]]
    assert len(raw) == 12
    smallest = min(raw)
    adjusted = [e["p_adjusted"] for e in result["effects"]] + [c["p_adjusted"] for c in result["controls"]]
    assert min(adjusted) == pytest.approx(min(1.0, 12 * smallest))


# Placebo -------------------------------------------------------------------------------------


def good_control(arm: str, task: str, **changes: Any) -> decision.ControlEffect:
    values = {
        "arm": arm, "task": task, "control_accuracy": 0.5, "candidate_accuracy": 0.9, "mean_delta": 0.4,
        "ci_low": 0.3, "ci_high": 0.5, "ci_level": 0.99, "p_value": 0.001, "p_adjusted": 0.01, "clusters": 80,
        "seed_deltas": {"1": 0.4, "2": 0.4, "3": 0.4}, "control_minus_base": 0.0, **changes,
    }
    return decision.ControlEffect(**values)


CONFIRMATORY = make_plan(
    label="confirmatory", evaluator="pythia_mlp_override", checkpoint_dir="x", checkpoint_digest="y", synthesis_seeds=(1, 2, 3)
)
TASKS = ("parity", "boolean", "comparison")
EFFECTS = tuple(
    decision.TaskEffect(t, 0.5, 0.9, 0.4, 0.3, 0.5, 0.99, 0.001, 0.01, 80, 240, {"1": 0.4, "2": 0.4, "3": 0.4}) for t in TASKS
)
RETENTION = decision.RetentionGate(0.95, 0.05, 40)


def confirmatory_decision(controls: tuple[decision.ControlEffect, ...], **kwargs: Any) -> decision.PromotionDecision:
    return decision.decide(
        CONFIRMATORY, plan_digest="p", manifest_digest="m", effects=EFFECTS, retention=RETENTION, controls=controls, **kwargs
    )


ALL_CONTROLS = tuple(good_control(arm, task) for arm in COMPARATORS for task in TASKS)


def test_candidate_that_clears_every_gate_and_control_promotes() -> None:
    assert confirmatory_decision(ALL_CONTROLS).promoted


def test_placebo_cannot_promote_a_candidate_that_only_matches_it(tmp_path: Path) -> None:
    """A placebo-level candidate fails against the placebo, whatever it does against base."""
    study = make_controlled(tmp_path, candidate_p=0.5)
    with client_for(study) as client:
        result = run_all(client, study["manifest"])
    assert not result["promoted"]
    assert {"placebo:parity", "placebo:boolean", "placebo:comparison"} <= {
        r["scope"] for r in result["rejection_reasons"] if r["code"] == "CONTROL_NOT_BEATEN"
    }


def test_placebo_that_beats_base_invalidates_even_a_strong_candidate() -> None:
    """If random edits score as an effect, the items reward noise, so nothing can promote."""
    bad = tuple(
        good_control(c.arm, c.task, control_minus_base=0.2) if c.arm == "placebo" else c for c in ALL_CONTROLS
    )
    result = confirmatory_decision(bad)
    assert not result.promoted
    assert {r.code for r in result.rejection_reasons} == {"PLACEBO_BEATS_BASE"}


def test_same_gain_as_a_strong_placebo_is_rejected_through_the_custodian(tmp_path: Path) -> None:
    study = make_controlled(tmp_path, candidate_p=0.95, arm_p={"placebo": 0.95})
    with client_for(study) as client:
        result = run_all(client, study["manifest"])
    assert {"CONTROL_NOT_BEATEN", "PLACEBO_BEATS_BASE"} <= codes(result)


def test_a_placebo_can_never_be_entered_as_a_candidate(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    manifest = json.loads(json.dumps(study["manifest"]))
    placebo = next(c for c in manifest["controls"] if c["arm"] == "placebo")
    manifest["candidates"][0].update(candidate_digest=placebo["candidate_digest"], artifact_path=placebo["artifact_path"])
    with client_for(study) as client:
        assert refusal(client.freeze, manifest) == "BAD_MANIFEST"


def test_placebo_matches_norm_and_rank_and_is_not_the_candidate() -> None:
    torch.manual_seed(0)
    base = {"w": torch.randn(40, 20, dtype=torch.float64)}
    candidate = {"w": base["w"] + torch.randn(40, 3, dtype=torch.float64) @ torch.randn(3, 20, dtype=torch.float64)}
    placebo = arms.placebo_weights(base, candidate, 5)
    expected, actual = arms.edit_profile(base, candidate)["w"], arms.edit_profile(base, placebo)["w"]
    assert actual["rank"] == expected["rank"] == 3
    assert actual["norm"] == pytest.approx(expected["norm"], rel=1e-9)
    delta_a, delta_b = (candidate["w"] - base["w"]).flatten(), (placebo["w"] - base["w"]).flatten()
    assert abs(torch.dot(delta_a, delta_b)) / (delta_a.norm() * delta_b.norm()) < 0.5
    assert torch.equal(placebo["w"], arms.placebo_weights(base, candidate, 5)["w"])
    assert not torch.equal(placebo["w"], arms.placebo_weights(base, candidate, 6)["w"])


def test_identity_candidate_has_no_placebo() -> None:
    base = {"w": torch.ones(4, 4, dtype=torch.float64)}
    with pytest.raises(ValueError, match="no placebo"):
        arms.placebo_weights(base, {"w": base["w"].clone()}, 1)


def test_placebo_with_a_different_norm_or_rank_is_refused(tmp_path: Path) -> None:
    """Otherwise an oversized placebo would make any candidate look better than its control."""
    study = make_controlled(tmp_path)
    manifest = json.loads(json.dumps(study["manifest"]))
    placebo = next(c for c in manifest["controls"] if c["arm"] == "placebo")
    path = Path(placebo["artifact_path"])
    placebo["candidate_digest"] = write_candidate(
        path, synthetic(0.5, "big", profile={"site": {"norm": 50.0, "rank": 1}})
    )
    with client_for(study) as client:
        assert refusal(client.freeze, manifest) == "PLACEBO_MISMATCH"


# Identity ------------------------------------------------------------------------------------


def test_identity_edit_scores_like_base_through_the_custodian(tmp_path: Path) -> None:
    plan = make_plan(synthesis_seeds=(1,))
    plan_path = tmp_path / "plan.json"
    digest = freeze_plan(plan, plan_path)
    path = tmp_path / "identity"
    artifact = json.dumps({"p": {}, "salt": "base", "agreement": 1.0, "kl": 0.0}).encode()
    entry = {"synthesis_seed": 1, "candidate_digest": write_candidate(path, artifact), "artifact_path": str(path)}
    manifest = {"plan_digest": digest, "candidates": [entry], "exposed_operands": dict.fromkeys(plan.tasks, [])}
    with CustodianClient(plan_path, digest, tmp_path / "state") as client:
        client.freeze(manifest)
        client.query(entry["candidate_digest"])
        result = client.decide()
    for effect in result["effects"]:
        assert effect["candidate_accuracy"] == effect["base_accuracy"]
        assert effect["mean_delta"] == effect["ci_low"] == effect["ci_high"] == 0.0
        assert effect["p_value"] == 1.0
    assert {"EFFECT_CI_BELOW_MINIMUM", "NOT_SIGNIFICANT_AFTER_CORRECTION"} <= codes(result)


def test_identity_edit_scores_like_base_through_the_custodian_on_pythia(tmp_path: Path) -> None:
    """Negative control through the real evaluator and the whole protocol, on a whole-model artifact."""
    checkpoint = Path(os.environ.get("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m"))
    if not (checkpoint / "model.safetensors").exists():
        pytest.skip("needs a Pythia-14M checkpoint")
    from neural_foundry.pythia import PythiaModel
    from neural_foundry.study.plan import digest_bytes

    plan = make_plan(
        evaluator="pythia_mlp_override",
        checkpoint_dir=str(checkpoint),
        checkpoint_digest=digest_bytes((checkpoint / "model.safetensors").read_bytes()),
        synthesis_seeds=(1,),
        clusters_per_task_per_draw=20,
        holdout_draws=1,
        holdout_prefixes=("Today:",),
        retention_items=20,
    )
    plan_path = tmp_path / "plan.json"
    digest = freeze_plan(plan, plan_path)
    tensors = PythiaModel.from_pretrained_dir(checkpoint).tensors
    path = tmp_path / "identity"
    entry = {
        "synthesis_seed": 1,
        "candidate_digest": write_candidate(path, encode_candidate_artifact(tensors)),
        "artifact_path": str(path),
    }
    manifest = {"plan_digest": digest, "candidates": [entry], "exposed_operands": dict.fromkeys(plan.tasks, [])}
    with CustodianClient(plan_path, digest, tmp_path / "state") as client:
        client.freeze(manifest)
        reply = client.query(entry["candidate_digest"])
        result = client.decide()
    for effect in result["effects"]:
        assert effect["candidate_accuracy"] == effect["base_accuracy"]
        assert effect["mean_delta"] == 0.0
    assert reply["aggregate"]["retention"]["agreement"]["mean"] == 1.0
    assert abs(reply["aggregate"]["retention"]["kl"]["high"]) < 1e-9


# A missing or failed arm ---------------------------------------------------------------------


@pytest.mark.parametrize("arm", CONTROL_ARMS)
def test_control_missing_from_the_manifest_invalidates_the_study(tmp_path: Path, arm: str) -> None:
    study = make_controlled(tmp_path)
    manifest = {**study["manifest"], "controls": [c for c in study["manifest"]["controls"] if c["arm"] != arm]}
    with client_for(study) as client:
        assert refusal(client.freeze, manifest) == "CONTROL_MISSING"
        # Nothing was frozen, so nothing can be scored for that incomplete study.
        assert refusal(client.query, study["manifest"]["candidates"][0]["candidate_digest"]) == "NOT_FROZEN"
        assert refusal(client.decide) == "NOT_FROZEN"
    assert not (study["state"] / "sealed" / "holdout.json").exists()


def test_control_missing_for_one_seed_is_also_invalid(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    controls = list(study["manifest"]["controls"])
    controls.remove(next(c for c in controls if c["arm"] == "lora"))
    with client_for(study) as client:
        assert refusal(client.freeze, {**study["manifest"], "controls": controls}) == "CONTROL_MISSING"


def test_controls_cannot_be_omitted_by_dropping_the_key(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    manifest = {k: v for k, v in study["manifest"].items() if k != "controls"}
    with client_for(study) as client:
        assert refusal(client.freeze, manifest) == "CONTROL_MISSING"


def test_unplanned_control_is_refused(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    manifest = study["manifest"]
    extra = {**next(c for c in manifest["controls"] if c["arm"] == "lora"), "arm": "other"}
    with client_for(study) as client:
        assert refusal(client.freeze, {**manifest, "controls": [*manifest["controls"], extra]}) in {
            "UNPLANNED_CONTROL",
            "BAD_MANIFEST",
        }


def test_confirmatory_plan_needs_every_control() -> None:
    assert CONFIRMATORY.control_arms == CONTROL_ARMS
    with pytest.raises(PlanError, match="every control arm"):
        make_plan(
            label="confirmatory", evaluator="pythia_mlp_override", checkpoint_dir="x", checkpoint_digest="y",
            synthesis_seeds=(1, 2, 3), control_arms=("base", "placebo"),
        )
    with pytest.raises(PlanError, match="base model"):
        make_plan(control_arms=("placebo",))


def test_decision_without_a_planned_control_cannot_promote() -> None:
    result = confirmatory_decision(tuple(c for c in ALL_CONTROLS if c.arm != "lora"))
    assert [(r.code, r.scope) for r in result.rejection_reasons] == [("MISSING_CONTROL_RESULTS", "lora")]


@pytest.mark.parametrize("failing", ["lora", "placebo", "base"])
def test_failed_arm_cannot_be_dropped(tmp_path: Path, failing: str) -> None:
    study = make_controlled(tmp_path)
    manifest = json.loads(json.dumps(study["manifest"]))
    entry = next(c for c in manifest["controls"] if c["arm"] == failing)
    state_custodian_failure = failing == "base"
    if not state_custodian_failure:
        path = Path(entry["artifact_path"])
        if failing == "lora":
            entry["candidate_digest"] = write_candidate(path, b"not json")
        else:
            entry["candidate_digest"] = write_candidate(path, json.dumps({"salt": "x", "agreement": 1, "kl": 0}).encode())
    with client_for(study) as client:
        client.freeze(manifest)
    if state_custodian_failure:
        # The ledger state a crash leaves behind: started, with no result recorded.
        Custodian(study["plan_path"], study["digest"], study["state"])._append(
            "query_started", candidate_digest=entry["candidate_digest"]
        )
    with client_for(study) as client:
        for other in (*manifest["candidates"], *manifest["controls"]):
            if other is entry:
                continue
            client.query(other["candidate_digest"])
        if not state_custodian_failure:
            assert refusal(client.query, entry["candidate_digest"]) == "EVALUATION_FAILED"
            assert refusal(client.query, entry["candidate_digest"]) == "ALREADY_QUERIED"
        else:
            assert refusal(client.query, entry["candidate_digest"]) == "ALREADY_QUERIED"
        result = client.decide()
    assert not result["promoted"]
    assert "CONTROL_EVALUATION_FAILED" in codes(result)
    assert result["failed_controls"] == [f"{failing}:{entry['candidate_digest']}"]


def test_failed_arm_blocks_a_confirmatory_decision() -> None:
    result = confirmatory_decision(ALL_CONTROLS, failed_controls=("lora:sha256:x",))
    assert not result.promoted


# Tuning --------------------------------------------------------------------------------------


def test_tuning_only_sees_development_prompts_and_holdout_excludes_what_it_saw(tmp_path: Path) -> None:
    study = make_controlled(tmp_path)
    plan, data = study["plan"], study["data"]
    seen: list[arms.DevelopmentData] = []

    def spy(_model: Any, config: bl.BaselineConfig, offered: arms.DevelopmentData) -> Any:
        seen.append(offered)
        cost = bl.FitCost(1, 1.0, 1, 1, 1, None, 0.0)
        return [bl.CheckpointScore(1, 0.5, 1.0, 1.0, 0.5, 1.0)], bl.BaselineFit(config, {}, cost, 0.0)

    arms.tune_arm(None, plan, [bl.BaselineConfig(kind="sft", lr=1e-3, steps=1)], data, trial=spy)
    assert seen
    with client_for(study) as client:
        client.freeze(study["manifest"])
    audit = audit_sealed_state(study["state"])

    offered = seen[0]
    tuning_text = [p for p, _ in (*offered.splits.dev, *offered.splits.val)] + list(offered.splits.refit_guard)
    holdout = audit["items"]
    assert set(tuning_text).isdisjoint(item.prompt for item in holdout)
    for prompt in tuning_text[: len(offered.splits.dev) + len(offered.splits.val)]:
        assert prompt.split(" ", 1)[0] in plan.development_prefixes
    for item in holdout:
        if item.task != "retention":
            assert item.prompt.split(" ", 1)[0] in plan.holdout_prefixes
    tuned_bodies = {cluster.body for cluster in offered.clusters}
    assert not any(body in item.prompt for item in holdout for body in tuned_bodies)
    # Every operand tuning touched was removed from the draw.
    tuned_operands = {operand for cluster in offered.clusters for operand in cluster.operands}
    drawn = {
        operand
        for item in holdout
        if item.task == "parity"
        for operand in item.prompt.split()[2:3]
    }
    assert drawn and tuned_operands.isdisjoint(drawn)
    # And the frozen manifest, tuning record included, is what was audited.
    assert audit["manifest"]["tuning"]["lora"]["data"]["prefixes"] == sorted(plan.development_prefixes)


def test_tune_arm_refuses_holdout_prefixes() -> None:
    plan = make_plan(control_arms=CONTROL_ARMS)
    data = development(plan)
    leaked = arms.DevelopmentData(data.train, data.validation, data.guard, data.retention_tune, plan.holdout_prefixes[0], data.validation_prefix)

    def never(*_: Any) -> Any:
        raise AssertionError("tuning ran on holdout-prefixed data")

    with pytest.raises(arms.TuningLeak):
        arms.tune_arm(None, plan, [bl.BaselineConfig(kind="sft", lr=1e-3, steps=1)], leaked, trial=never)


def test_arms_module_cannot_reach_the_custodian() -> None:
    """Tuning runs in the proposer: the sealed-state code must not even be importable from its modules."""
    code = (
        "import sys; import neural_foundry.study.arms; "
        "assert 'neural_foundry.study.custodian' not in sys.modules; print('ok')"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(neural_foundry.__file__).resolve().parents[1])}
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip() == "ok"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda tuning: tuning.pop("lora"), "TUNING_NOT_RECORDED"),
        (lambda tuning: tuning["fine_tune"].update(trials=[]), "TUNING_NOT_RECORDED"),
        (lambda tuning: tuning["lora"]["data"].update(prefixes=["Today:"]), "TUNING_NOT_DEVELOPMENT"),
        (lambda tuning: tuning["lora"]["data"]["operands"]["parity"].append("777777"), "TUNING_NOT_DECLARED"),
        (lambda tuning: tuning["lora"]["data"]["retention_prompts"].append("Undeclared prompt"), "TUNING_NOT_DECLARED"),
    ],
)
def test_bad_tuning_records_are_refused(tmp_path: Path, change: Any, expected: str) -> None:
    study = make_controlled(tmp_path)
    manifest = json.loads(json.dumps(study["manifest"]))
    change(manifest["tuning"])
    with client_for(study) as client:
        assert refusal(client.freeze, manifest) == expected


def test_real_lora_tuning_runs_and_selects_on_forced_choice() -> None:
    model = build_mini_pythia_model()
    vocab = model.tokenizer.vocab
    plan = make_plan(control_arms=CONTROL_ARMS, tasks=("parity",))
    data = arms.development_data(plan, 1, rules_per_task=2, retention_prompts=4)
    if not all(c.target in vocab and c.alternative in vocab for c in data.clusters):
        pytest.skip("the mini model's vocabulary doesn't have the study's answer tokens")
    grid = [bl.BaselineConfig(kind="lora", lr=0.01, steps=2, targets="final_down_proj", rank=2, checkpoints=(1, 2))]
    selected, record = arms.tune_arm(model, plan, grid, data)
    assert selected.config.steps in (1, 2)
    assert {score["steps"] for score in record["trials"][0]["scores"]} == {1, 2}
    fit = arms.fit_arm(model, selected, 7, data)
    assert fit.config.seed == 7


def test_forced_choice_accuracy_compares_the_two_answers() -> None:
    class Tok:
        vocab = {"a": 0, "b": 1, "c": 2}

        def encode(self, _prompt: str) -> list[int]:
            return [0]

    class Model:
        tokenizer = Tok()

        def forward_with_mlp_key(self, _ids: list[int]) -> tuple[torch.Tensor, None]:
            return torch.tensor([[1.0, 0.0, 9.0]]), None

    items = [generators.Item("1", "1", "t", "p", "a", "b"), generators.Item("2", "2", "t", "p", "b", "a")]
    assert arms.forced_choice_accuracy(Model(), items) == 0.5  # Top-1 would be "c" and score 0.


# Artifacts -----------------------------------------------------------------------------------


def test_whole_model_artifact_round_trips_and_foreign_tensors_are_refused() -> None:
    base = {"a": torch.ones(2, 3, dtype=torch.float64), "b": torch.zeros(4, dtype=torch.float64)}
    decoded = decode_candidate_artifact(encode_candidate_artifact(base), base)
    assert torch.equal(decoded["a"], base["a"]) and torch.equal(decoded["b"], base["b"])
    with pytest.raises(ValueError, match="unsupported"):
        decode_candidate_artifact(encode_candidate_artifact({"c": base["a"]}), base)
    with pytest.raises(ValueError, match="unsupported"):
        decode_candidate_artifact(encode_candidate_artifact({"a": torch.ones(3, 2, dtype=torch.float64)}), base)
    with pytest.raises(ValueError, match="finite"):
        decode_candidate_artifact(encode_candidate_artifact({"b": torch.full((4,), float("nan"), dtype=torch.float64)}), base)


def test_artifact_bytes_are_the_documented_little_endian_layout() -> None:
    import struct

    data = encode_candidate_artifact({"a": torch.tensor([1.5, -2.0], dtype=torch.float64)})
    assert data.endswith(struct.pack("<2d", 1.5, -2.0))


# Pluggable items -----------------------------------------------------------------------------


class ShiftedGenerator:
    """A stand-in for the replacement items: same contract, different sentences."""

    name = "test_shifted"
    confirmatory_ready = True
    shared_operand_tasks = ("parity", "comparison")
    corrupt = False

    def draw_clusters(
        self, task: str, seed: int, n: int, *, domain: tuple[int, int], exclude: Collection[str], tag: str
    ) -> list[generators.Cluster]:
        clusters = generators.draw_clusters(task, seed, n, domain=domain, exclude=exclude, tag=tag)
        out = [generators.Cluster(c.cluster_id, c.task, c.operands, f"Shifted {c.body}", c.target, c.alternative) for c in clusters]
        if self.corrupt:
            out = [generators.Cluster(c.cluster_id, c.task, c.operands, c.body, out[0].target, c.alternative) for c in out]
        return out

    def draw_retention(self, seed: int, n: int, *, exclude: Collection[str], tag: str) -> list[generators.Item]:
        return generators.draw_retention(seed, n, exclude=exclude, tag=tag)


class CorruptGenerator(ShiftedGenerator):
    corrupt = True


@pytest.fixture(scope="module", autouse=True)
def _registered() -> None:
    for name, factory in (("test_shifted", ShiftedGenerator), ("test_corrupt", CorruptGenerator)):
        try:
            generators.register_item_generator(name, factory)
        except ValueError:
            pass


def in_process(tmp_path: Path, plan: StudyPlan) -> tuple[Custodian, dict[str, Any], str]:
    plan_path = tmp_path / "plan.json"
    digest = freeze_plan(plan, plan_path)
    path = tmp_path / "c"
    entries = [
        {"synthesis_seed": s, "candidate_digest": write_candidate(tmp_path / f"c{s}", synthetic(0.9, str(s))), "artifact_path": str(tmp_path / f"c{s}")}
        for s in plan.synthesis_seeds
    ]
    del path
    state = tmp_path / "state"
    state.mkdir()
    manifest = {"plan_digest": digest, "candidates": entries, "exposed_operands": dict.fromkeys(plan.tasks, [])}
    return Custodian(plan_path, digest, state), manifest, digest


def test_plan_chooses_the_generator_and_the_holdout_comes_from_it(tmp_path: Path) -> None:
    custodian, manifest, _ = in_process(tmp_path, make_plan(item_generator="test_shifted"))
    custodian.freeze(manifest)
    assert custodian.items is not None
    task_items = [item for item in custodian.items if item.task != "retention"]
    assert task_items and all(" Shifted " in item.prompt for item in task_items)
    # It reproduces from the seal, so the commitment is checked against the same generator.
    assert audit_sealed_state(tmp_path / "state")["commitment"]


def test_unknown_generator_is_a_plan_error() -> None:
    with pytest.raises(PlanError, match="unknown item generator"):
        make_plan(item_generator="not_a_generator")
    with pytest.raises(ValueError, match="already registered"):
        generators.register_item_generator("templated_v1", generators.TemplatedGeneratorV1)


def test_generator_that_breaks_the_balance_contract_is_refused(tmp_path: Path) -> None:
    custodian, manifest, _ = in_process(tmp_path, make_plan(item_generator="test_corrupt"))
    with pytest.raises(Refused) as refused:
        custodian.freeze(manifest)
    assert refused.value.code == "BAD_GENERATOR"
    assert custodian.items is None


def test_generator_returning_an_excluded_operand_is_refused() -> None:
    clusters = generators.draw_clusters("parity", 1, 20)
    with pytest.raises(generators.GeneratorContractError, match="excluded"):
        generators.check_clusters(clusters, "parity", 20, {clusters[0].operands[0]}, "dev")


def test_current_items_cannot_freeze_a_confirmatory_study(tmp_path: Path) -> None:
    """The templated items have a known shortcut, so the confirmatory path stays shut until they're replaced."""
    assert not generators.get_item_generator("templated_v1").confirmatory_ready
    plan = make_plan(
        label="confirmatory", evaluator="pythia_mlp_override", checkpoint_dir="x", checkpoint_digest="y", synthesis_seeds=(1, 2, 3)
    )
    custodian, manifest, _ = in_process(tmp_path, plan)
    with pytest.raises(Refused) as refused:
        custodian.freeze({**manifest, "controls": []})
    assert refused.value.code == "GENERATOR_NOT_QUALIFIED"
    assert custodian._event("freeze") is None
