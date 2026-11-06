"""Controlled-study harness: sealing, single queries, plan freezing, statistics and decisions.

The custodian tests spawn a real separate process. They use the synthetic evaluator, so
they prove harness behaviour and say nothing about any edit method.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import neural_foundry
from neural_foundry.study import analysis, decision, generators
from neural_foundry.study.client import CustodianClient, CustodianRefused, declare_exposure, write_candidate
from neural_foundry.study.custodian import audit_sealed_state
from neural_foundry.study.plan import (
    PlanError,
    PlanTamperedError,
    StudyPlan,
    Thresholds,
    freeze_plan,
    load_frozen_plan,
)

THRESHOLDS = Thresholds(
    min_effect=0.1,
    min_accuracy=0.6,
    min_seed_pass_fraction=1.0,
    min_retention_agreement=0.8,
    max_retention_kl=0.5,
)


def make_plan(**changes: Any) -> StudyPlan:
    values: dict[str, Any] = {
        "study_id": "test-study",
        "label": "harness_test",
        "hypothesis": "synthetic candidate beats base",
        "tasks": ("parity", "boolean", "comparison"),
        "clusters_per_task_per_draw": 40,
        "holdout_draws": 2,
        "retention_items": 40,
        "development_prefixes": ("Task:", "Rule:"),
        "holdout_prefixes": ("Today:", "Check:", "Quiz:"),
        "number_domain": (0, 9999),
        "synthesis_seeds": (11, 22),
        "analysis_seed": 7,
        "primary_metric": "forced_choice_accuracy",
        "secondary_metrics": ("top1_accuracy",),
        "alpha": 0.05,
        "correction": "holm",
        "bootstrap_resamples": 300,
        "permutation_resamples": 300,
        "thresholds": THRESHOLDS,
        "evaluator": "synthetic_bernoulli",
        **changes,
    }
    return StudyPlan(**values)


def development_exposure(plan: StudyPlan) -> tuple[dict[str, list[str]], list[generators.Cluster]]:
    clusters = [
        cluster
        for task in plan.tasks
        for cluster in generators.draw_clusters(task, 1, 10, domain=plan.number_domain)
    ]
    return declare_exposure(clusters, tasks=plan.tasks), clusters


def make_study(tmp_path: Path, plan: StudyPlan | None = None, p: float = 0.95) -> dict[str, Any]:
    plan = plan or make_plan()
    plan_path = tmp_path / "plan.json"
    digest = freeze_plan(plan, plan_path)
    exposed, clusters = development_exposure(plan)
    candidates = []
    for seed in plan.synthesis_seeds:
        artifact = json.dumps(
            {"p": dict.fromkeys(plan.tasks, p), "salt": f"seed{seed}", "agreement": 0.99, "kl": 0.01}
        ).encode()
        path = tmp_path / f"candidate_{seed}.json"
        candidates.append(
            {"synthesis_seed": seed, "candidate_digest": write_candidate(path, artifact), "artifact_path": str(path)}
        )
    manifest = {"plan_digest": digest, "candidates": candidates, "exposed_operands": exposed}
    return {
        "plan": plan,
        "plan_path": plan_path,
        "digest": digest,
        "manifest": manifest,
        "state": tmp_path / "state",
        "clusters": clusters,
    }


def client_for(study: dict[str, Any]) -> CustodianClient:
    return CustodianClient(study["plan_path"], study["digest"], study["state"])


# Generators ---------------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["parity", "boolean", "comparison"])
def test_generators_are_seeded_balanced_and_large(task: str) -> None:
    first = generators.draw_clusters(task, 3, 300)
    assert first == generators.draw_clusters(task, 3, 300)
    assert first != generators.draw_clusters(task, 4, 300)
    assert len({c.body for c in first}) == 300
    positive = generators.TOKENS[task][0]
    assert sum(c.target == positive for c in first) == 150


@pytest.mark.parametrize("task", ["parity", "boolean", "comparison"])
def test_labels_are_correct(task: str) -> None:
    for c in generators.draw_clusters(task, 5, 200):
        if task == "parity":
            assert c.target == ("Ġodd" if int(c.operands[0]) % 2 else "Ġeven")
        elif task == "comparison":
            assert c.target == ("Ġyes" if int(c.operands[0]) > int(c.operands[1]) else "Ġno")
        else:
            # Python's own and/or/not have the precedence the generator claims.
            python = c.operands[0].replace("AND", "and").replace("OR", "or").replace("NOT", "not")
            assert c.target == ("ĠTrue" if eval(python) else "ĠFalse")  # noqa: S307


@pytest.mark.parametrize("task", ["parity", "boolean", "comparison"])
def test_exclusion_makes_draws_disjoint(task: str) -> None:
    development = generators.draw_clusters(task, 1, 200)
    excluded = {operand for c in development for operand in c.operands}
    holdout = generators.draw_clusters(task, 2, 200, exclude=excluded)
    assert not excluded & {operand for c in holdout for operand in c.operands}


def test_comparison_uses_each_number_once_per_draw() -> None:
    operands = [operand for c in generators.draw_clusters("comparison", 1, 300) for operand in c.operands]
    assert len(set(operands)) == len(operands)


def test_exhausted_domain_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        generators.draw_clusters("parity", 1, 20, domain=(0, 9))


# Plan ---------------------------------------------------------------------------------------


def test_changing_the_frozen_plan_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    digest = freeze_plan(make_plan(), path)
    assert load_frozen_plan(path, digest) == make_plan()
    doc = json.loads(path.read_text())
    doc["thresholds"]["min_effect"] = 0.0
    path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    with pytest.raises(PlanTamperedError):
        load_frozen_plan(path, digest)


def test_whitespace_change_to_frozen_plan_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    digest = freeze_plan(make_plan(), path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PlanTamperedError):
        load_frozen_plan(path, digest)


def test_freeze_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    freeze_plan(make_plan(), path)
    with pytest.raises(FileExistsError):
        freeze_plan(make_plan(alpha=0.5 - 0.01), path)


@pytest.mark.parametrize(
    "changes",
    [
        {"holdout_prefixes": ("Task:",)},
        {"correction": "none"},
        {"evaluator": "os:system"},
        {"clusters_per_task_per_draw": 10},
        {"clusters_per_task_per_draw": 41},
        {"synthesis_seeds": (1, 1)},
        {"schema_version": "nf.study.plan.v0"},
        {"label": "confirmatory"},  # with the synthetic evaluator
        {"evaluator": "pythia_mlp_override"},  # with no checkpoint
    ],
)
def test_bad_plans_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(PlanError):
        make_plan(**changes)


def test_confirmatory_plan_needs_hundreds_of_items_and_three_seeds() -> None:
    real = {"label": "confirmatory", "evaluator": "pythia_mlp_override", "checkpoint_dir": "x", "checkpoint_digest": "y"}
    with pytest.raises(PlanError, match="items per task"):
        make_plan(**real, clusters_per_task_per_draw=20, holdout_draws=1, holdout_prefixes=("Today:",))
    with pytest.raises(PlanError, match="synthesis seeds"):
        make_plan(**real)
    assert make_plan(**real, synthesis_seeds=(1, 2, 3)).items_per_task == 240


def test_unknown_plan_field_is_rejected() -> None:
    doc = make_plan().to_dict()
    doc["extra"] = 1
    with pytest.raises(PlanError, match="unknown"):
        StudyPlan.from_dict(doc)


# Statistics ---------------------------------------------------------------------------------


def test_duplicating_items_within_clusters_adds_no_precision() -> None:
    """Cluster means first: this is why the unit of resampling is the expression."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 5
    clustered = analysis.cluster_bootstrap_ci(values, [0] * len(values), level=0.95, resamples=4000, seed=1)
    naive = analysis.cluster_bootstrap_ci(values * 10, [0] * (len(values) * 10), level=0.95, resamples=4000, seed=1)
    assert clustered.mean == pytest.approx(naive.mean)
    assert clustered.high - clustered.low > 2.5 * (naive.high - naive.low)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1 * (i % 7) for i in range(100)]
    strata = [i % 2 for i in range(100)]
    a = analysis.cluster_bootstrap_ci(values, strata, level=0.9, resamples=500, seed=3)
    assert a == analysis.cluster_bootstrap_ci(values, strata, level=0.9, resamples=500, seed=3)
    assert a.low < a.mean < a.high
    assert a.clusters == 100


def test_sign_flip_p_value_separates_signal_from_noise() -> None:
    assert analysis.sign_flip_p_value([0.5] * 100, resamples=999, seed=1) < 0.01
    assert analysis.sign_flip_p_value([1.0, -1.0] * 50, resamples=999, seed=1) > 0.3
    # All ties carry no evidence and must never look significant.
    assert analysis.sign_flip_p_value([0.0] * 50, resamples=99, seed=1) == 1.0


def test_holm_matches_a_worked_example() -> None:
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert analysis.holm_adjust([0.9, 0.9]) == pytest.approx([1.0, 1.0])


# Decision -----------------------------------------------------------------------------------


def good_effect(task: str, **changes: Any) -> decision.TaskEffect:
    values = {
        "task": task,
        "base_accuracy": 0.5,
        "candidate_accuracy": 0.9,
        "mean_delta": 0.4,
        "ci_low": 0.3,
        "ci_high": 0.5,
        "ci_level": 0.98,
        "p_value": 0.001,
        "p_adjusted": 0.003,
        "clusters": 80,
        "items": 240,
        "seed_deltas": {"11": 0.4, "22": 0.4},
        **changes,
    }
    return decision.TaskEffect(**values)


GOOD_RETENTION = decision.RetentionGate(0.95, 0.05, 40)
CONFIRMATORY = make_plan(
    label="confirmatory", evaluator="pythia_mlp_override", checkpoint_dir="x", checkpoint_digest="y", synthesis_seeds=(1, 2, 3)
)


def run_decide(plan: StudyPlan, effects: tuple[decision.TaskEffect, ...], retention: Any = GOOD_RETENTION) -> Any:
    return decision.decide(plan, plan_digest="p", manifest_digest="m", effects=effects, retention=retention)


ALL_GOOD = tuple(good_effect(task) for task in ("parity", "boolean", "comparison"))


def test_confirmatory_study_that_clears_every_gate_promotes() -> None:
    result = run_decide(CONFIRMATORY, ALL_GOOD)
    assert result.promoted and not result.rejection_reasons


def test_non_confirmatory_label_can_never_promote() -> None:
    result = run_decide(make_plan(), ALL_GOOD)
    assert not result.promoted
    assert [r.code for r in result.rejection_reasons] == ["NON_CONFIRMATORY_LABEL"]


def test_every_failed_gate_is_recorded() -> None:
    bad = good_effect(
        "parity", ci_low=-0.1, p_adjusted=0.5, candidate_accuracy=0.4, seed_deltas={"11": 0.4, "22": -0.2}
    )
    result = run_decide(CONFIRMATORY, (bad, *ALL_GOOD[1:]), decision.RetentionGate(0.5, 2.0, 40))
    assert not result.promoted
    assert {(r.code, r.scope) for r in result.rejection_reasons} == {
        ("EFFECT_CI_BELOW_MINIMUM", "parity"),
        ("NOT_SIGNIFICANT_AFTER_CORRECTION", "parity"),
        ("ACCURACY_BELOW_MINIMUM", "parity"),
        ("INCONSISTENT_ACROSS_SEEDS", "parity"),
        ("RETENTION_AGREEMENT_BELOW_MINIMUM", "retention"),
        ("RETENTION_KL_ABOVE_MAXIMUM", "retention"),
    }


def test_missing_results_reject() -> None:
    codes = {r.code for r in run_decide(CONFIRMATORY, ALL_GOOD[:2], None).rejection_reasons}
    assert codes == {"MISSING_TASK_RESULTS", "MISSING_RETENTION_RESULTS"}


# Custodian protocol -------------------------------------------------------------------------


def test_repeated_query_is_refused(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        digest = study["manifest"]["candidates"][0]["candidate_digest"]
        assert "tasks" in client.query(digest)["aggregate"]
        with pytest.raises(CustodianRefused) as refused:
            client.query(digest)
        assert refused.value.code == "ALREADY_QUERIED"


def test_repeated_query_is_refused_after_custodian_restart(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    digest = study["manifest"]["candidates"][0]["candidate_digest"]
    with client_for(study) as client:
        client.freeze(study["manifest"])
        client.query(digest)
    with client_for(study) as client:
        with pytest.raises(CustodianRefused) as refused:
            client.query(digest)
        assert refused.value.code == "ALREADY_QUERIED"


def test_only_frozen_candidates_are_answered(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    digest = study["manifest"]["candidates"][0]["candidate_digest"]
    with client_for(study) as client:
        with pytest.raises(CustodianRefused) as refused:
            client.query(digest)
        assert refused.value.code == "NOT_FROZEN"
        client.freeze(study["manifest"])
        with pytest.raises(CustodianRefused) as refused:
            client.query("sha256:" + "f" * 64)
        assert refused.value.code == "UNKNOWN_CANDIDATE"


def test_second_manifest_is_refused_so_queries_cant_become_a_search(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        with pytest.raises(CustodianRefused) as refused:
            client.freeze(study["manifest"])
        assert refused.value.code == "ALREADY_FROZEN"


def test_artifact_changed_after_freeze_cannot_reach_evaluation(tmp_path: Path) -> None:
    study = make_study(tmp_path, p=0.5)
    entry = study["manifest"]["candidates"][0]
    with client_for(study) as client:
        client.freeze(study["manifest"])
        before = client.send_raw({"op": "query", "candidate_digest": entry["candidate_digest"]})
        Path(entry["artifact_path"]).write_text(json.dumps({"p": {}, "salt": "swapped", "agreement": 1, "kl": 0}))
        second = study["manifest"]["candidates"][1]
        Path(second["artifact_path"]).write_text("not json")
        # The custodian evaluates its own sealed copy, so the corrupted file doesn't matter.
        assert "tasks" in client.query(second["candidate_digest"])["aggregate"]
        assert before["candidate_digest"] == entry["candidate_digest"]


def test_manifest_with_wrong_digest_or_seeds_is_refused(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    manifest = json.loads(json.dumps(study["manifest"]))
    with client_for(study) as client:
        manifest["candidates"][0]["candidate_digest"] = "sha256:" + "e" * 64
        with pytest.raises(CustodianRefused) as refused:
            client.freeze(manifest)
        assert refused.value.code == "DIGEST_MISMATCH"
        with pytest.raises(CustodianRefused) as refused:
            client.freeze({**study["manifest"], "candidates": study["manifest"]["candidates"][:1]})
        assert refused.value.code == "SEED_MISMATCH"
        with pytest.raises(CustodianRefused) as refused:
            client.freeze({**study["manifest"], "plan_digest": "sha256:" + "d" * 64})
        assert refused.value.code == "WRONG_PLAN"
        # A refused manifest uses nothing up: the valid one still freezes.
        client.freeze(study["manifest"])


def test_custodian_refuses_tampered_plan_at_startup(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    study["plan_path"].write_bytes(study["plan_path"].read_bytes().replace(b'"alpha": 0.05', b'"alpha": 0.2'))
    with pytest.raises(CustodianRefused) as refused:
        client_for(study)
    assert refused.value.code == "PLAN_TAMPERED"


def test_custodian_refuses_plan_changed_mid_study(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    digest = study["manifest"]["candidates"][0]["candidate_digest"]
    original = study["plan_path"].read_bytes()
    with client_for(study) as client:
        client.freeze(study["manifest"])
        study["plan_path"].write_bytes(original.replace(b'"min_effect": 0.1', b'"min_effect": 0.0'))
        for call in (lambda: client.query(digest), client.decide):
            with pytest.raises(CustodianRefused) as refused:
                call()
            assert refused.value.code == "PLAN_TAMPERED"
        # A refused query must not spend the candidate's only look.
        study["plan_path"].write_bytes(original)
        assert "tasks" in client.query(digest)["aggregate"]


def test_state_directory_stays_bound_to_its_first_plan(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    with client_for(study):
        pass
    other = tmp_path / "other_plan.json"
    other_digest = freeze_plan(make_plan(alpha=0.2), other)
    with pytest.raises(CustodianRefused) as refused:
        CustodianClient(other, other_digest, study["state"])
    assert refused.value.code == "PLAN_CHANGED"


def test_broken_ledger_is_refused(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
    ledger = study["state"] / "ledger.jsonl"
    lines = ledger.read_bytes().splitlines()
    ledger.write_bytes(b"\n".join([lines[0].replace(b"registered", b"registerex"), *lines[1:]]) + b"\n")
    with pytest.raises(CustodianRefused) as refused:
        client_for(study)
    assert refused.value.code == "LEDGER_CORRUPT"


def test_failed_evaluation_spends_the_query_and_rejects_the_study(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    bad = tmp_path / "bad.json"
    entry = study["manifest"]["candidates"][0]
    entry["candidate_digest"] = write_candidate(bad, b"not json")
    entry["artifact_path"] = str(bad)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        for expected in ("EVALUATION_FAILED", "ALREADY_QUERIED"):
            with pytest.raises(CustodianRefused) as refused:
                client.query(entry["candidate_digest"])
            assert refused.value.code == expected
        client.query(study["manifest"]["candidates"][1]["candidate_digest"])
        result = client.decide()
    assert not result["promoted"]
    assert "CANDIDATE_EVALUATION_FAILED" in {r["code"] for r in result["rejection_reasons"]}


def test_crash_during_query_spends_it_and_rejects_the_study(tmp_path: Path) -> None:
    from neural_foundry.study.custodian import Custodian

    study = make_study(tmp_path)
    crashed, other = (c["candidate_digest"] for c in study["manifest"]["candidates"])
    with client_for(study) as client:
        client.freeze(study["manifest"])
    # The ledger state a crash leaves behind: started, with no result recorded.
    Custodian(study["plan_path"], study["digest"], study["state"])._append("query_started", candidate_digest=crashed)
    with client_for(study) as client:
        with pytest.raises(CustodianRefused) as refused:
            client.query(crashed)
        assert refused.value.code == "ALREADY_QUERIED"
        client.query(other)
        result = client.decide()
    assert not result["promoted"]
    assert result["failed_candidates"] == [crashed]


def test_decide_waits_for_every_candidate_and_is_stable(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    candidates = study["manifest"]["candidates"]
    with client_for(study) as client:
        client.freeze(study["manifest"])
        client.query(candidates[0]["candidate_digest"])
        with pytest.raises(CustodianRefused) as refused:
            client.decide()
        assert refused.value.code == "INCOMPLETE"
        client.query(candidates[1]["candidate_digest"])
        assert client.decide() == client.decide()


def test_synthetic_effect_is_recovered_and_null_is_rejected(tmp_path: Path) -> None:
    """Known-answer test through the whole protocol: strong and null candidates give opposite gates."""
    outcomes = {}
    for name, p in (("strong", 0.95), ("null", 0.5)):
        root = tmp_path / name
        root.mkdir()
        study = make_study(root, p=p)
        with client_for(study) as client:
            client.freeze(study["manifest"])
            for candidate in study["manifest"]["candidates"]:
                client.query(candidate["candidate_digest"])
            outcomes[name] = client.decide()
    strong = outcomes["strong"]
    assert all(effect["ci_low"] > 0.1 for effect in strong["effects"])
    # Only the label stops the strong candidate, which shows that the label gate has teeth.
    assert [r["code"] for r in strong["rejection_reasons"]] == ["NON_CONFIRMATORY_LABEL"]
    null_codes = {r["code"] for r in outcomes["null"]["rejection_reasons"]}
    assert {"EFFECT_CI_BELOW_MINIMUM", "ACCURACY_BELOW_MINIMUM"} <= null_codes


def test_holdout_is_disjoint_from_declared_exposure_and_fresh_each_study(tmp_path: Path) -> None:
    commitments = []
    prompts = []
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        study = make_study(root)
        with client_for(study) as client:
            commitments.append(client.freeze(study["manifest"])["holdout_commitment"])
        audit = audit_sealed_state(study["state"])
        assert audit["commitment"] == commitments[-1]
        items = audit["items"]
        task_items = [item for item in items if item.task != "retention"]
        assert len(task_items) == 40 * 2 * 3 * 3
        development_bodies = {cluster.body for cluster in study["clusters"]}
        assert not any(body in item.prompt for item in task_items for body in development_bodies)
        assert not {p.split()[0] for p in (i.prompt for i in task_items)} & set(study["plan"].development_prefixes)
        prompts.append({item.prompt for item in task_items})
    # Same plan and manifest shape, but entropy after freeze means a different holdout.
    assert commitments[0] != commitments[1]
    assert prompts[0] != prompts[1]


def test_a_guessed_operand_declared_in_the_manifest_is_never_scored(tmp_path: Path) -> None:
    """The old gym's flaw: an agent wrote holdout expressions itself. Declared guesses are now excluded."""
    study = make_study(tmp_path, make_plan(number_domain=(0, 400), clusters_per_task_per_draw=20, holdout_draws=1))
    guesses = [str(v) for v in range(0, 200)]
    study["manifest"]["exposed_operands"]["parity"] += guesses
    with client_for(study) as client:
        client.freeze(study["manifest"])
    used = {
        item.prompt.split()[2]
        for item in audit_sealed_state(study["state"])["items"]
        if item.task == "parity"
    }
    assert used and not used & set(guesses)


# Sealing: the proposer runs as its own process, and the auditor checks it only afterwards ----


PROPOSER = textwrap.dedent(
    """
    import gc, json, sys
    from pathlib import Path
    from neural_foundry.study.client import CustodianClient

    plan_path, digest, state, manifest_path, out = sys.argv[1:6]
    manifest = json.loads(Path(manifest_path).read_text())
    planted = manifest.pop("planted_control")
    keep_alive = [planted]

    with CustodianClient(plan_path, digest, state) as client:
        client.freeze(manifest)
        replies = [client.query(c["candidate_digest"]) for c in manifest["candidates"]]
        replies.append(client.decide())
        for candidate in manifest["candidates"]:
            try:
                client.query(candidate["candidate_digest"])
            except Exception as exc:
                replies.append(repr(exc))
        transcript = b"".join(client.transcript).decode("utf-8")

    # Every string reachable from any container in this process, including sys.modules.
    gc.collect()
    found = {transcript}
    def walk(value):
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, bytes):
            found.add(value.decode("utf-8", "replace"))
    for container in gc.get_objects():
        if isinstance(container, dict):
            for key, value in list(container.items()):
                walk(key); walk(value)
        elif isinstance(container, (list, tuple, set, frozenset)):
            for value in list(container):
                walk(value)
    modules = sorted(m for m in sys.modules if m.startswith("neural_foundry"))
    Path(out).write_text(json.dumps({"strings": sorted(found), "transcript": transcript,
                                      "replies": replies, "modules": modules}))
    """
)


def test_holdout_items_never_appear_in_the_proposer_process(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    control = "Task: Number 123456 planted positive control"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({**study["manifest"], "planted_control": control}))
    script = tmp_path / "proposer.py"
    script.write_text(PROPOSER)
    out = tmp_path / "proposer_dump.json"
    env = {**os.environ, "PYTHONPATH": str(Path(neural_foundry.__file__).resolve().parents[1])}
    subprocess.run(
        [sys.executable, str(script), str(study["plan_path"]), study["digest"], str(study["state"]), str(manifest_path), str(out)],
        check=True,
        env=env,
        timeout=600,
    )
    dump = json.loads(out.read_text())
    blob = "\0".join(dump["strings"])

    # Positive control: the scan can see a string that really is in the process.
    assert control in blob

    # Only now does the auditor open the seal. Check every prompt and operand expression.
    audit = audit_sealed_state(study["state"])
    items = audit["items"]
    assert len(items) > 500
    leaked = [item.prompt for item in items if item.prompt in blob]
    assert leaked == []
    assert dump["transcript"]
    for item in items:
        assert item.prompt not in dump["transcript"]
    expressions = {item.prompt.split(" ", 1)[1] for item in items if item.task != "retention"}
    assert not [body for body in expressions if body in blob]
    # Seeds would regenerate everything, so they must stay sealed too.
    sealed = json.loads((study["state"] / "sealed" / "holdout.json").read_text())
    for seed in [*sealed["seeds"], sealed["retention_seed"]]:
        assert str(seed) not in blob
    # And the proposer process never loaded the server-side module.
    assert "neural_foundry.study.custodian" not in dump["modules"]
    assert any("ALREADY_QUERIED" in reply for reply in dump["replies"] if isinstance(reply, str))


def test_egress_filter_blocks_a_prompt_in_a_response(tmp_path: Path) -> None:
    from neural_foundry.study.custodian import Custodian

    study = make_study(tmp_path)
    study["state"].mkdir()
    custodian = Custodian(study["plan_path"], study["digest"], study["state"])
    custodian.prompts = ["Today: Number 42 has parity"]
    assert b"EGRESS_BLOCKED" in custodian.egress({"ok": True, "result": "Today: Number 42 has parity"})
    assert b"EGRESS_BLOCKED" not in custodian.egress({"ok": True, "result": 0.5})


def test_sealed_files_are_private_where_the_filesystem_supports_it(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permissions only")
    study = make_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
    holdout = study["state"] / "sealed" / "holdout.json"
    assert holdout.stat().st_mode & 0o077 == 0
    assert (study["state"] / "sealed").stat().st_mode & 0o077 == 0


def test_plan_helpers_keep_the_plan_immutable() -> None:
    plan = make_plan()
    assert replace(plan, alpha=0.01).to_bytes() != plan.to_bytes()
    assert StudyPlan.from_dict(json.loads(plan.to_bytes())) == plan


# Real checkpoint: skipped, and reported as skipped, without one ------------------------------


def _pythia_plan(checkpoint: Path) -> StudyPlan:
    from neural_foundry.study.plan import digest_bytes

    return make_plan(
        evaluator="pythia_mlp_override",
        checkpoint_dir=str(checkpoint),
        checkpoint_digest=digest_bytes((checkpoint / "model.safetensors").read_bytes()),
    )


def test_identity_artifact_scores_exactly_like_base_on_pythia() -> None:
    """Negative control for the real evaluator: an unedited weight must change nothing."""
    checkpoint = Path(os.environ.get("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m"))
    if not (checkpoint / "model.safetensors").exists():
        pytest.skip("needs a Pythia-14M checkpoint")
    from neural_foundry.study.evaluators import PythiaEvaluator, encode_candidate_artifact

    evaluator = PythiaEvaluator(_pythia_plan(checkpoint))
    site = "gpt_neox.layers.5.mlp.dense_4h_to_h.weight"
    artifact = encode_candidate_artifact({site: evaluator._model.tensors[site]})
    items = generators.render(generators.draw_clusters("parity", 9, 20), ("Today:",))
    assert evaluator.score(items, artifact) == evaluator.score(items, None)
    retention = evaluator.retention(generators.draw_retention(9, 20), artifact)
    assert all(row["agreement"] == 1.0 for row in retention.values())
    assert all(abs(row["kl"]) < 1e-9 for row in retention.values())
