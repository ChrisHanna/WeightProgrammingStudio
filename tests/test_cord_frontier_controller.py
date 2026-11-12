from __future__ import annotations

import json
import urllib.request

import pytest
from neural_foundry import cord_frontier
from neural_foundry.gym_job import GymJobError
from neural_foundry.proof_pipeline import read_document


@pytest.fixture(autouse=True)
def forbid_live_transport(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline controller tests must not dispatch provider calls")

    monkeypatch.setattr(urllib.request, "build_opener", forbidden)


def test_prepare_preserves_exact_200_requests_and_does_not_freeze_or_initialize_budget(tmp_path) -> None:
    config = cord_frontier.prepare(tmp_path)
    source = cord_frontier._source()
    assert config["core_stable_released"] is False
    assert config["request_plan"]["wire_bytes"] == 403380
    requests = config["request_plan"]["requests"]
    assert [row["scenario_id"] for row in requests] == [
        row["scenario_id"] for panel in source["public_sets"] for row in panel["rows"]]
    assert len(requests) == 200 and 316818 <= config["request_plan"]["total_reservation_microusd"] < 317218
    assert config["approval"]["provider_transfer_user_approved"] is True
    assert config["approval"]["maximum_cost_microusd"] == 500000
    assert config == read_document(tmp_path / "ready-config.json")
    assert {path.name for path in tmp_path.iterdir()} == {"ready-config.json"}
    with pytest.raises(FileExistsError):
        cord_frontier.prepare(tmp_path)


@pytest.mark.parametrize("operation", [cord_frontier.freeze, cord_frontier.execute])
def test_freeze_and_execute_require_parent_release_before_any_other_action(tmp_path, operation) -> None:
    with pytest.raises(GymJobError, match="core-stable"):
        operation(tmp_path)
    assert not list(tmp_path.iterdir())


def test_ready_config_rejects_policy_or_roster_drift(tmp_path) -> None:
    cord_frontier.prepare(tmp_path)
    path = tmp_path / "ready-config.json"
    document = json.loads(path.read_bytes())
    document["request_plan"]["requests"].reverse()
    path.write_text(json.dumps(document))
    with pytest.raises(GymJobError, match="digest"):
        cord_frontier.freeze(tmp_path, core_stable=True)


def test_exact_approved_provider_policy_cannot_be_raised(tmp_path, monkeypatch) -> None:
    document = json.loads(cord_frontier.POLICY.read_bytes())
    document["policy"]["api"]["maximum_cost_microusd"] += 1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(cord_frontier, "POLICY", path)
    with pytest.raises(GymJobError, match="explicit user approval"):
        cord_frontier.prepare(tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("operation", [cord_frontier.freeze, cord_frontier.execute])
def test_approval_cannot_be_duplicated_into_a_second_ledger(tmp_path, operation) -> None:
    cord_frontier.prepare(tmp_path)
    with pytest.raises(GymJobError, match="one shared run root"):
        operation(tmp_path, core_stable=True)
    assert {path.name for path in tmp_path.iterdir()} == {"ready-config.json"}
