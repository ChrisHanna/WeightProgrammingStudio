from __future__ import annotations

import json
from pathlib import Path

import pytest
from neural_foundry import proof_plugins
from neural_foundry.company_facts_public_plugin import (
    CatalogFixture,
    compose,
    evidence,
    project,
    supported_fact,
)
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import ExecutionResponse, GymStudioService
from neural_foundry.proof_pipeline import score


def configuration():
    path = Path(__file__).resolve().parents[1] / "protocols" / "company-facts-public-fixture-v1.json"
    return json.loads(path.read_bytes())


def test_second_gym_runs_unchanged_generic_scorer_with_no_training_or_api(tmp_path, monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("company fixture must not train, load models, or call an API")

    for name in ("training_model", "automodel_trainer", "openai_provider"):
        monkeypatch.setattr(proof_plugins, name, forbidden)
    config = configuration()
    config["candidate"] = {"role": "second-gym-arbitrary-role", "method": "arbitrary-catalog-method"}
    spec, registry = compose(config)
    result = score(spec, role=config["candidate"]["role"], root=tmp_path, registry=registry)
    assert result["gate_passed"] and result["gates_complete"]
    assert result["scores"]["company-public"]["metrics"] == {"supported_fact": 1.0}
    assert result["model_identity"]["fixture"] and result["model_identity"]["new_api_calls"] == 0
    assert not result["model_identity"]["model_quality_accepted"]
    assert result["scores"]["company-public"]["responses"][-1]["response"] == {
        "value": "not listed", "source": "none"}


def test_company_source_cannot_materialize_training_or_private_splits() -> None:
    spec, registry = compose(configuration())
    task = registry.resolve("gym", spec["plugins"]["gym"])(spec["gym"])
    service = GymStudioService()
    task.bind(service)
    source = task.job.scenario_sources[0]
    provider = service.components.resolve("scenario_provider", source.provider_id,
                                          source.provider_version, source.provider_digest)
    assert len(provider(split="validation")) == 4
    for split in ("train", "holdout"):
        with pytest.raises(GymJobError, match="public-validation-only"):
            provider(split=split)


def test_company_plugin_rejects_schema_drift_and_foreign_receipts() -> None:
    config = configuration()
    model = CatalogFixture(config["catalog"])
    interaction = project(config["questions"][1], None)
    response = model.infer(interaction)
    evidence(model=model, interaction=interaction, result=response)
    assert response.response["value"] == "marine sensors"
    altered = {**response.response, "extra": "schema violation"}
    assert supported_fact(reference=response.response, response=altered)["supported_fact"] == 0.0
    with pytest.raises(GymJobError, match="receipt differs"):
        evidence(model=model, interaction=interaction,
                 result=ExecutionResponse({"value": "mining", "source": "none"}, response.interaction))
