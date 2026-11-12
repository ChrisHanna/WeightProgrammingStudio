"""Immutable project assets and bounded, explicitly offline product workflows."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from .models import actor_name, digest, document, encode, fail, identifier, keys, validate_spec
from .store import json_value, one

if TYPE_CHECKING:
    from psycopg import Connection

    from .bridges import TrustedRuntimeProfile
    from .service import ControlPlaneService

MAX_EXAMPLES = 1000
POLICY = "json_exact_v1"
PARAMETERS: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
    "steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
    "rank": {"type": "integer", "minimum": 1, "maximum": 4096},
    "alpha": {"type": "number", "minimum": 0, "maximum": 1000000},
    "batch_size": {"type": "integer", "minimum": 1, "maximum": 4096},
    "strength": {"type": "number", "minimum": 0, "maximum": 1},
    "seed": {"type": "integer", "minimum": 0, "maximum": 1000000},
    "weight": {"type": "number", "minimum": 0, "maximum": 1},
}
METHOD_PARAMETERS = {
    "lora": ("learning_rate", "epochs", "steps", "rank", "alpha", "batch_size", "seed"),
    "sft": ("learning_rate", "epochs", "steps", "batch_size", "seed"),
    "weight_edit": ("strength", "seed"),
    "merge": ("weight",),
}


def recipe_methods() -> list[dict[str, Any]]:
    return [{"id": method, "name": method.replace("_", " ").upper(),
             "description": "Saved plan only; execution requires a separately qualified model-specific runtime.",
             "parameters_schema": {"type": "object", "additionalProperties": False,
                                   "properties": {key: PARAMETERS[key] for key in fields}},
             **({"source_model_asset_ids_schema": {
                 "type": "array", "minItems": 2, "maxItems": 16, "uniqueItems": True,
                 "items": {"type": "string"}, "asset_kind": "model", "project_owned": True,
                 "include_model_asset_id": True,
                 "description": "Required merge sources: select two to sixteen distinct project models, including the base model."}}
                if method == "merge" else {})}
            for method, fields in METHOD_PARAMETERS.items()]


def text(value: Any, field: str, maximum: int = 128, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        fail("invalid_document", f"{field} must be bounded text.")
    return value


def score_json(expected: dict[str, Any], response: str) -> dict[str, Any]:
    try:
        actual = json.loads(response, parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
        valid = isinstance(actual, dict)
        exact = valid and encode(actual) == encode(expected)
    except (ValueError, RecursionError):
        valid, exact = False, False
    return {"valid_json": valid, "exact_match": exact}


def validate_dataset(value: Any) -> dict[str, Any]:
    value = document(value)
    keys(value, {"format", "examples"}, {"format", "examples"})
    assert isinstance(value, dict)
    examples = value["examples"]
    if value["format"] != "json_examples_v1" or not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
        fail("invalid_dataset", "Use json_examples_v1 with one to 1000 examples.")
    ids: set[str] = set()
    splits: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in examples:
        keys(row, {"id", "split", "input", "expected"}, {"id", "split", "input", "expected"})
        eid = identifier(row["id"], "example ID")
        if eid in ids or row["split"] not in ("train", "validation"):
            fail("invalid_dataset", "Example IDs must be unique and splits train or validation.")
        ids.add(eid)
        source = text(row["input"], "example input", 8192)
        if not isinstance(row["expected"], dict):
            fail("invalid_dataset", "Expected output must be a JSON object.")
        splits[row["split"]].add(" ".join(unicodedata.normalize("NFC", source).split()))
    if splits["train"] & splits["validation"]:
        fail("dataset_overlap", "Normalized inputs overlap between train and validation.")
    return value


def validate_gym(value: Any) -> dict[str, Any]:
    value = document(value)
    keys(value, {"template", "config", "dataset_asset_id"}, {"template", "config"})
    assert isinstance(value, dict)
    template, config = value["template"], value["config"]
    if template not in ("json_exact", "stateful_counter", "tool_workflow"):
        fail("invalid_gym", "Unknown trusted gym template.")
    keys(config, {"target", "horizon"} if template == "stateful_counter" else set(), set())
    for field, high in (("target", 32), ("horizon", 64)):
        if field in config and (type(config[field]) is not int or not 1 <= config[field] <= high):
            fail("invalid_gym", "Counter target/horizon exceeds its supported bound.")
    if template == "json_exact":
        identifier(value.get("dataset_asset_id"), "dataset asset ID")
    elif "dataset_asset_id" in value:
        fail("invalid_gym", "Software fixtures do not consume datasets.")
    return value


def templates() -> list[dict[str, Any]]:
    return [
        {"id": "json_exact", "name": "Owned JSON task", "description": "Exact JSON object evaluation on owned validation data.",
         "interaction": "response", "scope": "owned-data-task-not-model-evidence",
         "config_schema": {"type": "object", "additionalProperties": False}},
        {"id": "stateful_counter", "name": "Stateful counter", "description": "Deterministic CPU counter; no renderer or model.",
         "interaction": "actions", "scope": "software-only",
         "config_schema": {"type": "object", "additionalProperties": False, "properties": {
             "target": {"type": "integer", "minimum": 1, "maximum": 32, "default": 2},
             "horizon": {"type": "integer", "minimum": 1, "maximum": 64, "default": 10}}}},
        {"id": "tool_workflow", "name": "Tool workflow", "description": "In-memory lookup/finish fixture, no real tools.",
         "interaction": "actions", "scope": "software-only",
         "config_schema": {"type": "object", "additionalProperties": False}},
    ]


class ProductWorkspace:
    def __init__(self, service: ControlPlaneService) -> None:
        self.service = service

    def _project(self, conn: Connection[dict[str, Any]], project_id: str) -> dict[str, Any]:
        identifier(project_id, "project ID")
        row = conn.execute("SELECT id,name,description,created_at FROM cp_projects WHERE id=%s FOR UPDATE",
                           (project_id,)).fetchone()
        if row is None:
            fail("project_not_found", "Project does not exist.", 404)
        return row

    def _asset(self, conn: Connection[dict[str, Any]], project_id: str, asset_id: Any,
               kind: str | None = None) -> dict[str, Any]:
        identifier(asset_id, "asset ID")
        row = conn.execute("SELECT id,project_id,kind,name,version,parent_asset_id,artifact_id,created_at "
                           "FROM cp_assets WHERE id=%s AND project_id=%s", (asset_id, project_id)).fetchone()
        if row is None or (kind is not None and row["kind"] != kind):
            fail("asset_not_found", "Referenced project asset does not exist or has the wrong kind.", 404)
        return {**row, "document": self.service.artifacts.read(row["artifact_id"])}

    def capabilities(self) -> dict[str, Any]:
        profiles = self.service.runtime_profiles.values()
        training = any(p.qualified and p.registry is not None and (p.product_gym_bindings or p.descriptor)
                       and {"model_training", "model_export", "model_identity", "model_scoring"}
                       <= p.deployment_capabilities for p in profiles)
        reason = "Operator must install a qualified JSON gym registry/runtime with train/export/identity/score capabilities and exact saved-input proof bindings."
        return {
            "evaluate_saved": {"enabled": True, "reason": "Offline analysis of user-supplied recorded outputs; no fresh inference."},
            "fine_tune": {"enabled": training, "reason": "Qualified registry/runtime installed; exact model, method, dataset and gym proof bindings still required." if training else reason},
            "weight_edit": {"enabled": False, "reason": "No qualified product weight-edit bridge registered. Operator must qualify a model-specific bridge."},
            "merge": {"enabled": False, "reason": "No qualified merge bridge registered. Operator must qualify a merge runtime."},
        }

    def workspace(self) -> dict[str, Any]:
        from .service import _public_document
        with self.service.store.transaction() as conn:
            projects = [_public_document(row) for row in conn.execute(
                "SELECT id,name,description,created_at FROM cp_projects ORDER BY created_at,id")]
        return {"projects": projects, "capabilities": self.capabilities(), "gym_templates": templates(),
                "recipe_methods": recipe_methods(),
                "training_runtimes": self.training_runtimes(),
                "limits": {"max_examples": MAX_EXAMPLES, "max_evaluation_examples": 256, "max_document_bytes": 262144,
                           "max_preview_actions": 64, "max_response_chars": 8192, "max_evidence_candidates": 16}}

    def training_runtimes(self) -> list[dict[str, Any]]:
        from .local_runtime import runtime_descriptor
        return [runtime_descriptor(profile) for profile in self.service.runtime_profiles.values()]

    def create_project(self, body: dict[str, Any], *, actor: str) -> dict[str, Any]:
        from .service import _id, _public_document
        actor_name(actor)
        keys(body, {"name", "description", "idempotency_key"}, {"name", "description", "idempotency_key"})
        name = text(body["name"], "project name")
        description = text(body["description"], "project description", 4096, empty=True)
        key = identifier(body["idempotency_key"], "idempotency key")
        fingerprint = digest({"name": name, "description": description})
        with self.service.store.transaction() as conn:
            conn.execute("INSERT INTO cp_projects(id,actor,create_key,request_digest,name,description) "
                         "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(actor,create_key) DO NOTHING",
                         (_id("project"), actor, key, fingerprint, name, description))
            row = one(conn.execute("SELECT * FROM cp_projects WHERE actor=%s AND create_key=%s", (actor, key)))
            if row["request_digest"] != fingerprint:
                fail("idempotency_conflict", "Project key is already bound to another request.", 409)
            return _public_document({k: row[k] for k in ("id", "name", "description", "created_at")})

    def get_project(self, project_id: str) -> dict[str, Any]:
        from .service import _public_document
        with self.service.store.transaction() as conn:
            project = self._project(conn, project_id)
            assets = [self._asset(conn, project_id, row["id"]) for row in conn.execute(
                "SELECT id FROM cp_assets WHERE project_id=%s ORDER BY created_at,id", (project_id,))]
            runs = [self.service._dto(conn, self.service.store.run(conn, row["run_id"])) for row in conn.execute(
                "SELECT links.run_id FROM cp_project_runs links JOIN cp_runs runs ON runs.id=links.run_id "
                "WHERE links.project_id=%s ORDER BY runs.created_at DESC,runs.id DESC", (project_id,))]
        return _public_document({"project": project, "assets": assets, "runs": runs})

    def _validate_document(self, conn: Connection[dict[str, Any]], project_id: str,
                           kind: str, value: Any) -> dict[str, Any]:
        value = document(value)
        if not isinstance(value, dict):
            fail("invalid_document", "Asset document must be an object.")
        if kind == "dataset":
            return validate_dataset(value)
        if kind == "model":
            keys(value, {"model_id", "revision", "source", "weights_digest"}, {"model_id", "revision", "source"})
            text(value["model_id"], "model identity", 256)
            text(value["revision"], "model revision", 256)
            if value["source"] not in ("reference", "recorded", "software_fixture"):
                fail("invalid_model", "Model source must be reference, recorded or software_fixture.")
            if "weights_digest" in value and (not isinstance(value["weights_digest"], str) or
                                             not re.fullmatch(r"sha256:[0-9a-f]{64}", value["weights_digest"])):
                fail("invalid_model", "Weights digest must be SHA-256; a declaration is not checkpoint verification.")
        elif kind == "gym":
            validate_gym(value)
            if value["template"] == "json_exact":
                dataset = self._asset(conn, project_id, value["dataset_asset_id"], "dataset")
                if not any(r["split"] == "validation" for r in dataset["document"]["examples"]):
                    fail("invalid_gym", "JSON gym requires at least one validation example.")
        elif kind == "recipe":
            keys(value, {"method", "model_asset_id", "gym_asset_ids", "dataset_asset_id", "parameters",
                         "proof_spec_artifact_id", "role", "runtime_profile", "source_model_asset_ids"},
                 {"method", "model_asset_id", "gym_asset_ids", "parameters"})
            if value["method"] not in ("lora", "sft", "weight_edit", "merge"):
                fail("invalid_recipe", "Unsupported recipe method.")
            self._asset(conn, project_id, value["model_asset_id"], "model")
            if value["method"] == "merge":
                sources = value.get("source_model_asset_ids")
                if (not isinstance(sources, list) or not 2 <= len(sources) <= 16
                        or any(not isinstance(source, str) for source in sources)
                        or len(set(sources)) != len(sources) or value["model_asset_id"] not in sources):
                    fail("invalid_recipe", "Merge requires two to sixteen distinct project model sources including the base model.")
                for source in sources:
                    self._asset(conn, project_id, source, "model")
            elif "source_model_asset_ids" in value:
                fail("invalid_recipe", "Source model composition is supported only for merge recipes.")
            gyms = value["gym_asset_ids"]
            if not isinstance(gyms, list) or not 1 <= len(gyms) <= 16 or len(set(map(str, gyms))) != len(gyms):
                fail("invalid_recipe", "Specify one to sixteen distinct project gyms.")
            for gid in gyms:
                self._asset(conn, project_id, gid, "gym")
            if "dataset_asset_id" in value:
                self._asset(conn, project_id, value["dataset_asset_id"], "dataset")
            parameters = value["parameters"]
            keys(parameters, set(METHOD_PARAMETERS[value["method"]]), set())
            for name, number in parameters.items():
                rule = PARAMETERS[name]
                if type(number) not in (int, float) or not rule["minimum"] <= number <= rule["maximum"]:
                    fail("invalid_recipe", "Parameters must be finite bounded numbers, never executable code.")
                if rule["type"] == "integer" and type(number) is not int:
                    fail("invalid_recipe", "Count and seed parameters must be integers.")
            for name in ("role", "runtime_profile"):
                if name in value:
                    identifier(value[name], name)
            if "proof_spec_artifact_id" in value:
                from .artifacts import validate_artifact_id
                validate_artifact_id(value["proof_spec_artifact_id"])
                if conn.execute("SELECT id FROM cp_artifacts WHERE id=%s", (value["proof_spec_artifact_id"],)).fetchone() is None:
                    fail("artifact_not_found", "Proof specification artifact is not registered.", 404)
        elif kind == "evidence":
            keys(value, {"model_asset_id", "dataset_asset_id", "source_description", "rows"},
                 {"model_asset_id", "dataset_asset_id", "source_description", "rows"})
            self._asset(conn, project_id, value["model_asset_id"], "model")
            dataset = self._asset(conn, project_id, value["dataset_asset_id"], "dataset")
            text(value["source_description"], "recorded source description", 2048)
            expected = {r["id"] for r in dataset["document"]["examples"] if r["split"] == "validation"}
            rows = value["rows"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_EXAMPLES:
                fail("invalid_evidence", "Provide bounded recorded validation rows.")
            seen = set()
            for row in rows:
                keys(row, {"example_id", "response"}, {"example_id", "response"})
                eid = identifier(row["example_id"], "example ID")
                if eid in seen or eid not in expected:
                    fail("invalid_evidence", "Evidence rows must uniquely reference validation examples.")
                seen.add(eid)
                text(row["response"], "recorded response", 8192, empty=True)
            if seen != expected:
                fail("incomplete_evidence", "Evaluation policy requires exactly all validation examples.")
        else:
            fail("invalid_asset_kind", "Unsupported asset kind.")
        return value

    def create_asset(self, project_id: str, body: dict[str, Any], *, actor: str) -> dict[str, Any]:
        from .service import _id, _public_document
        actor_name(actor)
        keys(body, {"kind", "name", "document", "parent_asset_id", "idempotency_key"},
             {"kind", "name", "document", "idempotency_key"})
        name = text(body["name"], "asset name")
        kind = text(body["kind"], "asset kind")
        key = identifier(body["idempotency_key"], "idempotency key")
        fingerprint = digest({**body, "parent_asset_id": body.get("parent_asset_id")})
        with self.service.store.transaction() as conn:
            self._project(conn, project_id)
            prior = conn.execute("SELECT id,request_digest FROM cp_assets WHERE project_id=%s AND create_key=%s",
                                 (project_id, key)).fetchone()
            if prior:
                if prior["request_digest"] != fingerprint:
                    fail("idempotency_conflict", "Asset key is already bound to another request.", 409)
                return _public_document(self._asset(conn, project_id, prior["id"]))
            value = self._validate_document(conn, project_id, kind, body["document"])
            parent_id = body.get("parent_asset_id")
            latest = conn.execute("SELECT id,version FROM cp_assets WHERE project_id=%s AND kind=%s AND name=%s "
                                  "ORDER BY version DESC LIMIT 1", (project_id, kind, name)).fetchone()
            if parent_id is not None:
                parent = self._asset(conn, project_id, parent_id, kind)
                if parent["name"] != name or latest is None or latest["id"] != parent_id:
                    fail("version_conflict", "Parent must be the current version of this named asset.", 409)
            elif latest:
                fail("version_conflict", "An existing asset requires its current parent version.", 409)
            metadata = self.service.artifacts.put(value)
            conn.execute("INSERT INTO cp_artifacts(id,metadata,actor) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                         (metadata["id"], json_value(metadata), actor))
            asset_id = _id("asset")
            conn.execute("INSERT INTO cp_assets(id,project_id,kind,name,version,parent_asset_id,artifact_id,"
                         "create_key,request_digest) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (asset_id, project_id, kind, name, latest["version"] + 1 if latest else 1,
                          parent_id, metadata["id"], key, fingerprint))
            return _public_document(self._asset(conn, project_id, asset_id))

    def preview(self, project_id: str, body: dict[str, Any]) -> dict[str, Any]:
        keys(body, {"definition", "actions", "seed", "response", "example_id"}, {"definition", "actions", "seed"})
        actions, seed = body["actions"], body["seed"]
        if not isinstance(actions, list) or len(actions) > 64 or any(type(a) not in (str, int, float, bool, type(None)) for a in actions):
            fail("invalid_preview", "Preview accepts at most 64 JSON scalar actions.")
        document(actions)
        if type(seed) is not int or not 0 <= seed < 2**31:
            fail("invalid_preview", "Seed must be a bounded nonnegative integer.")
        with self.service.store.transaction() as conn:
            self._project(conn, project_id)
            definition = self._validate_document(conn, project_id, "gym", body["definition"])
            if definition["template"] == "json_exact":
                if actions:
                    fail("invalid_preview", "JSON task preview does not accept actions.")
                dataset = self._asset(conn, project_id, definition["dataset_asset_id"], "dataset")
                example = next((r for r in dataset["document"]["examples"] if r["id"] == body.get("example_id")
                                and r["split"] == "validation"), None)
                if example is None:
                    fail("example_not_found", "Select an owned validation example.", 404)
                response = text(body.get("response", ""), "preview response", 8192, empty=True)
                measured = score_json(example["expected"], response)
                return {"observation": {**example, "response": response, **measured}, "transitions": [],
                        "allowed_actions": [], "terminated": True, "truncated": False,
                        "metrics": [{"name": key, "version": POLICY, "value": int(val), "unit": "boolean"}
                                    for key, val in measured.items()], "closed": True,
                        "scope": "owned-data-task-not-model-evidence", "model_calls": False}
        from ..gym_job import GymJobError
        from ..gym_runtime import Payload, StepAllowance
        from ..gym_runtime_fixtures import LOOKUP_PERMISSION, CounterFactory, WorkflowFactory
        workflow = definition["template"] == "tool_workflow"
        config = definition["config"]
        factory = WorkflowFactory() if workflow else CounterFactory(target=config.get("target", 2),
                                                                   horizon=config.get("horizon", 10))
        session = factory.create(owner_id="manual-product-preview")
        transitions: list[dict[str, Any]] = []
        terminated, truncated, metrics = False, False, []
        try:
            observation = session.reset(seed=seed).document()["data"]
            for index, action in enumerate(actions):
                factory.spec.capabilities.action.validate(Payload(action))
                transition = session.step(Payload(action), allowance=StepAllowance(
                    simulated_seconds=1, tool_calls=1 if workflow else 0, recorded_bytes=262144,
                    permissions=(LOOKUP_PERMISSION,) if workflow else ()))
                observation = transition.observation.document()["data"]
                terminated, truncated = transition.terminated, transition.truncated
                metrics = [{"name": m.name, "version": m.version, "value": m.value, "unit": m.unit}
                           for m in transition.metrics]
                transitions.append({"index": index, "action": action, "observation": observation,
                                    "reward": transition.reward, "terminated": terminated, "truncated": truncated,
                                    "metrics": metrics})
        except GymJobError:
            fail("preview_step_failed", "Manual action failed trusted fixture validation or lifecycle order.")
        finally:
            session.close()
        allowed = [] if terminated or truncated else (
            (["lookup"] if observation == "lookup" else ["finish"]) if workflow else [-1, 1])
        return {"observation": observation, "transitions": transitions, "allowed_actions": allowed,
                "terminated": terminated, "truncated": truncated, "metrics": metrics,
                "closed": session.closed, "scope": "software-only-no-renderer", "model_calls": False}

    def experiment(self, project_id: str, body: dict[str, Any], *, actor: str) -> dict[str, Any]:
        from .service import _public_document
        actor_name(actor)
        keys(body, {"workstream", "name", "gym_asset_id", "evidence_asset_ids", "recipe_asset_id", "idempotency_key"},
             {"workstream", "name", "idempotency_key"})
        name = text(body["name"], "experiment name")
        key = identifier(body["idempotency_key"], "idempotency key")
        workstream = body["workstream"]
        if workstream not in ("evaluate_saved", "fine_tune", "weight_edit", "merge"):
            fail("invalid_workstream", "Unsupported workstream.")
        with self.service.store.transaction() as conn:
            self._project(conn, project_id)
            if workstream == "evaluate_saved":
                if body.get("recipe_asset_id") is not None:
                    fail("invalid_experiment", "Recorded evaluation does not execute recipes.")
                gym = self._asset(conn, project_id, body.get("gym_asset_id"), "gym")
                if gym["document"]["template"] != "json_exact":
                    fail("invalid_experiment", "Saved-output analysis requires an owned JSON gym.")
                dataset = self._asset(conn, project_id, gym["document"]["dataset_asset_id"], "dataset")
                evidence_ids = body.get("evidence_asset_ids")
                if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 16 or len(set(map(str, evidence_ids))) != len(evidence_ids):
                    fail("invalid_experiment", "Select one to sixteen distinct recorded-output assets.")
                candidates, stages = [], []
                count = sum(r["split"] == "validation" for r in dataset["document"]["examples"])
                if count > 256:
                    fail("budget_exceeded", "This worker permits at most 256 validation rows per candidate stage.")
                for index, evidence_id in enumerate(evidence_ids):
                    evidence = self._asset(conn, project_id, evidence_id, "evidence")
                    if evidence["document"]["dataset_asset_id"] != dataset["id"]:
                        fail("incompatible_evidence", "All candidates must share the exact frozen dataset version.")
                    model = self._asset(conn, project_id, evidence["document"]["model_asset_id"], "model")
                    cid = f"candidate-{index}"
                    candidates.append({"id": cid, "model": {"id": model["document"]["model_id"],
                                                           "revision": model["document"]["revision"]}, "method": "recorded-output-analysis"})
                    stages.append({"id": f"analysis-{index}", "operation": "analysis.saved_json", "depends_on": [],
                                   "candidate_id": cid, "config": {
                                       "gym_artifact_id": gym["artifact_id"], "dataset_artifact_id": dataset["artifact_id"],
                                       "evidence_artifact_id": evidence["artifact_id"], "model_artifact_id": model["artifact_id"],
                                       "gym_asset_id": gym["id"], "dataset_asset_id": dataset["id"],
                                       "evidence_asset_id": evidence["id"], "model_asset_id": model["id"],
                                       "gym_version": gym["version"], "dataset_version": dataset["version"],
                                       "evidence_version": evidence["version"], "model_version": model["version"],
                                       "metric_policy": POLICY, "example_count": count},
                                   "limits": {"steps": count, "episodes": 1, "wall_seconds": 20, "recorded_bytes": 262144}})
                spec = {"version": 1, "name": name, "candidates": candidates, "stages": stages}
            else:
                recipe = self._asset(conn, project_id, body.get("recipe_asset_id"), "recipe")
                if body.get("evidence_asset_ids") or (
                        body.get("gym_asset_id") is not None
                        and body["gym_asset_id"] not in recipe["document"]["gym_asset_ids"]):
                    fail("invalid_experiment", "Proof experiments must use only the saved recipe's gym bindings and no recorded-output substitution.")
                spec = self._proof_plan(conn, project_id, workstream, name, recipe, actor=actor)
            spec = validate_spec(spec, self.service._operations())
            run_id = self.service._insert_run(conn, spec, actor=actor,
                                              idempotency_key="product-" + digest([project_id, key])[7:])
            conn.execute("INSERT INTO cp_project_runs(project_id,run_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                         (project_id, run_id))
            return _public_document(self.service._dto(conn, self.service.store.run(conn, run_id)))

    def _proof_plan(self, conn: Connection[dict[str, Any]], project_id: str, workstream: str,
                    name: str, recipe: dict[str, Any], *, actor: str) -> dict[str, Any]:
        value = recipe["document"]
        reason = self.capabilities()[workstream]["reason"]
        if workstream != "fine_tune" or value["method"] not in ("lora", "sft"):
            fail("runtime_not_qualified", reason, 409)
        profile = self.service.runtime_profiles.get(value.get("runtime_profile", ""))
        required = {"model_training", "model_export", "model_identity", "model_scoring"}
        if not profile or not profile.qualified or not required <= profile.deployment_capabilities:
            fail("runtime_not_qualified", reason, 409)
        if profile.descriptor:
            return self._local_proof_plan(conn, project_id, name, recipe, profile, actor=actor)
        if not all(value.get(field) for field in ("proof_spec_artifact_id", "role", "dataset_asset_id")):
            fail("runtime_not_qualified", "Operator must supply a complete proof artifact, role and dataset for the qualified runtime.", 409)
        from ..gym_job import GymJobError
        from ..proof_pipeline import validate_spec as validate_proof_spec
        from ..proof_registry import builtin_registry
        proof = self.service.get_artifact(value["proof_spec_artifact_id"])["document"]
        try:
            validate_proof_spec(proof, profile.registry or builtin_registry()[0])
        except (GymJobError, ValueError, TypeError, KeyError):
            fail("runtime_not_qualified", "Operator must supply a valid complete proof specification.", 409)
        selected = next((c for c in proof["candidates"] if c["role"] == value["role"]), None)
        model_asset = self._asset(conn, project_id, value["model_asset_id"], "model")
        model = model_asset["document"]
        if (selected is None or proof.get("base_model", {}).get("model_id") != model["model_id"]
                or proof.get("base_model", {}).get("revision") != model["revision"]
                or selected.get("method") != value["method"]
                or not selected.get("recipe") or not selected.get("execution")):
            fail("runtime_not_qualified", "Proof candidate must explicitly match the selected declared model identity.", 409)
        if "weights_digest" in model and proof["base_model"].get("weights_digest") != model["weights_digest"]:
            fail("runtime_not_qualified", "Proof base model must match the selected declared weights digest.", 409)
        if any(encode(selected["recipe"].get(k)) != encode(v) for k, v in value["parameters"].items()):
            fail("runtime_not_qualified", "Saved parameters must match the pinned proof recipe.", 409)
        dataset_asset = self._asset(conn, project_id, value["dataset_asset_id"], "dataset")
        dataset = dataset_asset["document"]
        expected_train = [{"input": r["input"], "reference": r["expected"]} for r in dataset["examples"] if r["split"] == "train"]
        expected_validation = [{"input": r["input"], "reference": r["expected"]} for r in dataset["examples"] if r["split"] == "validation"]
        train = [{"input": r["input"], "reference": r["reference"]} for r in proof["training_rows"]]
        panels = proof.get("public_sets", [])
        if (not expected_train or not expected_validation or encode(train) != encode(expected_train) or len(panels) != 1
                or encode([{"input": r["input"], "reference": r["reference"]} for r in panels[0]["rows"]]) != encode(expected_validation)):
            fail("runtime_not_qualified", "Proof rows must exactly match the saved train/validation dataset.", 409)
        gym_assets = [self._asset(conn, project_id, gid, "gym") for gid in value["gym_asset_ids"]]
        from .bridges import require_product_gym_binding
        for gym_asset in gym_assets:
            gym = gym_asset["document"]
            if gym["template"] != "json_exact" or gym["dataset_asset_id"] != value["dataset_asset_id"]:
                fail("runtime_not_qualified", "Product proof gyms must pin the selected JSON dataset.", 409)
            require_product_gym_binding(
                profile, project_id=project_id, gym_asset_id=gym_asset["id"], gym_version=gym_asset["version"],
                gym_artifact_id=gym_asset["artifact_id"], proof_gym_reference=proof["plugins"]["gym"],
                proof_gym_configuration=proof["gym"], method=value["method"])
        binding_assets = [recipe, model_asset, dataset_asset, *gym_assets]
        pinned_proof = {**proof, "product_binding": {
            "project_id": project_id, "original_proof_artifact_id": value["proof_spec_artifact_id"],
            "method": value["method"], "role": value["role"],
            "assets": [{key: a[key] for key in ("id", "kind", "version", "artifact_id")} for a in binding_assets]}}
        metadata = self.service.artifacts.put(pinned_proof)
        conn.execute("INSERT INTO cp_artifacts(id,metadata,actor) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                     (metadata["id"], json_value(metadata), actor))
        return {"version": 1, "name": name, "candidates": [
            {"id": "candidate", "model": {"id": model["model_id"], "revision": model["revision"]}, "method": value["method"]}],
            "stages": [{"id": op, "operation": "proof." + op,
                        "depends_on": [("train", "export", "identity", "score")[i - 1]] if i else [],
                        "candidate_id": "candidate", "config": {"runtime_profile": profile.id,
                            "artifact_id": metadata["id"], "role": value["role"]},
                        "limits": {}} for i, op in enumerate(("train", "export", "identity", "score"))]}

    def _local_proof_plan(self, conn: Connection[dict[str, Any]], project_id: str,
                          name: str, recipe: dict[str, Any], profile: TrustedRuntimeProfile,
                          *, actor: str) -> dict[str, Any]:
        from pathlib import Path

        from .local_runtime import compile_proof, read_config, stage_limits
        value = recipe["document"]
        if value.get("proof_spec_artifact_id"):
            fail("runtime_binding_changed", "The built-in runtime compiles its own proof; do not supply a legacy artifact.")
        model = self._asset(conn, project_id, value["model_asset_id"], "model")
        gyms = [self._asset(conn, project_id, gid, "gym") for gid in value["gym_asset_ids"]]
        dataset_id = value.get("dataset_asset_id") or gyms[0]["document"].get("dataset_asset_id")
        dataset = self._asset(conn, project_id, dataset_id, "dataset")
        if any(gym["document"] != {"template": "json_exact", "config": {},
                                  "dataset_asset_id": dataset["id"]} for gym in gyms):
            fail("runtime_not_qualified", "Only the owned JSON gym pinned to this dataset is installed.", 409)
        binding = {"project_id": project_id, "method": value["method"],
                   "role": value.get("role", "candidate"),
                   "assets": [{key: asset[key] for key in ("id", "kind", "version", "artifact_id")}
                              for asset in (recipe, model, dataset, *gyms)]}
        config = read_config(Path(profile.operator_config))
        proof = compile_proof(config, profile.descriptor, model=model["document"],
                              dataset=dataset["document"], recipe=value, binding=binding)
        metadata = self.service.artifacts.put(proof)
        conn.execute("INSERT INTO cp_artifacts(id,metadata,actor) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                     (metadata["id"], json_value(metadata), actor))
        operations = ("train", "export", "identity", "score")
        return {"version": 1, "name": name, "candidates": [{
            "id": "candidate", "model": {"id": model["document"]["model_id"],
                "revision": model["document"]["revision"]}, "method": "lora"}],
            "stages": [{"id": op, "operation": "proof." + op,
                "depends_on": [operations[i - 1]] if i else [], "candidate_id": "candidate",
                "config": {"runtime_profile": profile.id, "artifact_id": metadata["id"], "role": binding["role"]},
                "limits": stage_limits(config, proof, "proof." + op)} for i, op in enumerate(operations)]}

    def bundle(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        artifacts = {a["artifact_id"] for a in project["assets"]}
        artifacts.update(a["document"]["proof_spec_artifact_id"] for a in project["assets"]
                         if a["kind"] == "recipe" and a["document"].get("proof_spec_artifact_id"))
        for run in project["runs"]:
            artifacts.update(s["artifact_id"] for s in run["stages"] if s["artifact_id"])
            artifacts.update(s["config"]["artifact_id"] for s in run["spec"]["stages"]
                             if s["operation"].startswith("proof."))
        return {"format": "product-provenance-v1", **project, "artifacts": [
            self.service.get_artifact(aid) | {"document": None} for aid in sorted(artifacts)],
            "release_authorized": False, "signed": False,
            "provenance": "Declared model references and user-supplied recorded outputs; not fresh or independent inference."}


def analyze_saved(service: ControlPlaneService, config: dict[str, Any], bound: dict[str, int]) -> dict[str, Any]:
    project_id = None
    with service.store.transaction() as conn:
        for kind in ("gym", "dataset", "evidence", "model"):
            row = conn.execute("SELECT project_id,kind,version,artifact_id FROM cp_assets WHERE id=%s",
                               (config[kind + "_asset_id"],)).fetchone()
            if (not row or row["kind"] != kind or row["version"] != config[kind + "_version"]
                    or row["artifact_id"] != config[kind + "_artifact_id"]
                    or (project_id is not None and row["project_id"] != project_id)):
                fail("invalid_analysis_pins", "Analysis requires exact project-owned immutable asset pins.")
            project_id = row["project_id"]
    gym = service.get_artifact(config["gym_artifact_id"])["document"]
    dataset = validate_dataset(service.get_artifact(config["dataset_artifact_id"])["document"])
    evidence = service.get_artifact(config["evidence_artifact_id"])["document"]
    model = service.get_artifact(config["model_artifact_id"])["document"]
    validate_gym(gym)
    if gym["template"] != "json_exact" or gym["dataset_asset_id"] != config["dataset_asset_id"]:
        fail("incompatible_evidence", "Pinned gym does not match the dataset.")
    if evidence.get("dataset_asset_id") != config["dataset_asset_id"] or evidence.get("model_asset_id") != config["model_asset_id"]:
        fail("incompatible_evidence", "Pinned evidence does not match declared dataset/model.")
    source = text(evidence.get("source_description"), "recorded source description", 2048)
    examples = {r["id"]: r for r in dataset["examples"] if r["split"] == "validation"}
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(examples) or not rows or len(rows) > bound["steps"]:
        fail("budget_exceeded", "Recorded validation rows exceed policy or approved bounds.")
    seen = set()
    measured = []
    for row in rows:
        keys(row, {"example_id", "response"}, {"example_id", "response"})
        eid = identifier(row["example_id"], "example ID")
        if eid not in examples or eid in seen:
            fail("invalid_evidence", "Recorded evidence does not cover validation examples exactly.")
        seen.add(eid)
        response = text(row["response"], "recorded response", 8192, empty=True)
        measured.append({"example_id": eid, "response": response, **score_json(examples[eid]["expected"], response)})
    if len(rows) != config["example_count"]:
        fail("invalid_evidence", "Pinned example count does not match.")
    result = {"execution": "completed", "analysis": "recorded-output-analysis", "metric_policy": POLICY,
              "declared_model": {"id": model["model_id"], "revision": model["revision"]},
              "evidence_class": "recorded_output_analysis",
              **{kind + "_asset_id": config[kind + "_asset_id"] for kind in ("gym", "dataset", "model", "evidence")},
              "provenance": "user-supplied-recorded-outputs-not-fresh-or-independent-inference",
              "source_description": source, "pins": config, "rows": measured, "steps": len(rows),
              "metrics": [{"name": name, "version": POLICY, "unit": "ratio",
                           "value": sum(r[field] for r in measured) / len(measured)}
                          for name, field in (("json_exact_match", "exact_match"), ("json_validity", "valid_json"))],
              "quality": "passed" if all(r["exact_match"] for r in measured) else "failed",
              "model_calls": False, "release_authorized": False, "runtime_profile": "cpu-recorded-analysis-v1"}
    if len(encode(result)) > bound["recorded_bytes"]:
        fail("budget_exceeded", "Recorded analysis exceeds approved byte limit.")
    return result
