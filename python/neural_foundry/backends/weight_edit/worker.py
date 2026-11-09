"""Bounded real-model calculation worker for the generic weight editor."""

from __future__ import annotations

import json
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...gym_artifacts import CandidateRef
from ...gym_adapters import _resolve_training_checkpoint
from ...gym_job import GymJobError, gym_job_from_document, validate_task_instance
from ...gym_service import _document_digest, _wire_document
from ...prompt_rendering import PromptRenderer
from ...provenance import parse_strict_json
from ...vllm_container_adapter import VllmCandidateServer
from .adapter import WeightEditBackendConfig, validate_recipe, weight_edit_trainer
from .apply import apply, save, seal_serving_probes
from .capture import capture
from .demonstrations import prepare_demonstration
from .discover import SiteMap, find_blocks, find_sites
from .loader import load_model
from .methods import LegacyKeyValueEdit
from .numerics import deterministic_execution, working_weight
from .probe import record_portability_probes
from .serving_probes import verify_portability
from .preservation import calibrate_one_ulp, float32_logits_of_exact_weights, metrics, verify_v2


def preservation_logits(loaded, interactions):
    renderer = PromptRenderer(loaded.tokenizer)
    device = next(loaded.model.parameters()).device
    values = []
    with deterministic_execution(loaded.model), torch.no_grad():
        for interaction in interactions:
            rendered = renderer.render(interaction)
            ids = torch.tensor([rendered.token_ids], dtype=torch.long, device=device)
            values.append(loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].detach().float().cpu())
    return values


def _edit_key_diagnostics(keys, outputs, active, correct_tokens, *, target_step, retention_weight):
    used_keys, used_outputs = keys[active], outputs[active]
    mean_diagonal = float(used_keys.square().sum() / used_keys.shape[1])
    mean_input_norm = float(used_keys.norm(dim=1).mean())
    mean_output_norm = float(used_outputs.norm(dim=1).mean())
    return {"gram_orientation": "K[input_dimension, active_key_count]; diagonal of K @ K.T",
            "mean_gram_diagonal": mean_diagonal, "fixed_ridge": 1e-3,
            "ridge_to_mean_gram_diagonal": 1e-3 / mean_diagonal if mean_diagonal > 0 else None,
            "key_count": keys.shape[0], "active_key_count": int(active.sum()),
            "already_correct_token_keys": int(correct_tokens.sum()),
            "active_already_correct_token_keys": int(correct_tokens[active].sum()),
            "incorrect_token_keys": int((~correct_tokens).sum()),
            "correctness_definition": "teacher-forced next-token top-1 equals approved token, including EOS",
            "mean_key_input_l2_norm": mean_input_norm,
            "mean_write_output_l2_norm": mean_output_norm,
            "hidden_norm_scope": "MLP write-site input and output at edited keys, not full residual stream",
            "target_shift_l2": target_step,
            "target_shift_fraction_of_mean_write_output_norm": target_step / mean_output_norm if mean_output_norm > 0 else None,
            "target_shift_fraction_of_mean_key_input_norm": target_step / mean_input_norm if mean_input_norm > 0 else None,
            "retention_nullspace_weight": retention_weight,
            "retention_penalty_enabled": retention_weight != 0,
            "diagnostics_only": True}


def calculate_edit(loaded, job, examples, recipe, *, site_layer: int | None = None, discovered_sites: SiteMap | None = None):
    recipe = validate_recipe(recipe)
    failure_selection = recipe.get("failure_selection")
    if failure_selection is not None:
        example_digests = [example.get("example_digest") for example in examples]
        if (len(examples) < 10 or len(examples) != failure_selection["failure_count"]
            or example_digests != failure_selection["selected_example_digests"]):
            raise GymJobError("weight-edit keys must be exactly the selected base-failing train examples, with at least ten cases")
    sites = discovered_sites or find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                                          checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    if not sites.checkpoint_names_verified:
        raise GymJobError("edit calculation requires a checkpoint-verified site map")
    writers = [site for site in sites.sites if site.role == "residual_write_mlp"]
    if not writers:
        raise GymJobError("qualified model has no MLP write site")
    if site_layer is None:
        site = writers[len(writers) // 2]
    else:
        matches = [candidate for candidate in writers if candidate.layer == site_layer]
        if len(matches) != 1:
            raise GymJobError("requested edit layer is not a unique discovered MLP write site")
        site = matches[0]
    selected = SiteMap((site,), checkpoint_names_verified=True)
    renderer = PromptRenderer(loaded.tokenizer)
    panel = [*recipe["preservation_interactions"], *recipe.get("protected_training_interactions", ())]
    for interaction in panel:
        if len(renderer.render(interaction).token_ids) > recipe["max_sequence_tokens"]:
            raise GymJobError("preservation prompt exceeds the frozen sequence budget")
    retention = capture(loaded, selected, panel, output_dtype=torch.float64)[site.tensor_name].inputs
    keys, gradients, outputs, correct_tokens = [], [], [], []
    total_tokens, key_count = 0, 0
    for example in examples:
        interaction = example["interaction"]
        if interaction.get("format") == "json_extraction_v1":
            if interaction.get("output_schema") != job.to_document()["task"]["output_schema"]:
                raise GymJobError("demonstration schema differs from the frozen job")
            response = parse_strict_json(interaction["messages"][-1]["content"].encode())
            validate_task_instance(job.task_output_schema, response, field="approved_response")
        demonstration = prepare_demonstration(renderer, interaction)
        total_tokens += len(demonstration.rendered.token_ids)
        key_count += len(demonstration.positions)
        if (len(demonstration.rendered.token_ids) > recipe["max_sequence_tokens"]
            or total_tokens > recipe["max_training_tokens"] or key_count > recipe["max_keys"]):
            raise GymJobError("weight-edit demonstrations exceed frozen token/key budgets")
        captured = capture(loaded, selected, [demonstration.rendered], positions=[demonstration.positions],
                           loss_labels=[demonstration.labels], output_dtype=torch.float64,
                           token_outcome_sink=correct_tokens.append)[site.tensor_name]
        keys.append(captured.inputs)
        gradients.append(captured.output_gradients)
        outputs.append(captured.outputs)
    if not keys:
        raise GymJobError("weight editing requires approved demonstrations")
    keys, gradients = torch.cat(keys), torch.cat(gradients)
    norms = gradients.norm(dim=1, keepdim=True)
    active = norms[:, 0] > 0
    if not bool(active.any()):
        raise GymJobError("target loss has no gradient at the selected site")
    weights = working_weight(loaded, site)
    targets = keys[active] @ weights.T - recipe["target_step"] * gradients[active] / norms[active]
    delta = LegacyKeyValueEdit(retention).compute(site, weights, keys[active], targets,
                {"retention_nullspace_weight": recipe["retention_nullspace_weight"]})
    evidence = {"target_objective": "teacher_forced_token_loss_output_gradient", "gradient_inspection_used": True,
                "training_tokens": total_tokens, "key_count": key_count, "zero_gradient_keys": int((~active).sum()),
                "preservation_digest": _document_digest({"source": "public_train", "interactions": panel}),
                "preservation_count": len(panel), "preservation_input_authority": "caller_supplied_public_panel",
                "failure_selection_digest": _document_digest(failure_selection) if failure_selection else None,
                "protected_training_interactions": len(recipe.get("protected_training_interactions", ())),
                "renderer_digest": renderer.digest,
                "key_diagnostics": _edit_key_diagnostics(keys, torch.cat(outputs), active, torch.cat(correct_tokens),
                    target_step=recipe["target_step"], retention_weight=recipe["retention_nullspace_weight"])}
    return sites, site, delta, evidence


def calculate_edits_for_sites(loaded, job, examples, recipe, *, sites: SiteMap,
                              retention_interactions: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Capture each demonstration once across all selected MLP sites, then reuse it for every strength."""
    recipe = validate_recipe(recipe)
    if not sites.checkpoint_names_verified or not sites.sites or not retention_interactions:
        raise GymJobError("multi-site calculation requires verified sites and protected interactions")
    renderer = PromptRenderer(loaded.tokenizer)
    for interaction in (*retention_interactions, *(example["interaction"] for example in examples)):
        if len(renderer.render(interaction).token_ids) > recipe["max_sequence_tokens"]:
            raise GymJobError("multi-site example exceeds the frozen sequence budget")
    retained: dict[str, list[torch.Tensor]] = {site.tensor_name: [] for site in sites.sites}
    keys: dict[str, list[torch.Tensor]] = {site.tensor_name: [] for site in sites.sites}
    gradients: dict[str, list[torch.Tensor]] = {site.tensor_name: [] for site in sites.sites}
    write_outputs: dict[str, list[torch.Tensor]] = {site.tensor_name: [] for site in sites.sites}
    correct_tokens = []

    def collect(destination, *, collect_gradients=False):
        def consume(name, inputs, outputs, output_gradient, prompt_digest):
            destination[name].append(inputs)
            if collect_gradients:
                if output_gradient is None:
                    raise GymJobError("multi-site gradient capture returned no output gradient")
                gradients[name].append(output_gradient)
                write_outputs[name].append(outputs)
        return consume

    capture(loaded, sites, retention_interactions, output_dtype=torch.float64, sink=collect(retained))
    total_tokens, key_count = 0, 0
    for example in examples:
        interaction = example["interaction"]
        if interaction.get("format") == "json_extraction_v1":
            if interaction.get("output_schema") != job.to_document()["task"]["output_schema"]:
                raise GymJobError("demonstration schema differs from frozen job")
            response = parse_strict_json(interaction["messages"][-1]["content"].encode())
            validate_task_instance(job.task_output_schema, response, field="approved_response")
        demonstration = prepare_demonstration(renderer, interaction)
        total_tokens += len(demonstration.rendered.token_ids)
        key_count += len(demonstration.positions)
        if (len(demonstration.rendered.token_ids) > recipe["max_sequence_tokens"]
            or total_tokens > recipe["max_training_tokens"] or key_count > recipe["max_keys"]):
            raise GymJobError("multi-site demonstrations exceed frozen token/key budgets")
        capture(loaded, sites, [demonstration.rendered], positions=[demonstration.positions],
                loss_labels=[demonstration.labels], output_dtype=torch.float64,
            sink=collect(keys, collect_gradients=True), token_outcome_sink=correct_tokens.append)
    if not examples:
        raise GymJobError("multi-site edit requires at least one base-failing training example")
    result = {}
    for site in sites.sites:
        name = site.tensor_name
        key = torch.cat(keys[name])
        gradient = torch.cat(gradients[name])
        norms = gradient.norm(dim=1, keepdim=True)
        active = norms[:, 0] > 0
        if not bool(active.any()):
            raise GymJobError(f"target loss has no gradient at {name}")
        weights = working_weight(loaded, site)
        targets = key[active] @ weights.T - recipe["target_step"] * gradient[active] / norms[active]
        protected = torch.cat(retained[name])
        delta = LegacyKeyValueEdit(protected).compute(site, weights, key[active], targets,
                    {"retention_nullspace_weight": recipe["retention_nullspace_weight"]})
        result[name] = (delta, {"layer": site.layer, "site": site.document(), "failing_key_count": int(active.sum()),
            "zero_gradient_keys": int((~active).sum()), "training_tokens": total_tokens,
            "target_objective": "teacher_forced_token_loss_output_gradient", "gradient_inspection_used": True,
            "protected_key_count": protected.shape[0], "preservation_digest": "sha256:" + hashlib.sha256(
                protected.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "renderer_digest": renderer.digest,
            "key_diagnostics": _edit_key_diagnostics(key, torch.cat(write_outputs[name]), active, torch.cat(correct_tokens),
                target_step=recipe["target_step"], retention_weight=recipe["retention_nullspace_weight"])})
    return result


def train_payload(payload: dict[str, Any], serving_factory=None) -> CandidateRef:
    from importlib import metadata
    if metadata.version("transformers") != "4.57.3" or metadata.version("torch").split("+")[0] != "2.9.1":
        raise GymJobError("weight-edit worker runtime differs from the pinned numerical environment")
    request = payload["request"]
    base = CandidateRef(**payload["base"])
    job = gym_job_from_document(payload["job"])
    probe = payload["probe"]
    expected_trainer = weight_edit_trainer(WeightEditBackendConfig(**payload["config"]), job)
    if expected_trainer.identity.implementation_digest != payload["implementation_digest"]:
        raise GymJobError("worker helper code or configuration changed after dispatch")
    if request.get("training_request_digest") != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise GymJobError("worker training request digest is invalid")
    if (job.digest != request["job_digest"] or probe.get("supported") is not True
        or probe.get("probe_digest") != _document_digest({key: value for key, value in probe.items() if key != "probe_digest"})
        or _document_digest({"examples": payload["examples"]}) != request["projected_examples_digest"]):
        raise GymJobError("worker inputs differ from the frozen job, probe, or demonstrations")
    if "weight_edit" not in {method.method_id for method in job.candidate_methods}:
        raise GymJobError("job has not authorized weight editing")
    checkpoint = _resolve_training_checkpoint(request, base)
    settings, recipe = request["inference_config"], validate_recipe(request["recipe"])
    failure_selection = recipe.get("failure_selection")
    if failure_selection is not None:
        example_digests = [example.get("example_digest") for example in payload["examples"]]
        if (failure_selection["filtered_dataset_digest"] != request["dataset_digest"]
            or failure_selection["base_state_digest"] != request["loaded_base"]["state_digest"]
            or failure_selection["failure_count"] != len(payload["examples"])
            or example_digests != failure_selection["selected_example_digests"]):
            raise GymJobError("worker examples differ from the frozen base-failure dataset selection")
    failure_selection = recipe.get("failure_selection")
    if failure_selection is not None and (failure_selection["filtered_dataset_digest"] != request["dataset_digest"]
        or failure_selection["base_state_digest"] != request["loaded_base"]["state_digest"]
        or [example.get("example_digest") for example in payload["examples"]] != failure_selection["selected_example_digests"]):
        raise GymJobError("worker examples differ from the frozen base-failure dataset selection")
    if recipe["max_sequence_tokens"] > settings["context_tokens"]:
        raise GymJobError("edit sequence budget exceeds model context")
    torch.manual_seed(recipe["seed"])
    loaded = load_model(checkpoint.path, dtype=getattr(torch, settings["precision"]), device=settings["device"])
    if (loaded.base_digest != probe["base_digest"] or loaded.base_digest != request["loaded_base"]["state_digest"]
        or loaded.text_config.model_type != base.family):
        raise GymJobError("worker loaded state differs from the qualified base")
    before = preservation_logits(loaded, recipe["preservation_interactions"])
    sites, site, delta, evidence = calculate_edit(loaded, job, payload["examples"], recipe)
    if sites.digest != probe["site_map_digest"]:
        raise GymJobError("worker site map differs from the qualified model")
    calibration, float32_before = None, None
    if recipe.get("preservation_policy_version", 1) == 2:
        calibration = calibrate_one_ulp(loaded, site, recipe["preservation_interactions"])
        calibration_path = Path(payload["output_dir"]).parent / "preservation-calibration-v2.json"
        frozen = _wire_document(calibration)
        if calibration_path.exists():
            if calibration_path.read_bytes() != frozen:
                raise GymJobError("preservation calibration differs from the previously frozen evidence")
        else:
            with calibration_path.open("xb") as stream:
                stream.write(frozen)
        float32_before = float32_logits_of_exact_weights(loaded, recipe["preservation_interactions"])
    edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="legacy_key_value_v1", settings=recipe)
    if edited.receipt["edited_tensors"] != [site.tensor_name]:
        raise GymJobError("edit did not produce exactly one effective tensor change")
    after = preservation_logits(loaded, recipe["preservation_interactions"])
    drift = max(float((left - right).abs().max()) for left, right in zip(before, after, strict=True))
    preservation_verification = None
    if calibration is not None:
        precise = metrics(float32_before, float32_logits_of_exact_weights(loaded, recipe["preservation_interactions"]))
        behaviour = metrics(before, after)
        preservation_verification = verify_v2(precise, behaviour, calibration)
        verification_path = Path(payload["output_dir"]).parent / "preservation-verification-v2.json"
        with verification_path.open("xb") as stream:
            stream.write(_wire_document(preservation_verification))
        if recipe.get("retention_mode", "generalist") == "generalist" and not preservation_verification["passed"]:
            raise GymJobError(f"candidate violates frozen preservation v2: {json.dumps(preservation_verification, allow_nan=False)}")
    elif drift > recipe["max_preservation_logit_drift"]:
        raise GymJobError(f"candidate violates the frozen preservation-logit drift bound: measured={drift}, limit={recipe['max_preservation_logit_drift']}")
    edited = replace(edited, receipt={**edited.receipt, **evidence, "preservation_max_logit_drift": drift,
                                     "retention_mode": recipe.get("retention_mode", "generalist"),
                                     "preservation_gating": recipe.get("retention_mode", "generalist") == "generalist",
                                     "trainer_implementation_digest": payload["implementation_digest"],
                                     "compatibility_probe_digest": probe["probe_digest"], "job_digest": job.digest,
                                     "scorer": job.scorer.to_document(), "dataset_digest": request["dataset_digest"]})
    if calibration is not None:
        edited = replace(edited, receipt={**edited.receipt, "preservation_policy_version": 2,
                                          "preservation_calibration": calibration,
                                          "preservation_verification": preservation_verification})
    portability = record_portability_probes(loaded)
    directory = Path(payload["output_dir"])
    reference = save(edited, directory, model_id=base.model_id, request_digest=request["training_request_digest"],
                     base_reference=base, base_revision=request["loaded_base"]["revision"])
    if next(loaded.model.parameters()).device.type == "cuda":
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    if serving_factory is None:
        serving_factory = lambda ref: VllmCandidateServer(ref, revision=request["loaded_base"]["revision"],
            context_tokens=settings["context_tokens"], precision=settings["precision"],
            gpu_memory_utilization=payload["config"]["gpu_memory_utilization"], timeout_seconds=300)
    creation_server = serving_factory(reference)
    with creation_server:
        identity = creation_server.record_identity_probes()
    verified = verify_portability(portability, identity, probe["portability_calibration"])
    reference = seal_serving_probes(reference, identity, verified)
    with serving_factory(reference) as served:
        if served.loaded_identity().get("served_model_identity_verified") is not True:
            raise GymJobError("fresh serving identity failed before publication")
    return reference


def main() -> None:
    payload = parse_strict_json(Path(sys.argv[1]).read_bytes())
    reference = train_payload(payload)
    print(json.dumps({"kind": "nf_weight_edit_worker_result_v1", "candidate": vars(reference)}, allow_nan=False))


if __name__ == "__main__":
    main()