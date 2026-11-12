"""CPU-testable capture/solver route, deliberately unavailable to live jobs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...gym_service import _wire_document
from ...prompt_rendering import PromptRenderer
from ...provenance import parse_strict_json
from .capture import capture, capture_statistics
from .loader import WeightEditInputError
from .methods.experimental import METHOD_ID, ValueScaledRetainedEdit, validate_settings
from .value_objective import prepare_value_demonstration


def validate_experimental_recipe(recipe):
    fields = {"method", "edit_method", "experimental_opt_in", "experimental_settings",
              "max_sequence_tokens", "max_training_tokens", "max_keys", "retention_interactions",
              "retention_source", "protected_training_examples", "protected_training_source", "value_fields"}
    if (not isinstance(recipe, Mapping) or set(recipe) != fields or recipe["method"] != "weight_edit"
        or recipe["edit_method"] != METHOD_ID or recipe["experimental_opt_in"] is not True):
        raise WeightEditInputError("experimental method requires its separate explicit opt-in recipe")
    validate_settings(recipe["experimental_settings"])
    for name, maximum in (("max_sequence_tokens", 4096), ("max_training_tokens", 1_000_000), ("max_keys", 8192)):
        if type(recipe[name]) is not int or not 1 <= recipe[name] <= maximum:
            raise WeightEditInputError(f"experimental {name} is outside its explicit budget")
    if recipe["retention_source"] != "public_train" or recipe["protected_training_source"] != "public_train":
        raise WeightEditInputError("experimental retention/protection must be caller-authorized public train only")
    for name, maximum in (("retention_interactions", 128), ("protected_training_examples", 1024)):
        if not isinstance(recipe[name], (list, tuple)) or not 1 <= len(recipe[name]) <= maximum:
            raise WeightEditInputError(f"experimental {name} must be nonempty and bounded")
    for example in recipe["protected_training_examples"]:
        if not isinstance(example, Mapping) or example.get("already_passing") is not True or "interaction" not in example:
            raise WeightEditInputError("protected examples require explicit already-passing train authority")
    fields = recipe["value_fields"]
    if fields is not None and (not isinstance(fields, (list, tuple)) or not fields
                               or any(not isinstance(field, str) or not field for field in fields)
                               or len(set(fields)) != len(fields)):
        raise WeightEditInputError("value_fields must be null for task spans or explicit unique JSON fields")
    return parse_strict_json(_wire_document(recipe))


def calculate_experimental_edit(loaded, examples, recipe, selected, site):
    recipe = validate_experimental_recipe(recipe)
    if not examples:
        raise WeightEditInputError("experimental editing requires approved value demonstrations")
    renderer = PromptRenderer(loaded.tokenizer)

    def demonstration(example):
        return prepare_value_demonstration(renderer, example["interaction"],
            spans=example.get("value_spans"), value_fields=recipe["value_fields"])

    demonstrations = [demonstration(example) for example in examples]
    protected = [demonstration(example) for example in recipe["protected_training_examples"]]
    protected_prompts = [renderer.render({**example["interaction"],
        "messages": example["interaction"]["messages"][:-1]}) for example in recipe["protected_training_examples"]]
    retention_prompts = [renderer.render(interaction) for interaction in recipe["retention_interactions"]]
    prompts = retention_prompts + protected_prompts
    count = sum(len(item.positions) for item, _ in demonstrations)
    full_positions = [list(range(len(prompt.token_ids))) for prompt in prompts]
    protected_positions = [list(range(len(prompt.token_ids))) for prompt in protected_prompts]
    extra_protected_positions = [
        [position for position in item.positions if position >= len(prompt.token_ids)]
        for (item, _), prompt in zip(protected, protected_prompts, strict=True)
    ]
    extra_protected = [(item.rendered, positions) for (item, _), positions in
                       zip(protected, extra_protected_positions, strict=True) if positions]
    capture_passes = {
        "retention_statistics": list(zip(prompts, full_positions, strict=True)),
        "protected_prompt_recapture": list(zip(protected_prompts, protected_positions, strict=True)),
        "protected_value_capture": extra_protected,
        "edit_value_capture": [(item.rendered, item.positions) for item, _ in demonstrations],
    }
    budget_stages = {
        name: {"forward_pass_count": len(passes),
               "forward_tokens": sum(len(prompt.token_ids) for prompt, _ in passes),
               "captured_key_positions": sum(len(positions) for _, positions in passes)}
        for name, passes in capture_passes.items()
    }
    budget_totals = {name: sum(stage[name] for stage in budget_stages.values())
                     for name in ("forward_pass_count", "forward_tokens", "captured_key_positions")}
    if (any(len(prompt.token_ids) > recipe["max_sequence_tokens"]
            for passes in capture_passes.values() for prompt, _ in passes)
        or budget_totals["forward_tokens"] > recipe["max_training_tokens"]):
        raise WeightEditInputError("experimental capture exceeds explicit token budgets")
    if budget_totals["captured_key_positions"] > recipe["max_keys"]:
        raise WeightEditInputError("experimental capture exceeds explicit key budget")
    stats = capture_statistics(loaded, selected, prompts, positions=full_positions)[site.tensor_name]
    passing_keys = [capture(loaded, selected, protected_prompts, positions=protected_positions,
                           output_dtype=torch.float64)[site.tensor_name].inputs]
    for rendered, positions in extra_protected:
        passing_keys.append(capture(loaded, selected, [rendered], positions=[positions],
                                   output_dtype=torch.float64)[site.tensor_name].inputs)
    keys, outputs, gradients, losses, groups = [], [], [], [], []
    for item, item_groups in demonstrations:
        values = capture(loaded, selected, [item.rendered], positions=[item.positions], loss_labels=[item.labels],
                         output_dtype=torch.float64, causal_value_objective=True, loss_denominator=count,
                         token_loss_sink=losses.append)[site.tensor_name]
        keys.append(values.inputs)
        outputs.append(values.outputs)
        gradients.append(values.output_gradients)
        groups.extend(item_groups)
    # Match the exact stored base used by apply, not a higher-precision source
    # checkpoint that may differ from the loaded serving tensor.
    stored = loaded.model.get_parameter(site.tensor_name).detach().cpu().double()
    W = stored.T.contiguous() if site.transposed else stored
    method = ValueScaledRetainedEdit(stats, torch.cat(passing_keys), torch.cat(outputs))
    delta = method.compute(site, W, torch.cat(keys), torch.cat(gradients), recipe["experimental_settings"])
    token_losses = torch.cat(losses).double()
    grouped = {}
    for group in dict.fromkeys(groups):
        mask = torch.tensor([value == group for value in groups])
        grouped[group] = {"value_token_count": int(mask.sum()), "mean_nll_nats": float(token_losses[mask].mean())}
    evidence = {"target_objective": "causal_semantic_value_only_global_mean_loss",
                "experimental_only": True, "live_qualified": False,
                "key_diagnostics": method.diagnostics, "value_token_groups": grouped,
                "retention_prompt_tokens": sum(len(prompt.token_ids) for prompt in retention_prompts),
                "protected_prompt_tokens": sum(map(len, protected_positions)),
                "protected_value_prediction_keys": sum(len(item.positions) for item, _ in protected),
                "protected_value_prediction_keys_not_already_in_prompt": sum(map(len, extra_protected_positions)),
                "capture_budget": {"stages": budget_stages, "totals": budget_totals,
                                   "key_counting": "all_captured_positions_including_repeated_passes"},
                "retention_prompt_digest": stats.prompt_digest, "renderer_digest": renderer.digest}
    return delta, evidence
