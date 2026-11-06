"""Adversarial tests for the edit, signing and artifact trust path.

Each test attacks one guarantee and asserts a specific rejection or an exact result. A test marked
`xfail(strict=True)` is a real open defect in code this suite doesn't own; it starts failing the moment
the defect is fixed, so the mark has to be removed then. Fixtures are small synthetic models, so nothing
here says anything about real checkpoints.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import os
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from neural_foundry import _native, edit_plan, export_formats, patch_ux, provenance
from neural_foundry.edit_plan import (
    EditConstraints,
    EditConstraintViolation,
    EditGraphError,
    EditPlan,
    Region,
    apply_edit_graph,
    apply_signed_edit_graph,
    sign_edit_graph,
)
from neural_foundry.records import canonicalize, content_digest

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("NF_CLI", ROOT / "target" / "debug" / "nf"))
# What a refusal may look like at the native boundary. KeyError, TypeError and friends are not refusals.
REFUSED = (ValueError, _native.NativeError)
SITE, TWIN, BIAS = "layer.weight", "layer2.weight", "layer.bias"


def xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=reason)


def _marked(names: Any, reasons: dict[str, str]) -> list[Any]:
    return [pytest.param(n, marks=xfail(reasons[n])) if n in reasons else n for n in names]


class Toy:
    """The smallest model the edit and export paths accept."""

    def __init__(self, seed: int = 3, config: dict[str, Any] | None = None) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.tensors = {
            SITE: torch.randn(4, 3, dtype=torch.float64, generator=generator),
            TWIN: torch.randn(4, 3, dtype=torch.float64, generator=generator),
            BIAS: torch.randn(3, dtype=torch.float64, generator=generator),
        }
        self.config = config or {"model_type": "toy"}
        self.model_name = "toy"


@pytest.fixture
def model() -> Toy:
    return Toy()


@pytest.fixture(scope="module")
def keys() -> dict[str, Any]:
    return patch_ux.generate_ed25519_keypair()


@pytest.fixture(scope="module")
def other_keys() -> dict[str, Any]:
    return patch_ux.generate_ed25519_keypair()


def _matrix(value: float = 0.5, shape: tuple[int, int] = (4, 3)) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float64)


def _graph_plan(model: Toy) -> EditPlan:
    return EditPlan(model).scale(SITE, 0.5, region=Region(rows=(0, 2))).add(TWIN, _matrix(0.25)).scale(SITE, 1.5)


# ---------------------------------------------------------------- regions


@pytest.mark.parametrize(
    "region",
    [
        Region(rows=(0, 0)),
        Region(rows=(2, 1)),
        Region(rows=(-1, 2)),
        Region(rows=(0, 5)),
        Region(cols=(0, 4)),
        Region(rows=(0, 2**64)),
        Region(cols=(2**64 - 1, 2**64 + 1)),
        pytest.param(Region(rows=(0, 1.5)), marks=xfail("D16 edit_plan.py: Region accepts a float bound and truncates it")),  # type: ignore[arg-type]
    ],
    ids=["empty", "inverted", "negative", "past_end", "cols_past_end", "u64_end", "u64_wrap", "float"],
)
def test_malformed_regions_are_refused_and_base_is_untouched(model: Toy, region: Region) -> None:
    before = model.tensors[SITE].clone()
    with pytest.raises(REFUSED):
        EditPlan(model).scale(SITE, 2.0, region=region).preview()
    assert torch.equal(model.tensors[SITE], before)


@pytest.mark.parametrize(
    "arguments",
    [
        {"region_start": [2**63, 0], "region_shape": [1, 1]},
        {"region_start": [2**64 - 1, 0], "region_shape": [2, 3]},
        {"region_start": [-1, 0], "region_shape": [1, 1]},
        {"region_start": [0, 0], "region_shape": [2**64, 3]},
        {"region_start": [0, 0], "region_shape": [0, 0]},
        {"region_start": [0], "region_shape": [1]},
        {"region_start": [0, 0, 0], "region_shape": [1, 1, 1]},
        {"region_start": ["0", "0"], "region_shape": [1, 1]},
    ],
    ids=["i64", "wrapping_u64", "negative", "huge_shape", "empty", "rank1", "rank3", "strings"],
)
def test_graph_region_overflow_cannot_reach_the_executor(model: Toy, arguments: dict[str, Any]) -> None:
    graph = EditPlan(model).scale(SITE, 2.0).to_graph()
    graph["steps"][0]["arguments"].update(arguments)
    with pytest.raises(REFUSED):
        apply_edit_graph(model, graph)


def test_region_edit_changes_exactly_that_region(model: Toy) -> None:
    out = EditPlan(model).scale(SITE, 2.0, region=Region(rows=(1, 3), cols=(0, 2))).preview().edited_tensors[SITE]
    expected = model.tensors[SITE].clone()
    expected[1:3, 0:2] *= 2.0
    assert torch.equal(out, expected)


# ---------------------------------------------------------------- shape changes


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.add(SITE, _matrix(shape=(3, 4))),
        lambda p: p.add(SITE, _matrix(shape=(1, 1))),
        lambda p: p.replace_slice(SITE, _matrix(shape=(2, 2)), region=Region(rows=(0, 1))),
        lambda p: p.mask(SITE, _matrix(shape=(4, 2))),
        lambda p: p.low_rank(SITE, _matrix(shape=(3, 1)), _matrix(shape=(1, 3))),
        lambda p: p.low_rank(SITE, _matrix(shape=(4, 1)), _matrix(shape=(2, 3))),
        lambda p: p.linear_map(SITE, _matrix(shape=(3, 4))),
        lambda p: p.linear_map(SITE, torch.zeros(3, 3, dtype=torch.float64), side="sideways"),
        lambda p: p.project(SITE, _matrix(1.0, shape=(4, 1))),
        lambda p: p.project(SITE, torch.eye(4, 2, dtype=torch.float64), side="sideways"),
        lambda p: p.permute(SITE, 0, [0, 1, 2]),
        lambda p: p.permute(SITE, 0, [0, 0, 1, 2]),
        lambda p: p.permute(SITE, 0, [0, 1, 2, -1]),
        lambda p: p.permute(SITE, 0, [0, 1, 2, 4]),
        lambda p: p.permute(SITE, 2, [0, 1, 2]),
    ],
    ids=[
        "add_transposed", "add_scalar", "slice_wrong_size", "mask_wrong", "low_rank_rows", "low_rank_rank_mismatch",
        "map_not_square", "map_bad_side", "project_not_orthonormal", "project_bad_side",
        "perm_short", "perm_duplicate", "perm_negative", "perm_out_of_range", "perm_axis",
    ],
)
def test_operands_that_do_not_fit_are_refused(model: Toy, build: Callable[[EditPlan], Any]) -> None:
    with pytest.raises(REFUSED):
        build(EditPlan(model)).preview()


def test_direct_edit_of_a_vector_is_refused(model: Toy) -> None:
    with pytest.raises(ValueError, match="matrix"):
        EditPlan(model).scale(BIAS, 2.0)


def test_unknown_tensor_is_refused(model: Toy) -> None:
    with pytest.raises(ValueError, match="Unknown tensor"):
        EditPlan(model).scale("nope", 2.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.adopt_weight(SITE, _matrix(1.0, shape=(1, 1))),
        lambda p: p.adopt_weight(SITE, torch.ones(3, dtype=torch.float64)),
        lambda p: p.interpolate(SITE, _matrix(1.0, shape=(1, 3)), 0.5),
    ],
    ids=["adopt_scalar_matrix", "adopt_row", "interpolate_row"],
)
@xfail("D5 edit_plan.py: adopt_weight and interpolate broadcast the operand, so a 1x1 matrix overwrites the whole tensor")
def test_derived_edits_do_not_broadcast_a_wrong_shape_over_the_tensor(model: Toy, build: Callable[[EditPlan], Any]) -> None:
    with pytest.raises(REFUSED):
        build(EditPlan(model)).preview()


# ---------------------------------------------------------------- tied weights


def _tied_model() -> Toy:
    toy = Toy(config={"model_type": "toy", "tie_word_embeddings": True})
    toy.tensors[TWIN] = toy.tensors[SITE]
    return toy


@xfail("D6 edit_plan.py: an edit changes one of two tied tensors and the tie is silently lost")
def test_edit_of_a_tied_weight_does_not_silently_split_the_tie() -> None:
    toy = _tied_model()
    try:
        out = EditPlan(toy).scale(SITE, 2.0).preview().model
    except REFUSED:
        return
    assert torch.equal(out.tensors[SITE], out.tensors[TWIN])


def test_graph_cannot_be_redirected_to_a_tensor_with_identical_bytes() -> None:
    toy = Toy()
    toy.tensors[TWIN] = toy.tensors[SITE].clone()
    graph = EditPlan(toy).scale(SITE, 2.0).to_graph()
    graph["steps"][0]["tensor"] = TWIN
    with pytest.raises(EditGraphError):
        apply_edit_graph(toy, graph)


# ---------------------------------------------------------------- non-finite and huge operands


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.scale(SITE, NAN),
        lambda p: p.scale(SITE, INF),
        lambda p: p.scale(SITE, 1e308).scale(SITE, 1e308),
        lambda p: p.add(SITE, _matrix(NAN)),
        lambda p: p.add(SITE, _matrix(INF)),
        lambda p: p.add(SITE, _matrix(1.7e308)).add(SITE, _matrix(1.7e308)),
        lambda p: p.low_rank(SITE, _matrix(1e200, (4, 1)), _matrix(1e200, (1, 3))),
        lambda p: p.low_rank(SITE, _matrix(1.0, (4, 1)), _matrix(1.0, (1, 3)), strength=INF),
        lambda p: p.replace_slice(SITE, _matrix(NAN)),
        lambda p: p.mask(SITE, _matrix(NAN)),
        pytest.param(lambda p: p.mask(SITE, _matrix(INF)), marks=xfail("D15 ir.py: an infinite mask escapes as OverflowError instead of a refusal")),
        lambda p: p.linear_map(SITE, _matrix(NAN, (4, 4))),
        lambda p: p.project(SITE, torch.eye(4, 2, dtype=torch.float64), strength=NAN),
        lambda p: p.spectral_shift(SITE, INF),
    ],
    ids=[
        "scale_nan", "scale_inf", "scale_overflow", "add_nan", "add_inf", "add_overflow", "low_rank_overflow",
        "low_rank_inf", "slice_nan", "mask_nan", "mask_inf", "map_nan", "project_nan", "shift_inf",
    ],
)
def test_non_finite_operands_and_results_are_refused(model: Toy, build: Callable[[EditPlan], Any]) -> None:
    with pytest.raises(REFUSED):
        build(EditPlan(model)).preview()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: p.spectral_clip(SITE, NAN), marks=xfail("D7 edit_plan.py: NaN clip limit is accepted")),
        lambda p: p.spectral_clip(SITE, -1.0),
        pytest.param(lambda p: p.remove_noise_floor(SITE, NAN), marks=xfail("D7 edit_plan.py: NaN threshold zeroes the whole tensor")),
        lambda p: p.soft_threshold(SITE, NAN),
        lambda p: p.clamp(SITE, NAN),
        lambda p: p.spectral_shift(SITE, NAN),
    ],
    ids=["clip_nan", "clip_negative", "noise_floor_nan", "threshold_nan", "clamp_nan", "shift_nan"],
)
def test_derived_edit_parameters_must_be_meaningful(model: Toy, build: Callable[[EditPlan], Any]) -> None:
    with pytest.raises(REFUSED):
        build(EditPlan(model)).preview()


def test_huge_finite_add_is_exact_or_refused(model: Toy) -> None:
    delta = _matrix(1e300)
    try:
        out = EditPlan(model).add(SITE, delta).preview().edited_tensors[SITE]
    except REFUSED:
        return
    assert torch.equal(out, model.tensors[SITE] + delta)


@pytest.mark.parametrize("field", ["max_frobenius_delta", "max_spectral_delta", "max_parameter_magnitude", "max_changed_parameters"])
@xfail("D3 edit_plan.py: value > NaN is False, so a NaN limit disables the constraint")
def test_nan_constraint_is_not_treated_as_no_limit(model: Toy, field: str) -> None:
    with pytest.raises((EditConstraintViolation, ValueError)):
        EditPlan(model, EditConstraints(**{field: NAN})).scale(SITE, 2.0).preview()


@xfail("D4 edit_plan.py: tensor constraints are only checked when a step is added")
def test_constraint_added_after_the_step_still_applies(model: Toy) -> None:
    plan = EditPlan(model).scale(SITE, 2.0).add_constraint(forbidden_tensors=(SITE,))
    with pytest.raises(EditConstraintViolation):
        plan.preview()


def test_constraint_added_after_preview_is_not_skipped_by_the_cache(model: Toy) -> None:
    plan = EditPlan(model).scale(SITE, 2.0)
    plan.preview()
    plan.add_constraint(max_frobenius_delta=1e-9)
    with pytest.raises(EditConstraintViolation):
        plan.execute()


def test_failed_step_leaves_no_candidate_and_no_change(model: Toy) -> None:
    before = {k: v.clone() for k, v in model.tensors.items()}
    plan = EditPlan(model).scale(SITE, 2.0).add(TWIN, _matrix(NAN))
    with pytest.raises(REFUSED):
        plan.preview()
    assert plan._last is None
    assert all(torch.equal(model.tensors[k], v) for k, v in before.items())


# ---------------------------------------------------------------- replay


def test_repeated_replay_is_bit_identical(model: Toy) -> None:
    plan = EditPlan(model).scale(SITE, 1 / 3).spectral_clip(TWIN, 0.5).rank_truncate(SITE, 2)
    original = plan.execute()
    graph = json.loads(json.dumps(plan.to_graph()))
    for _ in range(5):
        again = apply_edit_graph(Toy(), graph)
        assert again.plan_digest == original.plan_digest
        for name, weight in original.edited_tensors.items():
            assert torch.equal(again.edited_tensors[name], weight)
            assert weight.tobytes() == again.edited_tensors[name].tobytes() if hasattr(weight, "tobytes") else True


def test_signed_graph_survives_a_canonical_round_trip_exactly(model: Toy, keys: dict[str, Any]) -> None:
    plan = EditPlan(model).scale(SITE, 0.1 + 0.2).add(TWIN, _matrix(1 / 3))
    envelope = json.loads(json.dumps(sign_edit_graph(plan.to_graph(), signing_key=keys["private_key"])))
    replayed = apply_signed_edit_graph(Toy(), envelope, public_key=keys["public_key"])
    assert torch.equal(replayed.edited_tensors[SITE], plan.execute().edited_tensors[SITE])


def test_replay_does_not_modify_the_base_or_the_graph(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    snapshot, before = copy.deepcopy(graph), {k: v.clone() for k, v in model.tensors.items()}
    apply_edit_graph(model, graph)
    assert graph == snapshot and all(torch.equal(model.tensors[k], v) for k, v in before.items())


def _swap(graph: dict[str, Any]) -> None:
    graph["steps"][0], graph["steps"][1] = graph["steps"][1], graph["steps"][0]


GRAPH_REWRITES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "swap_adjacent": _swap,
    "swap_commuting_tensors": lambda g: g["steps"].insert(0, g["steps"].pop(1)),
    "remove_first": lambda g: g["steps"].pop(0),
    "remove_last": lambda g: g["steps"].pop(),
    "remove_middle": lambda g: g["steps"].pop(1),
    "duplicate_step": lambda g: g["steps"].append(copy.deepcopy(g["steps"][-1])),
    "empty_steps": lambda g: g["steps"].clear(),
    "stale_before": lambda g: g["steps"][0].update(before_parameter_digest="sha256:" + "0" * 64),
    "stale_after": lambda g: g["steps"][0].update(after_parameter_digest="sha256:" + "0" * 64),
    "stale_plan_digest": lambda g: g.update(plan_digest="sha256:" + "0" * 64),
    "no_plan_digest": lambda g: g.pop("plan_digest"),
    "changed_factor": lambda g: g["steps"][0]["arguments"].update(factor=0.75),
    "changed_operand": lambda g: g["steps"][1]["operands"]["delta_matrix"].update(
        data_b64=base64.b64encode(struct.pack("<12d", *[0.5] * 12)).decode()
    ),
    "wrong_kind": lambda g: g.update(kind="nf_edit_graph_v2"),
    "wrong_version": lambda g: g.update(schema_version=2),
    "planted_argument": lambda g: g["steps"][0]["arguments"].update(engine_name="x"),
    "planted_operand": lambda g: g["steps"][0]["operands"].update(junk=g["steps"][1]["operands"]["delta_matrix"]),
    "unknown_tensor": lambda g: g["steps"][0].update(tensor="nope"),
}


@pytest.mark.parametrize("name", GRAPH_REWRITES)
def test_rewritten_graph_is_refused(model: Toy, name: str) -> None:
    graph = _graph_plan(model).to_graph()
    GRAPH_REWRITES[name](graph)
    with pytest.raises(EditGraphError):
        apply_edit_graph(model, graph)


@pytest.mark.parametrize("name", ["swap_adjacent", "remove_last", "changed_factor", "planted_argument"])
def test_signed_graph_cannot_be_rewritten(model: Toy, keys: dict[str, Any], name: str) -> None:
    envelope = sign_edit_graph(_graph_plan(model).to_graph(), signing_key=keys["private_key"])
    GRAPH_REWRITES[name](envelope["patch"])
    with pytest.raises(ValueError, match="Tampered"):
        apply_signed_edit_graph(model, envelope, public_key=keys["public_key"])


def _set_operand(graph: dict[str, Any], **fields: Any) -> None:
    graph["steps"][1]["operands"]["delta_matrix"].update(fields)


MALFORMED_GRAPHS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "step_not_object": lambda g: g["steps"].__setitem__(0, "scale"),
    "step_null": lambda g: g["steps"].__setitem__(0, None),
    "no_tensor": lambda g: g["steps"][0].pop("tensor"),
    "no_arguments": lambda g: g["steps"][0].pop("arguments"),
    "no_operands": lambda g: g["steps"][0].pop("operands"),
    "no_before": lambda g: g["steps"][0].pop("before_parameter_digest"),
    "operands_list": lambda g: g["steps"][1].update(operands=[1]),
    "arguments_list": lambda g: g["steps"][0].update(arguments=[1]),
    "no_operation_type": lambda g: g["steps"][0]["arguments"].pop("operation_type"),
    "bad_operation_type": lambda g: g["steps"][0]["arguments"].update(operation_type="format_disk"),
    "factor_string": lambda g: g["steps"][0]["arguments"].update(factor="2.0"),
    "operand_not_object": lambda g: g["steps"][1]["operands"].update(delta_matrix=7),
    "operand_no_data": lambda g: g["steps"][1]["operands"]["delta_matrix"].pop("data_b64"),
    "operand_data_int": lambda g: _set_operand(g, data_b64=7),
    "operand_bad_base64": lambda g: _set_operand(g, data_b64="!!!not base64!!!"),
    "operand_short": lambda g: _set_operand(g, data_b64=base64.b64encode(b"\0" * 8).decode()),
    "operand_negative_shape": lambda g: _set_operand(g, shape=[-4, -3]),
    "operand_rank3": lambda g: _set_operand(g, shape=[4, 3, 1]),
    "operand_huge_shape": lambda g: _set_operand(g, shape=[2**40, 2**40]),
    "operand_string_shape": lambda g: _set_operand(g, shape="43"),
    "operand_dtype": lambda g: _set_operand(g, dtype="f32"),
    "steps_object": lambda g: g.update(steps={"0": 1}),
}


MALFORMED_GRAPH_D13 = "D13 edit_plan.py: a malformed graph escapes as KeyError, TypeError, AttributeError or RuntimeError"


@pytest.mark.parametrize(
    "name",
    _marked(
        MALFORMED_GRAPHS,
        {
            **dict.fromkeys(
                [
                    "step_not_object", "step_null", "no_tensor", "no_arguments", "no_operands", "no_before", "operands_list",
                    "no_operation_type", "operand_not_object", "operand_no_data", "operand_data_int", "operand_negative_shape",
                ],
                MALFORMED_GRAPH_D13,
            ),
            "operand_string_shape": "D13 edit_plan.py: the string shape '43' is read as [4, 3]",
        },
    ),
)
def test_malformed_graph_gets_a_defined_refusal(model: Toy, name: str) -> None:
    graph = _graph_plan(model).to_graph()
    MALFORMED_GRAPHS[name](graph)
    with pytest.raises(REFUSED):
        apply_edit_graph(model, graph)


def test_graph_operand_limit_is_enforced(model: Toy, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph_plan(model).to_graph()
    monkeypatch.setattr(edit_plan, "EDIT_GRAPH_MAX_OPERAND_BYTES", 8)
    with pytest.raises(EditGraphError, match="size limit"):
        apply_edit_graph(model, graph)


def test_base_model_differing_in_the_edited_tensor_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    other = Toy(seed=4)
    with pytest.raises(EditGraphError, match="base model"):
        apply_edit_graph(other, graph)


def test_base_model_differing_outside_the_edited_tensors_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    other = Toy()
    other.tensors[BIAS] = other.tensors[BIAS] + 1.0
    with pytest.raises(EditGraphError):
        apply_edit_graph(other, graph)


def test_base_model_with_extra_tensors_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    other = Toy()
    other.tensors["backdoor.weight"] = torch.ones(2, 2, dtype=torch.float64)
    with pytest.raises(EditGraphError):
        apply_edit_graph(other, graph)


@xfail("D1 edit_plan.py: base_model_id is never compared with anything")
def test_graph_declaring_a_different_base_model_id_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    graph["base_model_id"] = "some/other-model"
    with pytest.raises(EditGraphError):
        apply_edit_graph(model, graph)


@xfail("D14 edit_plan.py: depends_on is written but never checked")
def test_graph_with_a_forward_dependency_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    graph["steps"][0]["depends_on"] = ["step_2"]
    with pytest.raises(EditGraphError):
        apply_edit_graph(model, graph)


def test_graph_with_a_relabelled_operation_is_refused(model: Toy) -> None:
    graph = _graph_plan(model).to_graph()
    graph["steps"][0]["operation"] = "mask"
    with pytest.raises(EditGraphError):
        apply_edit_graph(model, graph)


@xfail("D8 native canonical parser: signing refuses over 1 MiB, so a 400x400 add_delta graph can't be signed despite the 256 MiB operand limit")
def test_signed_graph_for_a_real_sized_matrix_can_be_replayed(keys: dict[str, Any]) -> None:
    toy = Toy()
    toy.tensors = {SITE: torch.ones(400, 400, dtype=torch.float64)}
    graph = EditPlan(toy).add(SITE, torch.full((400, 400), 0.5, dtype=torch.float64)).to_graph()
    envelope = sign_edit_graph(graph, signing_key=keys["private_key"])
    assert apply_signed_edit_graph(toy, envelope, public_key=keys["public_key"]).changed_parameters == 160_000


# ---------------------------------------------------------------- signatures


def _body() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "nf_open_weight_patch_v1",
        "format_version": "1.0.0",
        "base_model": {"model_id": "m", "checkpoint": "sha256:" + "1" * 64, "parameter_digest": "sha256:" + "1" * 64},
        "candidate_checkpoint_digest": "sha256:" + "2" * 64,
        "total_parameters": 100,
        "operations": [{"id": "op_0", "operation": {"type": "low_rank_update", "target": {"tensor": "t"}, "frobenius_norm": 1.0}}],
        "sbom": {
            "base_checkpoint_digest": "sha256:" + "1" * 64,
            "candidate_checkpoint_digest": "sha256:" + "2" * 64,
            "changed_parameters": 4,
            "total_parameters": 100,
        },
        "rounding_contract": {"source_dtype": "f32", "accumulation_dtype": "f32", "rounding_mode": "nearest_ties_even"},
        "security_scan": {"backdoor_scan_passed": True},
        "evaluation_provenance": {
            "promote_eligible": True,
            "controls": {"candidate_accuracy": 0.9, "noop_accuracy": 0.1, "random_accuracy": 0.1},
        },
    }


def _open_patch(body: dict[str, Any], key: dict[str, Any], signer: str = "nf.test") -> dict[str, Any]:
    envelope = patch_ux.sign_patch_envelope(body, signing_key=key["private_key"], signer_id=signer)
    return {**body, "patch_digest": envelope["patch_digest"], "signature_envelope": envelope}


def _malleate(signature: str) -> str:
    """Add the group order to S: the classic Ed25519 malleation, same message and key."""
    prefix, raw = signature.split(":")
    data = bytes.fromhex(raw)
    order = 2**252 + 27742317777372353535851937790883648493
    s = int.from_bytes(data[32:], "little") + order
    return f"{prefix}:{data[:32].hex()}{s.to_bytes(32, 'little').hex()}"


ENVELOPE_TAMPERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "signer_swap": lambda e: e.update(signer_id="nf.someone_else"),
    "signer_removed": lambda e: e.pop("signer_id"),
    "signer_int": lambda e: e.update(signer_id=123),
    "signature_removed": lambda e: e.pop("signature"),
    "signature_empty": lambda e: e.update(signature=""),
    "signature_bit_flip": lambda e: e.update(signature=e["signature"][:-1] + ("0" if e["signature"][-1] != "0" else "1")),
    "signature_malleated": lambda e: e.update(signature=_malleate(e["signature"])),
    "signature_uppercase": lambda e: e.update(signature=e["signature"].upper()),
    "scheme_none": lambda e: e.update(signature_scheme="none"),
    "scheme_removed": lambda e: e.pop("signature_scheme"),
    "digest_swap": lambda e: e.update(patch_digest="sha256:" + "0" * 64),
    "kind_swap": lambda e: e.update(kind="nf_artifact_manifest"),
    "patch_removed": lambda e: e.pop("patch"),
    "patch_string": lambda e: e.update(patch="{}"),
    "future_schema": lambda e: e.update(schema_version=2),
    "unknown_field": lambda e: e.update(trusted=True),
}


@pytest.mark.parametrize("name", ENVELOPE_TAMPERS)
def test_tampered_envelope_is_refused(keys: dict[str, Any], name: str) -> None:
    envelope = patch_ux.sign_patch_envelope({"a": 1}, signing_key=keys["private_key"], signer_id="123")
    ENVELOPE_TAMPERS[name](envelope)
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(envelope, public_key=keys["public_key"])


def test_signature_cannot_move_to_another_document(keys: dict[str, Any]) -> None:
    first = patch_ux.sign_patch_envelope({"a": 1}, signing_key=keys["private_key"])
    second = patch_ux.sign_patch_envelope({"a": 2}, signing_key=keys["private_key"])
    moved = {**second, "signature": first["signature"]}
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(moved, public_key=keys["public_key"])


def test_signature_cannot_move_to_another_signer_id_or_key(keys: dict[str, Any], other_keys: dict[str, Any]) -> None:
    alice = patch_ux.sign_patch_envelope({"a": 1}, signing_key=keys["private_key"], signer_id="alice")
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature({**alice, "signer_id": "bob"}, public_key=keys["public_key"])
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(alice, public_key=other_keys["public_key"])
    forged = {**alice, "public_key": other_keys["public_key"], "key_fingerprint": other_keys["key_fingerprint"]}
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(forged, public_key=other_keys["public_key"])


@pytest.mark.parametrize("key", ["", "ed25519_pub:zz", "ed25519_priv:" + "0" * 64, "ed25519_pub:" + "0" * 63])
def test_malformed_trusted_key_is_refused(keys: dict[str, Any], key: str) -> None:
    envelope = patch_ux.sign_patch_envelope({"a": 1}, signing_key=keys["private_key"])
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(envelope, public_key=key)


def test_valid_signature_from_the_right_key_cannot_cross_document_types(model: Toy, keys: dict[str, Any], tmp_path: Path) -> None:
    public = keys["public_key"]
    graph_envelope = sign_edit_graph(_graph_plan(model).to_graph(), signing_key=keys["private_key"])
    with pytest.raises(ValueError, match="schema"):
        patch_ux.verify_open_patch_v1(graph_envelope, public_key=public)

    patch_envelope = _open_patch(_body(), keys)["signature_envelope"]
    with pytest.raises(EditGraphError, match="nf_edit_graph_v1"):
        apply_signed_edit_graph(model, patch_envelope, public_key=public)

    (tmp_path / provenance.ARTIFACT_MANIFEST_NAME).write_bytes(canonicalize(patch_envelope))
    (tmp_path / "x.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="malformed"):
        provenance.verify_artifact_manifest(tmp_path, public_key=public)

    path = provenance.write_artifact_manifest(tmp_path, signing_key=keys["private_key"], signer_id="s", open_patch_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="schema"):
        patch_ux.verify_open_patch_v1(json.loads(path.read_bytes()), public_key=public)
    with pytest.raises(EditGraphError):
        apply_signed_edit_graph(model, json.loads(path.read_bytes()), public_key=public)


# ---------------------------------------------------------------- Open Patch: signed but invalid


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(body: dict[str, Any]) -> None:
        parts = path.split(".")
        node: Any = body
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else node[part]
        node[parts[-1]] = value

    return apply


def _both_sets(*pairs: tuple[str, Any]) -> Callable[[dict[str, Any]], None]:
    def apply(body: dict[str, Any]) -> None:
        for path, value in pairs:
            _set(path, value)(body)

    return apply


_OP = "operations.0.operation"
_CTRL = "evaluation_provenance.controls"
SIGNED_BUT_INVALID: dict[str, Callable[[dict[str, Any]], None]] = {
    "schema_version_true": _set("schema_version", True),
    "schema_version_float": _set("schema_version", 1.0),
    "format_version_future": _set("format_version", "2.0.0"),
    "format_version_null": _set("format_version", None),
    "kind_other": _set("kind", "something_else"),
    "unknown_key": _set("trusted", True),
    "missing_section": lambda b: b.pop("rounding_contract"),
    "base_model_string": _set("base_model", "m"),
    "digest_uppercase": _both_sets(("candidate_checkpoint_digest", "sha256:" + "A" * 64), ("sbom.candidate_checkpoint_digest", "sha256:" + "A" * 64)),
    "digest_short": _both_sets(("candidate_checkpoint_digest", "sha256:ab"), ("sbom.candidate_checkpoint_digest", "sha256:ab")),
    "sbom_base_mismatch": _set("sbom.base_checkpoint_digest", "sha256:" + "3" * 64),
    "counts_booleans": _both_sets(("total_parameters", True), ("sbom.total_parameters", True), ("sbom.changed_parameters", True)),
    "counts_float": _both_sets(("total_parameters", 100.0), ("sbom.total_parameters", 100.0)),
    "changed_zero": _set("sbom.changed_parameters", 0),
    "changed_negative": _set("sbom.changed_parameters", -1),
    "changed_over_total": _set("sbom.changed_parameters", 101),
    "total_disagrees": _set("sbom.total_parameters", 99),
    "operations_empty": _set("operations", []),
    "operation_not_object": _set("operations", [1]),
    "op_type_empty": _set(f"{_OP}.type", ""),
    "op_type_number": _set(f"{_OP}.type", 1),
    "op_target_string": _set(f"{_OP}.target", "t"),
    "op_target_empty": _set(f"{_OP}.target", {}),
    "metric_negative": _set(f"{_OP}.frobenius_norm", -1.0),
    "metric_string": _set(f"{_OP}.linf_norm", "1"),
    "metric_boolean": _set(f"{_OP}.l2_norm", True),
    "metric_null": _set(f"{_OP}.frobenius_norm", None),
    "rounding_empty": _set("rounding_contract", {}),
    "rounding_null_field": _set("rounding_contract.rounding_mode", None),
    "rounding_empty_field": _set("rounding_contract.source_dtype", ""),
    "rounding_number": _set("rounding_contract.accumulation_dtype", 7),
    "scan_failed": _set("security_scan.backdoor_scan_passed", False),
    "scan_string": _set("security_scan.backdoor_scan_passed", "true"),
    "scan_int": _set("security_scan.backdoor_scan_passed", 1),
    "promote_false": _set("evaluation_provenance.promote_eligible", False),
    "promote_int": _set("evaluation_provenance.promote_eligible", 1),
    "controls_missing": lambda b: b["evaluation_provenance"].pop("controls"),
    "controls_list": _set(_CTRL, [0.9, 0.1, 0.1]),
    "control_missing": lambda b: b["evaluation_provenance"]["controls"].pop("random_accuracy"),
    "control_equal": _set(f"{_CTRL}.candidate_accuracy", 0.1),
    "control_worse": _set(f"{_CTRL}.noop_accuracy", 0.95),
    "control_range": _set(f"{_CTRL}.candidate_accuracy", 1.5),
    "control_string": _set(f"{_CTRL}.candidate_accuracy", "0.9"),
    "control_boolean": _set(f"{_CTRL}.candidate_accuracy", True),
    "control_null": _set(f"{_CTRL}.noop_accuracy", None),
}


def _signed_invalid(name: str, keys: dict[str, Any]) -> dict[str, Any]:
    body = _body()
    SIGNED_BUT_INVALID[name](body)
    return _open_patch(body, keys)


def _run_nf(*args: str) -> subprocess.CompletedProcess[str]:
    if not CLI.exists():
        pytest.skip(f"NF_CLI not built at {CLI}")
    return subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=60)


def _nf_verify(path: Path, public_key: str) -> bool:
    result = _run_nf("verify", str(path), "--pubkey", public_key)
    accepted = result.returncode == 0 and '"verified":true' in result.stdout.replace(" ", "")
    assert accepted or result.returncode != 0, result.stdout
    return accepted


def test_control_a_valid_open_patch_is_accepted_by_both(keys: dict[str, Any], tmp_path: Path) -> None:
    doc = _open_patch(_body(), keys)
    path = tmp_path / "p.json"
    path.write_bytes(json.dumps(doc).encode())
    assert patch_ux.verify_open_patch_v1(doc, public_key=keys["public_key"])["verified"] is True
    assert patch_ux.verify_open_patch_v1(path, public_key=keys["public_key"])["verified"] is True
    assert _nf_verify(path, keys["public_key"]) is True


@pytest.mark.parametrize("name", SIGNED_BUT_INVALID)
def test_python_refuses_signed_but_invalid_patch(keys: dict[str, Any], name: str) -> None:
    with pytest.raises(ValueError):
        patch_ux.verify_open_patch_v1(_signed_invalid(name, keys), public_key=keys["public_key"])


NF_SCHEMA_GAPS = {
    "format_version_future": "D9 studio.rs: format_version isn't checked",
    "format_version_null": "D9 studio.rs: format_version isn't checked",
    "op_type_empty": "D9 studio.rs: an empty operation type is accepted",
    "op_target_empty": "D9 studio.rs: an operation target naming no tensor is accepted",
    "metric_null": "D9 studio.rs: a null metric is accepted",
    "rounding_null_field": "D9 studio.rs: rounding_contract fields only have to exist, so null passes",
    "rounding_empty_field": "D9 studio.rs: rounding_contract fields only have to exist, so an empty string passes",
    "rounding_number": "D9 studio.rs: rounding_contract fields only have to exist, so a number passes",
}


@pytest.mark.parametrize("name", _marked(SIGNED_BUT_INVALID, NF_SCHEMA_GAPS))
def test_nf_verify_refuses_signed_but_invalid_patch(keys: dict[str, Any], name: str, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(json.dumps(_signed_invalid(name, keys)).encode())
    assert _nf_verify(path, keys["public_key"]) is False


# ---------------------------------------------------------------- Open Patch: hostile bytes and wrappers


def _dup_top(keys: dict[str, Any]) -> bytes:
    text = json.dumps(_open_patch(_body(), keys))
    return text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1).encode()


def _dup_signature(keys: dict[str, Any]) -> bytes:
    text = json.dumps(_open_patch(_body(), keys))
    return text.replace('"signature_envelope": {', '"signature_envelope": {"signer_id": "x", ', 1).encode()


def _dup_hides_failed_scan(keys: dict[str, Any]) -> bytes:
    text = json.dumps(_open_patch(_body(), keys))
    return text.replace('"backdoor_scan_passed": true', '"backdoor_scan_passed": false, "backdoor_scan_passed": true').encode()


def _bytes_cases() -> dict[str, Callable[[dict[str, Any]], bytes]]:
    valid = lambda k: json.dumps(_open_patch(_body(), k)).encode()  # noqa: E731
    return {
        "duplicate_top_level": _dup_top,
        "duplicate_in_envelope": _dup_signature,
        "duplicate_hides_failed_scan": _dup_hides_failed_scan,
        "utf8_bom": lambda k: b"\xef\xbb\xbf" + valid(k),
        "utf16": lambda k: valid(k).decode().encode("utf-16"),
        "utf32": lambda k: valid(k).decode().encode("utf-32"),
        "trailing_garbage": lambda k: valid(k) + b"garbage",
        "second_document": lambda k: valid(k) + b"{}",
        "truncated": lambda k: valid(k)[:-20],
        "empty": lambda k: b"",
        "invalid_utf8": lambda k: b'{"a":"\xff"}',
        "top_level_array": lambda k: b"[]",
        "top_level_number": lambda k: b"7",
        "top_level_null": lambda k: b"null",
        "nan_constant": lambda k: valid(k).replace(b"0.9", b"NaN"),
        "infinity_constant": lambda k: valid(k).replace(b"0.9", b"Infinity"),
        "float_overflow": lambda k: valid(k).replace(b"0.9", b"1e999"),
        "huge_integer": lambda k: valid(k).replace(b'"total_parameters": 100', b'"total_parameters": 1' + b"0" * 40),
        "deep_nesting_100k": lambda k: b"[" * 100_000 + b"]" * 100_000,
        "deep_nesting_100": lambda k: b"[" * 100 + b"]" * 100,
        "over_one_mib": lambda k: json.dumps({**_open_patch(_body(), k), "pad": "x" * 2_000_000}).encode(),
    }


@pytest.mark.parametrize("name", _bytes_cases())
def test_python_refuses_hostile_bytes_with_a_value_error(keys: dict[str, Any], name: str, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(_bytes_cases()[name](keys))
    with pytest.raises(ValueError):
        patch_ux.verify_open_patch_v1(path, public_key=keys["public_key"])


@pytest.mark.parametrize("name", _bytes_cases())
def test_nf_verify_refuses_hostile_bytes(keys: dict[str, Any], name: str, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(_bytes_cases()[name](keys))
    assert _nf_verify(path, keys["public_key"]) is False


def test_file_over_the_artifact_limit_is_refused_before_reading(keys: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    with path.open("wb") as handle:
        handle.truncate(patch_ux.OPEN_PATCH_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="maximum artifact size"):
        patch_ux.verify_open_patch_v1(path, public_key=keys["public_key"])
    assert _nf_verify(path, keys["public_key"]) is False


@pytest.mark.parametrize(
    "document",
    [lambda: {"a": {"b": 1}}, lambda: _deep_mapping(200), lambda: {"pad": "x" * 2_000_000}],
    ids=["not_a_patch", "deep_mapping", "oversized_mapping"],
)
def test_python_refuses_hostile_in_memory_documents_with_a_value_error(keys: dict[str, Any], document: Callable[[], dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        patch_ux.verify_open_patch_v1(document(), public_key=keys["public_key"])


def _deep_mapping(depth: int) -> dict[str, Any]:
    node: dict[str, Any] = {}
    for _ in range(depth):
        node = {"n": node}
    return {"kind": "nf_signed_patch_envelope", "patch": node, "patch_digest": "", "signature_scheme": "ed25519_v1"}


WRAPPER_ATTACKS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] = {
    "unsigned_body_edit": lambda d, k: d["security_scan"].update(note="unsigned"),
    "unsigned_extra_key": lambda d, k: d.update(trusted=True),
    "unsigned_number_alias": lambda d, k: d.update(total_parameters=100.0),
    "unsigned_top_digest": lambda d, k: d.update(patch_digest="sha256:" + "0" * 64),
    "envelope_future_schema": lambda d, k: d["signature_envelope"].update(schema_version=2),
    "envelope_unknown_field": lambda d, k: d["signature_envelope"].update(trusted=True),
    "envelope_is_string": lambda d, k: d.update(signature_envelope="x"),
    "envelope_is_null": lambda d, k: d.update(signature_envelope=None),
    "envelope_from_other_key": lambda d, k: d.update(signature_envelope=_open_patch(_body(), patch_ux.generate_ed25519_keypair())["signature_envelope"]),
}


@pytest.mark.parametrize("name", WRAPPER_ATTACKS)
def test_python_refuses_wrapper_attack(keys: dict[str, Any], name: str) -> None:
    doc = _open_patch(_body(), keys)
    WRAPPER_ATTACKS[name](doc, keys)
    with pytest.raises(ValueError):
        patch_ux.verify_open_patch_v1(doc, public_key=keys["public_key"])


NF_WRAPPER_GAPS = {
    "unsigned_top_digest": "D10 studio.rs: the unsigned top-level patch_digest isn't compared with the signed one",
    "envelope_future_schema": "D10 studio.rs: the envelope's schema_version isn't checked",
    "envelope_unknown_field": "D10 studio.rs: unknown envelope fields are accepted",
}


@pytest.mark.parametrize("name", _marked(WRAPPER_ATTACKS, NF_WRAPPER_GAPS))
def test_nf_verify_refuses_wrapper_attack(keys: dict[str, Any], name: str, tmp_path: Path) -> None:
    doc = _open_patch(_body(), keys)
    WRAPPER_ATTACKS[name](doc, keys)
    path = tmp_path / "p.json"
    path.write_bytes(json.dumps(doc).encode())
    assert _nf_verify(path, keys["public_key"]) is False


def test_nf_verify_requires_a_trusted_key(keys: dict[str, Any], other_keys: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(json.dumps(_open_patch(_body(), keys)).encode())
    assert _run_nf("verify", str(path)).returncode != 0
    assert _nf_verify(path, other_keys["public_key"]) is False


@xfail("D11 studio.rs: patch rollback reports rollback_verified:true after reading one JSON file and restoring nothing")
def test_nf_patch_rollback_does_not_claim_verification_it_did_not_do(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"base_checkpoint_digest": "sha256:" + "1" * 64}))
    result = _run_nf("patch", "rollback", str(path))
    assert result.returncode != 0 or '"rollback_verified":true' not in result.stdout.replace(" ", "")


@xfail("D12 studio.rs: patch audit reports audit_passed:true for any unsigned document")
def test_nf_patch_audit_does_not_pass_an_unsigned_patch(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_body()))
    result = _run_nf("patch", "audit", str(path))
    assert result.returncode != 0 or '"audit_passed":true' not in result.stdout.replace(" ", "")


# ---------------------------------------------------------------- artifact manifests


def _bundle(root: Path, keys: dict[str, Any], digest: str = "sha256:" + "a" * 64) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text("{}")
    (root / "sub").mkdir()
    (root / "sub" / "tok.json").write_text("{}")
    provenance.write_artifact_manifest(root, signing_key=keys["private_key"], signer_id="s", open_patch_digest=digest)
    return root


def _write_files(root: Path, keys: dict[str, Any], files: dict[str, Any], **extra: Any) -> None:
    """A correctly signed manifest with hostile contents: the signer's key may be trusted and still wrong."""
    body = {"schema_version": 1, "kind": "nf_artifact_manifest", "open_patch_digest": "sha256:" + "a" * 64, "files": files, **extra}
    envelope = patch_ux.sign_patch_envelope(body, signing_key=keys["private_key"], signer_id="s")
    (root / provenance.ARTIFACT_MANIFEST_NAME).write_bytes(canonicalize(envelope))


def _try_symlink(link: Path, target: Path | str) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")


def test_control_manifest_round_trip(keys: dict[str, Any], tmp_path: Path) -> None:
    result = provenance.verify_artifact_manifest(_bundle(tmp_path / "b", keys), public_key=keys["public_key"])
    assert result == {"verified": True, "open_patch_digest": "sha256:" + "a" * 64, "files_verified": 3}


@pytest.mark.parametrize(
    "name",
    ["../outside.bin", "../../outside.bin", "sub/../../outside.bin", "/etc/hostname", ".", "..", "", "sub/..", "./config.json", "sub//tok.json", "sub\\tok.json", "C:/Windows/win.ini"],
)
def test_manifest_paths_must_be_plain_and_inside(keys: dict[str, Any], tmp_path: Path, name: str) -> None:
    (tmp_path / "outside.bin").write_bytes(b"x")
    root = _bundle(tmp_path / "b", keys)
    files = {p: provenance.sha256_file(root / p) for p in ("model.safetensors", "config.json", "sub/tok.json")}
    _write_files(root, keys, {**files, name: "sha256:" + "0" * 64})
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_manifest_traversal_with_the_correct_digest_is_still_refused(keys: dict[str, Any], tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    root = _bundle(tmp_path / "b", keys)
    files = {p: provenance.sha256_file(root / p) for p in ("model.safetensors", "config.json", "sub/tok.json")}
    _write_files(root, keys, {**files, "../outside.bin": provenance.sha256_file(outside)})
    with pytest.raises(ValueError, match="escapes|path"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


@pytest.mark.parametrize("build", ["file_out", "dir_out", "file_in", "dir_in", "dangling"])
def test_symlink_planted_after_signing_is_refused(keys: dict[str, Any], tmp_path: Path, build: str) -> None:
    (tmp_path / "secret").write_bytes(b"secret")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "payload.py").write_bytes(b"evil")
    root = _bundle(tmp_path / "b", keys)
    link = root / "planted"
    target = {"file_out": tmp_path / "secret", "dir_out": tmp_path / "dir", "file_in": root / "config.json", "dir_in": root / "sub", "dangling": tmp_path / "missing"}[build]
    _try_symlink(link, target)
    with pytest.raises(ValueError, match="symlink|unlisted"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_listed_file_replaced_by_a_symlink_to_identical_outside_bytes_is_refused(keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", keys)
    outside = tmp_path / "same"
    outside.write_bytes((root / "config.json").read_bytes())
    (root / "config.json").unlink()
    _try_symlink(root / "config.json", outside)
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_listed_file_replaced_by_a_symlink_inside_the_directory_is_refused(keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", keys)
    (root / "sub" / "tok.json").unlink()
    _try_symlink(root / "sub" / "tok.json", root / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_writer_refuses_to_sign_through_a_symlink_or_outside_path(keys: dict[str, Any], tmp_path: Path) -> None:
    (tmp_path / "secret").write_bytes(b"secret")
    root = tmp_path / "b"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    _try_symlink(root / "link", tmp_path / "secret")
    with pytest.raises(ValueError, match="symlink"):
        provenance.write_artifact_manifest(root, signing_key=keys["private_key"], signer_id="s", open_patch_digest="sha256:" + "0" * 64)
    (root / "link").unlink()
    with pytest.raises(ValueError, match="escapes|path"):
        provenance.write_artifact_manifest(root, signing_key=keys["private_key"], signer_id="s", open_patch_digest="sha256:" + "0" * 64, files=["../secret"])
    assert not (root / provenance.ARTIFACT_MANIFEST_NAME).exists()


def test_extra_file_empty_directory_and_nested_manifest(keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", keys)
    (root / "extra.py").write_text("print('hi')")
    with pytest.raises(ValueError, match="unlisted"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])
    (root / "extra.py").unlink()
    (root / "sub" / provenance.ARTIFACT_MANIFEST_NAME).write_text("{}")
    with pytest.raises(ValueError, match="unlisted"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_modified_and_missing_files_are_refused(keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", keys)
    (root / "config.json").write_text('{"a":1}')
    with pytest.raises(ValueError, match="digest mismatch"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])
    (root / "config.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def _manifest_path(root: Path) -> Path:
    return root / provenance.ARTIFACT_MANIFEST_NAME


@pytest.mark.parametrize(
    "content",
    [
        lambda m: b'{"kind":"nf_signed_patch_envelope","kind":"x"}',
        lambda m: b"[]",
        lambda m: b"",
        lambda m: b"[" * 100_000,
        lambda m: b"\xef\xbb\xbf" + m,
        lambda m: m.decode().encode("utf-16"),
        lambda m: m[:-5],
        lambda m: m + b"x",
        lambda m: m.replace(b'"kind":', b'"kind":"x","kind":', 1),
    ],
    ids=["duplicate", "array", "empty", "deep", "bom", "utf16", "truncated", "trailing", "duplicate_in_signed_doc"],
)
def test_hostile_manifest_bytes_get_a_value_error(keys: dict[str, Any], tmp_path: Path, content: Callable[[bytes], bytes]) -> None:
    root = _bundle(tmp_path / "b", keys)
    original = _manifest_path(root).read_bytes()
    _manifest_path(root).write_bytes(content(original))
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_missing_manifest_gets_a_value_error(keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", keys)
    _manifest_path(root).unlink()
    with pytest.raises(ValueError, match="manifest"):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


@pytest.mark.parametrize(
    "extra",
    [{"files": []}, {"files": ["config.json"]}, {"files": {"config.json": 7}}, {"open_patch_digest": 7}, {"open_patch_digest": "not-a-digest"}, {"schema_version": 2}, {"note": "unknown"}],
    ids=["files_empty", "files_list", "digest_int", "patch_digest_int", "patch_digest_text", "future_schema", "unknown_field"],
)
def test_signed_manifest_with_a_malformed_body_is_refused(keys: dict[str, Any], tmp_path: Path, extra: dict[str, Any]) -> None:
    root = tmp_path / "b"
    root.mkdir()
    (root / "config.json").write_text("{}")
    fields = {"files": {"config.json": provenance.sha256_file(root / "config.json")}, **extra}
    _write_files(root, keys, **fields) if "files" in extra else _write_files(root, keys, fields.pop("files"), **extra)
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_manifest_key_must_be_the_trusted_one(keys: dict[str, Any], other_keys: dict[str, Any], tmp_path: Path) -> None:
    root = _bundle(tmp_path / "b", other_keys)
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


@pytest.mark.parametrize("content", [b"[]", b"{}", b'{"patch_digest":"sha256:0"}', b"not json"], ids=["array", "empty", "other_digest", "garbage"])
def test_enclosed_patch_must_be_the_signed_one(keys: dict[str, Any], tmp_path: Path, content: bytes) -> None:
    root = tmp_path / "b"
    root.mkdir()
    (root / "weight_patch.nf.json").write_bytes(content)
    provenance.write_artifact_manifest(root, signing_key=keys["private_key"], signer_id="s", open_patch_digest="sha256:" + "a" * 64)
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(root, public_key=keys["public_key"])


def test_resolve_checkpoint_revision_cannot_escape_the_cache(tmp_path: Path) -> None:
    (tmp_path / "evil").mkdir()
    (tmp_path / "evil" / "config.json").write_text("{}")
    cache = tmp_path / "cache"
    repo = cache / "models--org--m"
    (repo / "snapshots").mkdir(parents=True)
    with pytest.raises((provenance.CheckpointUnavailableError, ValueError)):
        provenance.resolve_checkpoint("org/m", revision="../../../evil", cache_dir=cache)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("../../../evil")
    with pytest.raises((provenance.CheckpointUnavailableError, ValueError)):
        provenance.resolve_checkpoint("org/m", cache_dir=cache)


# ---------------------------------------------------------------- safetensors files


def _st_file(path: Path, header: Any, payload: bytes, *, header_bytes: bytes | None = None) -> None:
    raw = header_bytes if header_bytes is not None else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _toy_st_parts(model: Toy) -> tuple[dict[str, Any], bytes]:
    header: dict[str, Any] = {}
    payload = bytearray()
    for name in sorted(model.tensors):
        tensor = model.tensors[name]
        start = len(payload)
        payload.extend(struct.pack(f"<{tensor.numel()}d", *tensor.reshape(-1).tolist()))
        header[name] = {"dtype": "F64", "shape": list(tensor.shape), "data_offsets": [start, len(payload)]}
    return header, bytes(payload)


def test_control_safetensors_round_trip_is_exact(model: Toy, tmp_path: Path) -> None:
    export_formats.export_huggingface_safetensors(model, tmp_path, dtype="f64")
    reloaded, receipt = export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])
    assert receipt.exported_tensor_count == 3
    assert all(torch.equal(reloaded.tensors[k], v) for k, v in model.tensors.items())


def _offsets(header: dict[str, Any], name: str, offsets: list[int]) -> dict[str, Any]:
    return {**header, name: {**header[name], "data_offsets": offsets}}


def _meta(header: dict[str, Any], name: str, **fields: Any) -> dict[str, Any]:
    return {**header, name: {**header[name], **fields}}


# Each builder returns (header, payload) or raw file bytes for one corrupted checkpoint.
def _st_cases(h: dict[str, Any], p: bytes) -> dict[str, Any]:
    first = sorted(h)[0]
    last = max(h, key=lambda k: h[k]["data_offsets"][1])
    dup_raw = json.dumps(h)[:-1] + ', ' + json.dumps(first) + ": " + json.dumps(h[first]) + "}"
    return {
        "truncated_payload": (h, p[:-8]),
        "payload_empty": (h, b""),
        "trailing_bytes": (h, p + b"\0" * 8),
        "negative_offset_reads_header": (_offsets(h, first, [-h[first]["data_offsets"][1], 0]), p),
        "offset_past_eof": (_offsets(h, last, [len(p), len(p) + 24]), p),
        "inverted_offsets": (_offsets(h, last, [h[last]["data_offsets"][1], h[last]["data_offsets"][0]]), p),
        "overlapping_tensors": (_offsets(h, sorted(h)[1], h[first]["data_offsets"]) if h[sorted(h)[1]]["shape"] == h[first]["shape"] else _offsets(h, last, h[sorted(h, key=lambda k: h[k]["data_offsets"][0])[0]]["data_offsets"]), p),
        "shape_smaller_than_span": (_meta(h, last, shape=[1]), p),
        "shape_larger_than_span": (_meta(h, last, shape=[1000]), p),
        "negative_dimension": (_meta(h, last, shape=[-1, -1]), p),
        "huge_dimension": (_meta(h, last, shape=[2**40, 2**40]), p),
        "fractional_dimension": (_meta(h, last, shape=[1.5]), p),
        "string_shape": (_meta(h, last, shape="3"), p),
        "unknown_dtype": (_meta(h, last, dtype="Q99"), p),
        "dtype_number": (_meta(h, last, dtype=7), p),
        "offsets_missing": ({**h, last: {"dtype": "F64", "shape": h[last]["shape"]}}, p),
        "meta_not_object": ({**h, last: 7}, p),
        "header_array": ([], p),
        "header_number": (7, p),
        "duplicate_tensor": (dup_raw.encode(), p),
        "invalid_json": (b'{"a":', p),
        "invalid_utf8": (b'{"\xff":1}', p),
        "deep_header": (b"[" * 100_000, p),
    }


CASE_NAMES = list(_st_cases(_toy_st_parts(Toy())[0], _toy_st_parts(Toy())[1]))


@pytest.mark.parametrize("name", CASE_NAMES)
def test_corrupt_safetensors_is_refused_on_reload(model: Toy, tmp_path: Path, name: str) -> None:
    head, payload = _toy_st_parts(model)
    case_header, case_payload = _st_cases(head, payload)[name]
    path = tmp_path / "model.safetensors"
    if isinstance(case_header, bytes):
        _st_file(path, None, case_payload, header_bytes=case_header)
    else:
        _st_file(path, case_header, case_payload)
    with pytest.raises(ValueError):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_corrupt_safetensors_is_refused_by_the_mmap_loader(model: Toy, tmp_path: Path, name: str) -> None:
    head, payload = _toy_st_parts(model)
    case_header, case_payload = _st_cases(head, payload)[name]
    path = tmp_path / "model.safetensors"
    if isinstance(case_header, bytes):
        _st_file(path, None, case_payload, header_bytes=case_header)
    else:
        _st_file(path, case_header, case_payload)
    with pytest.raises(ValueError):
        export_formats.ShardedMmapCheckpoint(tmp_path).close()


@pytest.mark.parametrize(
    "raw",
    [b"", b"\x01", struct.pack("<Q", 2**64 - 1), struct.pack("<Q", 2**63) + b"{}", struct.pack("<Q", 1024) + b"{}"],
    ids=["empty", "short_length", "max_length", "i64_length", "length_past_eof"],
)
def test_truncated_safetensors_prefix_is_refused(model: Toy, tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "model.safetensors").write_bytes(raw)
    with pytest.raises(ValueError):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])
    with pytest.raises(ValueError):
        export_formats.ShardedMmapCheckpoint(tmp_path).close()


def test_reload_must_match_the_reference_tensors(model: Toy, tmp_path: Path) -> None:
    head, payload = _toy_st_parts(model)
    path = tmp_path / "model.safetensors"
    # Missing tensor.
    without = dict(head)
    without.pop(max(head, key=lambda k: head[k]["data_offsets"][1]))
    end = max(v["data_offsets"][1] for v in without.values())
    _st_file(path, without, payload[:end])
    with pytest.raises(ValueError, match="missing|differ"):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])
    # Planted tensor.
    extra = {**head, "backdoor": {"dtype": "F64", "shape": [1], "data_offsets": [len(payload), len(payload) + 8]}}
    _st_file(path, extra, payload + b"\0" * 8)
    with pytest.raises(ValueError, match="unexpected|differ"):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])
    # Same bytes, transposed shape.
    transposed = _meta(head, SITE, shape=[3, 4])
    _st_file(path, transposed, payload)
    with pytest.raises(ValueError, match="shape"):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])


@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "inf", "neg_inf"])
def test_non_finite_checkpoint_is_refused_on_reload(model: Toy, tmp_path: Path, value: float) -> None:
    head, payload = _toy_st_parts(model)
    _st_file(tmp_path / "model.safetensors", head, struct.pack("<d", value) + payload[8:])
    with pytest.raises(ValueError, match="non-finite"):
        export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, [])


@pytest.mark.parametrize("shard", ["../evil.safetensors", "/etc/passwd", "sub/x.safetensors", "missing.safetensors"])
def test_shard_paths_cannot_leave_the_checkpoint(model: Toy, tmp_path: Path, shard: str) -> None:
    head, payload = _toy_st_parts(model)
    root = tmp_path / "ck"
    root.mkdir()
    _st_file(tmp_path / "evil.safetensors", head, payload)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {SITE: shard}}))
    with pytest.raises(ValueError):
        export_formats.ShardedMmapCheckpoint(root)


def test_sharded_index_must_agree_with_its_shards(model: Toy, tmp_path: Path) -> None:
    head, payload = _toy_st_parts(model)
    _st_file(tmp_path / "a.safetensors", head, payload)
    _st_file(tmp_path / "b.safetensors", head, payload)
    index = tmp_path / "model.safetensors.index.json"
    # The same tensor in two shards: which one is used must not be decided by file order.
    index.write_text(json.dumps({"weight_map": {SITE: "a.safetensors", TWIN: "b.safetensors"}}))
    with pytest.raises(ValueError, match="more than one|index"):
        export_formats.ShardedMmapCheckpoint(tmp_path)
    # The index names a tensor its shard doesn't hold.
    index.write_text(json.dumps({"weight_map": {"ghost": "a.safetensors"}}))
    with pytest.raises(ValueError, match="index"):
        export_formats.ShardedMmapCheckpoint(tmp_path)
    # Duplicate keys in the index, and malformed shapes of it.
    index.write_text('{"weight_map":{},"weight_map":{"x":"a.safetensors"}}')
    with pytest.raises(ValueError):
        export_formats.ShardedMmapCheckpoint(tmp_path)
    index.write_text("[]")
    with pytest.raises(ValueError):
        export_formats.ShardedMmapCheckpoint(tmp_path)


@pytest.mark.parametrize("value", [NAN, INF], ids=["nan", "inf"])
@pytest.mark.parametrize("target", ["safetensors", "gguf_f32", "gguf_q8"])
def test_non_finite_weights_are_not_exported(tmp_path: Path, value: float, target: str) -> None:
    toy = Toy()
    toy.tensors = {SITE: torch.ones(8, 4, dtype=torch.float64)}
    toy.tensors[SITE][0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        if target == "safetensors":
            export_formats.export_huggingface_safetensors(toy, tmp_path)
        else:
            export_formats.export_gguf_v3(toy, tmp_path / "m.gguf", quantization=target.split("_")[1].replace("q8", "q8_0"))


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_values_too_large_for_the_export_dtype_are_refused(tmp_path: Path, dtype: str) -> None:
    toy = Toy()
    toy.tensors = {SITE: torch.full((2, 2), 1e300, dtype=torch.float64)}
    with pytest.raises((OverflowError, ValueError)):
        export_formats.export_huggingface_safetensors(toy, tmp_path, dtype=dtype)
    with pytest.raises((OverflowError, ValueError)):
        export_formats.export_gguf_v3(toy, tmp_path / "m.gguf", quantization=dtype)


# ---------------------------------------------------------------- GGUF files


def _gguf_string(text: str | bytes, declared: int | None = None) -> bytes:
    raw = text.encode() if isinstance(text, str) else text
    return struct.pack("<Q", len(raw) if declared is None else declared) + raw


def _gguf(
    tensors: list[tuple[str, tuple[int, ...], int, int]],
    data: bytes,
    *,
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int | None = None,
    kv_count: int | None = None,
    alignment: int = 32,
) -> bytes:
    out = bytearray(magic + struct.pack("<IQQ", version, len(tensors) if tensor_count is None else tensor_count, 1 if kv_count is None else kv_count))
    out += _gguf_string("general.alignment") + struct.pack("<II", export_formats.GGUF_TYPE_UINT32, alignment)
    for name, shape, kind, offset in tensors:
        out += _gguf_string(name) + struct.pack("<I", len(shape))
        for dim in shape:
            out += struct.pack("<Q", dim)
        out += struct.pack("<IQ", kind, offset)
    out += b"\0" * (export_formats._align_offset(len(out), max(alignment, 1)) - len(out))
    return bytes(out) + data


def _gguf_toy() -> Toy:
    toy = Toy()
    toy.tensors = {SITE: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)}
    return toy


F32, Q8 = export_formats.GGML_TYPE_F32, export_formats.GGML_TYPE_Q8_0
GOOD_DATA = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
GOOD_TENSOR = (SITE, (2, 2), F32, 0)

GGUF_CASES: dict[str, bytes] = {
    "bad_magic": _gguf([GOOD_TENSOR], GOOD_DATA, magic=b"GGUG"),
    "bad_version": _gguf([GOOD_TENSOR], GOOD_DATA, version=2),
    "future_version": _gguf([GOOD_TENSOR], GOOD_DATA, version=4),
    "tensor_count_huge": _gguf([GOOD_TENSOR], GOOD_DATA, tensor_count=2**63),
    "kv_count_huge": _gguf([GOOD_TENSOR], GOOD_DATA, kv_count=2**63),
    "alignment_zero": _gguf([GOOD_TENSOR], GOOD_DATA, alignment=0),
    "alignment_not_power_of_two": _gguf([GOOD_TENSOR], GOOD_DATA, alignment=7),
    "truncated_tensor_data": _gguf([GOOD_TENSOR], GOOD_DATA[:-4]),
    "no_tensor_data": _gguf([GOOD_TENSOR], b""),
    "trailing_bytes": _gguf([GOOD_TENSOR], GOOD_DATA + b"\0" * 32),
    "offset_past_eof": _gguf([(SITE, (2, 2), F32, 2**40)], GOOD_DATA),
    "offset_near_u64_max": _gguf([(SITE, (2, 2), F32, 2**64 - 8)], GOOD_DATA),
    "unaligned_offset": _gguf([(SITE, (1, 1), F32, 4)], GOOD_DATA),
    "huge_dimensions": _gguf([(SITE, (2**32, 2**32), F32, 0)], GOOD_DATA),
    "unknown_type": _gguf([(SITE, (2, 2), 999, 0)], GOOD_DATA),
    "q8_on_a_vector": _gguf([(SITE, (32,), Q8, 0)], b"\0" * 34),
    "q8_three_dimensions": _gguf([(SITE, (2, 2, 8), Q8, 0)], b"\0" * 34),
    "q8_truncated": _gguf([(SITE, (2, 16), Q8, 0)], b"\0" * 33),
    "duplicate_tensor": _gguf([GOOD_TENSOR, GOOD_TENSOR], GOOD_DATA),
    "overlapping_tensors": _gguf([GOOD_TENSOR, ("other", (2, 2), F32, 0)], GOOD_DATA),
    "nan_value": _gguf([GOOD_TENSOR], struct.pack("<4f", NAN, 2.0, 3.0, 4.0)),
    "inf_value": _gguf([GOOD_TENSOR], struct.pack("<4f", INF, 2.0, 3.0, 4.0)),
    "transposed_vs_reference": _gguf([(SITE, (1, 4), F32, 0)], GOOD_DATA),
    "wrong_name": _gguf([("other", (2, 2), F32, 0)], GOOD_DATA),
    "no_tensors": _gguf([], b""),
    "empty_file": b"",
    "magic_only": b"GGUF",
    "header_cut": b"GGUF" + struct.pack("<I", 3) + b"\0" * 4,
    "name_length_huge": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**62) + b"key",
    "name_invalid_utf8": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _gguf_string(b"\xff\xfe") + struct.pack("<II", 4, 1),
    "name_short": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 100) + b"key",
    "kv_type_unknown": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _gguf_string("k") + struct.pack("<I", 99),
}


def test_control_gguf_builder_and_export_are_accepted(tmp_path: Path) -> None:
    toy = _gguf_toy()
    path = tmp_path / "ok.gguf"
    path.write_bytes(_gguf([GOOD_TENSOR], GOOD_DATA))
    reloaded, receipt = export_formats.reload_and_verify_gguf_v3(path, toy, [])
    assert receipt.exported_tensor_count == 1 and torch.equal(reloaded.tensors[SITE], toy.tensors[SITE])
    export_formats.export_gguf_v3(toy, tmp_path / "e.gguf")
    reloaded, _ = export_formats.reload_and_verify_gguf_v3(tmp_path / "e.gguf", toy, [])
    assert torch.equal(reloaded.tensors[SITE], toy.tensors[SITE])


@pytest.mark.parametrize("name", GGUF_CASES)
def test_corrupt_gguf_is_refused(tmp_path: Path, name: str) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(GGUF_CASES[name])
    with pytest.raises(ValueError):
        export_formats.reload_and_verify_gguf_v3(path, _gguf_toy(), [])


@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
@pytest.mark.parametrize("quantization", ["f32", "f16", "q8_0", "q4_0", "nf_q4"])
def test_every_truncation_of_a_real_export_is_refused(tmp_path: Path, quantization: str, fraction: float) -> None:
    toy = Toy()
    toy.tensors = {SITE: torch.linspace(-1, 1, 64, dtype=torch.float64).reshape(8, 8)}
    path = export_formats.export_gguf_v3(toy, tmp_path / "m.gguf", quantization=quantization)
    raw = path.read_bytes()
    path.write_bytes(raw[: int(len(raw) * fraction)])
    with pytest.raises(ValueError):
        export_formats.reload_and_verify_gguf_v3(path, toy, [])


# ---------------------------------------------------------------- hub bundle


def test_model_card_cannot_be_injected_through_the_patch(model: Toy, keys: dict[str, Any], tmp_path: Path) -> None:
    body = _body()
    body["sbom"]["base_model_id"] = "m\nlicense: mit\n---\n# Trusted and audited"
    with pytest.raises(ValueError, match="base_model"):
        export_formats.export_huggingface_hub_bundle(model, tmp_path, open_patch=_open_patch(body, keys), signing_key=keys["private_key"])


def test_hub_bundle_verifies_and_detects_a_swapped_patch(model: Toy, keys: dict[str, Any], tmp_path: Path) -> None:
    doc = _open_patch(_body(), keys)
    export_formats.export_huggingface_hub_bundle(model, tmp_path, open_patch=doc, signing_key=keys["private_key"])
    assert provenance.verify_artifact_manifest(tmp_path, public_key=keys["public_key"])["verified"] is True
    other = _body()
    other["sbom"]["changed_parameters"] = 5
    (tmp_path / "weight_patch.nf.json").write_bytes(canonicalize(_open_patch(other, keys)))
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(tmp_path, public_key=keys["public_key"])


# ---------------------------------------------------------------- diff, scan, rollback


def _edited(model: Toy, name: str, value: float) -> Toy:
    edited = copy.deepcopy(model)
    edited.tensors[name] = edited.tensors[name].clone()
    edited.tensors[name].reshape(-1)[0] = value
    return edited


@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "inf", "neg_inf"])
def test_diff_counts_a_non_finite_change_instead_of_crashing_or_hiding_it(model: Toy, value: float) -> None:
    diff = patch_ux.diff_model_weights(model, _edited(model, SITE, value))
    assert diff.layers_touched == (SITE,) and diff.total_changed_parameters == 1


@pytest.mark.parametrize("value", [NAN, INF], ids=["nan", "inf"])
def test_scan_reports_non_finite_injection_instead_of_crashing(model: Toy, value: float) -> None:
    report = patch_ux.scan_patch_for_backdoor_patterns(model, _edited(model, SITE, value), declared_target_sites=[SITE])
    assert report.backdoor_scan_passed is False
    assert "non_finite_parameter_injection" in report.flagged_reasons


def test_scan_sees_an_undeclared_nan_in_a_vector(model: Toy) -> None:
    report = patch_ux.scan_patch_for_backdoor_patterns(model, _edited(model, BIAS, NAN), declared_target_sites=[SITE])
    assert report.backdoor_scan_passed is False
    assert any("undeclared_layer_mutation" in r for r in report.flagged_reasons)


def test_scan_sees_an_undeclared_edit_to_a_tiny_tensor(model: Toy) -> None:
    tiny = copy.deepcopy(model)
    tiny.tensors[BIAS] = tiny.tensors[BIAS] * 1e-14
    edited = copy.deepcopy(tiny)
    edited.tensors[BIAS] = edited.tensors[BIAS] * 3.0
    report = patch_ux.scan_patch_for_backdoor_patterns(tiny, edited, declared_target_sites=[SITE])
    assert any("undeclared_layer_mutation" in r for r in report.flagged_reasons)


def test_diff_refuses_shape_changes_and_missing_tensors(model: Toy) -> None:
    reshaped = copy.deepcopy(model)
    reshaped.tensors[SITE] = reshaped.tensors[SITE].reshape(3, 4)
    with pytest.raises(ValueError, match="changed shape"):
        patch_ux.diff_model_weights(model, reshaped)
    missing = copy.deepcopy(model)
    del missing.tensors[BIAS]
    with pytest.raises(ValueError, match="missing"):
        patch_ux.diff_model_weights(model, missing)


def test_added_tensor_is_flagged_as_undeclared(model: Toy) -> None:
    edited = copy.deepcopy(model)
    edited.tensors["backdoor"] = torch.ones(2, 2, dtype=torch.float64)
    report = patch_ux.scan_patch_for_backdoor_patterns(model, edited, declared_target_sites=[SITE])
    assert any("undeclared_layer_mutation:backdoor" in r for r in report.flagged_reasons)


def test_tensor_digest_binds_shape(model: Toy) -> None:
    w = model.tensors[SITE]
    assert patch_ux._tensor_digest(w) != patch_ux._tensor_digest(w.reshape(3, 4))


@pytest.mark.parametrize(
    "tamper",
    [
        lambda h: h["tensors"].__setitem__(SITE, h["tensors"][SITE] + 1.0),
        lambda h: h["tensors"].__setitem__(SITE, h["tensors"][SITE].reshape(3, 4)),
        lambda h: h["tensors"].pop(BIAS),
        lambda h: h["tensors"].__setitem__("extra", torch.ones(1, dtype=torch.float64)),
        lambda h: h.update(base_parameter_digest="sha256:" + "0" * 64),
    ],
    ids=["values", "reshaped", "removed", "added", "digest"],
)
def test_tampered_rollback_handle_is_refused(model: Toy, tamper: Callable[[dict[str, Any]], Any]) -> None:
    handle = patch_ux.create_rollback_handle(model)
    tamper(handle)
    with pytest.raises(RuntimeError, match="integrity"):
        patch_ux.rollback_model_patch(_edited(model, SITE, 9.0), handle)


def test_rollback_restores_exact_bits_including_negative_zero(model: Toy) -> None:
    model.tensors[SITE][0, 0] = -0.0
    handle = patch_ux.create_rollback_handle(model)
    restored, receipt = patch_ux.rollback_model_patch(_edited(model, SITE, 9.0), handle)
    assert receipt["rollback_verified"] is True
    assert math.copysign(1.0, float(restored.tensors[SITE][0, 0])) == -1.0
    assert torch.equal(restored.tensors[SITE], model.tensors[SITE])


def test_patch_digest_is_stable_under_key_order(keys: dict[str, Any]) -> None:
    a, b = {"x": 1, "y": 2}, {"y": 2, "x": 1}
    assert content_digest(canonicalize(a)) == content_digest(canonicalize(b))
    envelope = patch_ux.sign_patch_envelope(a, signing_key=keys["private_key"])
    assert patch_ux.verify_patch_signature({**envelope, "patch": b}, public_key=keys["public_key"])["verified"] is True
