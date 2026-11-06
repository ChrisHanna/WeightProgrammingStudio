"""Cumulative edit search: scope is not fixed in advance; it grows only when a step pays for itself.

Declared before any run (see docs/retention-multisite/PROTOCOL.md):

    J = task + retention - 0.1 * KL - cost_per_parameter * changed_parameters

`task` is accuracy on the development and extra-prefix items. `retention` and `KL` are measured on the accept guard,
a group-wise part of the fit corpus that solves never see. A candidate is accepted when it has the highest J of its
round and raises J by at least MIN_GAIN. The search stops at the first round with no such candidate.

Reporting splits (validation, report prefixes, practice, tuning) are measured each round and never used to accept.
Every accepted step is an `EditPlan.low_rank` step, so the sequence replays to the same digest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import baselines as bl
from . import retention_multisite as ms
from . import retention_research as rr
from .edit_plan import EditConstraints, EditPlan, apply_edit_graph
from .pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from .pythia import PythiaModel

KL_WEIGHT = 0.1
MIN_GAIN = 0.005
PRIMARY_COST = 1e-7  # One 65,536-parameter tensor costs 0.0066, about one accept-guard prompt.
SENSITIVITY_COSTS = (0.0, 1e-6)
SAFETY_ROUNDS = 40  # A hard safety limit, not a design choice. The output says if it binds.
STRENGTHS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
TASK_ALPHAS = (1e3, 1e5)
REPAIR = ((1e3, 1.0), (1e5, 1.0), (1e3, 100.0), (1e5, 100.0))
RANK_LIMIT = 12
REPORT_PREFIXES = ("Example:", "Case:", "Sample:")  # Base rules only, used only for reporting.
EXECUTOR_SOURCE_LIMIT_BYTES = 16_777_216


def candidate_sites(model: Any) -> list[str]:
    """Every matrix with an implemented closed-form step. Other tensors are listed by `excluded_sites`."""
    layers = int(model.config["num_hidden_layers"])
    return [
        f"gpt_neox.layers.{l}.{name}.weight"
        for l in range(layers)
        for name in ("mlp.dense_4h_to_h", "attention.dense", "mlp.dense_h_to_4h")
    ]


def excluded_sites(model: Any) -> dict[str, list[str]]:
    open_sites = set(candidate_sites(model))
    out: dict[str, list[str]] = {"executor_source_bound": [], "no_closed_form_implemented": [], "not_a_matrix": []}
    for name, tensor in model.tensors.items():
        if name in open_sites:
            continue
        if tensor.ndim != 2:
            out["not_a_matrix"].append(name)
        elif tensor.numel() * 8 > EXECUTOR_SOURCE_LIMIT_BYTES:
            out["executor_source_bound"].append(name)
        else:
            out["no_closed_form_implemented"].append(name)
    return out


def accept_split(seed: int = 41, accept_share: float = 0.4) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Group-wise split of the corpus's fit part into (solve, accept). The tuning part stays for reporting."""
    fit = set(rr.split_corpus()[0])
    groups = [(g, [p for p in prompts if p in fit]) for g, prompts in sorted(rr.build_corpus().items())]
    groups = [g for g in groups if g[1]]
    random.Random(seed).shuffle(groups)
    target = accept_share * sum(len(p) for _, p in groups)
    solve: list[str] = []
    accept: list[str] = []
    for _, prompts in groups:
        (accept if len(accept) < target else solve).extend(prompts)
    return tuple(solve), tuple(accept)


@dataclass(frozen=True)
class Move:
    site: str
    mode: str  # "task" moves the task target and protects; "repair" cancels drift and holds the task effect.
    strength: float
    alpha: float
    ridge: float


def objective(task: float, retention: float, kl: float, changed: int, cost: float) -> float:
    return task + retention - KL_WEIGHT * kl - cost * changed


def round_moves(model: Any, first_round: bool) -> list[Move]:
    moves: list[Move] = []
    for site in candidate_sites(model):
        moves += [Move(site, "task", s, a, 1.0) for s in STRENGTHS for a in TASK_ALPHAS]
        if not first_round:  # With no edit yet there is no drift to repair.
            moves += [Move(site, "repair", 0.0, a, r) for a, r in REPAIR]
    return moves


class Capture:
    """Inputs at every site and the final residual at the last token, for one model state."""

    def __init__(self, model: Any, prompts: Sequence[str]) -> None:
        inputs: list[dict[str, torch.Tensor]] = []
        residuals: list[torch.Tensor] = []
        for p in prompts:
            sink: list[torch.Tensor] = []
            site: dict[str, torch.Tensor] = {}
            model.forward_with_mlp_key(model.tokenizer.encode(p), residual_sink=sink, site_inputs=site)
            inputs.append(site)
            residuals.append(sink[0][-1])
        self.residual = torch.stack(residuals, dim=1)
        self.inputs = {name: torch.stack([row[name] for row in inputs], dim=1) for name in inputs[0]}


class CumulativeSearch:
    def __init__(self, model: PythiaModel, cost: float) -> None:
        self.model = model
        self.cost = cost
        self.mlab = ms.MultiLab(model)
        self.lab = self.mlab.lab
        self.solve_guard, self.accept_guard = accept_split()
        self.report_items = [(f"{p} {e}", t) for p in REPORT_PREFIXES for e, t in GYM.base_rules]
        self.plan = EditPlan(model, EditConstraints())
        self.tensors: dict[str, torch.Tensor] = {}
        self.task_target = ms.contrast(self.lab.unit_targets)
        dev = [p for p, _ in self.lab.dev_items]
        self.dev_prompts = dev
        self.base_dev = Capture(model, dev)
        self.base_solve = Capture(model, self.solve_guard)

    def current(self) -> Any:
        return bl.merged_model(self.model, self.tensors)

    def changed(self, tensors: Mapping[str, torch.Tensor]) -> int:
        return ms.changed_parameters(self.model, tensors)

    def _solve(self, move: Move, dev: Capture, guard: Capture, drift_task: torch.Tensor, drift_guard: torch.Tensor, current: Any) -> tuple[torch.Tensor, torch.Tensor]:
        k = ms.contrast(dev.inputs[move.site])
        kp = guard.inputs[move.site]
        if move.mode == "task":
            y = self.task_target * move.strength - drift_task
            yp = None
        else:
            y = torch.zeros_like(self.task_target)
            yp = -drift_guard
        if "h_to_4h" in move.site:
            # The step reaches the residual through GELU and the down-projection. The mean GELU slope is taken as 1/2.
            down = current.tensors[move.site.replace("dense_h_to_4h", "dense_4h_to_h")]
            to_hidden = 2.0 * torch.linalg.pinv(down)
            y = to_hidden @ y
            yp = to_hidden @ yp if yp is not None else None
        delta = ms.ridge_map(k, y, kp, yp, move.alpha, move.ridge)
        return ms.low_rank(delta, RANK_LIMIT)

    def measure(self, tensors: Mapping[str, torch.Tensor], items: Sequence[tuple[str, str]]) -> float:
        return sum(self.mlab.task_hits(items, tensors)) / len(items)

    def accept_score(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
        task = self.measure(tensors, [*self.lab.dev_items, *self.lab.dev_aug_items])
        detail = self.mlab.retention_detail(self.accept_guard, tensors)
        return {
            "task": task,
            "retention": detail["retention"],
            "kl": detail["mean_kl"],
            "changed": self.changed(tensors),
            "J": objective(task, detail["retention"], detail["mean_kl"], self.changed(tensors), self.cost),
        }

    def report(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Splits that never decide a step."""
        tune = self.mlab.retention_detail(self.lab.tune, tensors)
        practice = self.mlab.retention_detail(ms.PRACTICE_RETENTION_PROMPTS, tensors)
        return {
            "dev": self.measure(tensors, self.lab.dev_items),
            "dev_aug": self.measure(tensors, self.lab.dev_aug_items),
            "val": self.measure(tensors, self.lab.val_items),
            "report_prefixes": self.measure(tensors, self.report_items),
            "tune": tune["retention"],
            "tune_kl": tune["mean_kl"],
            "practice": practice["retention"],
        }

    def run(self, log_path: Path) -> dict[str, Any]:
        start = time.time()
        state = self.accept_score({})
        reported = self.report({})
        rounds: list[dict[str, Any]] = []
        refused = 0
        stop = "safety_rounds_bound"
        for number in range(1, SAFETY_ROUNDS + 1):
            current = self.current()
            dev = Capture(current, self.dev_prompts)
            guard = Capture(current, self.solve_guard)
            drift_task = ms.contrast(dev.residual - self.base_dev.residual)
            drift_guard = guard.residual - self.base_solve.residual
            best: tuple[float, Move, torch.Tensor, torch.Tensor, dict[str, float]] | None = None
            moves = round_moves(self.model, number == 1)
            for move in moves:
                try:
                    left, right = self._solve(move, dev, guard, drift_task, drift_guard, current)
                except ValueError:  # The solve had nothing to do, which is an empty candidate and not an error to report.
                    refused += 1
                    continue
                trial = {**self.tensors, move.site: current.tensors[move.site] + left @ right}
                score = self.accept_score(trial)
                if best is None or score["J"] > best[0]:
                    best = (score["J"], move, left, right, score)
            if best is None or best[0] - state["J"] < MIN_GAIN:
                stop = "no_candidate_cleared_min_gain"
                rounds.append({"round": number, "stopped": True, "best_gain": None if best is None else best[0] - state["J"], "candidates": len(moves)})
                break
            _, move, left, right, score = best
            self.plan.low_rank(move.site, left.contiguous(), right.contiguous())
            candidate = self.plan.preview()  # Rust executes the whole sequence from the base model.
            self.tensors = dict(candidate.tensors if hasattr(candidate, "tensors") else candidate.model.tensors)
            self.tensors = {k: v for k, v in self.tensors.items() if not torch.equal(v, self.model.tensors[k])}
            after = self.report(self.tensors)
            row = {
                "round": number,
                "site": move.site,
                "mode": move.mode,
                "strength": move.strength,
                "alpha": move.alpha,
                "ridge": move.ridge,
                "candidates": len(moves),
                "J_gain_accept": score["J"] - state["J"],
                "accept": score,
                "report": after,
                "marginal_report_task_val": after["val"] - reported["val"],
                "marginal_report_task_prefixes": after["report_prefixes"] - reported["report_prefixes"],
                "marginal_report_retention_tune": after["tune"] - reported["tune"],
                "marginal_report_kl_tune": after["tune_kl"] - reported["tune_kl"],
                "cumulative_changed": score["changed"],
                "tensors_touched": sorted(self.tensors),
                "plan_digest": candidate.plan_digest,
                "state_digest": candidate.state_digest,
                "elapsed_seconds_rough": time.time() - start,
            }
            rounds.append(row)
            state, reported = score, after
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("round", "site", "mode", "strength", "J_gain_accept", "cumulative_changed", "report")}), flush=True)
        return {
            "cost_per_parameter": self.cost,
            "kl_weight": KL_WEIGHT,
            "min_gain": MIN_GAIN,
            "stop_reason": stop,
            "safety_rounds": SAFETY_ROUNDS,
            "safety_limit_bound": stop == "safety_rounds_bound",
            "excluded_sites": excluded_sites(self.model),
            "empty_candidates": refused,
            "rounds": rounds,
            "final_report": reported,
            "final_accept": state,
            "total_changed": state["changed"],
            "total_parameters": self.model.total_parameters,
            "wall_seconds_rough": time.time() - start,
        }


def graph_path(root: Path, cost: float) -> Path:
    return root / f"cumulative_graph_c{cost:g}.json"


def run_search(model: PythiaModel, cost: float, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    log = root / f"cumulative_rounds_c{cost:g}.jsonl"
    log.unlink(missing_ok=True)
    search = CumulativeSearch(model, cost)
    result = search.run(log)
    if search.plan._steps:
        graph = search.plan.to_graph()
        graph_path(root, cost).write_text(json.dumps(graph), encoding="utf-8")
        replayed = apply_edit_graph(model, graph)
        result["replay"] = {
            "plan_digest": replayed.plan_digest,
            "state_digest": replayed.state_digest,
            "matches": replayed.plan_digest == graph["plan_digest"] and replayed.state_digest == graph["candidate_state_digest"],
            "steps": len(graph["steps"]),
        }
    else:
        result["replay"] = None
    (root / f"cumulative_result_c{cost:g}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def replay_graph(model: PythiaModel, path: Path) -> dict[str, torch.Tensor]:
    candidate = apply_edit_graph(model, json.loads(path.read_text(encoding="utf-8")))
    return {k: v for k, v in candidate.model.tensors.items() if not torch.equal(v, model.tensors[k])}


def restricted_grid(kind: str) -> list[dict[str, Any]]:
    """The published learning rate and length for each kind, with both guard weights tried on the fit corpus."""
    if kind == "lora":
        return [{"kind": kind, "lr": 3e-4, "steps": 300, "weight": w, "rank": 4, "checkpoints": (100, 200, 300)} for w in (1.0, 10.0)]
    return [{"kind": kind, "lr": 3e-5, "steps": 150, "weight": w, "rank": 0, "checkpoints": (50, 100, 150)} for w in (1.0, 10.0)]


def train_restricted(model: PythiaModel, sites: Sequence[str], recipe: Mapping[str, Any], dev: Sequence[tuple[str, str]], guard: Sequence[str], on_checkpoint: Any = None, seed: int = 0) -> tuple[dict[str, torch.Tensor], float, int]:
    """LoRA or fine-tuning on the named tensors only. Same loss and optimizer as `baselines.fit_baseline`."""
    g = torch.Generator().manual_seed(seed)
    base = model.tensors
    guard = tuple(guard)
    with torch.no_grad():
        reference = {p: F.log_softmax(bl._last_logits(model, p), dim=-1) for p in guard}
    if recipe["kind"] == "sft":
        live = {s: base[s].clone().requires_grad_(True) for s in sites}
        params = list(live.values())
        state = lambda: live
    else:
        factors = {}
        for s in sites:
            out_dim, in_dim = base[s].shape
            rank = min(int(recipe["rank"]), int(out_dim), int(in_dim))
            a = torch.randn(rank, int(in_dim), dtype=torch.float64, generator=g) / int(in_dim) ** 0.5
            b = torch.zeros(int(out_dim), rank, dtype=torch.float64)
            factors[s] = (a.requires_grad_(True), b.requires_grad_(True))
        params = [t for pair in factors.values() for t in pair]
        state = lambda: {s: base[s] + b @ a for s, (a, b) in factors.items()}
    optimizer = torch.optim.Adam(params, lr=recipe["lr"])
    trained = bl.merged_model(model, {})
    started = time.perf_counter()
    excluded = 0.0
    for step in range(1, int(recipe["steps"]) + 1):
        trained.tensors = {**base, **state()}
        loss = bl._mean_ce(trained, dev)
        picks = torch.randperm(len(guard), generator=g)[:8]
        kl = torch.stack(
            [
                F.kl_div(F.log_softmax(bl._last_logits(trained, guard[int(i)]), dim=-1), reference[guard[int(i)]], log_target=True, reduction="sum")
                for i in picks
            ]
        ).mean()
        total = loss + recipe["weight"] * kl
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if on_checkpoint is not None and step in recipe["checkpoints"]:
            paused = time.perf_counter()
            with torch.no_grad():
                on_checkpoint(step, {k: v.detach().clone() for k, v in state().items()}, paused - started - excluded)
            excluded += time.perf_counter() - paused
    with torch.no_grad():
        final = {k: v.detach().clone() for k, v in state().items()}
    return final, time.perf_counter() - started - excluded, sum(int(p.numel()) for p in params)


def run_restricted(model: PythiaModel, cost: float, root: Path, weights_dir: Path) -> None:
    """Selects a recipe from public splits at the search's development bar, using the search's tensors."""
    result = json.loads((root / f"cumulative_result_c{cost:g}.json").read_text(encoding="utf-8"))
    if not result["replay"]:
        print("The search accepted nothing, so there are no tensors to restrict to", flush=True)
        (root / f"restricted_c{cost:g}.json").write_text(json.dumps({"skipped": "search accepted no step"}), encoding="utf-8")
        return
    sites = sorted({r["site"] for r in result["rounds"] if "site" in r})
    bar = result["final_report"]["dev"]
    mlab = ms.MultiLab(model)
    weights_dir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"sites": sites, "development_bar": bar, "trials": [], "selected": {}}
    for kind in ("lora", "sft"):
        best: tuple[tuple[float, ...], dict[str, Any]] | None = None
        for recipe in restricted_grid(kind):
            def record(step: int, weights: dict[str, torch.Tensor], wall: float, recipe: Mapping[str, Any] = recipe) -> None:
                nonlocal best
                scores = mlab.score_public(weights)
                row = {"kind": kind, "steps": step, "weight": recipe["weight"], "lr": recipe["lr"], "rank": recipe["rank"], "scores": scores, "wall_seconds_rough": wall}
                doc["trials"].append(row)
                print(row, flush=True)
                if scores["dev"] < bar - 1e-9 or "tune" not in scores:
                    return
                key = ((scores["practice"] + scores["tune"]) / 2, -scores["delta_norm"])
                if best is None or key > best[0]:
                    best = (key, row)
                    torch.save(weights, weights_dir / f"restricted_{kind}_c{cost:g}.pt")

            train_restricted(model, sites, recipe, mlab.lab.dev_items, mlab.lab.fit, record)
        if best is None:
            doc["selected"][kind] = None
            continue
        weights = torch.load(weights_dir / f"restricted_{kind}_c{cost:g}.pt")
        doc["selected"][kind] = {**best[1], "weights_digest": bl.weights_digest(weights), "changed_parameters": ms.changed_parameters(model, weights)}
    (root / f"restricted_c{cost:g}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def freeze_cumulative(doc: dict[str, Any], rows: Sequence[Mapping[str, Any]], root: Path, floor: float) -> None:
    """Adds the search results to the freeze. The comparator is the best single tensor at the same development bar."""
    doc["cumulative"] = {}
    doc["restricted"] = {}
    comparator_of: dict[str, str] = {}
    single_rows = [r for r in rows if r["family"] == "single" and "tune" in r.get("scores", {})]
    equal: dict[str, Any] = {}
    # The primary first, so a sensitivity arm that reached the identical model is recorded once and not tested twice.
    paths = sorted(root.glob("cumulative_result_c*.json"), key=lambda p: p.stem != f"cumulative_result_c{PRIMARY_COST:g}")
    seen: dict[str, str] = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        tag = path.stem.removeprefix("cumulative_result_c")
        name = f"cumulative_c{tag}"
        if not result["replay"]:
            doc["cumulative_empty"] = {**doc.get("cumulative_empty", {}), name: result["stop_reason"]}
            continue
        if result["replay"]["state_digest"] in seen:
            doc["cumulative_identical"] = {**doc.get("cumulative_identical", {}), name: seen[result["replay"]["state_digest"]]}
            continue
        seen[result["replay"]["state_digest"]] = name
        doc["cumulative"][name] = {
            "graph": str(root / f"cumulative_graph_c{tag}.json"),
            "state_digest": result["replay"]["state_digest"],
            "steps": result["replay"]["steps"],
            "public": result["final_report"],
        }
        bar = result["final_report"]["dev"]
        eligible = [r for r in single_rows if r["scores"]["dev"] >= bar - 1e-9]
        top = max(eligible, key=lambda r: ((r["scores"]["practice"] + r["scores"]["tune"]) / 2, -r["scores"]["delta_norm"]))
        single_name = f"single_at_dev_{bar:.3f}"
        equal[single_name] = {"config": top["config"], "public": top["scores"]}
        comparator_of[name] = single_name
        restricted_path = root / f"restricted_c{tag}.json"
        if restricted_path.exists():
            restricted = json.loads(restricted_path.read_text(encoding="utf-8"))
            for kind, entry in restricted.get("selected", {}).items():
                if entry:
                    label = f"restricted_{kind}_c{tag}"
                    doc["restricted"][label] = {**entry, "sites": restricted["sites"]}
                    comparator_of[label] = single_name
    doc["single_at_equal_task"] = equal
    primary = f"cumulative_c{PRIMARY_COST:g}"
    doc["comparisons"]["cumulative_primary"] = primary if primary in doc["cumulative"] else None
    doc["comparisons"]["comparator_of"] = comparator_of
    doc["comparisons"]["secondary"] = [*doc["comparisons"]["secondary"], *doc["cumulative"], *doc["restricted"]]


def load_extra_methods(lab: ms.MultiLab, frozen: Mapping[str, Any], weights_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    model = lab.model
    for name, entry in frozen.get("single_at_equal_task", {}).items():
        out[name] = lab.execute(lab.solve(ms.config_from(entry["config"]))).merged
    for name, entry in frozen.get("cumulative", {}).items():
        merged = replay_graph(model, Path(entry["graph"]))
        out[name] = merged
    primary = frozen["comparisons"].get("cumulative_primary")
    if primary:
        factors: dict[str, Any] = {}
        for site, weight in out[primary].items():
            left, right = ms.low_rank(weight - model.tensors[site], 128)
            factors[site] = (left, right, None)
        out["control_random_cumulative"] = lab.random_like(ms.MultiCandidate(dict(out[primary]), factors)).merged
    for name, entry in frozen.get("restricted", {}).items():
        weights = torch.load(weights_dir / f"{name}.pt")
        if bl.weights_digest(weights) != entry["weights_digest"]:
            raise RuntimeError(f"{name} weights do not match the freeze")
        out[name] = weights
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["search", "restricted"])
    parser.add_argument("--cost", type=float, default=PRIMARY_COST)
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--dir", default="docs/retention-multisite")
    parser.add_argument("--weights-dir", default="/tmp/nf-ms-weights")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name="EleutherAI/pythia-14m")
    if args.phase == "search":
        run_search(model, args.cost, Path(args.dir))
    else:
        run_restricted(model, args.cost, Path(args.dir), Path(args.weights_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
