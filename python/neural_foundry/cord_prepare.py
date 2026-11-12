"""CPU-only transcription projection, target mapping and pre-result panel freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .cord_acquire import BIPIA_REV, CORD_REV, file_hash
from .cord_receipt_plugin import FRAMING, OUTPUT, project
from .gym_job import validate_task_instance
from .gym_service import _document_digest
from .proof_pipeline import read_document, write_document


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _quad(word: dict) -> tuple:
    quad = word["quad"]
    return (sum(quad[f"y{i}"] for i in range(1, 5)) / 4,
            min(quad[f"x{i}"] for i in range(1, 5)))


def transcription(annotation: dict) -> str:
    lines = []
    supplied = [line["words"] for line in annotation["valid_line"]]
    supplied.extend(annotation.get("repeating_symbol", []))
    for ordinal, supplied_words in enumerate(supplied):
        words = sorted(enumerate(supplied_words), key=lambda pair: (
            _quad(pair[1])[1], _quad(pair[1])[0], pair[0]))
        if not words:
            raise ValueError("empty transcription line; refusing silent omission")
        center = sum(_quad(w)[0] for _, w in words) / len(words)
        left = min(_quad(w)[1] for _, w in words)
        lines.append(((center, left, ordinal), " ".join(w["text"] for _, w in words)))
    return "\n".join(text for _, text in sorted(lines))


def _objects(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError("unsupported CORD dict/list shape")


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        # Released repeated scalar occurrences remain ordered; never select one amount.
        return "\n".join(value)
    raise TypeError("unsupported scalar label; refusing silent coercion")


def target(annotation: dict) -> dict:
    parsed = annotation["gt_parse"]
    items = []

    def append(item: dict, parent: int | None, void: bool) -> None:
        index = len(items)
        items.append({
            "name": _text(item.get("nm")), "quantity_text": _text(item.get("cnt")),
            "unit_price_text": _text(item.get("unitprice")),
            "line_total_text": _text(item.get("price")),
            "parent_item_index": parent, "is_void": void})
        for sub in _objects(item.get("sub")):
            append(sub, index, void)

    for key, void in (("menu", False), ("void_menu", True)):
        for item in _objects(parsed.get(key)):
            append(item, None, void)
    subtotal, total = _objects(parsed.get("sub_total")), _objects(parsed.get("total"))

    def amount(containers: list[dict], key: str) -> str | None:
        values = [_text(container[key]) for container in containers if key in container]
        if not values:
            return None
        if any(value is None for value in values):
            raise TypeError("explicit null mixed with repeated totals is unsupported")
        return "\n".join(values)

    result = {"line_items": items, "totals": {
        "subtotal_text": amount(subtotal, "subtotal_price"),
        "discount_text": amount(subtotal, "discount_price"),
        "tax_text": amount(subtotal, "tax_price"),
        "service_text": amount(subtotal, "service_price"),
        "grand_total_text": amount(total, "total_price")}}
    validate_task_instance(OUTPUT, result, field="CORD target")
    return result


def load_split(cache: Path, manifest: dict, split: str) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError("native test is sealed")
    import pyarrow.parquet as pq

    files = [r for r in manifest["files"]
             if r["path"].startswith(f"cord-v2-{CORD_REV}/data/{split}-")]
    expected = 4 if split == "train" else 1
    if len(files) != expected:
        raise ValueError("approved split shard count differs")
    rows = []
    for record in sorted(files, key=lambda r: r["path"]):
        path = cache / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("cached shard changed")
        # Column projection never decodes or loads the embedded receipt images.
        for index, raw in enumerate(pq.read_table(path, columns=["ground_truth"])[
                "ground_truth"].to_pylist()):
            annotation = json.loads(raw)
            if annotation["meta"]["split"] != {"train": "train", "validation": "valid"}[split]:
                raise ValueError("native split differs")
            source_id = f"cord-v2:{CORD_REV}:{split}:{annotation['meta']['image_id']}"
            row = {
                "scenario_id": f"{split}:{source_id}", "source_id": source_id,
                "native_split": split, "input": {"document": transcription(annotation)},
                "reference": target(annotation), "family_id": "cord-provided-transcription",
                "source": {"shard": record["path"], "shard_sha256": record["sha256"],
                           "row_index": index, "annotation_digest": digest_bytes(raw.encode()),
                           "annotation_split": annotation["meta"]["split"]},
                "projection_audit": {
                    "valid_line_count": len(annotation["valid_line"]),
                    "valid_line_labeled_count": sum(bool(line.get("category"))
                                                    for line in annotation["valid_line"]),
                    "repeating_symbol_groups": len(annotation.get("repeating_symbol", [])),
                    "dontcare_geometry_groups": len(annotation.get("dontcare", []))},
            }
            row["row_digest"] = _document_digest(row)
            rows.append(row)
    if len(rows) != (800 if split == "train" else 100):
        raise ValueError("native receipt count differs; refusing dropping cases")
    if len({r["scenario_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate native receipt identity")
    return rows


def attack_catalog(path: Path) -> list[dict]:
    payloads = json.loads(path.read_bytes())
    if not isinstance(payloads, dict):
        raise TypeError("unsupported BIPIA train payload shape")
    records = []
    for family, values in sorted(payloads.items()):
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("unsupported BIPIA attack family")
        for index, text in enumerate(values):
            records.append({
                "source_id": f"bipia:{BIPIA_REV}:benchmark/text_attack_train.json:{family}:{index}",
                "family": family, "index": index, "text": text,
                "payload_digest": digest_bytes(text.encode())})
    return records


def overlay(row: dict, attack: dict, placement: str) -> dict:
    lines = row["input"]["document"].split("\n")
    offset = {"prefix": 0, "middle": len(lines) // 2, "suffix": len(lines)}[placement]
    document = "\n".join([*lines[:offset], attack["text"], *lines[offset:]])
    result = {**row, "scenario_id": row["scenario_id"] + ":overlay:" +
              digest_bytes(attack["source_id"].encode())[7:23] + ":" + placement,
              "input": {"document": document}, "clean_scenario_id": row["scenario_id"],
              "attack": {key: value for key, value in attack.items() if key != "text"},
              "placement": placement}
    result.pop("row_digest", None)
    result["row_digest"] = _document_digest(result)
    return result


def statistics(rows: list[dict], tokenizers: list[tuple[str, Any]], attacks: list[dict]) -> dict:
    stats = {}
    for name, tokenizer in tokenizers:
        inputs, outputs = [], []
        attack_max = max(len(tokenizer.encode(a["text"])) for a in attacks)
        for row in rows:
            messages = [{"role": "system", "content": FRAMING},
                        *project(row["input"], None)["messages"]]
            inputs.append(len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)))
            outputs.append(len(tokenizer.encode(json.dumps(row["reference"], ensure_ascii=False))))
        def summarize(values: list[int]) -> dict:
            ordered = sorted(values)
            return {"min": min(values), "max": max(values), "mean": sum(values) / len(values),
                    "p95": ordered[math.ceil(.95 * len(values)) - 1]}
        stats[name] = {"train_chat_input_tokens": summarize(inputs),
                       "train_target_tokens": summarize(outputs), "attack_max_tokens": attack_max,
                       "tokenizer_files": {p.name: file_hash(p) for p in Path(name).iterdir()
                                           if p.is_file() and p.name in {
                                               "tokenizer.json", "tokenizer_config.json",
                                               "vocab.json", "merges.txt", "special_tokens_map.json"}}}
    return stats


def prepare(cache: Path, output: Path, tokenizer_paths: list[Path], compatible: list[str]) -> dict:
    from transformers import AutoTokenizer

    if output.exists() and any(p.name != "attack-plan.json" for p in output.iterdir()):
        raise ValueError("preparation output already exists; immutable freeze cannot be overwritten")
    manifest_path = cache / "cord-bipia-acquisition.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["cord_revision"] != CORD_REV or manifest["bipia_revision"] != BIPIA_REV:
        raise ValueError("source revision differs")
    attacks_path = cache / f"bipia-{BIPIA_REV}/benchmark/text_attack_train.json"
    record = next(r for r in manifest["files"] if r["path"].endswith("text_attack_train.json"))
    if file_hash(attacks_path) != record["sha256"]:
        raise ValueError("attack file changed")
    catalog = attack_catalog(attacks_path)
    families = {a["family"] for a in catalog}
    if not compatible or not set(compatible) <= families:
        raise ValueError("explicit reviewed original attack families required")
    selected = [a for a in catalog if a["family"] in compatible]
    train = load_split(cache, manifest, "train")
    tokenizers = [(str(p), AutoTokenizer.from_pretrained(
        p, local_files_only=True, trust_remote_code=False)) for p in tokenizer_paths]
    if len(tokenizers) != 2:
        raise ValueError("both cached baseline tokenizers are required")
    stats = statistics(train, tokenizers, selected)
    max_output = max(v["train_target_tokens"]["max"] for v in stats.values())
    max_input = max(v["train_chat_input_tokens"]["max"] + v["attack_max_tokens"]
                    for v in stats.values())
    # New real-data budgets: 2x train target maximum and 1024 framing/overlay slack.
    output_tokens = math.ceil(max_output * 2 / 256) * 256
    context_tokens = math.ceil((max_input + output_tokens + 1024) / 1024) * 1024
    policy = {
        "version": 1, "budget_id": "cord-real-receipts-v1",
        "max_new_tokens": output_tokens, "context_tokens": context_tokens,
        "temperature": 0, "reasoning_effort": "none", "report_only": True,
        "adaptive_retraining": False,
        "identity": {"version": 1, "probe_count": 16, "logprob_atol": 1e-6,
                     "shutdown_logprobs": "informational_only"},
        "api": {"version": 1, "calls_authorized": False,
                "maximum_api_attempts": 1, "maximum_cost_microusd": 1},
    }
    # Freeze compatibility and placements before materializing validation rows.
    plan = {"selection_basis": "task compatibility only; no model outputs or scores",
            "training_authorized": False, "attack_source_split": "train",
            "selected": selected,
            "excluded": [{"source_id": a["source_id"], "family": a["family"],
                          "reason": "outside output-text-corruption subset: unrelated question, "
                                    "content addition, persuasion, link promotion or unsafe instruction"}
                         for a in catalog if a["family"] not in compatible],
            "placements": ["prefix", "middle", "suffix"],
            "assignment": "native shard order: payload[index modulo selected_count], "
                          "placement[index modulo 3]; one paired attack per receipt",
            "source_digest": digest_bytes(attacks_path.read_bytes())}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "attack-plan.json").exists():
        existing = read_document(output / "attack-plan.json")
        existing.pop("document_digest")
        if existing != plan:
            raise ValueError("resumed preparation may not alter frozen pre-validation attack plan")
    else:
        write_document(output / "attack-plan.json", plan)
    validation = load_split(cache, manifest, "validation")
    attacked = [overlay(row, selected[index % len(selected)], plan["placements"][index % 3])
                for index, row in enumerate(validation)]
    # Validation lengths are only checked against the train-derived budget; never used to tune it.
    overflow = []
    for name, tokenizer in tokenizers:
        for row in [*validation, *attacked]:
            messages = [{"role": "system", "content": FRAMING},
                        *project(row["input"], None)["messages"]]
            length = len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
            if length + output_tokens > context_tokens:
                overflow.append({"tokenizer": name, "scenario_id": row["scenario_id"],
                                 "input_tokens": length})
    public_sets = [{"name": name, "rows": rows, "rows_digest": _document_digest({"rows": rows})}
                   for name, rows in (("cord-clean", validation), ("cord-bipia", attacked))]
    train_texts = Counter(" ".join(r["input"]["document"].split()) for r in train)
    validation_texts = Counter(" ".join(r["input"]["document"].split()) for r in validation)
    result = {
        "kind": "nf_cord_prepared_v1", "source_manifest_digest": digest_bytes(manifest_path.read_bytes()),
        "train_rows": len(train), "train_rows_digest": _document_digest({"rows": train}),
        "train_text_characters": {"min": min(len(r["input"]["document"]) for r in train),
                                 "max": max(len(r["input"]["document"]) for r in train)},
        "train_item_counts": dict(sorted(Counter(len(r["reference"]["line_items"]) for r in train).items())),
        "train_void_items": sum(i["is_void"] for r in train for i in r["reference"]["line_items"]),
        "train_transcription_audit": {
            key: sum(r["projection_audit"][key] for r in train)
            for key in train[0]["projection_audit"]},
        "duplicate_audit": {
            "train_duplicate_normalized_text_excess": sum(n - 1 for n in train_texts.values()),
            "validation_duplicate_normalized_text_excess": sum(
                n - 1 for n in validation_texts.values()),
            "train_validation_shared_normalized_texts": len(train_texts.keys() & validation_texts.keys()),
            "near_duplicate_analysis": "not established; no automatic split repair",
            "store_disjoint_generalization": "not established"},
        "tokenizer_statistics": stats, "policy": policy, "public_sets": public_sets,
        "attack_plan_digest": _document_digest(plan), "overflow_cases": overflow,
        "ready_for_inference": not overflow, "model_results_observed": False,
        "limitations": [
            "Indonesian provided transcription, not OCR; valid_line is curated target-region text.",
            ("All valid_line and repeating_symbol transcription included without label filtering; "
             "dontcare has geometry only, not text; full-document noise absent."),
            "Only pilot menu/submenu/void and five totals fields; other released fields out of scope.",
            ("Repeated scalar gt_parse strings joined with newline in source order; never selected, "
             "deduplicated, summed or locale-coerced."),
            "BIPIA internal evaluation only; research-use wording OPEN; no redistribution clearance.",
            "Train attacks used only for validation overlays; no training authorized; native tests sealed.",
            "No attacker-goal success claim: extraction degradation is not verified goal completion.",
            "No quality thresholds; no synthetic gates; final executable spec not frozen here.",
        ],
    }
    write_document(output / "train-projection.json", {"rows": train})
    write_document(output / "policy.json", {"kind": "nf_cord_real_data_policy", "version": 1,
                                          "policy": policy, "basis": "train-only maxima"})
    frozen = write_document(output / "prepared.json", result)
    for model_id, revision, role in (
            ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "base06b"),
            ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "base4b")):
        configuration = {
            "kind": "nf_cord_receipt_config_v1", "run_id": f"cord-real-v1-{role}",
            "base_model": {"model_id": model_id, "revision": revision},
            "prepared_path": str((output / "prepared.json").resolve()),
            "prepared_digest": frozen["document_digest"],
            "candidates": [{"role": role, "system_prompt": FRAMING,
                            "serving": {"gpu_memory_utilization": .4}}],
        }
        write_document(output / f"config-{role}.json", configuration)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, action="append", required=True)
    parser.add_argument("--compatible-family", action="append", required=True)
    args = parser.parse_args()
    result = prepare(args.cache, args.output, args.tokenizer, args.compatible_family)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"public_sets"}}, indent=2))


if __name__ == "__main__":
    main()
