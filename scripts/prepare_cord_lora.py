"""Freeze approved train-only overlap exclusion and one-round CORD LoRA configuration."""

import argparse
import json
from pathlib import Path

from neural_foundry.cord_receipt_plugin import project
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.proof_pipeline import read_document, write_document
from neural_foundry.provenance import resolve_checkpoint, sha256_file
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    original = read_document(source / "train-projection.json")
    prepared = read_document(source / "prepared.json")
    baseline = read_document(source / "guided-one" / "spec-base4b.json")
    rows = original["rows"]
    if len(rows) != 800 or _document_digest({"rows": rows}) != prepared["train_rows_digest"]:
        raise GymJobError("expected the original frozen800 native CORD train rows")
    clean = next(panel["rows"] for panel in baseline["public_sets"] if panel["name"] == "cord-clean")
    texts = {" ".join(row["input"]["document"].split()) for row in clean}
    retained, excluded = [], []
    for row in rows:
        if row["native_split"] != "train" or row["row_digest"] != _document_digest({
            key: value for key, value in row.items() if key != "row_digest"}):
            raise GymJobError("original training row provenance differs")
        (excluded if " ".join(row["input"]["document"].split()) in texts else retained).append(row)
    if not excluded or not retained:
        raise GymJobError("expected disclosed overlap exclusion without emptying the training set")
    policy_path = Path(__file__).resolve().parents[1] / "protocols" / "cord-lora-training-policy-v1.json"
    policy = json.loads(policy_path.read_bytes())
    base = baseline["base_model"]
    checkpoint = resolve_checkpoint(base["model_id"], revision=base["revision"], allow_download=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True, trust_remote_code=False)
    messages = [{"role": "system", "content": baseline["candidates"][0]["system_prompt"]}]
    lengths = []
    demonstrations = []
    for row in retained:
        chat = [*messages, *project(row["input"], row["reference"])["messages"]]
        lengths.append(len(tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, enable_thinking=False)))
        demonstrations.append({"messages": chat})
    recipe = {**policy["recipe"], "steps": len(retained)}
    if (max(lengths) > recipe["max_sequence_tokens"]
        or sum(lengths) > recipe["max_dataset_tokens"] or len(retained) > policy["maximum_rows"]):
        raise GymJobError("retained full-chat rows exceed the approved frozen training policy")
    output.mkdir(parents=True, exist_ok=False)
    manifest = write_document(output / "training-manifest.json", {
        "kind": "nf_cord_approved_lora_training_manifest_v1",
        "original_rows": 800, "retained_rows": len(retained), "excluded_rows": len(excluded),
        "original_training_digest": original["document_digest"],
        "retained_rows_digest": _document_digest({"rows": retained}),
        "excluded": [{"scenario_id": row["scenario_id"], "row_digest": row["row_digest"]} for row in excluded],
        "exclusion_rule": "collapse whitespace; exclude ALL train rows exactly matching clean validation text",
        "original_order_preserved": True, "recipe": recipe,
        "full_chat_maximum_tokens": max(lengths), "full_chat_total_tokens": sum(lengths),
        "success_policy": policy["success"],
        "baseline_v2_score_digest": read_document(
            Path("artifacts") / "cord-scorer-v2-20261111" / "offline-rescore.json")["document_digest"],
        "training_policy_reference": {"path": str(policy_path), "file_digest": sha256_file(policy_path)},
        "user_approval": {
            "overlap": "Exclude overlapping training receipt(s), record the exact resulting count and freeze a new training manifest",
            "round": "Approve this one-round recipe, limits and success criteria"},
        "test_accessed": False, "training_started": False, "automatic_retries_authorized": False,
    })
    config = write_document(output / "config-lora4b.json", {
        "kind": "nf_cord_receipt_config_v1", "run_id": "cord-lora4b-one-approved-round-20261111",
        "base_model": base, "scorer_version": "2",
        "prepared_path": str(source / "prepared.json"), "prepared_digest": prepared["document_digest"],
        "training_rows": retained, "training_manifest_digest": manifest["document_digest"],
        "training_policy_reference": manifest["training_policy_reference"],
        "candidates": [{"role": "lora4b", "method": "lora", "recipe": recipe,
                        "system_prompt": baseline["candidates"][0]["system_prompt"],
                        "serving": {"gpu_memory_utilization": 0.5}}],
    })
    with (output / "training-demonstrations.jsonl").open("x", encoding="utf-8") as stream:
        for row in demonstrations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"retained_rows": len(retained), "excluded_rows": len(excluded),
                      "configuration_digest": config["document_digest"],
                      "maximum_tokens": max(lengths), "total_tokens": sum(lengths)}))


if __name__ == "__main__":
    main()
