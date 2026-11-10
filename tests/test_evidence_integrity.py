"""Regression tests: evidence must be measured and authenticated, or be rejected."""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest
from neural_foundry import flagship, patch_ux, prd_adapters, pythia
from neural_foundry.records import canonicalize, content_digest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "neural_foundry"
CLI = Path(os.environ.get("NF_CLI", ROOT / "target" / "debug" / "nf"))


def _valid_body() -> dict:
    return {
        "schema_version": 1,
        "kind": "nf_open_weight_patch_v1",
        "format_version": "1.0.0",
        "base_model": {"model_id": "m", "parameter_digest": "sha256:" + "1" * 64},
        "candidate_checkpoint_digest": "sha256:" + "2" * 64,
        "total_parameters": 100,
        "operations": [{"id": "op_0", "operation": {"type": "low_rank_update", "target": {"tensor": "t"}, "frobenius_norm": 1.0}}],
        "sbom": {
            "base_checkpoint_digest": "sha256:" + "1" * 64,
            "candidate_checkpoint_digest": "sha256:" + "2" * 64,
            "changed_parameters": 4,
            "total_parameters": 100,
        },
        "rounding_contract": {
            "source_dtype": "f32",
            "accumulation_dtype": "f32",
            "rounding_mode": "nearest_ties_even",
        },
        "security_scan": {"backdoor_scan_passed": True},
        "evaluation_provenance": {
            "promote_eligible": True,
            "controls": {"candidate_accuracy": 0.9, "noop_accuracy": 0.1, "random_accuracy": 0.1},
        },
    }


def _signed_open_patch(body: dict, key: dict) -> dict:
    envelope = patch_ux.sign_patch_envelope(body, signing_key=key["private_key"])
    return {**body, "patch_digest": envelope["patch_digest"], "signature_envelope": envelope}


def test_forgery_without_the_private_key_is_rejected() -> None:
    key = patch_ux.generate_ed25519_keypair()
    envelope = patch_ux.sign_patch_envelope({"a": 1}, signing_key=key["private_key"])
    evil = {"a": 666}
    forged = dict(envelope, patch=evil, patch_digest=str(content_digest(canonicalize(evil))))
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(forged, public_key=key["public_key"])


def test_attacker_key_embedded_in_envelope_is_not_trusted() -> None:
    trusted = patch_ux.generate_ed25519_keypair()
    attacker = patch_ux.generate_ed25519_keypair()
    envelope = patch_ux.sign_patch_envelope({"a": 1}, signing_key=attacker["private_key"])
    with pytest.raises(ValueError):
        patch_ux.verify_patch_signature(envelope, public_key=trusted["public_key"])


def test_keys_are_random_and_passphrases_are_not_keys() -> None:
    first = patch_ux.generate_ed25519_keypair()
    assert first["private_key"] != patch_ux.generate_ed25519_keypair()["private_key"]
    with pytest.raises(ValueError):
        patch_ux.sign_patch_envelope({"a": 1}, signing_key="my-passphrase")


def test_legacy_keyless_scheme_is_rejected() -> None:
    key = patch_ux.generate_ed25519_keypair()
    envelope = patch_ux.sign_patch_envelope({"a": 1}, signing_key=key["private_key"])
    with pytest.raises(ValueError, match="scheme"):
        patch_ux.verify_patch_signature(
            dict(envelope, signature_scheme="hmac_sha256_v1"), public_key=key["public_key"]
        )


@pytest.mark.parametrize(
    "section", ["sbom", "rounding_contract", "security_scan", "evaluation_provenance"]
)
def test_missing_evidence_section_fails_closed(section: str) -> None:
    key = patch_ux.generate_ed25519_keypair()
    body = _valid_body()
    del body[section]
    with pytest.raises(ValueError, match=f"{section}|missing"):
        patch_ux.verify_open_patch_v1(_signed_open_patch(body, key), public_key=key["public_key"])


def test_missing_promote_flag_fails_closed() -> None:
    key = patch_ux.generate_ed25519_keypair()
    body = _valid_body()
    del body["evaluation_provenance"]["promote_eligible"]
    with pytest.raises(ValueError, match="promote_eligible"):
        patch_ux.verify_open_patch_v1(_signed_open_patch(body, key), public_key=key["public_key"])


def test_unsigned_top_level_edit_is_rejected() -> None:
    key = patch_ux.generate_ed25519_keypair()
    body = _valid_body()
    body["evaluation_provenance"]["promote_eligible"] = False
    patch = _signed_open_patch(body, key)
    patch["evaluation_provenance"] = {**body["evaluation_provenance"], "promote_eligible": True}
    with pytest.raises(ValueError, match="differs"):
        patch_ux.verify_open_patch_v1(patch, public_key=key["public_key"])


def test_valid_patch_verifies() -> None:
    key = patch_ux.generate_ed25519_keypair()
    receipt = patch_ux.verify_open_patch_v1(
        _signed_open_patch(_valid_body(), key), public_key=key["public_key"]
    )
    assert receipt["verified"] is True


def test_cli_repair_refuses_instead_of_fabricating() -> None:
    result = subprocess.run(
        [str(CLI), "repair", "--model", "some/model"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "nf_open_weight_patch_v1" not in result.stdout


def test_cli_verify_requires_trusted_key_and_interoperates(tmp_path: Path) -> None:
    key = patch_ux.generate_ed25519_keypair()
    path = tmp_path / "patch.json"
    path.write_bytes(canonicalize(_signed_open_patch(_valid_body(), key)))
    assert subprocess.run([str(CLI), "verify", str(path)], capture_output=True, check=False).returncode != 0
    ok = subprocess.run(
        [str(CLI), "verify", str(path), "--pubkey", key["public_key"]],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(ok.stdout)["result"]["verified"] is True
    wrong = patch_ux.generate_ed25519_keypair()["public_key"]
    assert (
        subprocess.run([str(CLI), "verify", str(path), "--pubkey", wrong], capture_output=True, check=False).returncode
        != 0
    )


def test_synthesis_source_contains_no_holdout_expression() -> None:
    gym_block = ast.parse((PACKAGE / "pythia.py").read_text(encoding="utf-8"))
    study = next(
        n
        for n in ast.walk(gym_block)
        if isinstance(n, ast.FunctionDef) and n.name == "run_pythia_gym_task_learning_study"
    )
    literals = {
        n.value for n in ast.walk(study) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    for gym in (pythia.ALGORITHMIC_THREE_SKILL_GYM,):
        assert not literals & {expr for expr, _ in gym.holdout_rules}


def test_overlapping_holdout_input_is_rejected_before_exposure_reporting() -> None:
    from neural_foundry.gym_data import GymDataError

    gym = pythia.ALGORITHMIC_THREE_SKILL_GYM
    leaky = type(gym)(**{**gym.__dict__, "base_rules": (*gym.base_rules, *gym.holdout_rules)})
    with pytest.raises(GymDataError, match="occurs in multiple splits"):
        pythia.run_pythia_gym_task_learning_study(
            pythia.build_mini_pythia_model(), gym=leaky, target_strength=80.0
        )


def test_agent_session_does_not_report_unmeasured_numbers() -> None:
    agent = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet")
    report = agent.run_reasoning_loop_repair_experiment(pythia.build_mini_pythia_model())
    assert report.repaired_loop_rate_before is None
    assert report.repaired_loop_rate_after is None
    assert report.ablation_results == ()


def test_flagship_requires_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        flagship.run_flagship_loop_repair_study_and_publish_bundle(output_dir=tmp_path)  # type: ignore[call-arg]


def test_flagship_measures_the_defective_model_and_replicates(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    report = flagship.run_flagship_loop_repair_study_and_publish_bundle(model, output_dir=tmp_path)
    manifest = json.loads((tmp_path / "flagship_study_manifest.json").read_bytes())
    assert report.defect_provenance == "injected_by_study"
    assert manifest["holdout_accuracy_after"] == report.holdout_accuracy_after
    assert report.promote_eligible is (report.open_patch_rejection_reason is None)
    assert not list(tmp_path.rglob("*private*"))
    assert "ed25519_priv:" not in "".join(
        p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json")
    )
    result = flagship.replicate_published_flagship_bundle(
        tmp_path, public_key=report.public_key, reference_model=model
    )
    assert result["replicated_holdout_accuracy"] == report.holdout_accuracy_after
    with pytest.raises(ValueError):
        flagship.replicate_published_flagship_bundle(
            tmp_path,
            public_key=patch_ux.generate_ed25519_keypair()["public_key"],
            reference_model=model,
        )
