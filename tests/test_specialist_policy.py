from __future__ import annotations

import json
from pathlib import Path

from neural_foundry.specialist_policy import assess_specialist_v2, paired_bootstrap_improvement


def test_approved_specialist_v2_uses_paired_case_bootstrap_and_marks_partial_implementation():
    policy_path = Path(__file__).resolve().parents[1] / "protocols" / "specialist-retention-v2.json"
    policy = json.loads(policy_path.read_bytes())
    assert policy["status"] == "approved_frozen"
    assert policy["implementation_status"].startswith("base_failure_selection_and_specialist_gate_integrated")
    assert policy["edit_calculation"]["single_site_zero_retention_wired"] is True
    assert policy["edit_calculation"]["multi_layer_application_implemented"] is False
    assert policy["task_gate"]["minimum_net_case_improvement"] == 4
    assert policy["task_gate"]["confidence_interval"].startswith("paired bootstrap 95%")

    base = [1] * 45 + [0] * 19
    candidate = [1] * 50 + [0] * 14
    interval = paired_bootstrap_improvement(base, candidate, seed=7, replicates=2_000)
    assert interval["net_case_improvement"] == 5
    assert interval["selection_adjusted"] is False
    evidence = {"split": "validation", "base_case_outcomes": base, "candidate_case_outcomes": candidate}
    required = {"split": "validation", "format_valid": 1.0, "exact_match": 1.0}
    result = assess_specialist_v2(evidence, required, minimum_net_cases=4)
    assert result["policy_status"] == "approved_frozen" and result["policy_version"] == 2
    assert result["checks"]["confidence_interval_lower_bound"] is True
    assert not result["holdout_accessed"] and not result["generalist_authorized"] and not result["release_authorized"]
