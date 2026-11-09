from __future__ import annotations

import json
from pathlib import Path

from neural_foundry.specialist_policy import assess_specialist_v2, paired_bootstrap_improvement


def test_specialist_v2_proposal_uses_paired_case_bootstrap_without_activation():
    policy_path = Path(__file__).resolve().parents[1] / "protocols" / "specialist-retention-v2-proposal.json"
    policy = json.loads(policy_path.read_bytes())
    assert policy["status"] == "proposed_pending_user_approval"
    assert policy["edit_calculation"]["active"] is False
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
    assert result["policy_status"] == "proposal_pending_approval"
    assert not result["holdout_accessed"] and not result["generalist_authorized"] and not result["release_authorized"]
