#!/usr/bin/env python3
"""Run one real, billed agent session. Needs ANTHROPIC_API_KEY or OPENAI_API_KEY.

Example:
    python scripts/live_agent_session.py --checkpoint /path/to/pythia-14m --budget 5 --output session.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from neural_foundry import copilot_cli, prd_adapters, pythia


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local Pythia checkpoint directory")
    parser.add_argument("--agent", default="claude-4-5-sonnet", choices=["claude-4-5-sonnet", "gpt-6-astra"])
    parser.add_argument("--provider-model", default=None, help="Exact provider model id")
    parser.add_argument("--api-endpoint", default=None, help="Alternative endpoint in the chosen agent's wire format")
    parser.add_argument("--api-key-env", default=None, help="Environment variable holding the key for that endpoint")
    parser.add_argument("--copilot", action="store_true", help="Use your GitHub Copilot subscription through the copilot CLI")
    parser.add_argument("--copilot-model", default=None, help="Model name for the copilot CLI")
    parser.add_argument("--max-premium-requests", type=int, default=30, help="Hard limit on Copilot premium requests")
    parser.add_argument("--budget", type=float, default=5.0, help="Hard spending limit in USD")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--output", required=True, help="Where to write the session report JSON")
    args = parser.parse_args()

    transport = None
    if args.copilot:
        transport = copilot_cli.CopilotCliTransport(
            model=args.copilot_model, max_premium_requests=args.max_premium_requests
        )
    agent = prd_adapters.FrontierAgentProgrammer(
        args.agent,
        llm_transport=transport,
        require_live_provider=True,
        api_endpoint=args.api_endpoint,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
        provider_model=args.provider_model,
        budget_usd=args.budget,
        max_total_tokens=2_000_000,
    )
    report = agent.run_autonomous_tool_loop(
        pythia.PythiaModel.from_pretrained_dir(args.checkpoint), max_turns=args.max_turns
    )
    doc = dataclasses.asdict(dataclasses.replace(report, submitted_weights_by_layer=None))
    Path(args.output).write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "transport": report.transport_mode,
                "turns": len(report.transcript),
                "stop_reason": report.stop_reason or None,
                "cost_usd": report.total_cost_usd,
                "copilot_premium_requests": transport.premium_requests if transport else None,
                "copilot_models": sorted(transport.models_seen) if transport else None,
                "submitted_candidate": report.rust_candidate_digest or None,
                "holdout_accuracy": report.held_out_accuracy_after,
                "sealed_retention": report.sealed_retention_accuracy,
                "submitted_practice_scores": report.submitted_practice_scores,
                "holdout_access_denied": report.holdout_access_denied_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
