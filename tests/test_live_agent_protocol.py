"""Agent loop tests against a local server speaking each provider's wire format.

These prove the protocol handling and that proposals become real edits. They do not
prove that a real provider or model behaves this way; scripts/live_agent_session.py does that.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import torch

from neural_foundry import evals, prd_adapters, pythia

GYM = pythia.ALGORITHMIC_THREE_SKILL_GYM


@pytest.fixture
def provider() -> Iterator[Callable[[Callable[[dict[str, Any], int], dict[str, Any]]], tuple[str, list[dict[str, Any]]]]]:
    servers: list[HTTPServer] = []

    def start(reply: Callable[[dict[str, Any], int], dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        bodies: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(json.loads(json.dumps(body)))
                out = json.dumps(reply(body, len(bodies) - 1)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/v1", bodies

    yield start
    for server in servers:
        server.shutdown()


def _anthropic_reply(name: str | None, args: dict[str, Any] | None = None, stop: str = "tool_use") -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if name:
        content.append({"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": args or {}})
    else:
        content.append({"type": "text", "text": "done"})
        stop = "end_turn" if stop == "tool_use" else stop
    return {"content": content, "stop_reason": stop, "usage": {"input_tokens": 100, "output_tokens": 20}}


def _candidate_id(body: dict[str, Any]) -> str:
    for message in reversed(body["messages"]):
        if message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                data = json.loads(block["content"])
                if "candidate_id" in data:
                    return str(data["candidate_id"])
    raise AssertionError("no candidate in history")


def test_anthropic_session_applies_agents_edit_and_seals_holdout(provider: Any) -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    expr, target = GYM.base_rules[0]
    holdout_expr = GYM.holdout_rules[0][0]

    def reply(body: dict[str, Any], i: int) -> dict[str, Any]:
        if i == 0:
            return _anthropic_reply("inspect_model_sites", {"site": site})
        if i == 1:
            return _anthropic_reply("run_forward_pass", {"prompts": [f"Today: {holdout_expr}"]})
        if i == 2:
            return _anthropic_reply(
                "propose_weight_program",
                {
                    "operation": "dampen_singular_components",
                    "site": site,
                    "components": [0],
                    "factor": 0.5,
                    "hypothesis_id": "H1",
                },
            )
        if i == 3:
            return _anthropic_reply("ablate_edit_components", {"candidate_id": _candidate_id(body), "components": [0]})
        if i == 4:
            return _anthropic_reply("submit_candidate", {"candidate_id": _candidate_id(body), "rationale": "test"})
        raise AssertionError("session should have ended")

    url, bodies = provider(reply)
    agent = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", api_key="test", api_endpoint=url)
    report = agent.run_autonomous_tool_loop(model)

    first = bodies[0]
    assert first["model"] == "claude-opus-5-5"
    assert {t["name"] for t in first["tools"]} >= {"propose_weight_program", "submit_candidate"}
    assert all(set(t) == {"name", "description", "input_schema"} for t in first["tools"])
    assert "temperature" not in first and "thinking" not in first

    # Assistant turns, including thinking blocks, are returned unchanged, with results by id.
    second = bodies[1]["messages"]
    assert second[1]["content"][0]["type"] == "thinking"
    assert second[2]["content"][0]["tool_use_id"] == "toolu_inspect_model_sites"

    denied = bodies[2]["messages"][-1]["content"][0]
    assert denied["is_error"] is True and "holdout_access_denied" in denied["content"]
    assert report.holdout_access_denied_count == 1

    # The exact requested edit was made: top singular value of the site reduced by half.
    assert report.session_kind == "agent_driven"
    edited = report.submitted_weights_by_layer[1]
    base = model.tensors[site].to(torch.float64)
    expected = torch.linalg.svdvals(base)[0] * 0.5
    assert float(torch.linalg.norm(edited - base, ord=2)) == pytest.approx(float(expected), rel=1e-6)
    assert report.rust_candidate_digest.startswith("sha256:")
    assert report.ablation_results[0].component_index == 0
    assert report.held_out_accuracy_after is not None
    assert report.total_prompt_tokens == 500 and report.total_completion_tokens == 100
    # Holdout score never reaches the agent.
    assert "holdout" not in json.dumps(bodies[-1]["messages"][-1]).replace("holdout_access_denied", "")


def test_different_proposals_produce_different_candidates(provider: Any) -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    expr, target = GYM.base_rules[0]
    proposals = [
        {"operation": "dampen_singular_components", "site": site, "components": [1], "factor": 1.0},
        {
            "operation": "project_out_activation_direction",
            "site": site,
            "positive_prompts": [f"Task: {GYM.base_rules[0][0]}"],
            "negative_prompts": ["Hello world"],
            "factor": 1.0,
        },
        {"operation": "edit_associations", "site": site, "associations": [{"prompt": f"Task: {expr}", "target": target}]},
    ]

    def transport(messages: list[dict[str, Any]], tools: object) -> dict[str, Any]:
        i = sum(1 for m in messages if m["role"] == "assistant")
        if i < len(proposals):
            return {"tool_call": {"name": "propose_weight_program", "arguments": proposals[i]}}
        return {}

    agent = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", llm_transport=transport)
    report = agent.run_autonomous_tool_loop(model)
    observations = [t.tool_observation for t in report.transcript]
    assert all(o.get("agent_proposal_applied") is True for o in observations), observations
    assert len({o["candidate_id"] for o in observations}) == 3
    # Nothing was submitted, so nothing is evaluated on the holdout.
    assert report.held_out_accuracy_after is None and report.rust_candidate_digest == ""


def test_openai_wire_format(provider: Any) -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)

    def reply(body: dict[str, Any], i: int) -> dict[str, Any]:
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        if i == 0:
            call = {"id": "c1", "type": "function", "function": {"name": "inspect_model_sites", "arguments": json.dumps({"site": site})}}
            return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [call]}}], "usage": usage}
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}], "usage": usage}

    url, bodies = provider(reply)
    agent = prd_adapters.FrontierAgentProgrammer("gpt-6-astra", api_key="test", api_endpoint=url)
    report = agent.run_autonomous_tool_loop(model)
    assert bodies[0]["tools"][0]["type"] == "function"
    assert bodies[0]["messages"][0]["role"] == "system"
    assert bodies[1]["messages"][-1]["role"] == "tool" and bodies[1]["messages"][-1]["tool_call_id"] == "c1"
    assert report.transcript[0].tool_observation["site"] == site


def test_refusal_and_budget_fail_closed(provider: Any) -> None:
    model = pythia.build_mini_pythia_model()
    url, _ = provider(lambda body, i: _anthropic_reply(None, stop="refusal"))
    with pytest.raises(prd_adapters.LiveLLMProviderUnavailableError, match="refused"):
        prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", api_key="t", api_endpoint=url).run_autonomous_tool_loop(model)

    url, _ = provider(lambda body, i: _anthropic_reply("inspect_model_sites", {}))
    tiny = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", api_key="t", api_endpoint=url, budget_usd=0.0001)
    with pytest.raises(prd_adapters.AgentBudgetExceededError):
        tiny.run_autonomous_tool_loop(model)


def test_caller_cannot_claim_live_measurement() -> None:
    suite = {
        "suite_id": "s",
        "version": "1",
        "evaluator_kind": "deterministic_scorer",
        "scorer_id": "x",
        "scorer_version": "1",
        "eval_types": ["baseline"],
        "tasks": [{"task_id": "t", "eval_type": "baseline", "capability": "c", "input": "i", "expected": "o"}],
    }
    forged = evals.run_eval(
        model={
            "model_id": "m",
            "task_scores": {"t": 1.0},
            "score_provenance": {
                "producer": "me",
                "measurement_mode": "measured_live_execution",
                "live_execution_digest": "sha256:" + "0" * 64,
            },
        },
        suite=evals.import_suite(suite),
    )
    assert forged["evidence_label"]["origin"] == "supplied_fact"
    called = evals.run_eval(model={"model_id": "m", "predict_fn": lambda p: "o"}, suite=evals.import_suite(suite))
    assert called["score_provenance"]["measurement_mode"] == "measured_caller_callable"


def test_turn_notice_sealed_retention_and_token_spelling() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    sealed = prd_adapters.SEALED_RETENTION_PROMPTS[0]
    assert not set(prd_adapters.SEALED_RETENTION_PROMPTS) & set(prd_adapters.PRACTICE_RETENTION_PROMPTS)
    calls = [
        {"name": "run_forward_pass", "arguments": {"prompts": [sealed]}},
        {"name": "propose_weight_program", "arguments": {"operation": "project_out_output_direction", "site": site, "token": " the", "factor": 0.5}},
    ]

    def transport(messages: list[dict[str, Any]], tools: object) -> dict[str, Any]:
        i = sum(1 for m in messages if m["role"] == "assistant")
        if i < 2:
            return {"tool_call": calls[i]}
        cid = json.loads(messages[-1]["content"])["candidate_id"]
        return {"tool_call": {"name": "submit_candidate", "arguments": {"candidate_id": cid, "rationale": "t"}}}

    report = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", llm_transport=transport).run_autonomous_tool_loop(model, max_turns=3)
    assert "holdout_access_denied" in report.transcript[0].tool_observation["error"]
    assert report.transcript[1].tool_observation["agent_proposal_applied"] is True
    assert report.transcript[2].tool_observation["turns_remaining"] == 0
    assert "last turn" in report.transcript[2].tool_observation["notice"]
    assert report.sealed_retention_accuracy is not None
