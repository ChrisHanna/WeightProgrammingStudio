"""Copilot transport behaviour, using canned CLI output. No subscription is used."""

from __future__ import annotations

import json

import pytest

from neural_foundry import copilot_cli


def _output(content: str, tool_requests: list[object] | None = None) -> str:
    return "\n".join(
        [
            json.dumps({"type": "assistant.message", "data": {"model": "m", "content": content, "toolRequests": tool_requests or []}}),
            json.dumps({"type": "result", "usage": {"premiumRequests": 1}}),
        ]
    )


class Canned(copilot_cli.CopilotCliTransport):
    def __init__(self, outputs: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.outputs = outputs
        self.prompts: list[str] = []

    def _run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


TOOLS = [{"name": "t", "description": "d", "input_schema": {"type": "object"}, "version": "2"}]


def test_seal_flags_disable_every_builtin_capability() -> None:
    flags = copilot_cli.CopilotCliTransport()._seal_flags()
    for required in ("--available-tools=nf_no_such_tool", "--no-custom-instructions", "--disable-builtin-mcps", "--no-ask-user"):
        assert required in flags
    assert not any(f.startswith(("--allow-all", "--yolo", "--add-dir")) for f in flags)


def test_only_new_messages_are_sent_after_the_first_turn() -> None:
    transport = Canned([_output('```json\n{"tool_calls": [{"name": "t", "arguments": {"a": 1}}]}\n```'), _output('{"done": true}')])
    messages = [{"role": "user", "content": "goal"}]
    first = transport(messages, TOOLS)
    assert first["tool_calls"] == [{"name": "t", "arguments": {"a": 1}}]
    assert "goal" in transport.prompts[0] and '"name": "t"' in transport.prompts[0]
    messages += [{"role": "assistant", "content": "x"}, {"role": "user", "content": '{"r": 2}'}]
    assert transport(messages, TOOLS)["tool_calls"] == []
    assert transport.prompts[1] == 'Tool result: {"r": 2}'
    assert transport.premium_requests == 2


def test_builtin_tool_request_stops_the_session() -> None:
    transport = Canned([_output_tool := json.dumps({"type": "tool.execution_start", "data": {}})])
    with pytest.raises(copilot_cli.CopilotCliError, match="Seal broken"):
        transport._ask("x")
    transport = Canned([_output("{}", [{"name": "view"}])])
    with pytest.raises(copilot_cli.CopilotCliError, match="Seal broken"):
        transport._ask("x")


def test_request_limit_is_enforced() -> None:
    transport = Canned([_output('{"done": true}')], max_premium_requests=1)
    transport._ask("x")
    with pytest.raises(copilot_cli.CopilotCliError, match="limit"):
        transport._ask("y")


def test_invalid_json_gets_one_retry() -> None:
    transport = Canned([_output("no json"), _output('{"done": true}')])
    transport._ask = lambda prompt: transport.outputs and json.loads(transport.outputs.pop(0).splitlines()[0])["data"]["content"]  # type: ignore[method-assign]
    assert transport([{"role": "user", "content": "g"}], TOOLS)["tool_calls"] == []
