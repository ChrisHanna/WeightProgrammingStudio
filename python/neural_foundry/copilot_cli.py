"""Use a GitHub Copilot subscription as the agent's model, through the official `copilot` CLI.

The CLI is itself an agent that can read files and run commands, and this repository holds holdout
answers. Each call therefore runs with no tools, no MCP servers and no custom instructions, from an
empty directory. A response that requests a tool anyway stops the session.

Copilot's own tool calling is unavailable with its tools off, so the agent's tools are described in
the prompt and the model answers in JSON. Usage is counted in premium requests, not tokens.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .prd_adapters import AGENT_SYSTEM_PROMPT, AgentSessionStopped

_RUNNER = """param([string]$PromptFile, [Parameter(ValueFromRemainingArguments=$true)]$Rest)
$prompt = Get-Content -Raw -Encoding utf8 $PromptFile
& copilot @Rest -p $prompt
exit $LASTEXITCODE
"""

_PROTOCOL = """You have no built-in tools. You act only through the tools listed below, by replying with JSON and nothing else:
{"tool_calls": [{"name": "<tool>", "arguments": {...}}]}
You will get the results in the next message. To end without submitting a candidate, reply {"done": true, "reason": "..."}.

Tools:
"""


class CopilotCliError(RuntimeError):
    """The Copilot CLI failed, broke the seal, or exhausted its request limit."""


class CopilotRequestLimitReached(CopilotCliError, AgentSessionStopped):
    """The premium request limit was reached; the session report is still produced."""


def _wsl_path(path: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def _default_sealed_dir() -> tuple[Path, bool]:
    """An empty directory the CLI can run in, plus whether it is being reached from WSL."""
    if shutil.which("pwsh.exe") and shutil.which("wslpath"):
        temp = subprocess.run(
            ["pwsh.exe", "-NoProfile", "-Command", "$env:TEMP"], capture_output=True, text=True, check=True
        ).stdout.strip()
        converted = subprocess.run(["wslpath", "-u", temp], capture_output=True, text=True, check=True).stdout.strip()
        return Path(converted) / f"nf-copilot-{uuid.uuid4().hex}", True
    return Path(tempfile.gettempdir()) / f"nf-copilot-{uuid.uuid4().hex}", False


def parse_reply(text: str) -> dict[str, Any]:
    """Extract the JSON object from a reply, tolerating code fences or surrounding prose."""
    start = text.find("{")
    if start < 0:
        raise ValueError("Reply contains no JSON object")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Reply is not a JSON object")
    return parsed


class CopilotCliTransport:
    """An `llm_transport` for `FrontierAgentProgrammer`, backed by one resumable Copilot session."""

    def __init__(
        self,
        *,
        model: str | None = None,
        max_premium_requests: int = 30,
        timeout_seconds: int = 600,
    ) -> None:
        self.model = model
        self.max_premium_requests = int(max_premium_requests)
        self.timeout_seconds = int(timeout_seconds)
        self.session_id = str(uuid.uuid4())
        self.premium_requests = 0.0
        self._last_reported = 0.0
        self.models_seen: set[str] = set()
        self._sent = 0
        self._started = False
        self._session_exists = False
        self._dir: Path | None = None
        self._wsl = False

    def _seal_flags(self) -> list[str]:
        flags = [
            "--available-tools=nf_no_such_tool",
            "--no-custom-instructions",
            "--disable-builtin-mcps",
            "--no-ask-user",
            "--no-auto-update",
            "--disallow-temp-dir",
            "--output-format",
            "json",
            f"--resume={self.session_id}" if self._session_exists else f"--session-id={self.session_id}",
        ]
        if self.model:
            flags.append(f"--model={self.model}")
        return flags

    def _run(self, prompt: str) -> str:
        if self._dir is None:
            self._dir, self._wsl = _default_sealed_dir()
            control = self._dir.parent / f"{self._dir.name}-control"
            self._dir.mkdir(parents=True)
            control.mkdir(parents=True)
            self._control = control
            (control / "runner.ps1").write_text(_RUNNER, encoding="utf-8")
        to_native = _wsl_path if self._wsl else str
        prompt_file = self._control / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        command = [
            "pwsh.exe" if self._wsl else "pwsh",
            "-NoProfile",
            "-File",
            to_native(self._control / "runner.ps1"),
            to_native(prompt_file),
            "-C",
            to_native(self._dir),
            *self._seal_flags(),
        ]
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=self.timeout_seconds, check=False)
        if done.returncode != 0:
            raise CopilotCliError(f"copilot exited {done.returncode}: {(done.stderr or done.stdout)[-500:]}")
        self._session_exists = True
        return done.stdout

    def _ask(self, prompt: str) -> str:
        if self.premium_requests >= self.max_premium_requests:
            raise CopilotRequestLimitReached(f"Reached the limit of {self.max_premium_requests} premium requests")
        reply = ""
        for line in self._run(prompt).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            data = event.get("data", {})
            if kind == "assistant.message":
                if data.get("toolRequests"):
                    raise CopilotCliError("Seal broken: Copilot requested a built-in tool")
                reply = str(data.get("content", ""))
                if data.get("model"):
                    self.models_seen.add(str(data["model"]))
            elif kind == "tool.execution_start":
                raise CopilotCliError("Seal broken: Copilot started a built-in tool")
            elif kind == "result":
                # Resumed sessions report a running total, so count only the increase.
                reported = float(event.get("usage", {}).get("premiumRequests", 1))
                self.premium_requests += reported - self._last_reported if reported > self._last_reported else reported
                self._last_reported = reported
        if not reply:
            raise CopilotCliError("Copilot returned no reply")
        return reply

    def __call__(self, messages: list[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        parts: list[str] = []
        if not self._started:
            catalog = json.dumps(
                [{k: t[k] for k in ("name", "description", "input_schema") if k in t} for t in tools]
            )
            parts.append(f"{AGENT_SYSTEM_PROMPT}\n\n{_PROTOCOL}{catalog}")
            self._started = True
        parts.extend(
            str(m["content"]) if self._sent == 0 else f"Tool result: {m['content']}"
            for m in messages[self._sent:]
            if m.get("role") == "user"
        )
        self._sent = len(messages)
        prompt = "\n\n".join(parts)
        reply = self._ask(prompt)
        try:
            parsed = parse_reply(reply)
        except ValueError:
            parsed = parse_reply(self._ask("Your last reply was not valid JSON. Reply with only the JSON object."))
        calls = parsed.get("tool_calls") or []
        return {
            "tool_calls": [
                {"name": str(c["name"]), "arguments": dict(c.get("arguments", {}))} for c in calls
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
