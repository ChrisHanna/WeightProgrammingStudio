"""Trusted dispatch supervisor; only the --callback subprocess imports assets.

The supervisor is non-dumpable before spawning: same-UID callbacks cannot open
its /proc descriptors/memory or ptrace it. Child stdout is a private, bounded
reply channel, NEVER the supervisor's lifecycle output. Fresh random challenges
bind replies to dispatches, including a final shutdown. Callback values remain
adversarial claims; dispatch evidence does not prove malicious code's semantics.
The outer qualified sandbox supplies CPU, memory, process and deadline limits.
"""
import contextlib
import ctypes
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAX_REPLY = 65536


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"


def resolve(entry):
    module = importlib.import_module(entry["file"][:-3].replace("/", ".").removesuffix(".__init__"))
    if Path(module.__file__).resolve() != Path("/job/assets", entry["file"]):
        raise ValueError("entry escaped selected module")
    value = module
    for part in entry["symbol"].split("."):
        value = getattr(value, part)
    return value


def callback():
    # No supervisor output descriptor or challenge for a future dispatch is
    # inherited. Even import-time os.write(1)/os._exit cannot emit a trace.
    sys.path.insert(0, "/job/assets")
    functions = {}
    environment = None
    for line in sys.stdin.buffer:
        command = json.loads(line)
        operation = command["operation"]
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "load":
                    functions = {name: resolve(entry) for name, entry in command["entries"].items()}
                    value = None
                elif operation == "construct":
                    environment = functions["factory"]()
                    value = None
                elif operation == "shutdown":
                    value = None
                else:
                    function = (getattr(environment, operation) if operation in
                                ("reset", "step", "result", "close") else functions[operation])
                    value = function(*command.get("args", []), **command.get("kwargs", {}))
            result = {"ok": True, "value": value}
            # Serialization is inside the callback, not the trusted interpreter.
            wire = encode({"challenge": command["challenge"], "result": result})
        except Exception as error:  # noqa: BLE001 -- arbitrary callback values/errors
            wire = encode({"challenge": command["challenge"],
                           "result": {"ok": False, "error": type(error).__name__}})
        sys.stdout.buffer.write(wire)
        sys.stdout.buffer.flush()
        if operation == "shutdown":
            return


class Dispatcher:
    def __init__(self, entries):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
            raise RuntimeError("non-dumpable supervisor unavailable")
        self.child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "/job/main.py", "--callback"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            close_fds=True, cwd="/out", env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        if self.call("load", entries=entries) != {"ok": True, "value": None}:
            raise RuntimeError("entry resolution failed")

    def call(self, operation, *args, **kwargs):
        challenge = os.urandom(32).hex()
        command = {"challenge": challenge, "operation": operation, "args": args}
        if operation == "load":
            command.update(kwargs)
        else:
            command["kwargs"] = kwargs
        wire = encode(command)
        if len(wire) > MAX_REPLY:
            raise ValueError("dispatch request exceeded")
        self.child.stdin.write(wire)
        self.child.stdin.flush()
        line = self.child.stdout.readline(MAX_REPLY + 1)
        if not line.endswith(b"\n") or len(line) > MAX_REPLY:
            raise RuntimeError("callback reply missing or exceeded")
        reply = json.loads(line)
        if (type(reply) is not dict or set(reply) != {"challenge", "result"}
                or reply["challenge"] != challenge):
            raise RuntimeError("callback protocol violation")
        result = reply["result"]
        if (type(result) is not dict or type(result.get("ok")) is not bool
                or (result["ok"] and set(result) != {"ok", "value"})
                or (not result["ok"] and set(result) != {"ok", "error"})):
            raise RuntimeError("callback result schema")
        return result

    def finish(self):
        if self.call("shutdown") != {"ok": True, "value": None}:
            raise RuntimeError("callback shutdown failed")
        self.child.stdin.close()
        if self.child.wait() != 0 or self.child.stdout.read(1):
            raise RuntimeError("callback exit or trailing output")


def episode(dispatch, cell):
    trace = {"reset": {"ok": False}, "steps": [], "evaluation": {"ok": False}}
    constructed = dispatch.call("construct")["ok"]
    try:
        if constructed:
            trace["reset"] = dispatch.call("reset", inputs=cell["input"], seed=cell["seed"])
            if trace["reset"]["ok"]:
                for action in cell["actions"]:
                    step = dispatch.call("step", action)
                    trace["steps"].append(step)
                    if not step["ok"]:
                        break
                    value = step["value"]
                    if isinstance(value, dict) and (value.get("terminated") is True
                                                    or value.get("truncated") is True):
                        trace["evaluation"] = dispatch.call("result")
                        break
    finally:
        trace["close"] = dispatch.call("close") if constructed else {"ok": False}
    return trace


def scoring(dispatch, cell, output, observed_name):
    row = {"oracle": dispatch.call("oracle", cell["input"]),
           "good_score": dispatch.call("score", cell["input"], cell["expected"], cell["expected"]),
           "invalid_score": dispatch.call("score", cell["input"], None, cell["expected"]),
           "wrong_scores": [dispatch.call("score", cell["input"], wrong, cell["expected"])
                            for wrong in cell["wrong"]]}
    if output["ok"]:
        row[observed_name] = dispatch.call("score", cell["input"], output["value"], cell["expected"])
    elif observed_name == "score":
        row[observed_name] = dispatch.call("score", cell["input"], None, cell["expected"])
    return row


def main():
    request = json.load(sys.stdin)
    mode = request.get("mode", "validation")
    if mode == "policy":
        entries = {"evaluate": request["entry"]}
    elif mode == "episode":
        entries = {"factory": request["factory"]}
    elif mode == "gym":
        entries = {key: request[key] for key in ("oracle", "score")}
    else:
        interface = request["interface"]
        entries = {key: interface[key] for key in ("oracle", "score")}
        entries["evaluate" if interface["kind"] == "single_shot" else "factory"] = interface[
            "evaluate" if interface["kind"] == "single_shot" else "factory"]
    dispatch = Dispatcher(entries)
    samples = []
    if mode == "validation":
        for index, cell in enumerate(request["cases"]):
            for seed in request["seeds"]:
                for repeat in range(2):
                    sample = {"case": index, "seed": seed, "repeat": repeat}
                    if "factory" in entries:
                        sample["episode"] = episode(dispatch, {**cell, "seed": seed})
                        sample["evaluation"] = sample["episode"]["evaluation"]
                    else:
                        sample["evaluation"] = dispatch.call("evaluate", cell["input"], seed=seed)
                    sample.update(scoring(dispatch, cell, sample["evaluation"], "observed_score"))
                    samples.append(sample)
    else:
        for index, cell in enumerate(request["cells"]):
            sample = {"index": index}
            if mode == "policy":
                sample["output"] = dispatch.call("evaluate", cell["input"], seed=cell["seed"])
            elif mode == "episode":
                sample["episode"] = episode(dispatch, cell)
            else:
                sample.update(scoring(dispatch, cell, {"ok": True, "value": cell["output"]}, "score"))
            samples.append(sample)
    dispatch.finish()
    sys.stdout.buffer.write(encode({"schema_version": 1, "samples": samples}))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--callback"]:
        callback()
    else:
        main()
