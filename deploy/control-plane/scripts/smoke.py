"""Exercise authenticated, bounded CPU fixtures; never prints credential values."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def fixture_spec() -> dict:
    definitions = [
        ("local", "local_eval", []),
        ("provider", "provider", []),
        ("rescore", "offline_rescore", ["local"]),
        ("train", "training", []),
        ("edit", "weight_edit", []),
        ("export", "export", ["edit"]),
        ("identity", "identity", ["export"]),
        ("eval", "eval", ["identity"]),
        ("episode", "episode", []),
        ("workflow", "workflow", []),
    ]
    return {
        "version": 1,
        "name": "Single-host deployment CPU smoke",
        "purpose": "development",
        "candidates": [{
            "id": "cpu-fixture",
            "model": {"id": "inline-software-fixture", "revision": "1"},
            "method": "bounded-development-smoke",
        }],
        "stages": [{
            "id": name, "operation": f"software.{operation}", "depends_on": dependencies,
            "candidate_id": "cpu-fixture", "config": {},
            "limits": {"steps": 8, "episodes": 1, "wall_seconds": 20, "recorded_bytes": 32768},
        } for name, operation, dependencies in definitions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run", help="Recheck a previously completed run after restart")
    parser.add_argument("--allow-qualified-runtime", action="store_true",
                        help="Permit explicitly qualified proof profiles; the smoke itself still runs CPU fixtures only")
    args = parser.parse_args()
    port = int(os.environ.get("CP_HTTP_PORT", "8787"))
    token = (ROOT / "artifacts" / "control-plane-local" / "secrets" / "operator_token").read_text().strip()
    base = f"http://127.0.0.1:{port}"

    def request(path, body=None, *, authenticated=True):
        headers = {"Authorization": f"Bearer {token}"} if authenticated else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        with urllib.request.urlopen(
            urllib.request.Request(base + path, data=data, headers=headers), timeout=10
        ) as response:
            return json.load(response)

    try:
        request("/v1/runs", authenticated=False)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise SystemExit(f"Unexpected unauthenticated status: {error.code}") from None
    else:
        raise SystemExit("Operator API unexpectedly permits unauthenticated access")
    if not request("/ready").get("ready"):
        raise SystemExit("API database/schema readiness failed")
    catalog = request("/v1/catalog")
    templates = catalog.get("templates", [])
    if len(templates) != 6:
        raise SystemExit("Installed-wheel catalog must expose six CPU fixture templates")
    if catalog.get("proof_registry_status", {}).get("available") is not False:
        raise SystemExit("Repository-only proof registry must be unavailable in the deployment wheel")
    proof_plugins = [
        plugin for plugin in catalog.get("plugins", [])
        if plugin.get("id", "").startswith(("proof.", "live."))
    ]
    if not proof_plugins:
        raise SystemExit("Expected explicit proof/live capability descriptors")
    for plugin in proof_plugins:
        if plugin.get("enabled") is False:
            continue
        if (not args.allow_qualified_runtime or plugin["id"].startswith("live.")
                or not plugin.get("qualified_profiles")):
            raise SystemExit("Unqualified live/proof operations are not permitted by this smoke mode")

    run_id = args.verify_run
    if run_id is None:
        run = request("/v1/runs", {
            "spec": fixture_spec(), "idempotency_key": "deploy-" + uuid.uuid4().hex,
        })
        run_id = run["id"]
        planned = request(f"/v1/runs/{run_id}/plan", {})
        approval = request(f"/v1/runs/{run_id}/approve", {
            "plan_digest": planned["plan"]["digest"], "limits": planned["plan"]["limits"],
        })
        request(f"/v1/runs/{run_id}/start", {
            "approval_id": approval["id"], "idempotency_key": "deploy-" + uuid.uuid4().hex,
        })
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", run_id):
        raise SystemExit("Invalid run ID")
    deadline = time.monotonic() + 120
    while True:
        run = request(f"/v1/runs/{run_id}")
        if run["state"] == "completed":
            break
        if run["state"] in {"failed", "cancelled", "needs_attention"} or time.monotonic() >= deadline:
            raise SystemExit(f"Fixture run {run_id} did not complete: {run['state']}")
        time.sleep(0.5)
    artifacts = [stage["artifact_id"] for stage in run["stages"]]
    if not artifacts or any(not item for item in artifacts):
        raise SystemExit("Completed stages are missing artifacts")
    for artifact in artifacts:
        document = request(f"/v1/artifacts/{artifact}")["document"]
        if document.get("model_calls") is not False or document.get("release_authorized") is not False:
            raise SystemExit("Fixture artifact is not explicitly bounded/non-release")
    events = request(f"/v1/runs/{run_id}/events")
    if not any(event["type"] == "run.completed" for event in events):
        raise SystemExit("Completion audit event missing")
    print(json.dumps({"run_id": run_id, "state": run["state"],
                      "artifact_count": len(artifacts), "auth_rejection": 401,
                      "catalog_templates": len(templates), "proof_registry_available": False,
                      "runtime": "cpu-software-fixture-v1"}))


if __name__ == "__main__":
    main()
