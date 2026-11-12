"""HTTP-only operator CLI; credentials are read from the environment or a secret file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("CP_API_URL", "http://127.0.0.1:8787"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("health", "catalog", "list"):
        commands.add_parser(name)
    create = commands.add_parser("create")
    create.add_argument("spec", type=Path)
    create.add_argument("--idempotency-key", required=True)
    for name in ("plan", "approve", "start", "cancel", "events"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        if name == "approve":
            command.add_argument("--plan-digest", required=True)
            command.add_argument("--limits", type=Path, required=True)
        if name == "start":
            command.add_argument("--approval-id", required=True)
            command.add_argument("--idempotency-key", required=True)
        if name == "events":
            command.add_argument("--after", type=int, default=0)
    args = parser.parse_args(argv)
    parsed = urlsplit(args.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or (
        parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        parser.error("--url must be an HTTP(S) origin without credentials or query parameters")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("Remote operator connections require HTTPS")
    try:
        headers = {"Accept": "application/json"}
        if args.command != "health":
            from .api import operator_token

            headers["Authorization"] = "Bearer " + operator_token()
        body = None
        method = "GET"
        path = {"health": "/health", "catalog": "/v1/catalog", "list": "/v1/runs"}.get(
            args.command, ""
        )
        if args.command == "create":
            path, method = "/v1/runs", "POST"
            body = {"spec": json.loads(args.spec.read_text(encoding="utf-8")),
                    "idempotency_key": args.idempotency_key}
        elif args.command in ("plan", "approve", "start", "cancel", "events"):
            path = f"/v1/runs/{quote(args.run_id, safe='')}/{args.command}"
            method = "GET" if args.command == "events" else "POST"
            if args.command == "events":
                path += f"?after={args.after}"
            elif args.command == "approve":
                body = {"plan_digest": args.plan_digest,
                        "limits": json.loads(args.limits.read_text(encoding="utf-8"))}
            elif args.command == "start":
                body = {"approval_id": args.approval_id, "idempotency_key": args.idempotency_key}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(args.url.rstrip("/") + path, data=data, headers=headers, method=method)
        with build_opener(NoRedirect).open(request, timeout=30) as response:
            print(json.dumps(json.load(response), indent=2))
        return 0
    except HTTPError as exc:
        # Never echo arbitrary proxy responses, which may contain secrets or HTML.
        print(f"Operator request rejected (HTTP {exc.code})", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, URLError):
        print("Operator request failed; check configuration and input JSON", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
