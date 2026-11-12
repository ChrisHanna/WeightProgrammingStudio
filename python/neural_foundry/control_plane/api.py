"""Private, same-origin HTTP adapter for the durable control-plane service."""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from starlette.exceptions import HTTPException

MAX_BODY_BYTES = 1_048_576
STATIC = Path(__file__).with_name("static")


def operator_token() -> str:
    """Read the secret at runtime; never expose it in command arguments or responses."""
    token_file = os.environ.get("CP_OPERATOR_TOKEN_FILE")
    try:
        token = (
            Path(token_file).read_text(encoding="utf-8").strip()
            if token_file
            else os.environ.get("CP_OPERATOR_TOKEN", "").strip()
        )
    except OSError:
        raise RuntimeError("Cannot read CP_OPERATOR_TOKEN_FILE") from None
    if not token or len(token) > 4096 or any(char.isspace() for char in token):
        raise RuntimeError("Configure a nonempty operator token without whitespace")
    return token


def error_response(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def service_error_response(exc: Exception) -> JSONResponse:
    from .models import ControlPlaneError

    if isinstance(exc, ControlPlaneError):
        return error_response(exc.code, exc.message, exc.status)
    return error_response("internal_error", "Request could not be completed", 500)


class PrivateBoundary:
    """Bound bodies before parsing and enforce a private same-origin browser boundary."""

    def __init__(self, app: Any, hosts: set[str]) -> None:
        self.app = app
        self.hosts = hosts

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode("latin-1")
        try:
            parsed_host = urlsplit("//" + host)
            hostname = parsed_host.hostname
            if (
                parsed_host.username or parsed_host.password or parsed_host.path
                or parsed_host.query or parsed_host.fragment
                or parsed_host.port == 0
            ):
                hostname = None
            origin = headers.get(b"origin", b"").decode("latin-1")
            parsed_origin = urlsplit(origin)
            valid_origin = not origin or (
                parsed_origin.scheme == scope["scheme"]
                and parsed_origin.netloc.lower() == host.lower()
                and parsed_origin.path in ("", "/")
                and not parsed_origin.query
                and not parsed_origin.fragment
                and not parsed_origin.username
            )
        except ValueError:
            hostname, valid_origin = None, False

        started = False

        async def secured_send(message: Any) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message.setdefault("headers", []).extend(
                    [
                        (b"content-security-policy", (b"default-src 'self'; script-src 'self'; "
                         b"style-src 'self'; connect-src 'self'; img-src 'self'; "
                         b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                         b"form-action 'self'")),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cache-control", b"no-store"),
                    ]
                )
            await send(message)

        async def reject(code: str, message: str, status: int) -> None:
            await error_response(code, message, status)(scope, receive, secured_send)

        if hostname not in self.hosts:
            await reject("invalid_host", "Host is not allowed", 400)
            return
        if not valid_origin or headers.get(b"sec-fetch-site") == b"cross-site":
            await reject("invalid_origin", "Cross-origin requests are not allowed", 403)
            return
        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.extend(message.get("body", b""))
            if len(chunks) > MAX_BODY_BYTES:
                await reject("request_too_large", "Request body exceeds 1 MiB", 413)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered_receive() -> Any:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(chunks), "more_body": False}
            return await receive()

        try:
            await self.app(scope, buffered_receive, secured_send)
        except Exception as exc:
            # Handle before ServerErrorMiddleware can re-raise/log a secret-bearing traceback.
            if started:
                raise
            await service_error_response(exc)(scope, buffered_receive, secured_send)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateRun(Payload):
    spec: dict[str, Any]
    idempotency_key: StrictStr = Field(min_length=1, max_length=200)


class Approval(Payload):
    plan_digest: StrictStr = Field(min_length=1, max_length=200)
    limits: dict[str, Any]


class StartRun(Payload):
    approval_id: StrictStr = Field(min_length=1, max_length=200)
    idempotency_key: StrictStr = Field(min_length=1, max_length=200)


def create_app(service: Any = None) -> FastAPI:
    """Construct an app without opening a database until ASGI lifespan startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> Any:
        application.state.operator_token = operator_token().encode("utf-8")
        if application.state.service is None:
            dsn = os.environ.get("CP_DATABASE_URL", "")
            root = os.environ.get("CP_ARTIFACT_ROOT", "")
            if not dsn or not root:
                raise RuntimeError("CP_DATABASE_URL and CP_ARTIFACT_ROOT are required")
            from .service import ControlPlaneService

            application.state.service = ControlPlaneService(dsn, Path(root))
        yield

    application = FastAPI(
        title="Neural Foundry private control plane",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.service = service
    application.state.operator_token = None
    hosts = {
        host.strip().lower()
        for host in os.environ.get("CP_ALLOWED_HOSTS", "localhost,127.0.0.1,::1").split(",")
        if host.strip()
    }
    application.add_middleware(PrivateBoundary, hosts=hosts)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response("invalid_request", "Request body or parameters are invalid", 422)

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        messages = {401: "Operator authentication required", 404: "Resource not found",
                    405: "Method not allowed", 503: "Service is not ready"}
        return error_response("http_error", messages.get(exc.status_code, "Request rejected"),
                              exc.status_code)

    @application.exception_handler(Exception)
    async def service_error(request: Request, exc: Exception) -> JSONResponse:
        return service_error_response(exc)

    def authenticated(request: Request) -> Any:
        expected = request.app.state.operator_token
        authorization = request.headers.get("authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if expected is None:
            raise HTTPException(503)
        if not hmac.compare_digest(supplied.encode("utf-8"), expected):
            raise HTTPException(401)
        if request.app.state.service is None:
            raise HTTPException(503)
        return request.app.state.service

    facade_dependency = Depends(authenticated)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    def ready(request: Request) -> JSONResponse:
        facade = request.app.state.service
        try:
            result = facade.health() if facade is not None else False
            healthy = (
                result.get("ready", result.get("ok", result.get("status") == "ok"))
                if isinstance(result, dict) else result is True
            )
        except Exception:  # noqa: BLE001 -- readiness must not expose driver failures
            healthy = False
        return JSONResponse({"ready": bool(healthy)}, status_code=200 if healthy else 503)

    @application.get("/v1/catalog")
    def catalog(facade: Any = facade_dependency) -> Any:
        return facade.catalog_document()

    @application.get("/v1/workspace")
    def workspace(facade: Any = facade_dependency) -> Any:
        return facade.workspace()

    @application.post("/v1/projects")
    def create_project(body: dict[str, Any], facade: Any = facade_dependency) -> Any:
        return facade.create_project(body, actor="operator")

    @application.get("/v1/projects/{project_id}")
    def project(project_id: str, facade: Any = facade_dependency) -> Any:
        return facade.get_project(project_id)

    @application.post("/v1/projects/{project_id}/assets")
    def asset_create(project_id: str, body: dict[str, Any], facade: Any = facade_dependency) -> Any:
        return facade.create_asset(project_id, body, actor="operator")

    @application.post("/v1/projects/{project_id}/preview")
    def preview(project_id: str, body: dict[str, Any], facade: Any = facade_dependency) -> Any:
        return facade.preview_gym(project_id, body)

    @application.post("/v1/projects/{project_id}/experiments")
    def experiment(project_id: str, body: dict[str, Any], facade: Any = facade_dependency) -> Any:
        return facade.create_experiment(project_id, body, actor="operator")

    @application.get("/v1/projects/{project_id}/bundle")
    def bundle(project_id: str, facade: Any = facade_dependency) -> Any:
        return facade.project_bundle(project_id)

    @application.post("/v1/runs")
    def create(body: CreateRun, facade: Any = facade_dependency) -> Any:
        return facade.create_run(body.spec, actor="operator", idempotency_key=body.idempotency_key)

    @application.get("/v1/runs")
    def runs(limit: int = Query(100, ge=1, le=100),
             facade: Any = facade_dependency) -> Any:
        return facade.list_runs(limit=limit)

    @application.get("/v1/runs/{run_id}")
    def run(run_id: str, facade: Any = facade_dependency) -> Any:
        return facade.get_run(run_id)

    @application.post("/v1/runs/{run_id}/plan")
    def plan(run_id: str, facade: Any = facade_dependency) -> Any:
        return facade.plan_run(run_id, actor="operator")

    @application.post("/v1/runs/{run_id}/approve")
    def approve(run_id: str, body: Approval, facade: Any = facade_dependency) -> Any:
        return facade.approve_run(
            run_id, actor="operator", plan_digest=body.plan_digest, limits=body.limits
        )

    @application.post("/v1/runs/{run_id}/start")
    def start(run_id: str, body: StartRun, facade: Any = facade_dependency) -> Any:
        return facade.start_run(
            run_id, actor="operator", approval_id=body.approval_id,
            idempotency_key=body.idempotency_key,
        )

    @application.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, facade: Any = facade_dependency) -> Any:
        return facade.cancel_run(run_id, actor="operator")

    @application.get("/v1/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(0, ge=0, le=9_223_372_036_854_775_807),
               facade: Any = facade_dependency) -> Any:
        return facade.get_events(run_id, after=after)

    @application.post("/v1/artifacts")
    def import_artifact(body: dict[str, Any], facade: Any = facade_dependency) -> Any:
        return facade.import_artifact(body, actor="operator")

    @application.get("/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str, facade: Any = facade_dependency) -> Any:
        return facade.get_artifact(artifact_id)

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "workspace.html")

    @application.get("/operations", include_in_schema=False)
    def operations() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return application


app = create_app()
