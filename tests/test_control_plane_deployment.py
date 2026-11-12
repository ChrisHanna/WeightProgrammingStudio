"""Focused deployment helper tests; scratch state stays under ignored artifacts."""

import importlib.util
import os
import shutil
import sys
import urllib.error
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "deploy" / "control-plane" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cp_deployment_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bootstrap_state(monkeypatch):
    module = load_script("bootstrap")
    root = ROOT / "artifacts" / "control-plane-local" / "deployment-tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    monkeypatch.setattr(
        module, "__file__", str(root / "deploy" / "control-plane" / "scripts" / "bootstrap.py")
    )
    try:
        yield module, root / "artifacts" / "control-plane-local" / "secrets"
    finally:
        shutil.rmtree(root)


@pytest.mark.skipif(os.name != "posix", reason="POSIX secret mode verification runs in WSL")
def test_bootstrap_private_idempotent_and_silent(bootstrap_state, capsys):
    module, directory = bootstrap_state
    module.main()
    before = {path.name: path.read_text() for path in directory.iterdir()}
    module.main()
    assert before == {path.name: path.read_text() for path in directory.iterdir()}
    assert len(set(before.values())) == 2
    output = capsys.readouterr().out
    assert all(value.strip() not in output for value in before.values())
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in directory.iterdir())
    assert directory.stat().st_mode & 0o777 == 0o700


def test_bootstrap_refuses_invalid_existing_secret(bootstrap_state):
    module, directory = bootstrap_state
    directory.mkdir(parents=True)
    (directory / "postgres_password").write_text("short")
    with pytest.raises(SystemExit, match="unsafe existing secret"):
        module.main()


def test_database_password_urlencoding(monkeypatch):
    module = load_script("entrypoint")
    password = "colon:slash/@question?percent% space+unicodeé"
    monkeypatch.setenv("CP_DATABASE_PASSWORD_FILE", "mounted_password")
    monkeypatch.setattr(module.Path, "read_text", lambda self: password + "\n")
    parsed = urlsplit(module.database_url())
    assert parsed.hostname == "postgres"
    assert parsed.port == 5432
    assert parsed.username == "control_plane"
    assert unquote(parsed.password) == password
    assert parsed.path == "/control_plane"


def test_api_probe_checks_database_readiness_without_leaking_errors(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "entrypoint", load_script("entrypoint"))
    module = load_script("healthcheck")
    monkeypatch.setenv("CP_DATABASE_PASSWORD_FILE", "mounted_password")
    monkeypatch.setenv("CP_OPERATOR_TOKEN_FILE", "mounted_token")
    monkeypatch.setattr(module.Path, "read_text", lambda self: "secret-test-value\n")
    requests = []

    def respond(request, timeout):
        requests.append(request)
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    assert module.main() == 0
    assert requests[0].full_url.endswith("/ready")
    assert requests[0].get_header("Authorization") == "Bearer secret-test-value"

    def fail(request, timeout):
        raise urllib.error.URLError("credential-bearing failure detail")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    assert module.main() == 1
    assert capsys.readouterr() == ("", "")
