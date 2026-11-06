"""Shared pytest policy: in a qualification run, skipped is not passed (docs/milestones.md)."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import pytest


def _qualification_run() -> bool:
    return os.environ.get("NF_QUALIFICATION") == "1"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "qualification: needs a real checkpoint, runtime or backend. "
        "With NF_QUALIFICATION=1, skipping this test is a failure.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if (
        _qualification_run()
        and report.skipped
        and not hasattr(report, "wasxfail")
        and item.get_closest_marker("qualification") is not None
    ):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"NF_QUALIFICATION=1: qualification test was skipped: {reason}"


def pytest_collectreport(report: pytest.CollectReport) -> None:
    # A module skipped at import hides its markers, so any such skip fails a qualification run.
    if _qualification_run() and report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"NF_QUALIFICATION=1: module was skipped at collection: {reason}"
