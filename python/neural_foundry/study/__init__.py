"""Controlled-study harness: frozen plans, sealed holdouts and recorded promotion decisions.

Order of work: freeze plan -> synthesize on development items -> freeze manifest ->
custodian draws the holdout -> one query per candidate -> decide.
"""

from .client import CustodianClient, CustodianRefused, declare_exposure, write_candidate
from .plan import PlanError, PlanTamperedError, StudyPlan, Thresholds, freeze_plan, load_frozen_plan

__all__ = [
    "CustodianClient",
    "CustodianRefused",
    "PlanError",
    "PlanTamperedError",
    "StudyPlan",
    "Thresholds",
    "declare_exposure",
    "freeze_plan",
    "load_frozen_plan",
    "write_candidate",
]
