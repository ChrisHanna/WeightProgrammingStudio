"""Retired shared-interpreter shim. Native validation uses reference_worker.py.

Do not import gym code here: the trusted dispatch supervisor must remain in a
separate non-dumpable process with private output and challenge-bound replies.
"""
raise SystemExit("retired shim: use the native trusted dispatch supervisor")
