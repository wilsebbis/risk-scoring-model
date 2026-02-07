from __future__ import annotations

import pytest

from risk_scoring.monitoring import _psi_triage_action


def test_psi_triage_ok() -> None:
    assert _psi_triage_action(0.0) == "OK"
    assert _psi_triage_action(0.05) == "OK"
    assert _psi_triage_action(0.099) == "OK"


def test_psi_triage_investigate() -> None:
    assert _psi_triage_action(0.10) == "investigate"
    assert _psi_triage_action(0.15) == "investigate"
    assert _psi_triage_action(0.25) == "investigate"


def test_psi_triage_retrain() -> None:
    assert _psi_triage_action(0.251) == "retrain"
    assert _psi_triage_action(0.30) == "retrain"
    assert _psi_triage_action(1.0) == "retrain"
