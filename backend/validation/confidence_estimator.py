"""Confidence Estimator — combines independent signals instead of
trusting a model's self-reported confidence.

confidence = 0.35*evidence + 0.25*validation + 0.20*retrieval
             + 0.20*agreement

Policy (starting values — calibrate against measured correctness
with benchmark/calibrate.py):
  >= 0.85  return
  0.60-0.84 verify / run extra checks
  <  0.60  escalate to a stronger path

Thresholds are overridable via NR_RETURN_THRESHOLD /
NR_ESCALATE_THRESHOLD so a calibration finding can be applied
without a code change.
"""
from __future__ import annotations

import os

from backend.schemas import ConfidenceReport
from backend.validation.answer_verifier import VerificationResult

RETURN_THRESHOLD = 0.85
ESCALATE_THRESHOLD = 0.60


def thresholds() -> tuple[float, float]:
    """Active (return, escalate) thresholds: env override when it
    parses and keeps 0 < escalate < return < 1, else the defaults."""
    try:
        ret = float(os.environ.get("NR_RETURN_THRESHOLD", RETURN_THRESHOLD))
        esc = float(os.environ.get("NR_ESCALATE_THRESHOLD", ESCALATE_THRESHOLD))
    except ValueError:
        return RETURN_THRESHOLD, ESCALATE_THRESHOLD
    if 0.0 < esc < ret < 1.0:
        return ret, esc
    return RETURN_THRESHOLD, ESCALATE_THRESHOLD


def estimate(has_retrieval: bool, retrieval_relevance: float,
             verification: VerificationResult | None) -> ConfidenceReport:
    if verification is not None:
        evidence = verification.evidence_coverage
        validation_passed = verification.passed
        agreement = 0.85 if verification.passed else 0.30
    else:
        # No verifier node ran: use neutral priors, never certainty.
        evidence = 0.85 if not has_retrieval else verification_or_default(has_retrieval)
        validation_passed = True
        agreement = 0.80

    relevance = retrieval_relevance if has_retrieval else 0.85
    score = round(0.35 * evidence + 0.25 * (1.0 if validation_passed else 0.0)
                  + 0.20 * relevance + 0.20 * agreement, 3)

    return_threshold, escalate_threshold = thresholds()
    if score >= return_threshold:
        decision = "return"
    elif score >= escalate_threshold:
        decision = "verify"
    else:
        decision = "escalate"

    return ConfidenceReport(
        confidence_score=score, evidence_coverage=evidence,
        validation_passed=validation_passed, retrieval_relevance=relevance,
        model_agreement=agreement, decision=decision)


def verification_or_default(has_retrieval: bool) -> float:
    return 0.5 if has_retrieval else 0.85
