"""Verification Engine — evidence-grounded checks, not a second
opinion. Asking another LLM 'is this right?' produces another
unreliable opinion; checking whether the answer's claims are
supported by the retrieved evidence produces a measurement.

Phase 1 check: content-word overlap between the answer and the
evidence documents (coverage). Phase 2 adds claim extraction,
source-consistency and task-specific validators (code execution,
math checking) behind this same `verify()` contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_STOP = set("the a an and or of to in is are was were for on with by this that "
            "it as at from be has have had not but".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


@dataclass
class VerificationResult:
    passed: bool
    evidence_coverage: float    # 0..1, share of answer content grounded in evidence
    detail: str


def verify(answer: str, context_docs: list[dict] | None) -> VerificationResult:
    if not answer or not answer.strip():
        return VerificationResult(False, 0.0, "Empty answer")
    if not context_docs:
        # Nothing to ground against: cannot fail on evidence, but
        # coverage is reported as neutral-low by the caller.
        return VerificationResult(True, 0.5, "No evidence set; basic checks passed")
    evidence = set()
    for d in context_docs:
        evidence |= content_words(d["title"] + " " + d["text"])
    words = content_words(answer)
    if not words:
        return VerificationResult(False, 0.0, "Answer has no content words")
    coverage = round(len(words & evidence) / len(words), 3)
    return VerificationResult(
        coverage >= 0.25, coverage,
        f"{coverage:.0%} of answer content words appear in retrieved evidence")
