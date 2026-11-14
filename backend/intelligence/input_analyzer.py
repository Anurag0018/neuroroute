"""Input Analyzer — modality detection, task classification, risk.

Phase 1 uses deterministic heuristics so routing is testable offline.
Phase 2 can replace/augment these with a classifier model, keeping
the InputAnalysis contract unchanged.
"""
from __future__ import annotations

import itertools

from backend.schemas import InputAnalysis

_counter = itertools.count(1)

_DOC_EXT = (".pdf", ".doc", ".docx", ".txt", ".md")
_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_AUD_EXT = (".wav", ".mp3", ".m4a", ".ogg")

_TASK_KEYWORDS = {
    "summarization": ("summar", "tl;dr", "tldr", "key points of"),
    "coding": ("code", "debug", "function", "bug", "error in", "write a program",
               "script to", "algorithm to"),
    "research": ("research", "analyse", "analyze", "compare", "review", "paper",
                 "study", "evidence", "literature"),
    "math": ("calculate", "solve", "equation", "integral", "probability of"),
}
_HIGH_RISK = ("medical", "diagnos", "legal advice", "lawsuit", "investment advice",
              "prescription", "suicide")
_MEDIUM_RISK = ("financial", "password", "security", "personal data")


def _detect_modality(attachments: list[str]) -> str:
    for a in attachments:
        low = a.lower()
        if low.endswith(_DOC_EXT):
            return "document"
        if low.endswith(_IMG_EXT):
            return "image"
        if low.endswith(_AUD_EXT):
            return "audio"
    return "text"


def _classify_task(query: str) -> str:
    q = query.lower()
    for task, words in _TASK_KEYWORDS.items():
        if any(w in q for w in words):
            return task
    return "general_qa"


def _assess_risk(query: str, task_type: str) -> str:
    q = query.lower()
    if any(w in q for w in _HIGH_RISK):
        return "high"
    if any(w in q for w in _MEDIUM_RISK) or task_type == "coding":
        return "medium"
    return "low"


def analyze(query: str, attachments: list[str] | None = None,
            request_id: str | None = None) -> InputAnalysis:
    attachments = attachments or []
    task = _classify_task(query)
    return InputAnalysis(
        request_id=request_id or f"req_{next(_counter):04d}",
        query=query,
        modality=_detect_modality(attachments),
        task_type=task,
        risk_level=_assess_risk(query, task),
        attachments=attachments,
    )
