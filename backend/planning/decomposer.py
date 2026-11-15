"""Decomposer — splits a multi-part request into sub-questions.

Phase 3's decomposed_research strategy answers each sub-question on
its own parallel branch and synthesises the results. Splitting is
deterministic (testable, free); an LLM splitter could replace it
behind this same function contract.

Recognised shapes:
- Two or more questions:   "What is X? How does Y work?"
- Semicolon clauses:       "Summarise A; summarise B"
- Compare requests:        "Compare A and B" -> explain A / explain B
- Conjoined imperatives:   "Explain A and describe B"
Anything else returns [] (not decomposable).
"""
from __future__ import annotations

import re

MAX_SUBQUESTIONS = 3  # keeps branch model calls within the default budget

_STARTERS = ("what", "how", "why", "when", "where", "which", "who",
             "explain", "describe", "compare", "list", "define",
             "summarise", "summarize", "analyse", "analyze", "give",
             "tell", "discuss")

_COMPARE_RE = re.compile(
    r"^compare\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+)$", re.IGNORECASE)

# A right-hand clause that refers back to the left one ("... and why
# IT matters") is a follow-up, not an independent sub-question.
_ANAPHORA_RE = re.compile(
    r"^(what|how|why|when|which)\s+(it|this|that|they)\b", re.IGNORECASE)


def _looks_like_subquestion(text: str) -> bool:
    first = re.split(r"\s+", text.strip().lower(), maxsplit=1)[0]
    return first.strip(",") in _STARTERS


def split_subquestions(query: str) -> list[str]:
    q = query.strip()
    parts: list[str] = []

    if q.count("?") >= 2:
        parts = [p.strip() + "?" for p in q.split("?") if p.strip()]
    elif ";" in q:
        parts = [p.strip() for p in q.split(";") if p.strip()]
    else:
        m = _COMPARE_RE.match(q.rstrip("?. "))
        if m and all(len(g.split()) >= 2 for g in m.groups()):
            # Both sides must be real topics ("Compare supervised and
            # unsupervised learning" is one topic pair for a single
            # grounded answer, not two research branches).
            parts = [f"Explain {m.group(1).strip()}",
                     f"Explain {m.group(2).strip()}"]
        elif " and " in q.lower():
            idx = q.lower().index(" and ")
            left, right = q[:idx].strip(), q[idx + 5:].strip()
            if (_looks_like_subquestion(left)
                    and _looks_like_subquestion(right)
                    and not _ANAPHORA_RE.match(right)):
                parts = [left, right]

    parts = [p for p in parts if len(p) >= 8]
    if len(parts) < 2:
        return []
    return parts[:MAX_SUBQUESTIONS]
