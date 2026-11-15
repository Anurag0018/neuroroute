"""Tool Engine — subsystem 3 of the architecture ("Tool Engine:
APIs / Python / Search"; execution layer: "Tool executor — APIs,
calculations, web search").

Design rules, mirroring the rest of NeuroRoute:
- Tools are a closed registry, never free-form: a plan may only
  invoke a tool by its registered name (the graph validator checks
  this), and no tool evaluates arbitrary code — the calculator
  walks an AST allowlist instead of using eval().
- Tools are deterministic where the domain allows it (calculator,
  datetime), so their output can serve as ground-truth evidence
  for the confidence estimator (the architecture's "tool
  validation" signal: a calculation either checks out or the node
  fails).
- A tool that cannot produce a result raises ToolError; the node
  executor turns that into a failed node with no evidence, and the
  pipeline's confidence/escalation machinery handles the rest.
"""
from __future__ import annotations

import ast
import datetime as _dt
import operator
import re

import httpx


class ToolError(Exception):
    pass


# ------------------------------------------------------------------
# Calculator — arithmetic extracted from the query, AST-evaluated.
# ------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_PERCENT_OF = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*%\s*of\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_EXPR = re.compile(
    r"\d[\d,]*(?:\.\d+)?(?:\s*[-+*/×]\s*\d[\d,]*(?:\.\d+)?)+")


def _eval_ast(node) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left),
                                      _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    raise ToolError("Expression contains non-arithmetic elements")


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _calculator(query: str) -> str:
    m = _PERCENT_OF.search(query)
    if m:
        pct = float(m.group(1).replace(",", ""))
        base = float(m.group(2).replace(",", ""))
        return (f"Calculator: {m.group(1)}% of {m.group(2)} = "
                f"{_fmt(pct / 100.0 * base)}")
    m = _EXPR.search(query.replace("×", "*"))
    if not m:
        raise ToolError("No arithmetic expression found in the query")
    expr = m.group(0).replace(",", "")
    try:
        value = _eval_ast(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        raise ToolError("Division by zero")
    except (SyntaxError, ValueError) as exc:
        raise ToolError(f"Could not evaluate expression: {exc}")
    return f"Calculator: {expr.strip()} = {_fmt(value)}"


# ------------------------------------------------------------------
# Datetime — date arithmetic over dates named in the query.
# ------------------------------------------------------------------

_MONTHS = {name: i for i, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
_DATE_MDY = re.compile(
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
    re.IGNORECASE)
_DATE_DMY = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\s+(\d{4})",
    re.IGNORECASE)
_DATE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_SLASH = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")  # DD/MM/YYYY
_ADD_DAYS = re.compile(r"(\d+)\s*days?\s+(?:from|after|later than)\b",
                       re.IGNORECASE)


def _find_dates(text: str) -> list[_dt.date]:
    found: list[tuple[int, _dt.date]] = []
    for m in _DATE_MDY.finditer(text):
        found.append((m.start(), _dt.date(int(m.group(3)),
                                          _MONTHS[m.group(1).lower()],
                                          int(m.group(2)))))
    for m in _DATE_DMY.finditer(text):
        found.append((m.start(), _dt.date(int(m.group(3)),
                                          _MONTHS[m.group(2).lower()],
                                          int(m.group(1)))))
    for m in _DATE_ISO.finditer(text):
        found.append((m.start(), _dt.date(int(m.group(1)), int(m.group(2)),
                                          int(m.group(3)))))
    for m in _DATE_SLASH.finditer(text):
        found.append((m.start(), _dt.date(int(m.group(3)), int(m.group(2)),
                                          int(m.group(1)))))
    return [d for _, d in sorted(found)]


def _long(d: _dt.date) -> str:
    return d.strftime("%d %B %Y")


def _datetime_tool(query: str) -> str:
    dates = _find_dates(query)
    if not dates:
        raise ToolError("No recognisable date in the query")
    q = query.lower()
    add = _ADD_DAYS.search(query)
    if add and len(dates) >= 1:
        result = dates[0] + _dt.timedelta(days=int(add.group(1)))
        return (f"Date calculation: {_long(dates[0])} + {add.group(1)} days "
                f"= {_long(result)} (a {result.strftime('%A')})")
    if len(dates) >= 2:
        delta = (dates[1] - dates[0]).days
        return (f"Date calculation: there are {abs(delta)} days between "
                f"{_long(dates[0])} and {_long(dates[1])}")
    d = dates[0]
    return f"Date calculation: {_long(d)} is a {d.strftime('%A')}"


# ------------------------------------------------------------------
# Web search — DuckDuckGo Instant Answers (no API key). A real
# external call with real failure modes; both are handled.
# ------------------------------------------------------------------

def _fetch_ddg(query: str) -> dict:
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        resp = client.get("https://api.duckduckgo.com/",
                          params={"q": query, "format": "json",
                                  "no_html": 1, "skip_disambig": 1})
        resp.raise_for_status()
        return resp.json()


def _web_search(query: str) -> str:
    try:
        data = _fetch_ddg(query)
    except Exception as exc:
        raise ToolError(f"Web search unavailable: {exc.__class__.__name__}")
    parts = []
    if data.get("AbstractText"):
        src = data.get("AbstractSource") or "web"
        parts.append(f"{data['AbstractText']} (source: {src})")
    for topic in (data.get("RelatedTopics") or [])[:3]:
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(topic["Text"])
    if not parts:
        raise ToolError("Web search returned no usable results")
    return "Web search results: " + " | ".join(parts)


# ------------------------------------------------------------------
# Registry + detection
# ------------------------------------------------------------------

TOOL_REGISTRY = {
    "calculator": {"description": "Arithmetic and percentages, AST-evaluated",
                   "run": _calculator},
    "datetime": {"description": "Weekdays, days-between and date offsets",
                 "run": _datetime_tool},
    "web_search": {"description": "DuckDuckGo Instant Answers lookup",
                   "run": _web_search},
}


def run_tool(name: str, query: str) -> str:
    """Execute a registered tool against the raw query text."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise ToolError(f"Unknown tool: {name}")
    return entry["run"](query)


_SEARCH_TRIGGERS = ("search the web", "search online", "web search",
                    "google search", "google it", "latest news about",
                    "search for the latest")


def detect_tools(query: str) -> list[str]:
    """Which registered tools this query needs (deterministic, narrow
    patterns — a false positive would hijack a normal question into
    the tool path, so every trigger requires concrete evidence in
    the text: digits around an operator, a parseable date with a
    date question, or an explicit search request)."""
    q = query.lower()
    tools: list[str] = []
    # Strip ISO/slash dates first: "2026-11-15" is not subtraction.
    no_dates = _DATE_ISO.sub(" ", query)
    no_dates = _DATE_SLASH.sub(" ", no_dates)
    if _PERCENT_OF.search(query) or _EXPR.search(no_dates):
        tools.append("calculator")
    if _find_dates(query) and any(w in q for w in (
            "day of the week", "which day", "how many days",
            "days between", "days from", "days after", "what date")):
        tools.append("datetime")
    if any(t in q for t in _SEARCH_TRIGGERS):
        tools.append("web_search")
    return tools[:2]
