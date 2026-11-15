"""Multimodal Engine — subsystem 3 of the architecture ("Vision /
Audio / OCR"): for multimodal requests, route through vision or
OCR preprocessing only when required (architecture section A).

The perceive step turns an image attachment into evidence text —
a description plus an exact transcription of any visible text
(the OCR duty, done by the vision model) — which then flows
through the ordinary grounded pipeline: a text model answers from
the perception, and the confidence estimator treats it as evidence
like any retrieved document.

Modes mirror the LLM adapters: mock by default (a labelled
placeholder — no visual content is invented), Ollama vision when
NR_MODEL_MODE=ollama (model: NR_VISION_MODEL, default llava:7b;
moondream is a lighter alternative). A missing or unreachable
image raises PerceptionError and the node fails with no evidence,
exactly like a failed tool.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

from backend.models.llm_adapter import Generation, parse_generation
from backend.models.model_registry import estimate_cost_usd

VISION_PROMPT = (
    "Describe this image in detail, in English only. Transcribe any "
    "visible text exactly as written, preserving numbers, labels and "
    "values. If it shows a chart or graph, state the values it presents.")

_UPLOADS = (Path(__file__).resolve().parent.parent.parent
            / "data" / "uploads")


class PerceptionError(Exception):
    pass


def resolve_attachment(name: str) -> Path | None:
    """Find an attachment on disk: the path as given (e.g. a repo-
    relative benchmark asset), else the uploads directory."""
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return p
    candidate = _UPLOADS / Path(name).name
    return candidate if candidate.is_file() else None


class VisionPerceiver:
    def __init__(self, mode: str | None = None, model: str | None = None,
                 base_url: str | None = None):
        self.mode = (mode or os.environ.get(
            "NR_MODEL_MODE", "mock")).lower()
        self.model = model or os.environ.get("NR_VISION_MODEL", "llava:7b")
        self.base_url = (base_url or os.environ.get(
            "NR_OLLAMA_URL", "http://localhost:11434")).rstrip("/")

    def perceive(self, attachment: str, query: str = "") -> Generation:
        if self.mode != "ollama":
            text = (f"[Mock perception — no vision model runs in mock "
                    f"mode] Image '{Path(attachment).name}' was received "
                    f"as an attachment. With NR_MODEL_MODE=ollama the "
                    f"vision model ({self.model}) would describe it and "
                    f"transcribe its text here.")
            tokens = 48
            return Generation(text=text, tokens=tokens,
                              cost_usd=estimate_cost_usd("medium", tokens),
                              latency_ms=30)
        path = resolve_attachment(attachment)
        if path is None:
            raise PerceptionError(f"Image attachment not found: {attachment}")
        prompt = VISION_PROMPT
        if query:
            prompt += f"\nThe user asks about the image: {query}"
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "images": [base64.b64encode(path.read_bytes()).decode()],
                   "options": {"temperature": 0.2, "num_predict": 384}}
        try:
            with httpx.Client(timeout=180.0, trust_env=False) as client:
                resp = client.post(f"{self.base_url}/api/generate",
                                   json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise PerceptionError(
                f"Ollama is not reachable at {self.base_url}") from exc
        except httpx.HTTPStatusError as exc:
            raise PerceptionError(
                f"Vision model '{self.model}' failed ({exc.response.status_code}); "
                f"pull it first: ollama pull {self.model}") from exc
        # Priced at the medium tier's reference rate (see registry).
        return parse_generation(resp.json(), prompt, "medium")
