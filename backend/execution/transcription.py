"""Multimodal Engine — the audio path (architecture section E:
"Audio model — speech recognition"; section A: route through
audio preprocessing only when required).

Mirrors the vision perceiver: a `transcribe` preprocessing node
turns an audio attachment into evidence text (the transcript),
which then flows through the ordinary grounded pipeline.

Modes (NR_AUDIO_MODE):
- mock (default): a labelled placeholder — no speech content is
  invented. Also the automatic fallback when faster-whisper is
  not installed.
- whisper: local speech recognition with faster-whisper
  (optional package; model size NR_WHISPER_MODEL, default
  "tiny"). A missing file or a failed transcription raises
  TranscriptionError and the node fails with no evidence, like a
  failed tool or perception.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from backend.execution.perception import resolve_attachment
from backend.models.llm_adapter import Generation
from backend.models.model_registry import estimate_cost_usd


class TranscriptionError(Exception):
    pass


class AudioTranscriber:
    def __init__(self, mode: str | None = None, model: str | None = None):
        self.model_name = model or os.environ.get("NR_WHISPER_MODEL", "tiny")
        requested = (mode or os.environ.get(
            "NR_AUDIO_MODE", "auto")).lower()
        if requested == "whisper":
            self.mode = "whisper"
        elif requested == "mock":
            self.mode = "mock"
        else:  # auto: whisper when the package is available
            try:
                import faster_whisper  # noqa: F401
                self.mode = "whisper"
            except ImportError:
                self.mode = "mock"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "NR_AUDIO_MODE=whisper needs the optional package: "
                    "pip install faster-whisper") from exc
            self._model = WhisperModel(self.model_name,
                                       device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, attachment: str) -> Generation:
        if self.mode != "whisper":
            text = (f"[Mock transcript — no speech model runs in mock "
                    f"mode] Audio '{Path(attachment).name}' was received "
                    f"as an attachment. With faster-whisper installed, "
                    f"the {self.model_name} model would transcribe it here.")
            tokens = 42
            return Generation(text=text, tokens=tokens,
                              cost_usd=estimate_cost_usd("small", tokens),
                              latency_ms=25)
        path = resolve_attachment(attachment)
        if path is None:
            raise TranscriptionError(
                f"Audio attachment not found: {attachment}")
        start = time.monotonic()
        try:
            segments, _info = self._load().transcribe(str(path))
            text = " ".join(s.text.strip() for s in segments).strip()
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for {attachment}: {exc}") from exc
        if not text:
            raise TranscriptionError(
                f"No speech detected in {attachment}")
        latency = int((time.monotonic() - start) * 1000)
        tokens = len(text.split())
        return Generation(text=text, tokens=tokens,
                          cost_usd=estimate_cost_usd("small", tokens),
                          latency_ms=latency)
