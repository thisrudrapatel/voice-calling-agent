"""Speech-to-text, fully local.

- whisper: faster-whisper (CTranslate2). Best accuracy; `base.en` runs in
  roughly 0.3-1 s per utterance on a modern laptop CPU.
- vosk: very light, works on a Raspberry Pi, lower accuracy.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from .audio import SAMPLE_RATE, pcm_to_float

log = logging.getLogger(__name__)


class STT(Protocol):
    def transcribe(self, pcm: bytes) -> str: ...


class WhisperSTT:
    def __init__(self, model: str = "base.en", compute_type: str = "int8", hint: str = ""):
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper %s ...", model)
        self.model = WhisperModel(model, device="auto", compute_type=compute_type)
        # Domain words (company/product names) nudge Whisper to spell them right.
        self.hint = hint
        self.english_only = model.endswith(".en")

    def transcribe(self, pcm: bytes) -> str:
        segments, _ = self.model.transcribe(
            pcm_to_float(pcm),
            language="en" if self.english_only else None,
            beam_size=1,
            vad_filter=False,  # we already cut utterances with our own VAD
            condition_on_previous_text=False,
            initial_prompt=self.hint or None,
        )
        text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()
        return _drop_hallucinations(text)


class VoskSTT:
    def __init__(self, model_path: str):
        from vosk import KaldiRecognizer, Model

        log.info("loading vosk model from %s ...", model_path)
        self.model = Model(model_path)
        self._rec_cls = KaldiRecognizer

    def transcribe(self, pcm: bytes) -> str:
        rec = self._rec_cls(self.model, SAMPLE_RATE)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()


# Whisper sometimes "hears" these on noise or breathing. ("Thank you" and
# "bye" are deliberately not listed: callers really say them to hang up.)
_HALLUCINATIONS = {
    "", "you", "you.", ".", "so", "uh", "um", "hmm", "hmm.",
    "thanks for watching!", "thank you for watching.", "thanks for watching.",
}


def _drop_hallucinations(text: str) -> str:
    return "" if text.lower().strip() in _HALLUCINATIONS else text


def create_stt(settings, hint: str = "") -> STT:
    try:
        if settings.stt_engine == "vosk":
            return VoskSTT(settings.vosk_model_path)
        return WhisperSTT(settings.whisper_model, settings.whisper_compute_type, hint)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the {settings.stt_engine} speech-to-text model ({exc}). "
            "Run scripts/setup.sh once with internet access to download it, "
            "or point WHISPER_MODEL / VOSK_MODEL_PATH at a local model folder."
        ) from exc
