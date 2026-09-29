"""Text-to-speech, fully local.

- piper: neural voices that sound natural, fast on CPU (~0.1-0.3 s per sentence).
- espeak: robotic but needs no model download; good for smoke tests.

Both return 16 kHz mono int16 PCM.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Protocol

from .audio import SAMPLE_RATE, resample, wav_to_pcm

log = logging.getLogger(__name__)


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class PiperTTS:
    def __init__(self, voice_path: str, rate: float = 1.0):
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        log.info("loading piper voice %s ...", voice_path)
        self.voice = PiperVoice.load(voice_path)
        # length_scale > 1 is slower speech, so invert the "rate" knob.
        self.config = SynthesisConfig(length_scale=1.0 / max(rate, 0.1))

    def synthesize(self, text: str) -> bytes:
        out = bytearray()
        for chunk in self.voice.synthesize(text, syn_config=self.config):
            out += resample(chunk.audio_int16_bytes, chunk.sample_rate, SAMPLE_RATE)
        return bytes(out)


class EspeakTTS:
    def __init__(self, rate: float = 1.0, voice: str = "en-us"):
        self.bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.bin:
            raise RuntimeError("espeak-ng not found (apt install espeak-ng / brew install espeak-ng)")
        self.wpm = str(int(165 * rate))
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        wav = subprocess.run(
            [self.bin, "-v", self.voice, "-s", self.wpm, "--stdout", text],
            capture_output=True,
            check=True,
        ).stdout
        return wav_to_pcm(wav, SAMPLE_RATE)


def create_tts(settings) -> TTS:
    if settings.tts_engine == "piper":
        try:
            return PiperTTS(settings.piper_voice, settings.speech_rate)
        except Exception as exc:
            log.warning("piper unavailable (%s); falling back to espeak", exc)
    return EspeakTTS(settings.speech_rate)
