"""Turn-taking: decide when the caller starts talking and when they're done.

Feeds 20 ms frames through WebRTC VAD. A turn starts after ~200 ms of speech
and ends after `end_of_turn_ms` of silence. A little audio from before the
start is kept so the first syllable isn't clipped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import webrtcvad

from .audio import FRAME_BYTES, FRAME_MS, SAMPLE_RATE


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "utterance"
    audio: bytes = b""


class Endpointer:
    def __init__(
        self,
        aggressiveness: int = 2,
        end_of_turn_ms: int = 700,
        start_ms: int = 200,
        preroll_ms: int = 300,
        max_utterance_ms: int = 20000,
        min_energy: float = 250.0,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.start_frames = start_ms // FRAME_MS
        self.end_frames = end_of_turn_ms // FRAME_MS
        self.max_frames = max_utterance_ms // FRAME_MS
        self.min_energy = min_energy
        self.window: deque[tuple[bytes, bool]] = deque(maxlen=max(preroll_ms // FRAME_MS, self.start_frames))
        self.buf = bytearray()
        self.frames: list[bytes] = []
        self.in_speech = False
        self.silent_run = 0

    def _is_speech(self, frame: bytes) -> bool:
        # The energy gate stops line hiss / keyboard clicks from triggering.
        rms = float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2)))
        return rms >= self.min_energy and self.vad.is_speech(frame, SAMPLE_RATE)

    def feed(self, pcm: bytes) -> list[VadEvent]:
        """Accepts 16 kHz int16 PCM of any length; returns events."""
        self.buf += pcm
        events: list[VadEvent] = []
        while len(self.buf) >= FRAME_BYTES:
            frame = bytes(self.buf[:FRAME_BYTES])
            del self.buf[:FRAME_BYTES]
            speech = self._is_speech(frame)
            if not self.in_speech:
                self.window.append((frame, speech))
                recent = list(self.window)[-self.start_frames:]
                if sum(s for _, s in recent) >= int(self.start_frames * 0.8):
                    self.in_speech = True
                    self.silent_run = 0
                    self.frames = [f for f, _ in self.window]
                    self.window.clear()
                    events.append(VadEvent("speech_start"))
            else:
                self.frames.append(frame)
                self.silent_run = 0 if speech else self.silent_run + 1
                if self.silent_run >= self.end_frames or len(self.frames) >= self.max_frames:
                    # drop most of the trailing silence
                    keep = len(self.frames) - max(self.silent_run - 5, 0)
                    events.append(VadEvent("utterance", b"".join(self.frames[:keep])))
                    self.reset()
        return events

    def flush(self) -> list[VadEvent]:
        """End the current turn now (e.g. the line stopped sending audio,
        which some phones do during silence)."""
        if not self.in_speech:
            return []
        audio = b"".join(self.frames)
        self.reset()
        return [VadEvent("utterance", audio)]

    def reset(self) -> None:
        self.in_speech = False
        self.silent_run = 0
        self.frames = []
        self.window.clear()
