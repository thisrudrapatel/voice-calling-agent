"""Small PCM helpers. Internally everything is 16 kHz, mono, signed 16-bit."""

from __future__ import annotations

import io
import wave

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2  # 640 bytes per 20 ms frame


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resampler; plenty for telephone-grade speech."""
    if src_rate == dst_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(round(len(samples) * dst_rate / src_rate))
    if n_out == 0:
        return b""
    x_old = np.arange(len(samples))
    x_new = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_new, x_old, samples).astype(np.int16).tobytes()


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def wav_to_pcm(data: bytes, dst_rate: int = SAMPLE_RATE) -> bytes:
    """Decode a WAV blob to mono int16 PCM at dst_rate."""
    with wave.open(io.BytesIO(data)) as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("only 16-bit WAV is supported")
    if channels > 1:
        frames = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels).mean(axis=1).astype(np.int16).tobytes()
    return resample(frames, rate, dst_rate)


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def silence(ms: int, rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * (rate * ms // 1000)
