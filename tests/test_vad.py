from agent.audio import silence
from agent.vad import Endpointer
from tests.conftest import with_trailing_silence


def feed_in_frames(ep, pcm, chunk=640):
    events = []
    for i in range(0, len(pcm), chunk):
        events += ep.feed(pcm[i : i + chunk])
    return events


def test_detects_one_utterance(speech_pcm):
    ep = Endpointer(end_of_turn_ms=500)
    events = feed_in_frames(ep, with_trailing_silence(speech_pcm))
    assert [e.kind for e in events] == ["speech_start", "utterance"]
    # utterance should hold roughly the speech, not the long silences
    dur = len(events[1].audio) / 32000
    assert 0.8 * len(speech_pcm) / 32000 < dur < len(speech_pcm) / 32000 + 0.6


def test_silence_and_hiss_do_not_trigger():
    import numpy as np

    ep = Endpointer()
    hiss = (np.random.default_rng(0).normal(0, 60, 16000 * 2)).astype(np.int16).tobytes()
    assert feed_in_frames(ep, silence(2000) + hiss) == []


def test_short_pause_does_not_end_turn(speech_pcm):
    ep = Endpointer(end_of_turn_ms=700)
    pcm = speech_pcm + silence(300) + speech_pcm + silence(1000)
    events = feed_in_frames(ep, pcm, chunk=1234)  # odd chunk size on purpose
    assert [e.kind for e in events] == ["speech_start", "utterance"]


def test_flush_ends_open_turn(speech_pcm):
    ep = Endpointer()
    assert [e.kind for e in feed_in_frames(ep, speech_pcm)] == ["speech_start"]
    assert [e.kind for e in ep.flush()] == ["utterance"]
    assert ep.flush() == []
