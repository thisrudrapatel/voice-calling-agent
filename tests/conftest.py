import shutil
from pathlib import Path

import pytest

from agent.audio import silence
from agent.config import Settings
from agent.conversation import VoiceAgent
from agent.knowledge import KnowledgeBase
from agent.tts import EspeakTTS

ROOT = Path(__file__).resolve().parent.parent
needs_espeak = pytest.mark.skipif(not (shutil.which("espeak-ng") or shutil.which("espeak")), reason="espeak-ng not installed")


class FakeSTT:
    """Pretends to transcribe: returns scripted lines in order."""

    def __init__(self, *lines: str):
        self.lines = list(lines)
        self.calls = 0

    def transcribe(self, pcm: bytes) -> str:
        self.calls += 1
        return self.lines.pop(0) if self.lines else ""


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    async def available(self) -> bool:
        return True

    async def stream(self, messages):
        self.messages = messages
        for word in self.reply.split(" "):
            yield word + " "


class FakeTTS:
    """100 ms of quiet tone per word, so timing is predictable."""

    def synthesize(self, text: str) -> bytes:
        return b"\x10\x00" * (1600 * len(text.split()))


@pytest.fixture(scope="session")
def speech_pcm() -> bytes:
    """About 1.5 s of real synthetic speech, 16 kHz int16."""
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        pytest.skip("espeak-ng not installed")
    return EspeakTTS().synthesize("How much does the plus plan cost per month?")


@pytest.fixture
def settings() -> Settings:
    """The small Acme Fiber demo profile: stable facts for pipeline tests."""
    s = Settings(profile="acme_fiber")
    s.greeting = "Hello."
    s.end_of_turn_ms = 400
    return s


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase(ROOT / "profiles" / "acme_fiber" / "knowledge")


@pytest.fixture(scope="session")
def uow_settings() -> Settings:
    return Settings(profile="uow_india")


@pytest.fixture(scope="session")
def uow_kb(uow_settings) -> KnowledgeBase:
    return KnowledgeBase(uow_settings.data_dir, synonyms=uow_settings.synonyms)


def make_agent(settings, kb, stt, tts=None, llm=None) -> VoiceAgent:
    agent = VoiceAgent(settings, stt=stt, tts=tts or FakeTTS(), kb=kb, llm=llm or FakeLLM(""))
    agent.llm_ok = llm is not None
    return agent


def with_trailing_silence(pcm: bytes, ms: int = 1000) -> bytes:
    return silence(300) + pcm + silence(ms)
