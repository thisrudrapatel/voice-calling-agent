"""The call brain. Transport-agnostic: the browser WebSocket and the Asterisk
phone bridge both push caller audio in and receive agent audio + events out.

    caller audio -> Endpointer (VAD) -> STT -> retrieve -> LLM (streamed)
                 -> sentence -> TTS -> audio back to caller

Events sent to the transport:
    {"type": "state", "state": "listening" | "thinking" | "speaking"}
    {"type": "transcript", "role": "user" | "agent", "text": "..."}
    {"type": "clear"}    stop playing any queued agent audio (caller barged in)
    {"type": "hangup"}   the agent ended the call
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable

from .audio import SAMPLE_RATE
from .config import Settings
from .knowledge import KnowledgeBase, followup_query
from .llm import ExtractiveResponder, OllamaLLM, build_messages, sentences_from_stream
from .speech import speakable
from .stt import STT, create_stt
from .tts import TTS, create_tts
from .vad import Endpointer

log = logging.getLogger(__name__)

SendAudio = Callable[[bytes], Awaitable[None]]
SendEvent = Callable[[dict], Awaitable[None]]

GOODBYE = re.compile(
    r"\b(bye|goodbye|good bye|that'?s all|that is all|nothing else|no,? thanks?|hang up|end the call)\b",
    re.I,
)


class VoiceAgent:
    """Holds the heavy, shared pieces (models, knowledge base). One per process."""

    def __init__(
        self,
        settings: Settings,
        stt: STT | None = None,
        tts: TTS | None = None,
        kb: KnowledgeBase | None = None,
        llm: OllamaLLM | None = None,
    ):
        self.settings = settings
        self.kb = kb or KnowledgeBase(settings.data_dir, settings.ollama_url, settings.embed_model, settings.synonyms)
        self.stt = stt or create_stt(settings, hint=settings.stt_hint)
        self.tts = tts or create_tts(settings)
        self.llm = llm if llm is not None else OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_temperature)
        self.fallback = ExtractiveResponder(settings.handoff, self.kb.synonyms)
        self.llm_ok = False

    async def check_llm(self) -> bool:
        self.llm_ok = bool(self.llm) and await self.llm.available()
        if self.llm_ok:
            log.info("LLM ready: %s via Ollama", self.settings.llm_model)
        else:
            log.warning("Ollama LLM not available -> answering by quoting the knowledge base")
        return self.llm_ok

    def new_session(self, send_audio: SendAudio, send_event: SendEvent) -> "CallSession":
        return CallSession(self, send_audio, send_event)


class CallSession:
    def __init__(self, agent: VoiceAgent, send_audio: SendAudio, send_event: SendEvent):
        self.agent = agent
        self.s = agent.settings
        self.send_audio = send_audio
        self.send_event = send_event
        self.endpointer = Endpointer(self.s.vad_aggressiveness, self.s.end_of_turn_ms)
        self.history: list[dict] = []
        self.turn: asyncio.Task | None = None
        self.pending_audio = b""        # caller speech not answered yet
        self.speaking_until = 0.0       # monotonic time the queued agent audio ends
        self.closed = False
        self.last_audio_at = time.monotonic()
        self.watchdog: asyncio.Task | None = None

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        greeting = self.s.greeting.format(company=self.s.company_name, agent=self.s.agent_name)
        self.turn = asyncio.create_task(self._say_only(greeting))
        self.watchdog = asyncio.create_task(self._watch_for_dead_air())

    async def close(self) -> None:
        self.closed = True
        for task in (self.turn, self.watchdog):
            if task and not task.done():
                task.cancel()

    async def _watch_for_dead_air(self) -> None:
        # Phones with silence suppression stop sending audio instead of
        # sending silence, so the VAD alone would wait forever.
        timeout = self.s.end_of_turn_ms / 1000
        while not self.closed:
            await asyncio.sleep(0.1)
            if self.endpointer.in_speech and time.monotonic() - self.last_audio_at > timeout:
                for ev in self.endpointer.flush():
                    await self._handle(ev)

    @property
    def agent_speaking(self) -> bool:
        return time.monotonic() < self.speaking_until

    # ---------------------------------------------------------------- input

    async def feed_audio(self, pcm: bytes) -> None:
        """Caller audio, 16 kHz mono int16, any chunk size."""
        if self.closed:
            return
        self.last_audio_at = time.monotonic()
        for ev in self.endpointer.feed(pcm):
            await self._handle(ev)

    async def _handle(self, ev) -> None:
        if ev.kind == "speech_start":
            if self.agent_speaking and self.s.barge_in:
                log.info("caller barged in")
                await self._interrupt()
        elif ev.kind == "utterance":
            if self.agent_speaking and not self.s.barge_in:
                return  # half-duplex mode: ignore talk-over
            if self.turn and not self.turn.done():
                # Caller kept talking while we were still thinking: start over
                # with everything they said, so a mid-sentence pause is harmless.
                await self._interrupt()
            self.pending_audio += ev.audio
            self.turn = asyncio.create_task(self._respond(self.pending_audio))

    async def _interrupt(self) -> None:
        if self.turn and not self.turn.done():
            self.turn.cancel()
        if self.agent_speaking:
            self.speaking_until = 0.0
            await self.send_event({"type": "clear"})
        await self.send_event({"type": "state", "state": "listening"})

    # --------------------------------------------------------------- output

    async def _speak(self, text: str) -> None:
        # The transcript shows the written text; the voice gets a spoken form
        # ("+91 97734..." read digit by digit, "IELTS" as "eye elts", ...).
        pcm = await asyncio.to_thread(self.agent.tts.synthesize, speakable(text, self.s.lexicon))
        if not pcm:
            return
        await self.send_event({"type": "transcript", "role": "agent", "text": text})
        await self.send_event({"type": "state", "state": "speaking"})
        await self.send_audio(pcm)
        now = time.monotonic()
        self.speaking_until = max(now, self.speaking_until) + len(pcm) / 2 / SAMPLE_RATE

    async def _wait_until_spoken(self) -> None:
        while self.agent_speaking:
            await asyncio.sleep(0.05)

    async def _say_only(self, text: str) -> None:
        await self._speak(text)
        self.history.append({"role": "assistant", "content": text})
        await self._wait_until_spoken()
        await self.send_event({"type": "state", "state": "listening"})

    # ----------------------------------------------------------------- turn

    async def _respond(self, audio: bytes) -> None:
        spoken: list[str] = []
        user_text = ""
        try:
            await self.send_event({"type": "state", "state": "thinking"})
            t0 = time.monotonic()
            user_text = (await asyncio.to_thread(self.agent.stt.transcribe, audio)).strip()
            if not user_text:
                self.pending_audio = b""
                await self.send_event({"type": "state", "state": "listening"})
                return
            t_stt = time.monotonic() - t0
            await self.send_event({"type": "transcript", "role": "user", "text": user_text})

            last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            query = followup_query(user_text, last_user)
            context = await asyncio.to_thread(self.agent.kb.search, query, self.s.top_k)

            first = True
            async for sentence in self._answer(user_text, query, context):
                if first:
                    log.info("latency: stt %.2fs, first sentence %.2fs", t_stt, time.monotonic() - t0)
                    first = False
                self.pending_audio = b""
                await self._speak(sentence)
                spoken.append(sentence)

            reply = " ".join(spoken)
            log.info("caller: %s | agent: %s", user_text, reply)
            await self._wait_until_spoken()
            if GOODBYE.search(user_text) and len(user_text.split()) <= 8:
                await asyncio.sleep(0.3)
                await self.send_event({"type": "hangup"})
                return
            await self.send_event({"type": "state", "state": "listening"})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn failed")
            self.pending_audio = b""
            await self._speak("Sorry, I ran into a problem. Could you say that again?")
        finally:
            if user_text and spoken:
                self._remember(user_text, " ".join(spoken))

    async def _answer(self, question: str, query: str, context):
        """Yield speakable sentences, from the LLM if possible."""
        a = self.agent
        if a.llm_ok:
            msgs = build_messages(
                question, context, self.history, self.s.agent_name, self.s.company_name, self.s.persona, self.s.handoff
            )
            said_something = False
            try:
                async for sentence in sentences_from_stream(a.llm.stream(msgs)):
                    said_something = True
                    yield sentence
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("LLM failed (%s); using extractive answer", exc)
                if said_something:
                    return
        # The fallback matches words, so give it the follow-up-aware query.
        fallback_q = question if a.fallback.is_small_talk(question) else query
        async for sentence in a.fallback.stream(fallback_q, context):
            yield sentence

    def _remember(self, user: str, assistant: str) -> None:
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.history = self.history[-2 * self.s.max_history_turns :]
