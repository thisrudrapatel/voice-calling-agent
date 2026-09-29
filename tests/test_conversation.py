import asyncio

from agent.audio import silence
from tests.conftest import FakeLLM, FakeSTT, FakeTTS, make_agent, with_trailing_silence


class Line:
    """Collects what the agent sends back, like a phone line would."""

    def __init__(self):
        self.audio = bytearray()
        self.events: list[dict] = []

    async def send_audio(self, pcm):
        self.audio += pcm

    async def send_event(self, ev):
        self.events.append(ev)

    def said(self, role):
        return [e["text"] for e in self.events if e["type"] == "transcript" and e["role"] == role]

    def has(self, kind):
        return any(e["type"] == kind for e in self.events)


async def feed_realtime(session, pcm, chunk=640, speed=4.0):
    for i in range(0, len(pcm), chunk):
        await session.feed_audio(pcm[i : i + chunk])
        await asyncio.sleep(chunk / 32000 / speed)


async def wait_for(cond, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


async def test_full_turn_with_fallback_answer(settings, kb, speech_pcm):
    agent = make_agent(settings, kb, FakeSTT("when is my payment due"))
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await s.start()
    await wait_for(lambda: line.said("agent") == ["Hello."])
    await feed_realtime(s, with_trailing_silence(speech_pcm))
    await wait_for(lambda: len(line.said("agent")) >= 2)
    assert line.said("user") == ["when is my payment due"]
    assert "15th" in " ".join(line.said("agent")[1:])
    assert len(line.audio) > 0
    await s.close()


async def test_llm_gets_context_and_is_spoken_sentence_by_sentence(settings, kb, speech_pcm):
    llm = FakeLLM("The Plus plan is 55 dollars a month. Want me to sign you up?")
    agent = make_agent(settings, kb, FakeSTT("how much is the plus plan"), llm=llm)
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await feed_realtime(s, with_trailing_silence(speech_pcm))
    await wait_for(lambda: len(line.said("agent")) == 2)
    assert line.said("agent") == ["The Plus plan is 55 dollars a month.", "Want me to sign you up?"]
    assert "55 dollars" in llm.messages[0]["content"]  # knowledge reached the prompt
    await wait_for(lambda: s.history and s.history[-1]["role"] == "assistant")
    assert s.history[-2] == {"role": "user", "content": "how much is the plus plan"}
    await s.close()


async def test_caller_can_interrupt(settings, kb, speech_pcm):
    long_reply = " ".join(["word"] * 60) + "."  # 6 s of agent audio
    agent = make_agent(settings, kb, FakeSTT("tell me everything", "stop, what about pro"), llm=FakeLLM(long_reply))
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await feed_realtime(s, with_trailing_silence(speech_pcm, 600))
    await wait_for(lambda: s.agent_speaking)
    await feed_realtime(s, speech_pcm)  # talk over the agent
    await wait_for(lambda: line.has("clear"))
    assert not s.agent_speaking
    await feed_realtime(s, silence(800))
    await wait_for(lambda: "stop, what about pro" in line.said("user"))
    await s.close()


async def test_goodbye_hangs_up(settings, kb, speech_pcm):
    agent = make_agent(settings, kb, FakeSTT("ok thanks, bye"))
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await feed_realtime(s, with_trailing_silence(speech_pcm))
    await wait_for(lambda: line.has("hangup"))
    assert "goodbye" in " ".join(line.said("agent")).lower()


async def test_noise_transcribed_as_nothing_is_ignored(settings, kb, speech_pcm):
    stt = FakeSTT("")
    agent = make_agent(settings, kb, stt)
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await feed_realtime(s, with_trailing_silence(speech_pcm))
    await wait_for(lambda: stt.calls == 1)
    await asyncio.sleep(0.1)
    assert line.said("agent") == [] and line.said("user") == []
    await s.close()


async def test_turn_ends_when_line_goes_quiet(settings, kb, speech_pcm):
    """Phones with silence suppression just stop sending packets."""
    agent = make_agent(settings, kb, FakeSTT("how do I cancel"))
    line = Line()
    s = agent.new_session(line.send_audio, line.send_event)
    await s.start()
    await feed_realtime(s, silence(300) + speech_pcm)  # no trailing silence at all
    await wait_for(lambda: "how do I cancel" in line.said("user"), timeout=3)
    await s.close()
