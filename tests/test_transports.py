import asyncio
import json

from fastapi.testclient import TestClient

from agent.audio import resample
from server.app import app
from telephony.audiosocket import KIND_AUDIO, KIND_HANGUP, KIND_UUID, pack, read_message, serve
from tests.conftest import FakeSTT, make_agent, with_trailing_silence


def test_browser_call_over_websocket(settings, kb, speech_pcm):
    app.state.agent = make_agent(settings, kb, FakeSTT("how do I cancel"))
    app.state.enable_audiosocket = False
    with TestClient(app) as client:
        status = client.get("/api/status").json()
        assert status["chunks"] > 0
        assert "Voice Agent" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            got_audio, texts = 0, []
            pcm = with_trailing_silence(speech_pcm)
            for i in range(0, len(pcm), 640):
                ws.send_bytes(pcm[i : i + 640])
            while not any("cancel" in t for t in texts):
                msg = ws.receive()
                if msg.get("bytes"):
                    got_audio += len(msg["bytes"])
                elif msg.get("text"):
                    ev = json.loads(msg["text"])
                    if ev["type"] == "transcript":
                        texts.append(ev["text"])
            assert texts[0] == "Hello."
            assert "how do I cancel" in texts
            assert got_audio > 0
            ws.send_text(json.dumps({"type": "hangup"}))


async def test_phone_call_over_audiosocket(settings, kb, speech_pcm):
    agent = make_agent(settings, kb, FakeSTT("is installation free", "bye"))
    server = await serve(agent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(pack(KIND_UUID, bytes(range(16))))

    received = bytearray()
    kinds = []

    async def listen():
        while True:
            kind, payload = await read_message(reader)
            kinds.append(kind)
            if kind == KIND_HANGUP:
                return
            assert kind == KIND_AUDIO and len(payload) == 320  # 20 ms @ 8 kHz
            received.extend(payload)

    listener = asyncio.create_task(listen())
    # Caller speaks twice, as 8 kHz phone audio in 20 ms frames, in real time.
    for _ in range(2):
        phone_pcm = resample(with_trailing_silence(speech_pcm), 16000, 8000)
        for i in range(0, len(phone_pcm) - 319, 320):
            writer.write(pack(KIND_AUDIO, phone_pcm[i : i + 320]))
            await writer.drain()
            await asyncio.sleep(0.005)
        await asyncio.sleep(1.0)
    await asyncio.wait_for(listener, 20)
    assert kinds[-1] == KIND_HANGUP  # agent said goodbye and hung up
    assert len(received) > 8000 * 2 * 1  # at least a second of agent speech
    writer.close()
    server.close()
