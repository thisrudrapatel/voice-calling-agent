"""Phone-call bridge using Asterisk's AudioSocket protocol.

Asterisk (free, open-source PBX) answers a SIP call and streams the raw call
audio to this TCP server; we stream the agent's voice back. Any SIP phone or
softphone app (Linphone, Zoiper, MicroSIP...) can then call the agent, and a
SIP trunk can connect a real phone number later without changing this code.

Wire format, every message:  kind (1 byte) | length (2 bytes, big endian) | payload
    0x00 hangup   0x01 call UUID (16 bytes)   0x03 DTMF digit
    0x10 audio: signed 16-bit little-endian PCM, 8 kHz mono (20 ms = 320 bytes)
    0xff error
"""

from __future__ import annotations

import asyncio
import logging
import struct
import uuid

from agent.audio import SAMPLE_RATE, resample
from agent.conversation import VoiceAgent

log = logging.getLogger(__name__)

KIND_HANGUP, KIND_UUID, KIND_DTMF, KIND_AUDIO, KIND_ERROR = 0x00, 0x01, 0x03, 0x10, 0xFF
PHONE_RATE = 8000
PHONE_FRAME = 320  # bytes: 20 ms @ 8 kHz int16


def pack(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


async def read_message(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    header = await reader.readexactly(3)
    kind, length = struct.unpack(">BH", header)
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


class AudioSocketCall:
    def __init__(self, agent: VoiceAgent, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self.outbox: asyncio.Queue[bytes] = asyncio.Queue()
        self.leftover = b""
        self.session = agent.new_session(self.send_audio, self.send_event)
        self.call_id = "?"
        self.done = asyncio.Event()

    async def send_audio(self, pcm16k: bytes) -> None:
        data = self.leftover + resample(pcm16k, SAMPLE_RATE, PHONE_RATE)
        whole = len(data) - len(data) % PHONE_FRAME
        for i in range(0, whole, PHONE_FRAME):
            self.outbox.put_nowait(data[i : i + PHONE_FRAME])
        self.leftover = data[whole:]

    async def send_event(self, event: dict) -> None:
        if event["type"] == "clear":
            self.leftover = b""
            while not self.outbox.empty():
                self.outbox.get_nowait()
        elif event["type"] == "hangup":
            log.info("[%s] agent hung up", self.call_id)
            await self._write(pack(KIND_HANGUP))
            self.done.set()
        elif event["type"] == "transcript":
            log.info("[%s] %s: %s", self.call_id, event["role"], event["text"])

    async def _write(self, data: bytes) -> None:
        if not self.writer.is_closing():
            self.writer.write(data)
            await self.writer.drain()

    async def _pace_output(self) -> None:
        """Asterisk plays frames as they arrive, so send exactly one per 20 ms."""
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        while not self.done.is_set():
            frame = await self.outbox.get()
            now = loop.time()
            next_t = max(next_t, now)
            if next_t > now:
                await asyncio.sleep(next_t - now)
            await self._write(pack(KIND_AUDIO, frame))
            next_t += 0.02

    async def run(self) -> None:
        pacer = asyncio.create_task(self._pace_output())
        reader_task = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait({pacer, reader_task, asyncio.create_task(self.done.wait())}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (pacer, reader_task):
                t.cancel()
            await self.session.close()
            self.writer.close()
            log.info("[%s] call ended", self.call_id)

    async def _read_loop(self) -> None:
        started = False
        try:
            while True:
                kind, payload = await read_message(self.reader)
                if kind == KIND_UUID:
                    self.call_id = str(uuid.UUID(bytes=payload))[:8] if len(payload) == 16 else payload.hex()
                    log.info("[%s] incoming phone call", self.call_id)
                    if not started:
                        started = True
                        await self.session.start()
                elif kind == KIND_AUDIO:
                    if not started:  # some setups skip the UUID message
                        started = True
                        await self.session.start()
                    await self.session.feed_audio(resample(payload, PHONE_RATE, SAMPLE_RATE))
                elif kind == KIND_DTMF:
                    log.info("[%s] DTMF %s", self.call_id, payload.decode(errors="ignore"))
                elif kind == KIND_HANGUP:
                    log.info("[%s] caller hung up", self.call_id)
                    break
                elif kind == KIND_ERROR:
                    log.warning("[%s] asterisk reported error %s", self.call_id, payload.hex())
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.done.set()


async def serve(agent: VoiceAgent, host: str, port: int) -> asyncio.base_events.Server:
    async def handle(reader, writer):
        await AudioSocketCall(agent, reader, writer).run()

    server = await asyncio.start_server(handle, host, port)
    log.info("AudioSocket phone bridge listening on %s:%d", host, port)
    return server
