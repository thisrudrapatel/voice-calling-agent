"""Web server: serves the in-browser "phone" and a WebSocket that carries the
call audio. Also starts the Asterisk AudioSocket bridge so a real SIP phone
(e.g. a softphone app on your mobile) can dial the agent.

Run:  python -m server.app
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.config import settings
from agent.conversation import VoiceAgent

log = logging.getLogger("server")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests inject their own agent (with fake models) before startup.
    agent: VoiceAgent = getattr(app.state, "agent", None) or VoiceAgent(settings)
    app.state.agent = agent
    await agent.check_llm()
    bridge = None
    if getattr(app.state, "enable_audiosocket", True):
        from telephony.audiosocket import serve as serve_audiosocket

        bridge = await serve_audiosocket(agent, settings.host, settings.audiosocket_port)
    log.info("open http://localhost:%d and press Call", settings.port)
    yield
    if bridge:
        bridge.close()


app = FastAPI(title="Local Voice Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    agent: VoiceAgent = app.state.agent
    return {
        "agent": settings.agent_name,
        "company": settings.company_name,
        "llm": settings.llm_model if agent.llm_ok else "offline (quoting knowledge base)",
        "stt": type(agent.stt).__name__,
        "tts": type(agent.tts).__name__,
        "chunks": len(agent.kb.chunks),
        "semantic_search": agent.kb.vectors is not None,
    }


@app.websocket("/ws")
async def call(ws: WebSocket):
    """Browser call. Client -> server: binary 16 kHz int16 mono PCM frames.
    Server -> client: binary PCM (same format) + JSON text events."""
    await ws.accept()
    agent: VoiceAgent = app.state.agent
    send_lock = asyncio.Lock()
    open_ = True

    async def send_audio(pcm: bytes) -> None:
        if open_:
            async with send_lock:
                await ws.send_bytes(pcm)

    async def send_event(event: dict) -> None:
        if open_:
            async with send_lock:
                await ws.send_text(json.dumps(event))

    session = agent.new_session(send_audio, send_event)
    await session.start()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.feed_audio(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "hangup":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        open_ = False
        await session.close()
        log.info("browser call ended")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
