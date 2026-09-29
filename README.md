# Voice Calling Agent (local, free POC)

A voice agent you can **call** that talks naturally and answers questions from **your own documents**.
Everything runs on your machine with open-source models: no paid APIs, no API keys, no per-minute costs.

```
 caller ──audio──▶ turn detection ──▶ speech-to-text ──▶ search your docs ──▶ local LLM ──▶ text-to-speech ──audio──▶ caller
 (browser or        (WebRTC VAD)       (Whisper)          (BM25 + embeddings)  (Ollama)      (Piper)
  SIP phone)
```

| Piece | Tool (all free / open source) | Notes |
|---|---|---|
| Call: browser | WebSocket + Web Audio | Open a page, press **Call** |
| Call: real phone | Asterisk + AudioSocket | Any SIP softphone app (Linphone, Zoiper, MicroSIP) can dial in |
| Turn-taking | `webrtcvad` | Detects when you start/stop talking; lets you interrupt the agent |
| Speech-to-text | `faster-whisper` (`base.en`) | Alternative: Vosk for low-end machines |
| Answering | Ollama (`llama3.2:3b`) with retrieval | Without Ollama it still works by quoting the best matching passage |
| Text-to-speech | Piper (`en_US-lessac-medium`) | Natural neural voice; espeak-ng as zero-setup fallback |

What makes it feel like a conversation rather than a walkie-talkie:
- **Streaming replies.** The LLM's answer is cut into sentences and the first sentence is spoken while the rest is still being generated.
- **Barge-in.** Start talking while the agent is speaking and it stops immediately and listens.
- **Pause-tolerant.** If you pause mid-sentence and continue, both parts are treated as one question.
- **Phone-style answers.** The prompt keeps replies to one to three short spoken sentences, and the agent says it doesn't know rather than making things up.
- **Follow-ups.** "And how much does *it* cost?" is linked to the previous question.
- **Hangs up** after you say goodbye.

## Quick start

Requirements: Python 3.10+, about 8 GB RAM, and internet access once to download the models (about 2.5 GB in total).

```bash
# 1. Install Ollama (local LLM runtime): https://ollama.com/download
# 2. Install everything else and download the models:
./scripts/setup.sh

# 3. Run
. .venv/bin/activate
python -m server.app
```

Open **http://localhost:8000**, press **Call** and ask something like *"How much is the Plus plan?"* or *"What happens if I pay late?"*.
Use headphones. Browser echo cancellation works, but headphones stop the agent from hearing itself through your speakers.

> The browser only allows microphone access on `localhost` or HTTPS. To call from your phone's browser on the same Wi-Fi, put the server behind HTTPS (for example `caddy reverse-proxy --from <your-ip>.nip.io --to :8000`), or use the SIP route below.

## Use your own data

Put your files in `data/` and restart. Supported formats: `.md`, `.txt`, `.pdf`, `.csv` (each row becomes a fact), `.json`.
The sample data is a made-up internet provider ("Acme Fiber"); delete it and add your own.

Set the persona in `.env`:

```
COMPANY_NAME=Your Company
AGENT_NAME=Ava
```

To check answer quality **by typing** before trying a call:

```bash
python -m scripts.chat
```

Tips for good answers:
- Use one topic per section, with a Markdown heading (`# Refund policy`). Headings are attached to each chunk and help search a lot.
- Write facts the way they should be spoken: "55 dollars a month" works better than "$55/mo".

## Take real phone calls (SIP)

Asterisk is a free, open-source phone system. It answers the call and streams the audio to the agent over its AudioSocket protocol. The agent listens for it on port `9092`.

```bash
docker compose up -d asterisk     # Linux; uses host networking for SIP/RTP
python -m server.app              # the agent, same as before
```

Without Docker: `sudo apt install asterisk`, copy `telephony/asterisk/*.conf` into `/etc/asterisk/`, then run `sudo systemctl restart asterisk`.

Then, on your phone (on the same Wi-Fi):
1. Install a free softphone app: **Linphone** or **Zoiper**.
2. Add a SIP account with username `1001`, password `agentdemo`, domain `<your computer's LAN IP>`, transport UDP.
3. Dial `100`. The agent answers.

**A real phone number** (so anyone can dial it from a normal phone) needs a SIP trunk / DID from a telecom provider. That is the one part that is never fully free, typically a few dollars a month. Add the trunk to `pjsip.conf` and route its inbound calls to the `voice-agent` context; the agent code does not change.

## Configuration (`.env`)

| Variable | Default | What it does |
|---|---|---|
| `DATA_DIR` | `./data` | Folder with your documents |
| `WHISPER_MODEL` | `base.en` | `tiny.en` is faster; `small.en` is more accurate |
| `STT_ENGINE` | `whisper` | `vosk` for very light machines (`pip install vosk`, set `VOSK_MODEL_PATH`) |
| `LLM_MODEL` | `llama3.2:3b` | Any Ollama model: `qwen2.5:3b`, `llama3.2:1b` (faster), `llama3.1:8b` (smarter) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for semantic search |
| `TTS_ENGINE` | `piper` | `espeak` needs no download but sounds robotic |
| `PIPER_VOICE` | `models/piper/en_US-lessac-medium.onnx` | Other voices: `python -m piper.download_voices --help` |
| `SPEECH_RATE` | `1.0` | Speaking speed |
| `END_OF_TURN_MS` | `700` | Silence that ends your turn. Lower makes replies snappier but can cut you off |
| `BARGE_IN` | `true` | Let callers interrupt the agent |
| `GREETING` | "Hi, thanks for calling {company}…" | First thing the agent says |

## How fast is it?

Latency depends on your hardware. These are rough targets, not measurements from this repo: on a recent laptop CPU (no GPU), expect about **1 to 2 seconds** from the end of your sentence to the agent's first word (Whisper `base.en` about 0.3 to 0.8 s, a 3B LLM's first sentence about 0.5 to 1 s, Piper about 0.1 s).
The server logs the actual numbers for every turn (`latency: stt …, first sentence …`).
If it feels slow, try `WHISPER_MODEL=tiny.en` and `LLM_MODEL=llama3.2:1b` first. With an NVIDIA GPU, Whisper and Ollama use it automatically.

## Project layout

```
agent/
  config.py        settings from environment / .env
  knowledge.py     load + chunk documents, BM25 (+ optional embedding) search
  llm.py           Ollama streaming, sentence splitting, no-LLM fallback
  stt.py           faster-whisper / Vosk
  tts.py           Piper / espeak-ng
  vad.py           turn detection (start/end of speech)
  conversation.py  the call loop: turns, barge-in, history, hang-up
server/
  app.py           FastAPI: web page, /ws call socket, /api/status
  static/          browser phone (mic capture worklet + playback)
telephony/
  audiosocket.py   Asterisk AudioSocket bridge (8 kHz phone audio)
  asterisk/        Asterisk configs + Dockerfile
scripts/
  setup.sh         install + download models
  chat.py          text-mode chat against your data
tests/             pytest suite (no models needed)
```

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest
```

The tests use scripted speech-to-text and synthetic speech from espeak-ng (`apt install espeak-ng`), so they don't need any model downloads. They cover retrieval, sentence streaming, turn detection, barge-in, hang-up, a full browser call over the WebSocket, and a phone call over the AudioSocket protocol.

## What was verified, and what wasn't

- **Verified in development:**
  - The test suite passes.
  - A real **Asterisk 20** instance answered a call and bridged it to the agent: greeting, question, answer from the knowledge base, goodbye, and the agent hung up.
  - A **headless Chromium** call through the web page, using a fake microphone, completed three turns and ended with the agent hanging up.
- **Not yet verified:** the development sandbox could not download Whisper, Piper voices or Ollama models. Those three were therefore not run end to end. The integration code follows each library's current API (faster-whisper 1.2, piper-tts 1.3+, Ollama `/api/chat` and `/api/embed`). The first run on your machine is the real check; `python -m scripts.chat` is a quick way to try the LLM on its own.

## Known POC limitations

- English only by default. For other languages, use a multilingual Whisper model (`base`, `small`), a Piper voice for that language, and an LLM that speaks it.
- The knowledge index is rebuilt at every start, which is fine for up to a few thousand pages.
- No authentication on the web page or SIP account beyond the demo password. Don't expose it to the internet as is.
- Each concurrent call shares the same models; a laptop handles one or two calls at a time comfortably.
