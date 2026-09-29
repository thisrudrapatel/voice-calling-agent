# Voice Calling Agent (local, free POC)

A voice agent you can **call** that talks naturally and answers questions from **your own documents**.
Everything runs on your machine with open-source models: no paid APIs, no API keys, no per-minute costs.

It is currently set up as **Maya, a virtual admissions assistant for UOW India** (the University of Wollongong's
campus in GIFT City, Gandhinagar). Maya answers questions about courses, admissions, entry requirements,
scholarships, fees, accommodation, campus life and careers. The knowledge was compiled from UOW India's public
website; see [`profiles/uow_india/SOURCES.md`](profiles/uow_india/SOURCES.md) for sources and deliberate gaps.

> **Unofficial proof of concept.** This project is not affiliated with or endorsed by the University of Wollongong.
> Fees, intakes and scholarships change every year. Have the university verify the knowledge files before real
> students or parents use it.

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

Open **http://localhost:8000**, press **Call** and ask something like *"What courses do you offer?"*, *"Is there a hostel?"*,
*"I got 88 percent in Class 12. Which scholarship can I get?"* or *"What IELTS score do I need for the Master of Computing?"*.
Use headphones. Browser echo cancellation works, but headphones stop the agent from hearing itself through your speakers.

> The browser only allows microphone access on `localhost` or HTTPS. To call from your phone's browser on the same Wi-Fi, put the server behind HTTPS (for example `caddy reverse-proxy --from <your-ip>.nip.io --to :8000`), or use the SIP route below.

## Profiles: what the agent knows and how it behaves

Everything specific to one organisation lives in a profile folder. `PROFILE` picks one (default `uow_india`):

```
profiles/uow_india/
  profile.json    agent name, organisation, greeting, who to hand off to, speech-to-text vocabulary hint
  persona.md      extra instructions for the LLM: tone, audience, rules (e.g. "never quote a fee amount")
  knowledge/      the documents answers come from (.md .txt .pdf .csv .json)
  synonyms.txt    caller words -> document words, e.g. "hostel, accommodation, housing, paying guest"
  lexicon.txt     pronunciation fixes for the voice, e.g. "IELTS => eye elts", "GIFT City => Gift City"
  SOURCES.md      where the knowledge came from (not read by the agent)
profiles/acme_fiber/   a small made-up example, used by the tests
```

**Tuning the UOW India agent:**
- **Add fees or other updates.** Drop the official *UOW India Course Guide* PDF, or any new page saved as `.md`/`.txt`, into `profiles/uow_india/knowledge/` and restart. PDFs are indexed automatically.
- **It misses a question.** Add the caller's words to `synonyms.txt`, or add a short section with a clear heading to the knowledge files.
- **It says something wrong or awkwardly.** Add a rule to `persona.md`.
- **It mispronounces a word.** Add it to `lexicon.txt`.
- **It mishears a name.** Add the name to `stt_hint` in `profile.json`.

Phone numbers, emails, websites, `AUD`/`₹` amounts, lakh/crore figures and percentages are converted to spoken form automatically.
The on-screen transcript keeps the written form.

To check answers **by typing** before trying a call (`--spoken` also shows exactly what the voice will say):

```bash
python -m scripts.chat --spoken
python -m scripts.chat --profile acme_fiber
```

**New organisation:** copy `profiles/uow_india` to `profiles/<name>`, replace the contents, and set `PROFILE=<name>` in `.env`.

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
| `PROFILE` | `uow_india` | Which folder under `profiles/` to use |
| `DATA_DIR` | the profile's `knowledge/` | Override the documents folder |
| `AGENT_NAME`, `COMPANY_NAME`, `GREETING`, `HANDOFF`, `STT_HINT` | from `profile.json` | Override the profile without editing it |
| `WHISPER_MODEL` | `base.en` | `tiny.en` is faster. `small.en` is noticeably better with Indian-accented English, if your machine can keep up |
| `STT_ENGINE` | `whisper` | `vosk` for very light machines (`pip install vosk`, set `VOSK_MODEL_PATH`) |
| `LLM_MODEL` | `llama3.2:3b` | Any Ollama model: `qwen2.5:3b`, `llama3.2:1b` (faster), `llama3.1:8b` (smarter) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for semantic search |
| `TTS_ENGINE` | `piper` | `espeak` needs no download but sounds robotic |
| `PIPER_VOICE` | `models/piper/en_US-lessac-medium.onnx` | Other voices: `python -m piper.download_voices --help` |
| `SPEECH_RATE` | `1.0` | Speaking speed |
| `END_OF_TURN_MS` | `700` | Silence that ends your turn. Lower makes replies snappier but can cut you off |
| `BARGE_IN` | `true` | Let callers interrupt the agent |

## How fast is it?

Latency depends on your hardware. These are rough targets, not measurements from this repo: on a recent laptop CPU (no GPU), expect about **1 to 2 seconds** from the end of your sentence to the agent's first word (Whisper `base.en` about 0.3 to 0.8 s, a 3B LLM's first sentence about 0.5 to 1 s, Piper about 0.1 s).
The server logs the actual numbers for every turn (`latency: stt …, first sentence …`).
If it feels slow, try `WHISPER_MODEL=tiny.en` and `LLM_MODEL=llama3.2:1b` first. With an NVIDIA GPU, Whisper and Ollama use it automatically.

## Project layout

```
agent/
  config.py        settings from environment / .env, profile loading
  knowledge.py     load + chunk documents, BM25 (+ synonyms, optional embeddings) search
  llm.py           prompt, Ollama streaming, sentence splitting, no-LLM fallback
  speech.py        written text -> spoken form (numbers, emails, lexicon)
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
profiles/
  uow_india/       the UOW India admissions assistant (default)
  acme_fiber/      small made-up example used by the tests
scripts/
  setup.sh         install + download models
  chat.py          text-mode chat against a profile
tests/             pytest suite (no models needed)
```

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest
```

The tests use scripted speech-to-text and synthetic speech from espeak-ng (`apt install espeak-ng`), so they don't need any model downloads. They cover retrieval, sentence streaming, turn detection, barge-in, hang-up, a full browser call over the WebSocket, and a phone call over the AudioSocket protocol.
For UOW India, `tests/test_uow_india.py` checks that everyday caller wording ("is there any hostel facility", "do you have an MBA", "can I pay in instalments") finds the right section. Add a line there whenever you fix a missed question, so it stays fixed.

## What was verified, and what wasn't

- **Verified in development:**
  - The call-pipeline tests pass. The UOW India profile, the speech normaliser and their tests (`test_uow_india.py`, `test_speech.py`) were written while the build environment's shell was unavailable, and have **not been run yet**.
  - A real **Asterisk 20** instance answered a call and bridged it to the agent: greeting, question, answer from the knowledge base, goodbye, and the agent hung up.
  - A **headless Chromium** call through the web page, using a fake microphone, completed three turns and ended with the agent hanging up.
- **Not yet verified:** the development sandbox could not download Whisper, Piper voices or Ollama models. Those three were therefore not run end to end. The integration code follows each library's current API (faster-whisper 1.2, piper-tts 1.3+, Ollama `/api/chat` and `/api/embed`). The first run on your machine is the real check; `python -m scripts.chat` is a quick way to try the LLM on its own.

## Known POC limitations

- English only by default. Many callers in India may prefer Hindi or Gujarati. That needs a multilingual Whisper model (`small`, not `small.en`), a Piper voice for the language if one exists, an LLM that speaks it, and translated or bilingual knowledge files.
- The voice is a US English Piper voice. It works, but local names (Gandhinagar, Pragya) may sound slightly off; fix the worst ones in `lexicon.txt`.
- The knowledge index is rebuilt at every start, which is fine for up to a few thousand pages.
- No authentication on the web page or SIP account beyond the demo password. Don't expose it to the internet as is.
- Each concurrent call shares the same models; a laptop handles one or two calls at a time comfortably.
