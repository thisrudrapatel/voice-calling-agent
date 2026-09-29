"""All settings come from environment variables (or a .env file) so the
same code runs on a laptop, a server, or behind Asterisk without edits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Knowledge base
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    top_k: int = int(_env("TOP_K", "4"))

    # Speech-to-text: "whisper" (faster-whisper) or "vosk"
    stt_engine: str = _env("STT_ENGINE", "whisper")
    whisper_model: str = _env("WHISPER_MODEL", "base.en")
    whisper_compute_type: str = _env("WHISPER_COMPUTE_TYPE", "int8")
    vosk_model_path: str = _env("VOSK_MODEL_PATH", str(ROOT / "models" / "vosk"))

    # LLM served locally by Ollama (free, runs offline). If Ollama is not
    # reachable the agent falls back to reading the best matching passage.
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = _env("LLM_MODEL", "llama3.2:3b")
    embed_model: str = _env("EMBED_MODEL", "nomic-embed-text")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.3"))

    # Text-to-speech: "piper" (natural neural voice) or "espeak" (robotic, zero setup)
    tts_engine: str = _env("TTS_ENGINE", "piper")
    piper_voice: str = _env("PIPER_VOICE", str(ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"))
    speech_rate: float = float(_env("SPEECH_RATE", "1.0"))

    # Conversation behaviour
    agent_name: str = _env("AGENT_NAME", "Ava")
    company_name: str = _env("COMPANY_NAME", "Acme Fiber")
    greeting: str = _env(
        "GREETING",
        "Hi, thanks for calling {company}. This is {agent}. How can I help you today?",
    )
    barge_in: bool = _env_bool("BARGE_IN", True)
    # How long the caller must be silent before we treat the turn as finished.
    end_of_turn_ms: int = int(_env("END_OF_TURN_MS", "700"))
    vad_aggressiveness: int = int(_env("VAD_AGGRESSIVENESS", "2"))
    max_history_turns: int = int(_env("MAX_HISTORY_TURNS", "6"))

    # Servers
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))
    audiosocket_port: int = int(_env("AUDIOSOCKET_PORT", "9092"))


settings = Settings()
