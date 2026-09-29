"""All settings come from environment variables (or a .env file) so the
same code runs on a laptop, a server, or behind Asterisk without edits.

Who the agent is and what it knows comes from a *profile* folder:

    profiles/<name>/
        profile.json   agent name, organisation, greeting, hand-off, STT hint
        persona.md     extra instructions for the LLM (tone, rules)
        lexicon.txt    pronunciation fixes:  GIFT City => Gift City
        synonyms.txt   caller words -> document words, for search
        knowledge/     the documents the agent answers from

Pick one with PROFILE=<name> (default: uow_india). Env vars such as
AGENT_NAME or DATA_DIR still override what the profile says.
"""

from __future__ import annotations

import json
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
    profile: str = _env("PROFILE", "uow_india")
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
    barge_in: bool = _env_bool("BARGE_IN", True)
    # How long the caller must be silent before we treat the turn as finished.
    end_of_turn_ms: int = int(_env("END_OF_TURN_MS", "700"))
    vad_aggressiveness: int = int(_env("VAD_AGGRESSIVENESS", "2"))
    max_history_turns: int = int(_env("MAX_HISTORY_TURNS", "6"))

    # Servers
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))
    audiosocket_port: int = int(_env("AUDIOSOCKET_PORT", "9092"))

    # Filled from the profile in __post_init__ (env vars take precedence).
    data_dir: Path = field(default=None)
    agent_name: str = ""
    company_name: str = ""
    greeting: str = ""
    handoff: str = ""
    stt_hint: str = ""
    persona: str = ""
    lexicon: dict[str, str] = field(default_factory=dict)
    synonyms: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.load_profile(self.profile)

    @property
    def profile_dir(self) -> Path:
        path = Path(self.profile)
        return path if path.is_dir() else ROOT / "profiles" / self.profile

    def load_profile(self, name: str) -> None:
        self.profile = name
        pdir = self.profile_dir
        if not pdir.is_dir():
            raise FileNotFoundError(f"profile {name!r} not found (looked in {pdir})")
        meta_file = pdir / "profile.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        self.agent_name = _env("AGENT_NAME", meta.get("agent_name", "Ava"))
        self.company_name = _env("COMPANY_NAME", meta.get("company_name", "our company"))
        self.greeting = _env(
            "GREETING",
            meta.get("greeting", "Hi, thanks for calling {company}. This is {agent}. How can I help you today?"),
        )
        self.handoff = _env("HANDOFF", meta.get("handoff", "speaking with a member of our team"))
        self.stt_hint = _env("STT_HINT", meta.get("stt_hint", f"{self.company_name}, {self.agent_name}."))
        self.data_dir = Path(_env("DATA_DIR", str(pdir / "knowledge")))
        persona_file = pdir / "persona.md"
        self.persona = persona_file.read_text(encoding="utf-8").strip() if persona_file.exists() else ""
        self.lexicon = dict(_read_rules(pdir / "lexicon.txt", "=>"))
        self.synonyms = [
            [w.strip() for w in line.split(",") if w.strip()] for line in _read_lines(pdir / "synonyms.txt")
        ]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = (l.strip() for l in path.read_text(encoding="utf-8").splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def _read_rules(path: Path, sep: str) -> list[tuple[str, str]]:
    return [tuple(p.strip() for p in l.split(sep, 1)) for l in _read_lines(path) if sep in l]


settings = Settings()
