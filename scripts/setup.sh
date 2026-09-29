#!/usr/bin/env bash
# One-time setup: installs Python deps and downloads the free local models.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "==> Piper voice (natural TTS, ~60 MB)"
mkdir -p models/piper
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium

echo "==> Whisper speech-to-text model (~140 MB, cached by faster-whisper)"
python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("base.en", device="cpu", compute_type="int8")
PY

if command -v ollama >/dev/null 2>&1; then
  echo "==> Local LLM + embedding model via Ollama"
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
else
  echo "!! Ollama not found. Install it from https://ollama.com/download, then run:"
  echo "   ollama pull llama3.2:3b && ollama pull nomic-embed-text"
  echo "   (Without it the agent still works but answers by quoting your documents.)"
fi

[ -f .env ] || cp .env.example .env
echo "Done. Start with:  . .venv/bin/activate && python -m server.app"
