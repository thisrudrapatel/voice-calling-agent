"""Answer generation.

Primary: a local model served by Ollama (free, offline, no API key). Replies
are streamed and cut into sentences so the first sentence can be spoken while
the rest is still being generated -- that is what makes the call feel live.

Fallback: if Ollama is not running, answer by reading the most relevant
sentences straight from the knowledge base.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import AsyncIterator

import httpx

from .knowledge import Chunk, tokenize

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are {agent}, a friendly phone agent for {company}. You are on a live voice call. Today's date is {today}.

How to speak:
- Reply in 1 to 3 short sentences, like a person on the phone. No lists, no markdown, no emojis.
- Say numbers and prices the way people say them out loud. Give a phone number, email or website only when it helps, and at most one per reply.
- If the caller's question is vague, ask one short clarifying question.
- If the caller says goodbye or thanks you and is done, say a brief goodbye.

What you know:
- Answer ONLY from the reference information below. If the answer is not there, say you don't have that information and suggest {handoff}. Never invent prices, dates, policies, percentages or phone numbers.
- Dates in the reference information that are before today have already happened; don't describe them as upcoming.
{persona}
Reference information:
{context}"""

NO_CONTEXT = "(nothing relevant found)"


def build_messages(
    question: str,
    context: list[Chunk],
    history: list[dict],
    agent: str,
    company: str,
    persona: str = "",
    handoff: str = "speaking with a member of our team",
    today: str | None = None,
) -> list[dict]:
    ctx = "\n\n".join(f"[{c.source}] {c.text}" for c in context) or NO_CONTEXT
    system = SYSTEM_PROMPT.format(
        agent=agent,
        company=company,
        context=ctx,
        handoff=handoff,
        today=today or date.today().strftime("%d %B %Y"),
        persona=f"\nYour role and extra rules:\n{persona}\n" if persona else "",
    )
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


async def sentences_from_stream(tokens: AsyncIterator[str], min_chars: int = 12) -> AsyncIterator[str]:
    """Re-chunk a token stream into speakable sentences."""
    buf = ""
    async for tok in tokens:
        buf += tok
        parts = _SENTENCE_END.split(buf)
        # Everything but the last part is a finished sentence.
        ready, buf = parts[:-1], parts[-1]
        pending = ""
        for part in ready:
            pending = f"{pending} {part}".strip()
            if len(pending) >= min_chars:
                yield clean_for_speech(pending)
                pending = ""
        if pending:
            buf = f"{pending} {buf}"
    if buf.strip():
        yield clean_for_speech(buf.strip())


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>|]+", "", text)          # markdown symbols
    text = re.sub(r"^\s*[-•]\s*", "", text)          # bullets
    text = re.sub(r"\[[^\]]*\.(md|txt|pdf|csv|json)\]", "", text)  # [source.md] tags
    text = text.replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


class OllamaLLM:
    def __init__(self, url: str, model: str, temperature: float = 0.3):
        self.url, self.model, self.temperature = url, model, temperature
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=3))

    async def available(self) -> bool:
        try:
            resp = await self.client.get(f"{self.url}/api/tags")
            names = {m["name"] for m in resp.json().get("models", [])}
            if self.model not in names and f"{self.model}:latest" not in names:
                log.warning("Ollama is up but model %r is not pulled (have: %s)", self.model, sorted(names))
                return False
            return True
        except Exception:
            return False

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": self.temperature, "num_predict": 160},
        }
        async with self.client.stream("POST", f"{self.url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                tok = data.get("message", {}).get("content", "")
                if tok:
                    yield tok
                if data.get("done"):
                    break


_GREETING = re.compile(r"^\W*(hi|hello|hey|good (morning|afternoon|evening))\b[\w\s,!.']{0,20}$", re.I)
_THANKS = re.compile(r"\b(thanks?|thank you|bye|goodbye)\b", re.I)
_PRICE_Q = re.compile(r"\b(how much|price|cost|fee|charge|pay)\b", re.I)
_TIME_Q = re.compile(r"\b(when|how long|what time|hours|days|timings?|open|opening|duration)\b", re.I)
_TIME_WORDS = re.compile(
    r"\b(\d+\s?(am|pm)|hours?|days?|weeks?|months?|years?|trimesters?|monday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)


class ExtractiveResponder:
    """No-LLM fallback: speak the sentences that best overlap the question."""

    def __init__(self, handoff: str = "speaking with a member of our team", synonyms: dict[str, set[str]] | None = None):
        self.handoff = handoff
        self.synonyms = synonyms or {}

    @staticmethod
    def is_small_talk(question: str) -> bool:
        return bool(_GREETING.match(question) or (_THANKS.search(question) and len(question.split()) <= 6))

    def answer(self, question: str, context: list[Chunk]) -> str:
        if _GREETING.match(question):
            return "Hi there! What can I help you with today?"
        if _THANKS.search(question) and len(question.split()) <= 6:
            return "You're welcome! Thanks for calling, goodbye."
        q = set(tokenize(question))
        if not context or not q:
            return f"Sorry, I don't have information about that. I'd suggest {self.handoff}."
        related = {r for t in q for r in self.synonyms.get(t, ())} - q
        scored = []
        for rank, chunk in enumerate(context[:2]):
            for i, sent in enumerate(split_sentences(chunk.body)):
                words = set(tokenize(sent))
                overlap = len(q & words) + 0.5 * len(related & words)
                if overlap and _PRICE_Q.search(question) and re.search(r"\d|dollar|free", sent, re.I):
                    overlap += 1.5
                if overlap and _TIME_Q.search(question) and _TIME_WORDS.search(sent):
                    overlap += 1
                if overlap:
                    # prefer the best chunk and earlier sentences on ties
                    scored.append((overlap, -rank, -i, sent))
        if not scored:
            return "Sorry, I couldn't find that. Could you rephrase the question?"
        scored.sort(reverse=True)
        best = [scored[0][-1]]
        # Add a second sentence only if it is nearly as relevant.
        if len(scored) > 1 and scored[1][0] >= max(2, 0.75 * scored[0][0]):
            best.append(scored[1][-1])
        return clean_for_speech(" ".join(best))

    async def stream(self, question: str, context: list[Chunk]) -> AsyncIterator[str]:
        for sentence in split_sentences(self.answer(question, context)):
            yield sentence
