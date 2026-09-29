"""Knowledge base: load your documents, chunk them, and retrieve the passages
most relevant to what the caller asked.

Retrieval is keyword based (BM25, pure Python, zero downloads). If Ollama is
running with an embedding model, semantic vectors are blended in so that
paraphrased questions ("how much is it" vs "pricing") still match.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

log = logging.getLogger(__name__)

STOPWORDS = set(
    """a an and are as at be but by can do does for from have how i if in is it
    its me my of on or our so that the their them there this to us was we what
    when where which who why will with you your yours please tell know about
    could would should hi hello hey okay ok um uh like just""".split()
)


@dataclass
class Chunk:
    text: str  # includes the section heading, which helps retrieval
    source: str
    heading: str = ""

    @property
    def body(self) -> str:
        if self.heading and self.text.startswith(self.heading):
            return self.text[len(self.heading) :].strip()
        return self.text


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def _stem(word: str) -> str:
    # Tiny suffix stripper so "plans"/"plan", "charged"/"charge" match.
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


# ---------------------------------------------------------------- loading

def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed, skipping %s", path.name)
        return ""
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _read_csv(path: Path) -> str:
    # Each row becomes a sentence-like paragraph: "name: Basic, price: $30, ..."
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return "\n\n".join(
        ", ".join(f"{_humanize(k)}: {v}" for k, v in row.items() if k and v) for row in rows
    )


def _humanize(key: str) -> str:
    return re.sub(r"[_\-]+", " ", key).strip()


def _read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    parts = []
    for item in items:
        if isinstance(item, dict):
            parts.append(", ".join(f"{_humanize(str(k))}: {v}" for k, v in item.items()))
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


READERS = {
    ".txt": lambda p: p.read_text(encoding="utf-8"),
    ".md": lambda p: p.read_text(encoding="utf-8"),
    ".pdf": _read_pdf,
    ".csv": _read_csv,
    ".json": _read_json,
}


def chunk_text(text: str, source: str, max_chars: int = 600) -> list[Chunk]:
    """Split on blank lines / headings, then pack paragraphs up to max_chars.
    Markdown headings are carried into each chunk so it has context."""
    chunks: list[Chunk] = []
    heading = ""
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            body = buf.strip()
            chunks.append(Chunk(f"{heading}\n{body}" if heading else body, source, heading))
        buf = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            flush()
            lines = para.splitlines()
            heading = lines[0].lstrip("#").strip()
            para = "\n".join(lines[1:]).strip()
            if not para:
                continue
        if len(buf) + len(para) > max_chars:
            flush()
        while len(para) > max_chars:  # very long paragraph: hard split on sentences
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            buf = para[:cut]
            flush()
            para = para[cut:].strip()
        buf = f"{buf}\n{para}" if buf else para
    flush()
    return chunks


def load_documents(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*")):
        reader = READERS.get(path.suffix.lower())
        if not reader or path.name.startswith("."):
            continue
        try:
            text = reader(path)
        except Exception as exc:  # a bad file should not take the agent down
            log.warning("could not read %s: %s", path, exc)
            continue
        chunks.extend(chunk_text(text, path.name))
    log.info("loaded %d chunks from %s", len(chunks), data_dir)
    return chunks


# --------------------------------------------------------------- retrieval

_REFERS_BACK = re.compile(r"\b(it|its|that|this|those|these|they|them|one|same|also|too)\b", re.I)


def followup_query(question: str, previous_question: str) -> str:
    """"And how much does it cost?" means nothing alone; glue it to the
    previous question so retrieval finds the right passage."""
    if previous_question and _REFERS_BACK.search(question):
        return f"{previous_question} {question}"
    return question


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
            for t in query:
                f = tf.get(t)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return out


def build_synonym_map(groups: list[list[str]]) -> dict[str, set[str]]:
    """["hostel", "accommodation"] -> {"hostel": {"accommodation"}, ...} on stemmed tokens."""
    out: dict[str, set[str]] = {}
    for group in groups:
        tokens = {t for word in group for t in tokenize(word)}
        for t in tokens:
            out.setdefault(t, set()).update(tokens - {t})
    return out


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        ollama_url: str | None = None,
        embed_model: str | None = None,
        synonyms: list[list[str]] | None = None,
    ):
        self.chunks = load_documents(data_dir)
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.synonyms = build_synonym_map(synonyms or [])
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.vectors: np.ndarray | None = None
        if ollama_url and embed_model and self.chunks:
            self.vectors = self._embed([c.text for c in self.chunks])
            if self.vectors is not None:
                log.info("semantic search enabled (%s)", embed_model)

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        try:
            resp = httpx.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embed_model, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
            return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        except Exception as exc:
            log.info("embeddings unavailable (%s); using keyword search only", exc)
            self.ollama_url = None
            return None

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        spoken = tokenize(query)
        kw = self.bm25.scores(spoken)
        # Callers say "hostel", documents say "accommodation": search both,
        # but trust the words actually spoken more.
        related = {r for t in spoken for r in self.synonyms.get(t, ())} - set(spoken)
        if related:
            kw = kw + 0.5 * self.bm25.scores(sorted(related))
        score = kw / kw.max() if kw.max() > 0 else kw
        if self.vectors is not None:
            q = self._embed([query])
            if q is not None:
                sem = self.vectors @ q[0]
                score = 0.35 * score + 0.65 * np.clip(sem, 0, None)
        if score.max() <= 0:
            return []
        order = np.argsort(-score)[:k]
        return [self.chunks[i] for i in order if score[i] > 0]
