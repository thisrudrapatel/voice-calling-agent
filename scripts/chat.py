"""Type questions instead of speaking them: fastest way to check that the
agent answers well from your data before trying it over a call.

    python -m scripts.chat
"""

from __future__ import annotations

import asyncio
import logging

from agent.config import settings
from agent.knowledge import KnowledgeBase, followup_query
from agent.llm import ExtractiveResponder, OllamaLLM, build_messages, sentences_from_stream


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    kb = KnowledgeBase(settings.data_dir, settings.ollama_url, settings.embed_model)
    llm = OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_temperature)
    use_llm = await llm.available()
    print(f"{len(kb.chunks)} chunks loaded; LLM: {settings.llm_model if use_llm else 'offline (extractive answers)'}")
    history: list[dict] = []
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        query = followup_query(q, last_user)
        ctx = kb.search(query, settings.top_k)
        if use_llm:
            msgs = build_messages(q, ctx, history, settings.agent_name, settings.company_name)
            parts = []
            print("agent> ", end="", flush=True)
            async for s in sentences_from_stream(llm.stream(msgs)):
                print(s, end=" ", flush=True)
                parts.append(s)
            print()
            reply = " ".join(parts)
        else:
            fb = ExtractiveResponder()
            reply = fb.answer(q if fb.is_small_talk(q) else query, ctx)
            print(f"agent> {reply}")
        print(f"       (sources: {', '.join(sorted({c.source for c in ctx})) or 'none'})")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
        history = history[-12:]


if __name__ == "__main__":
    asyncio.run(main())
