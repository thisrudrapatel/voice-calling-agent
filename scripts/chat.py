"""Type questions instead of speaking them: fastest way to check that the
agent answers well from your data before trying it over a call.

    python -m scripts.chat                      # default profile (uow_india)
    python -m scripts.chat --profile acme_fiber
    python -m scripts.chat --spoken             # also show how the voice will read it
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from agent.config import settings
from agent.knowledge import KnowledgeBase, followup_query
from agent.llm import ExtractiveResponder, OllamaLLM, build_messages, sentences_from_stream
from agent.speech import speakable


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="profile folder name under profiles/")
    parser.add_argument("--spoken", action="store_true", help="print the text as the voice will say it")
    args = parser.parse_args()
    if args.profile:
        settings.load_profile(args.profile)

    logging.basicConfig(level=logging.WARNING)
    kb = KnowledgeBase(settings.data_dir, settings.ollama_url, settings.embed_model, settings.synonyms)
    llm = OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_temperature)
    use_llm = await llm.available()
    fallback = ExtractiveResponder(settings.handoff, kb.synonyms)
    print(f"profile: {settings.profile} | {len(kb.chunks)} chunks | "
          f"LLM: {settings.llm_model if use_llm else 'offline (extractive answers)'}")
    print(f"agent> {settings.greeting.format(company=settings.company_name, agent=settings.agent_name)}")
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
            msgs = build_messages(
                q, ctx, history, settings.agent_name, settings.company_name, settings.persona, settings.handoff
            )
            parts = []
            print("agent> ", end="", flush=True)
            async for s in sentences_from_stream(llm.stream(msgs)):
                print(s, end=" ", flush=True)
                parts.append(s)
            print()
            reply = " ".join(parts)
        else:
            reply = fallback.answer(q if fallback.is_small_talk(q) else query, ctx)
            print(f"agent> {reply}")
        if args.spoken:
            print(f"voice> {speakable(reply, settings.lexicon)}")
        print(f"       (sources: {', '.join(sorted({c.source for c in ctx})) or 'none'})")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
        history = history[-12:]


if __name__ == "__main__":
    asyncio.run(main())
