import pytest

from agent.knowledge import Chunk
from agent.llm import ExtractiveResponder, build_messages, clean_for_speech, sentences_from_stream, split_sentences


async def _tokens(text, size=3):
    for i in range(0, len(text), size):
        yield text[i : i + size]


@pytest.mark.asyncio
async def test_stream_is_cut_into_sentences():
    text = "Sure! The Plus plan is 55 dollars a month. It includes a free router."
    out = [s async for s in sentences_from_stream(_tokens(text))]
    # "Sure!" is too short to speak alone, so it is merged with the next sentence.
    assert out == ["Sure! The Plus plan is 55 dollars a month.", "It includes a free router."]


@pytest.mark.asyncio
async def test_prices_with_decimals_are_not_split():
    out = [s async for s in sentences_from_stream(_tokens("It costs $55.99 per month. Anything else?"))]
    assert out[0] == "It costs $55.99 per month."


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Plus** plan: 1 Gbps & free router [company_faq.md]") == "Plus plan: 1 Gbps and free router"


def test_split_sentences():
    assert split_sentences("One. Two? Three!\nFour") == ["One.", "Two?", "Three!", "Four"]


def test_prompt_contains_context_and_history():
    msgs = build_messages(
        "how much?",
        [Chunk("Plus is 55 dollars.", "faq.md")],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "Ava",
        "Acme",
    )
    assert msgs[0]["role"] == "system" and "Plus is 55 dollars." in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]


def test_extractive_picks_the_relevant_sentence(kb):
    q = "when is my payment due"
    answer = ExtractiveResponder().answer(q, kb.search(q))
    assert "due by the 15th" in answer


def test_extractive_small_talk_and_unknown():
    r = ExtractiveResponder()
    assert "help" in r.answer("hello there", [])
    assert "goodbye" in r.answer("thanks, bye", [])
    assert "don't have" in r.answer("what's the meaning of life", [])
