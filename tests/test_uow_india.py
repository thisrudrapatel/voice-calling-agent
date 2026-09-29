"""The UOW India profile: callers' everyday wording must reach the right facts."""

import pytest

from agent.knowledge import followup_query
from agent.llm import ExtractiveResponder, build_messages


def top_heading(kb, question):
    hits = kb.search(question, k=1)
    return hits[0].heading if hits else None


def test_profile_loads(uow_settings):
    s = uow_settings
    assert s.agent_name == "Maya"
    assert "Wollongong" in s.company_name
    assert s.data_dir.name == "knowledge" and s.data_dir.parent.name == "uow_india"
    assert "virtual assistant" in s.greeting
    assert "WhatsApp" in s.handoff
    assert "GIFT City" in s.stt_hint
    assert s.lexicon["IELTS"] == "eye elts"
    assert any("hostel" in g for g in s.synonyms)


@pytest.mark.parametrize(
    "question, heading",
    [
        ("is there any hostel facility", "Accommodation and hostels"),
        ("where will I stay, do you provide rooms", "Accommodation and hostels"),
        ("what is the address of the campus", "Campus address and opening hours"),
        ("what courses do you offer", "Courses offered at UOW India"),
        ("do you have an MBA program", "Courses offered at UOW India"),
        ("how do I apply", "How to apply"),
        ("what IELTS score do I need", "English language requirements"),
        ("tell me about the high achiever scholarship for masters", "High Achiever Scholarship for postgraduates"),
        ("what is the early bird discount", "Early Bird Scholarship"),
        ("can I pay the fees in instalments", "Paying your fees"),
        ("what about placements and jobs", "Careers and placements"),
        ("do I need a visa", "Do I need a visa to study at UOW India?"),
        ("can I move to the Australia campus later", "Can I transfer to UOW in Australia?"),
        ("is the degree the same as in Australia", "Is the UOW India degree the same as the Australian degree?"),
        ("what is the phone number for admissions", "Contact the admissions team"),
    ],
)
def test_caller_wording_finds_the_right_section(uow_kb, question, heading):
    assert top_heading(uow_kb, question) == heading


def test_fees_question_retrieves_fee_policy_not_invented_numbers(uow_kb):
    hits = uow_kb.search("how much is the fee for the master of fintech", k=4)
    assert "Tuition fees" in [c.heading for c in hits]
    text = " ".join(c.text for c in hits)
    assert "Australian dollars" in text


def test_followup_about_same_course(uow_kb):
    q = followup_query("and how long does it take", "tell me about the graduate certificate in computing")
    assert top_heading(uow_kb, q) == "Graduate Certificate in Computing"


def test_fallback_answers_from_uow_facts(uow_kb, uow_settings):
    fb = ExtractiveResponder(uow_settings.handoff, uow_kb.synonyms)

    def ask(q):
        return fb.answer(q, uow_kb.search(q))

    assert "does not run its own student housing" in ask("is there a hostel")
    assert "Monday to Saturday" in ask("what are the campus timings")
    assert "WhatsApp" in ask("what is the capital of france")


def test_prompt_carries_persona_handoff_and_date(uow_kb, uow_settings):
    s = uow_settings
    msgs = build_messages("hi", [], [], s.agent_name, s.company_name, s.persona, s.handoff, today="29 September 2026")
    system = msgs[0]["content"]
    assert "29 September 2026" in system
    assert "virtual admissions assistant" in system
    assert "never state a fee amount" in system
    assert "+91 97734 43748" in system
