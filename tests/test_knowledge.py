from agent.knowledge import BM25, chunk_text, followup_query, load_documents, tokenize


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("What are the Plans?") == ["plan"]
    assert tokenize("moving") == tokenize("move")


def test_chunks_keep_heading_context():
    text = "# Billing\n\nBills are sent monthly.\n\n# Moving\n\nWe transfer service for free."
    chunks = chunk_text(text, "faq.md")
    assert [c.heading for c in chunks] == ["Billing", "Moving"]
    assert chunks[1].text.startswith("Moving\n")
    assert chunks[1].body == "We transfer service for free."


def test_long_paragraph_is_split():
    text = "Sentence number one is here. " * 60
    chunks = chunk_text(text, "x.txt", max_chars=300)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)


def test_loads_markdown_and_csv(settings):
    chunks = load_documents(settings.data_dir)
    sources = {c.source for c in chunks}
    assert {"company_faq.md", "add_ons.csv"} <= sources
    csv_text = next(c.text for c in chunks if c.source == "add_ons.csv")
    assert "price per month: 8 dollars" in csv_text


def test_retrieval_finds_right_section(kb):
    top = kb.search("what's the late fee if I pay my bill late", k=1)[0]
    assert top.heading == "Billing and payments"
    top = kb.search("my internet is not working", k=1)[0]
    assert top.heading == "Outages and troubleshooting"


def test_retrieval_returns_nothing_for_unrelated(kb):
    assert kb.search("zebra quantum volcano", k=3) == []


def test_bm25_ranks_rarer_terms_higher():
    bm = BM25([["plan", "price"], ["plan", "router"], ["plan"]])
    scores = bm.scores(["router"])
    assert scores.argmax() == 1


def test_followup_query_only_merges_references():
    assert followup_query("and how much does it cost", "can I get a static ip").startswith("can I get")
    assert followup_query("is there a contract", "how do I cancel") == "is there a contract"
