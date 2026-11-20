import random
import string

import pytest

from ctxpack import Chunk, estimate_tokens, pack


def _chunk(i: int, text: str, score: float | None = None) -> Chunk:
    return Chunk(id=f"c{i}", text=text, score=score, source=f"doc{i}")


def test_never_exceeds_budget_randomised():
    rng = random.Random(7)
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 9))) for _ in range(300)]
    for trial in range(200):
        n = rng.randint(1, 40)
        chunks = [
            _chunk(i, " ".join(rng.choices(words, k=rng.randint(5, 120))), rng.random())
            for i in range(n)
        ]
        budget = rng.randint(20, 1500)
        result = pack(chunks, budget, diversity=rng.random())
        assert result.tokens_used <= budget, trial
        assert estimate_tokens(result.render()) <= budget, trial
        assert len(result.chunks) + len(result.dropped) == n


def test_prefers_relevant_chunks_under_tight_budget():
    chunks = [
        _chunk(0, "alpha beta gamma " * 10, score=0.1),
        _chunk(1, "delta epsilon zeta " * 10, score=0.9),
        _chunk(2, "eta theta iota " * 10, score=0.5),
    ]
    one_chunk = estimate_tokens("[3] doc1\n" + chunks[1].text)
    result = pack(chunks, one_chunk + 2)
    assert [c.id for c in result.chunks] == ["c1"]
    assert {d.reason for d in result.dropped} == {"over_budget"}


def test_near_duplicates_are_dropped_keeping_best():
    base = "the quick brown fox jumps over the lazy dog near the riverbank at dawn"
    chunks = [
        _chunk(0, base, score=0.7),
        _chunk(1, base + " today", score=0.9),
        _chunk(2, "completely different content about vector databases and indexes", score=0.5),
    ]
    result = pack(chunks, 10_000)
    ids = {c.id for c in result.chunks}
    assert ids == {"c1", "c2"}
    (dup,) = result.dropped
    assert dup.chunk.id == "c0" and dup.reason == "duplicate" and "c1" in dup.detail


def test_dedup_can_be_disabled():
    chunks = [_chunk(0, "same text here", 0.5), _chunk(1, "same text here", 0.4)]
    assert len(pack(chunks, 1000, dedup_threshold=None).chunks) == 2


def test_diversity_promotes_novel_chunk():
    chunks = [
        _chunk(0, "python asyncio event loop tasks coroutines await", 1.0),
        _chunk(1, "python asyncio event loop coroutines scheduling await futures", 0.95),
        _chunk(2, "postgres connection pooling with pgbouncer transaction mode", 0.6),
    ]
    budget = max(estimate_tokens(f"[3] doc{i}\n{c.text}") for i, c in enumerate(chunks)) * 2 + 2
    relevance_only = pack(chunks, budget, diversity=0.0, dedup_threshold=None)
    diverse = pack(chunks, budget, diversity=0.7, dedup_threshold=None)
    assert {c.id for c in relevance_only.chunks} == {"c0", "c1"}
    assert {c.id for c in diverse.chunks} == {"c0", "c2"}


def test_bm25_used_when_scores_missing():
    chunks = [
        Chunk(id="a", text="recipes for sourdough bread and starter care"),
        Chunk(id="b", text="token budgets for retrieval augmented generation context windows"),
    ]
    result = pack(chunks, 10_000, query="context window token budget", order="relevance")
    assert result.chunks[0].id == "b"
    assert result.relevance["b"] == 1.0


def test_edges_order_puts_top_two_at_the_ends():
    chunks = [
        _chunk(i, f"unique topic number {i} " + "x" * i, score=s)
        for i, s in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])
    ]
    result = pack(chunks, 10_000, order="edges", dedup_threshold=None)
    ids = [c.id for c in result.chunks]
    assert ids[0] == "c1"  # most relevant first
    assert ids[-1] == "c3"  # second most relevant last
    assert ids[len(ids) // 2] == "c0"  # weakest in the middle


def test_original_order_is_preserved():
    chunks = [_chunk(i, f"distinct body {i} " * 3, score=1 - i / 10) for i in range(4)]
    result = pack(chunks, 10_000, order="original")
    assert [c.id for c in result.chunks] == ["c0", "c1", "c2", "c3"]


def test_min_relevance_filters():
    chunks = [_chunk(0, "keep me please", 1.0), _chunk(1, "drop me now", 0.0)]
    result = pack(chunks, 1000, min_relevance=0.5)
    assert [c.id for c in result.chunks] == ["c0"]
    assert result.dropped[0].reason == "low_relevance"


def test_accepts_dicts_and_renders_citations():
    result = pack([{"id": "x", "text": "hello world", "source": "a.md", "lang": "en"}], 100)
    assert result.render() == "[1] a.md\nhello world"
    assert result.citations()[1].metadata == {"lang": "en"}
    assert "packed 1/1 chunks" in result.summary()


def test_custom_counter_is_respected():
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    result = pack([_chunk(0, "one two three")], 5, counter=counter)
    assert calls and result.tokens_used == 5  # "[1] doc0" + 3 words


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"budget": 0}, "budget"),
        ({"budget": 10, "diversity": 2}, "diversity"),
        ({"budget": 10, "order": "random"}, "order"),
    ],
)
def test_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        pack([_chunk(0, "text")], **kwargs)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        pack([Chunk("a", "x"), Chunk("a", "y")], 100)


def test_empty_input():
    result = pack([], 100)
    assert result.chunks == [] and result.tokens_used == 0
