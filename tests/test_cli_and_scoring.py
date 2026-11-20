import json
from collections import Counter
from pathlib import Path

from ctxpack import BM25, estimate_tokens, tokenize
from ctxpack.cli import main
from ctxpack.scoring import cosine, jaccard, shingles

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_tokenize_drops_stopwords():
    assert tokenize("The Vector DB is FAST!") == ["vector", "db", "fast"]


def test_bm25_ranks_matching_doc_first():
    docs = [tokenize("cats and dogs"), tokenize("vector search with bm25 ranking"), []]
    s = BM25(docs).scores(tokenize("bm25 ranking"))
    assert s[1] > s[0] == s[2] == 0.0


def test_similarity_helpers():
    a, b = Counter(["a", "b", "c"]), Counter(["a", "b", "d"])
    assert 0 < cosine(a, b) < 1
    assert cosine(a, Counter()) == 0.0
    sa, sb = shingles(list("abcdef")), shingles(list("abcdeg"))
    assert jaccard(sa, sa) == 1.0 and 0 < jaccard(sa, sb) < 1


def test_estimator_is_subadditive():
    parts = ["Hello, world!", "def f(x): return x ** 2", "naïve café ünïcode", " ", ""]
    for a in parts:
        for b in parts:
            assert estimate_tokens(a + b) <= estimate_tokens(a) + estimate_tokens(b)


def test_cli_json(capsys):
    rc = main(["pack", str(EXAMPLES / "knowledge.jsonl"), "--budget", "160", "-f", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tokens_used"] <= 160
    reasons = {d["reason"] for d in out["dropped"]}
    assert "duplicate" in reasons  # mmr-2 is a near copy of mmr-1
    assert out["chunks"][0]["id"] == "mmr-1"


def test_cli_text_from_directory(tmp_path, capsys):
    (tmp_path / "a.md").write_text("Rate limits apply per key.\n\nRetries use backoff.")
    (tmp_path / "b.txt").write_text("Unrelated gardening tips.")
    rc = main(["pack", str(tmp_path), "-q", "rate limits retries", "-b", "200", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1] a.md#0" in out or "[1] a.md#1" in out
