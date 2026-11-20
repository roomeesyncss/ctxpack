"""Lexical relevance (BM25) and redundancy (cosine, shingle Jaccard) primitives.

All pure Python, no NumPy. Built for the tens to low thousands of candidate chunks a
RAG retriever returns, not for indexing a corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TERM_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric terms with English stopwords removed."""
    return [t for t in _TERM_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a small in-memory collection.

    >>> bm = BM25([["vector", "database"], ["cooking", "pasta"]])
    >>> s = bm.scores(["vector"])
    >>> s[0] > s[1]
    True
    """

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(d) for d in docs]
        self._lens = [len(d) for d in docs]
        n = len(docs)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        q = set(query)
        out: list[float] = []
        for tf, dl in zip(self._tfs, self._lens, strict=True):
            norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self._idf[term] * (f * (self.k1 + 1)) / (f + norm)
            out.append(s)
        return out


def cosine(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if not dot:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def shingles(terms: Sequence[str], k: int = 3) -> frozenset[tuple[str, ...]]:
    """Contiguous k-term shingles; falls back to unigrams for very short texts."""
    if len(terms) < k:
        return frozenset((t,) for t in terms)
    return frozenset(tuple(terms[i : i + k]) for i in range(len(terms) - k + 1))


def jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
