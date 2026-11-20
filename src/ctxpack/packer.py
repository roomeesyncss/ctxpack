"""Budget-constrained context selection.

Pipeline:

1. **Relevance**: use the retriever's scores, or BM25 against ``query`` when scores
   are missing. Normalised to [0, 1].
2. **Dedup**: drop near-duplicates (shingle Jaccard >= ``dedup_threshold``), keeping
   the more relevant copy.
3. **Select**: greedy Maximal Marginal Relevance under a hard token budget. Each step
   picks the chunk that best trades relevance against similarity to what is already
   selected, among the chunks that still fit.
4. **Order**: arrange the selection for the model. ``"edges"`` puts the strongest
   evidence at the start and end of the context to counter the "lost in the middle"
   effect.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .render import DEFAULT_SEPARATOR, DEFAULT_TEMPLATE, render_chunks
from .scoring import BM25, cosine, jaccard, shingles, tokenize
from .tokens import TokenCounter, estimate_tokens

Order = Literal["edges", "relevance", "original"]
DropReason = Literal["duplicate", "over_budget", "low_relevance"]


@dataclass(frozen=True)
class Chunk:
    """A candidate piece of context, typically one retriever hit."""

    id: str
    text: str
    score: float | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, index: int = 0) -> Chunk:
        known = {"id", "text", "score", "source", "metadata"}
        extra = {k: v for k, v in d.items() if k not in known}
        meta = {**dict(d.get("metadata") or {}), **extra}
        score = d.get("score")
        return cls(
            id=str(d.get("id", index)),
            text=str(d["text"]),
            score=None if score is None else float(score),
            source=d.get("source"),
            metadata=meta,
        )


@dataclass(frozen=True)
class Dropped:
    chunk: Chunk
    reason: DropReason
    detail: str = ""


@dataclass(frozen=True)
class PackResult:
    """The packed context plus a full account of what was left out and why."""

    chunks: list[Chunk]
    tokens_used: int
    budget: int
    dropped: list[Dropped]
    relevance: dict[str, float]
    template: str = DEFAULT_TEMPLATE
    separator: str = DEFAULT_SEPARATOR

    @property
    def utilization(self) -> float:
        return self.tokens_used / self.budget if self.budget else 0.0

    def render(self) -> str:
        """Render the packed chunks with numbered citation headers."""
        return render_chunks(self.chunks, template=self.template, separator=self.separator)

    def citations(self) -> dict[int, Chunk]:
        """Map citation number (as rendered) to chunk."""
        return {i: c for i, c in enumerate(self.chunks, start=1)}

    def summary(self) -> str:
        reasons = Counter(d.reason for d in self.dropped)
        dropped = ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())) or "none"
        total = len(self.chunks) + len(self.dropped)
        return (
            f"packed {len(self.chunks)}/{total} chunks · "
            f"{self.tokens_used}/{self.budget} tokens ({self.utilization:.1%}) · "
            f"dropped: {dropped}"
        )


def _normalise(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _relevance(chunks: Sequence[Chunk], query: str | None) -> list[float]:
    if all(c.score is not None for c in chunks):
        return _normalise([float(c.score) for c in chunks])  # type: ignore[arg-type]
    if query:
        bm = BM25([tokenize(c.text) for c in chunks])
        raw = bm.scores(tokenize(query))
        top = max(raw, default=0.0)
        return [s / top if top > 0 else 0.0 for s in raw]
    # No scores, no query: trust the retriever's order with a gentle positional prior.
    n = len(chunks)
    return [1.0 - (i / (2 * n)) for i in range(n)]


def _arrange(selected: list[int], rel: Sequence[float], order: Order) -> list[int]:
    if order == "original":
        return sorted(selected)
    by_rel = sorted(selected, key=lambda i: (-rel[i], i))
    if order == "relevance":
        return by_rel
    # "edges": ranks 0, 2, 4... fill from the front; ranks 1, 3, 5... from the back.
    front = by_rel[0::2]
    back = by_rel[1::2]
    return front + back[::-1]


def pack(
    chunks: Iterable[Chunk | Mapping[str, Any]],
    budget: int,
    *,
    query: str | None = None,
    diversity: float = 0.3,
    dedup_threshold: float | None = 0.8,
    min_relevance: float = 0.0,
    order: Order = "edges",
    counter: TokenCounter = estimate_tokens,
    template: str = DEFAULT_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> PackResult:
    """Select and order chunks so the rendered context fits in ``budget`` tokens.

    Args:
        chunks: Candidates as :class:`Chunk` or dicts with ``text`` (and optional
            ``id``, ``score``, ``source``, ``metadata``).
        budget: Maximum tokens for the rendered context (headers and separators included).
        query: Used for BM25 relevance when any chunk lacks a ``score``.
        diversity: MMR trade-off in [0, 1]. 0 means relevance only, 1 means
            redundancy avoidance only. 0.2 to 0.4 works well for QA.
        dedup_threshold: Jaccard similarity of 3-term shingles at or above which a chunk
            is treated as a near-duplicate. ``None`` disables dedup.
        min_relevance: Drop chunks whose normalised relevance is below this.
        order: ``"edges"`` (default), ``"relevance"`` or ``"original"``.
        counter: Token counter. Defaults to a conservative estimator.
        template: Per-chunk format string with ``{n}``, ``{source}``, ``{id}`` and ``{text}``.
        separator: String placed between rendered chunks.
    """
    if budget <= 0:
        raise ValueError("budget must be a positive number of tokens")
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be within [0, 1]")
    if order not in ("edges", "relevance", "original"):
        raise ValueError(f"unknown order: {order!r}")

    items = [
        c if isinstance(c, Chunk) else Chunk.from_dict(c, index=i) for i, c in enumerate(chunks)
    ]
    if not items:
        return PackResult([], 0, budget, [], {}, template, separator)
    ids = [c.id for c in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"chunk ids must be unique; repeated: {dupes[:5]}")

    rel = _relevance(items, query)
    terms = [tokenize(c.text) for c in items]
    tfs = [Counter(t) for t in terms]

    # Cost of each chunk as rendered. The citation number is sized for the largest
    # possible index so the real render can only be equal or smaller.
    widest_n = len(items)
    sep_cost = counter(separator)
    costs = [
        counter(template.format(n=widest_n, id=c.id, source=c.source or c.id, text=c.text))
        for c in items
    ]

    dropped: list[Dropped] = []
    candidates: list[int] = []
    kept_shingles: list[tuple[int, frozenset[tuple[str, ...]]]] = []

    for i in sorted(range(len(items)), key=lambda i: (-rel[i], i)):
        if rel[i] < min_relevance:
            dropped.append(Dropped(items[i], "low_relevance", f"relevance={rel[i]:.3f}"))
            continue
        if dedup_threshold is not None:
            sh = shingles(terms[i])
            dup_of = next((j for j, s in kept_shingles if jaccard(sh, s) >= dedup_threshold), None)
            if dup_of is not None:
                dropped.append(
                    Dropped(items[i], "duplicate", f"near-duplicate of {items[dup_of].id}")
                )
                continue
            kept_shingles.append((i, sh))
        candidates.append(i)

    # Greedy MMR under a knapsack constraint. max_sim[i] tracks the highest similarity
    # between candidate i and anything already selected, updated incrementally.
    selected: list[int] = []
    max_sim = dict.fromkeys(candidates, 0.0)
    remaining = budget
    pool = set(candidates)
    while pool:
        overhead = sep_cost if selected else 0
        best, best_score = None, float("-inf")
        for i in pool:
            if costs[i] + overhead > remaining:
                continue
            score = (1 - diversity) * rel[i] - diversity * max_sim[i]
            if score > best_score or (score == best_score and best is not None and i < best):
                best, best_score = i, score
        if best is None:
            break
        selected.append(best)
        pool.discard(best)
        remaining -= costs[best] + overhead
        for i in pool:
            s = cosine(tfs[i], tfs[best])
            if s > max_sim[i]:
                max_sim[i] = s

    for i in sorted(pool, key=lambda i: (-rel[i], i)):
        dropped.append(Dropped(items[i], "over_budget", f"needs {costs[i]} tokens"))

    arranged = _arrange(selected, rel, order)
    return PackResult(
        chunks=[items[i] for i in arranged],
        tokens_used=budget - remaining,
        budget=budget,
        dropped=dropped,
        relevance={items[i].id: round(rel[i], 4) for i in range(len(items))},
        template=template,
        separator=separator,
    )
