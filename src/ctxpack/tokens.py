"""Token counting.

ctxpack never imports a tokenizer. The default :func:`estimate_tokens` is a fast,
dependency-free estimator that deliberately errs on the high side so packed context
does not overflow. For exact numbers, pass any ``Callable[[str], int]`` as ``counter``
(for example a wrapper around your provider's token-counting endpoint).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_PIECE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Estimate the BPE token count of ``text``.

    Takes the max of two estimates:

    * ~4 characters per token, which holds for English prose, and
    * ~1.1 tokens per word or punctuation mark, which holds for code and
      symbol-heavy text.

    The estimate is sub-additive: ``estimate(a + b) <= estimate(a) + estimate(b)``.
    Because of that, packing by summing per-chunk estimates never exceeds the budget
    when the joined context is measured.
    """
    if not text:
        return 0
    by_chars = len(text) / 4.0
    by_pieces = len(_PIECE_RE.findall(text)) * 1.1
    return max(1, math.ceil(max(by_chars, by_pieces)))
