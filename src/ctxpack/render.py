"""Rendering packed chunks into a prompt-ready, citeable context block."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .packer import Chunk

DEFAULT_TEMPLATE = "[{n}] {source}\n{text}"
DEFAULT_SEPARATOR = "\n\n"


def render_chunks(
    chunks: Sequence[Chunk],
    *,
    template: str = DEFAULT_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Join chunks with numbered headers so the model can cite them as ``[n]``."""
    return separator.join(
        template.format(n=n, id=c.id, source=c.source or c.id, text=c.text)
        for n, c in enumerate(chunks, start=1)
    )
