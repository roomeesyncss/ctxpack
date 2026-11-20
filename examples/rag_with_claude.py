"""Answer a question with Claude over a budgeted, citeable context built by ctxpack.

pip install "ctxpack[claude] @ git+https://github.com/roomeesyncss/ctxpack"
export ANTHROPIC_API_KEY=...
python examples/rag_with_claude.py "Why put the best chunks at the edges of the prompt?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

from ctxpack import Chunk, pack

MODEL = "claude-opus-5-5"
CONTEXT_BUDGET = 400  # tokens reserved for retrieved context

SYSTEM = (
    "Answer using only the numbered sources inside <context>. Cite each claim with its "
    "source number in square brackets, for example [2]. If the sources do not contain "
    "the answer, say that plainly instead of guessing."
)


def load_hits() -> list[Chunk]:
    path = Path(__file__).with_name("knowledge.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.from_dict(json.loads(line)) for line in lines if line.strip()]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I keep RAG context inside a token budget?"
    packed = pack(load_hits(), CONTEXT_BUDGET, query=question)
    print(f"ctxpack: {packed.summary()}\n", file=sys.stderr)

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        # If a safety classifier declines, the API retries on a fallback model in-call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<context>\n{packed.render()}\n</context>\n\nQuestion: {question}",
            }
        ],
    )
    if response.stop_reason == "refusal":
        raise SystemExit("The model declined this request.")

    print("".join(block.text for block in response.content if block.type == "text"))
    print("\nSources:")
    for n, chunk in packed.citations().items():
        print(f"  [{n}] {chunk.source}  (id={chunk.id})")


if __name__ == "__main__":
    main()
