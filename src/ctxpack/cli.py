"""Command-line interface.

ctxpack pack hits.jsonl --query "how does MMR work?" --budget 1500
cat hits.jsonl | ctxpack pack - --budget 800 --format json
ctxpack pack docs/ --query "rate limits" --budget 2000   # .md/.txt, split on blank lines
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from . import __version__
from .packer import Chunk, pack

_TEXT_SUFFIXES = {".txt", ".md", ".rst"}


def _read_jsonl(lines: Iterator[str], origin: str) -> Iterator[Chunk]:
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{origin}:{lineno}: invalid JSON ({exc.msg})") from exc
        if "text" not in obj:
            raise SystemExit(f"{origin}:{lineno}: object has no 'text' field")
        obj.setdefault("id", f"{origin}:{lineno}")
        yield Chunk.from_dict(obj)


def _read_text_file(path: Path) -> Iterator[Chunk]:
    paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
    for k, para in enumerate(p for p in paragraphs if p):
        yield Chunk(id=f"{path.name}#{k}", text=para, source=f"{path.name}#{k}")


def load_chunks(target: str) -> list[Chunk]:
    if target == "-":
        return list(_read_jsonl(iter(sys.stdin), "<stdin>"))
    path = Path(target)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in _TEXT_SUFFIXES)
        return [c for f in files for c in _read_text_file(f)]
    if not path.exists():
        raise SystemExit(f"ctxpack: {target}: no such file or directory")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as fh:
            return list(_read_jsonl(fh, path.name))
    return list(_read_text_file(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctxpack", description="Pack retrieved chunks into a token budget."
    )
    parser.add_argument("--version", action="version", version=f"ctxpack {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack", help="select, dedupe and order chunks to fit a budget")
    p.add_argument("input", help=".jsonl file, text/markdown file, directory, or '-' for stdin")
    p.add_argument("-b", "--budget", type=int, required=True, help="token budget")
    p.add_argument("-q", "--query", help="query for BM25 scoring when chunks have no score")
    p.add_argument("--diversity", type=float, default=0.3, help="MMR diversity weight [0-1]")
    p.add_argument("--dedup", type=float, default=0.8, help="near-duplicate threshold; 0 = off")
    p.add_argument("--min-relevance", type=float, default=0.0)
    p.add_argument("--order", choices=["edges", "relevance", "original"], default="edges")
    p.add_argument("-f", "--format", choices=["text", "json"], default="text")
    p.add_argument("--quiet", action="store_true", help="do not print the summary to stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = load_chunks(args.input)
    result = pack(
        chunks,
        args.budget,
        query=args.query,
        diversity=args.diversity,
        dedup_threshold=args.dedup or None,
        min_relevance=args.min_relevance,
        order=args.order,
    )
    if args.format == "json":
        payload = {
            "context": result.render(),
            "tokens_used": result.tokens_used,
            "budget": result.budget,
            "chunks": [
                {"n": n, "id": c.id, "source": c.source, "relevance": result.relevance[c.id]}
                for n, c in result.citations().items()
            ],
            "dropped": [
                {"id": d.chunk.id, "reason": d.reason, "detail": d.detail} for d in result.dropped
            ],
        }
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result.render() + "\n")
    if not args.quiet:
        print(f"ctxpack: {result.summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
