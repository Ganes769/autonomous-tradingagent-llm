"""
Build a news JSONL file (date/text) from the synthetic event-extraction dataset.

The repo's evaluation expects a JSONL with lines like:
  {"date": "YYYY-MM-DD", "text": "..."}

But data/events/*.jsonl is stored as chat-style records with:
  {"messages":[{"role":"user","content":"<news text>"}, ...]}

This script extracts the user 'content' as the news text and assigns dates
sequentially across a provided date range (business days).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


def _iter_business_days(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _extract_user_text(obj: dict) -> Optional[str]:
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            txt = m.get("content")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return None


def _read_event_jsonl(paths: Iterable[Path]) -> List[str]:
    texts: List[str] = []
    # Many synthetic datasets store "assistant" JSON unescaped, which breaks json.loads.
    # We therefore extract the user message content via regex first, and only fall back
    # to full JSON parsing when possible.
    user_pat = re.compile(r'"role"\s*:\s*"user"\s*,\s*"content"\s*:\s*"([^"]+)"')
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = user_pat.search(line)
            if m:
                texts.append(m.group(1).strip())
                continue
            # Fallback: try JSON parse for well-formed rows
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = _extract_user_text(obj)
            if txt:
                texts.append(txt)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Create data/news/news.jsonl from data/events/*.jsonl")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["data/events/train.jsonl", "data/events/valid.jsonl", "data/events/test.jsonl"],
        help="Input JSONL files (chat-style).",
    )
    ap.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end-date", default="2024-01-01", help="End date (YYYY-MM-DD)")
    ap.add_argument("--out", default="data/news/news.jsonl", help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of rows (0 = all)")
    args = ap.parse_args()

    in_paths = [Path(x) for x in args.inputs]
    texts = _read_event_jsonl(in_paths)
    if args.limit and args.limit > 0:
        texts = texts[: args.limit]

    y, m, d = (int(x) for x in args.start_date.split("-"))
    start = date(y, m, d)
    y, m, d = (int(x) for x in args.end_date.split("-"))
    end = date(y, m, d)

    days = list(_iter_business_days(start, end))
    if not days:
        raise SystemExit("No business days in the provided date range.")
    if not texts:
        raise SystemExit("No news texts found in the provided input files.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Assign dates in a round-robin manner to spread texts over the range.
    with out_path.open("w", encoding="utf-8") as f:
        for i, txt in enumerate(texts):
            dt = days[i % len(days)].isoformat()
            f.write(json.dumps({"date": dt, "text": txt}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(texts)} rows → {out_path}")


if __name__ == "__main__":
    main()

