"""Split the 367 format-passing EviGen reports into N batches for parallel
Claude-as-clinician review.

Reads format_check.jsonl (format_pass==True) and the original
evigen_mini_psrm395.jsonl, emits N batch files under
outputs/psrm_typed/claude_judge/batches/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format-check",
                    default="outputs/psrm_typed/format_check.jsonl")
    ap.add_argument("--evigen-jsonl",
                    default="outputs/psrm_typed/evigen_mini_psrm395.jsonl")
    ap.add_argument("--out-dir",
                    default="outputs/psrm_typed/claude_judge/batches")
    ap.add_argument("--n-batches", type=int, default=8)
    args = ap.parse_args()

    # 1. Find format-passing subject_ids.
    pass_sids: set[int] = set()
    for line in Path(args.format_check).open():
        r = json.loads(line)
        if r.get("format_pass"):
            pass_sids.add(int(r["subject_id"]))
    print(f"format_pass sids: {len(pass_sids)}")

    # 2. Pull (subject_id, parsed_probability, parsed_report) for each passer.
    records: list[dict] = []
    for line in Path(args.evigen_jsonl).open():
        r = json.loads(line)
        sid = int(r["subject_id"])
        if sid in pass_sids and r.get("parse_ok"):
            records.append({
                "subject_id": sid,
                "parsed_probability": r.get("parsed_probability"),
                "parsed_report": r["parsed_report"],
            })
    records.sort(key=lambda x: x["subject_id"])
    print(f"records carried into batches: {len(records)}")

    # 3. Split into N roughly equal batches.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n_batches
    chunk = math.ceil(len(records) / n)
    for i in range(n):
        batch = records[i * chunk : (i + 1) * chunk]
        path = out_dir / f"batch_{i}.json"
        path.write_text(json.dumps(batch, indent=2))
        print(f"  batch {i}: {len(batch)} records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
