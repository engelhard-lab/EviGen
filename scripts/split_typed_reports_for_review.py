"""Build per-batch JSON files for the v2 Claude clinician review.

Stage A passes typed_format_check.jsonl (format_pass==True) → pull each
patient's full annotated typed-report text from typed_reports_per_patient.json
→ split into N batches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format-check",
                    default="outputs/psrm_typed/typed_format_check.jsonl")
    ap.add_argument("--per-patient",
                    default="outputs/psrm_typed/typed_reports_per_patient.json")
    ap.add_argument("--evigen-jsonl",
                    default="outputs/psrm_typed/evigen_mini_psrm395.jsonl")
    ap.add_argument("--out-dir",
                    default="outputs/psrm_typed/claude_judge/batches_v2")
    ap.add_argument("--n-batches", type=int, default=8)
    args = ap.parse_args()

    pass_sids = []
    for line in Path(args.format_check).open():
        r = json.loads(line)
        if r.get("format_pass"):
            pass_sids.append(int(r.get("sid", r.get("subject_id"))))
    pass_sids.sort()
    print(f"typed_format_pass sids: {len(pass_sids)}")

    per_pat = json.loads(Path(args.per_patient).read_text())
    prob_by_sid = {}
    for line in Path(args.evigen_jsonl).open():
        r = json.loads(line)
        prob_by_sid[int(r["subject_id"])] = r.get("parsed_probability")

    records = [
        {
            "subject_id": sid,
            "parsed_probability": prob_by_sid.get(sid),
            "typed_report": per_pat[str(sid)],
        }
        for sid in pass_sids
    ]
    print(f"records carried into batches: {len(records)}")

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
