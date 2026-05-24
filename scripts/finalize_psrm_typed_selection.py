"""Stage C: combine Stage A format check + Stage B clinical judge verdicts
into the final selected subject_id list and a markdown summary.

Final selection criterion:
    format_pass==True AND n_flagged<=3 AND overall_trust=="trust"

Writes:
  outputs/psrm_typed/selected_subject_ids.json   — final sid list + funnel
  outputs/psrm_typed/selection_summary.md        — markdown summary
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path("/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery")
DEFAULT_FORMAT = ROOT / "outputs/psrm_typed/format_check.jsonl"
DEFAULT_JUDGE = ROOT / "outputs/psrm_typed/clinical_judge.jsonl"
DEFAULT_OUT_JSON = ROOT / "outputs/psrm_typed/selected_subject_ids.json"
DEFAULT_OUT_MD = ROOT / "outputs/psrm_typed/selection_summary.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format-check", type=Path, default=DEFAULT_FORMAT)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--max-flagged", type=int, default=3)
    args = ap.parse_args()

    fmt: dict[int, dict] = {}
    with args.format_check.open() as f:
        for line in f:
            r = json.loads(line)
            fmt[int(r["subject_id"])] = r

    judge: dict[int, dict] = {}
    if args.judge.exists():
        with args.judge.open() as f:
            for line in f:
                r = json.loads(line)
                # last-write-wins is fine (cache deduplicates).
                judge[int(r["subject_id"])] = r

    total = len(fmt)
    format_pass = sum(1 for r in fmt.values() if r.get("format_pass"))
    psrm_pass = sum(
        1 for r in fmt.values()
        if r.get("format_pass")
        and r.get("n_flagged") is not None
        and r["n_flagged"] <= args.max_flagged
    )

    eligible_for_judge = {
        sid for sid, r in fmt.items()
        if r.get("format_pass")
        and r.get("n_flagged") is not None
        and r["n_flagged"] <= args.max_flagged
    }
    judged_sids = eligible_for_judge & set(judge.keys())

    trust_counts: Counter[str] = Counter()
    boolean_counts = {
        "factors_clinically_plausible": 0,
        "evidence_supports_factor": 0,
        "language_appropriately_hedged": 0,
        "no_obvious_hallucination": 0,
    }
    for sid in judged_sids:
        v = judge[sid].get("verdict") or {}
        trust = (v.get("overall_trust") or "").strip().lower()
        trust_counts[trust] += 1
        for k in boolean_counts:
            if v.get(k) is True:
                boolean_counts[k] += 1

    selected = sorted(
        sid for sid in judged_sids
        if (judge[sid].get("verdict") or {}).get("overall_trust", "").lower()
        == "trust"
    )
    rejected = sorted(
        sid for sid in judged_sids
        if (judge[sid].get("verdict") or {}).get("overall_trust", "").lower()
        == "reject"
    )

    # Sample examples for the markdown.
    rng = random.Random(0)
    trust_examples = rng.sample(selected, min(5, len(selected)))
    reject_examples = rng.sample(rejected, min(5, len(rejected)))

    funnel = {
        "total_input": total,
        "format_pass": format_pass,
        "format_pass_AND_psrm_le_3": psrm_pass,
        "judged_by_gpt4o": len(judged_sids),
        "trust": trust_counts.get("trust", 0),
        "partial": trust_counts.get("partial", 0),
        "reject": trust_counts.get("reject", 0),
        "other_trust_label": (
            len(judged_sids)
            - sum(trust_counts[k] for k in ("trust", "partial", "reject"))
        ),
        "final_selected": len(selected),
    }

    out_obj = {
        "selected_subject_ids": selected,
        "funnel": funnel,
        "judge_boolean_pass_counts": boolean_counts,
        "max_flagged_threshold": args.max_flagged,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_obj, indent=2))
    print(f"Wrote {args.out_json}")

    lines = [
        "# PSRM-typed report selection summary",
        "",
        "## Funnel",
        "",
        f"- Total input (EviGen + mini both predict 1): **{funnel['total_input']}**",
        f"- Stage A: format_pass=True: **{funnel['format_pass']}**",
        f"- Stage A + PSRM n_flagged<={args.max_flagged}: "
        f"**{funnel['format_pass_AND_psrm_le_3']}**",
        f"- Stage B: judged by gpt-4o: **{funnel['judged_by_gpt4o']}**",
        f"- Stage C: final selected (overall_trust=='trust'): "
        f"**{funnel['final_selected']}**",
        "",
        "## Trust verdict distribution (Stage B)",
        "",
        f"- trust:   {funnel['trust']}",
        f"- partial: {funnel['partial']}",
        f"- reject:  {funnel['reject']}",
        f"- other/unparseable: {funnel['other_trust_label']}",
        "",
        "## Per-criterion pass counts (out of judged)",
        "",
    ]
    for k, v in boolean_counts.items():
        lines.append(f"- {k}: {v}/{len(judged_sids)}")
    lines += [
        "",
        "## Example TRUST sids",
        "",
    ]
    for sid in trust_examples:
        v = judge[sid].get("verdict") or {}
        lines.append(f"- `{sid}` — {v.get('notes', '').strip()}")
    lines += [
        "",
        "## Example REJECT sids",
        "",
    ]
    for sid in reject_examples:
        v = judge[sid].get("verdict") or {}
        lines.append(f"- `{sid}` — {v.get('notes', '').strip()}")

    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md}")

    print("\nFunnel:")
    for k, v in funnel.items():
        print(f"  {k:35s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
