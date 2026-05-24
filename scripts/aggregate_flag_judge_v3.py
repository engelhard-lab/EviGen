"""Aggregate Claude subagent verdicts into per-type / per-variant precision.

Reads outputs/psrm_typed_v3/flag_batches/batch_{0..7}_verdicts.jsonl plus
batch_{0..7}.json (task payloads). Joins on flag_id. Produces
outputs/psrm_typed_v3/flag_agreement_summary.md and flag_judge_all.jsonl.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery")
V3_DIR = ROOT / "outputs/psrm_typed_v3"
BATCH_DIR = V3_DIR / "flag_batches"
V2_VERDICTS = ROOT / "outputs/psrm_typed/claude_judge/clinical_judge_all_v2.jsonl"
OUT_MD = V3_DIR / "flag_agreement_summary.md"
OUT_JSONL = V3_DIR / "flag_judge_all.jsonl"


def load_tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for p in sorted(BATCH_DIR.glob("batch_*.json")):
        for t in json.loads(p.read_text()):
            tasks[t["flag_id"]] = t
    return tasks


def load_verdicts() -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    for p in sorted(BATCH_DIR.glob("batch_*_verdicts.jsonl")):
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                v = json.loads(line)
                verdicts[v["flag_id"]] = v
    return verdicts


def load_v2_buckets() -> dict[int, str]:
    out: dict[int, str] = {}
    with V2_VERDICTS.open() as f:
        for line in f:
            r = json.loads(line)
            out[int(r["subject_id"])] = r["overall_trust"]
    return out


def main() -> int:
    tasks = load_tasks()
    verdicts = load_verdicts()
    print(f"[agg] tasks={len(tasks)} verdicts={len(verdicts)}")

    rows = []
    missing = []
    for fid, task in tasks.items():
        v = verdicts.get(fid)
        if v is None:
            missing.append(fid)
            continue
        rows.append({**task, **{f"verdict_{k}": vv for k, vv in v.items() if k != "flag_id"}})
    if missing:
        print(f"[agg] WARN: {len(missing)} flag IDs missing verdicts: {missing[:10]}...")

    # Write joined JSONL
    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[agg] wrote {len(rows)} joined rows -> {OUT_JSONL}")

    v2 = load_v2_buckets()

    # --- per-type / per-variant precision -----------------------------------

    # Each row may have psrm_v3a and/or psrm_v3b. The error_type can differ.
    per_type_v3a: dict[str, Counter] = defaultdict(Counter)
    per_type_v3b: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        agree = r.get("verdict_agree", "unsure")
        if r.get("psrm_v3a"):
            t = r["psrm_v3a"]["error_type"]
            per_type_v3a[t][agree] += 1
        if r.get("psrm_v3b"):
            t = r["psrm_v3b"]["error_type"]
            per_type_v3b[t][agree] += 1

    def precision_rows(d: dict[str, Counter]) -> list[tuple]:
        out = []
        for t in sorted(d):
            c = d[t]
            total = sum(c.values())
            agree = c.get("agree", 0)
            disagree = c.get("disagree", 0)
            unsure = c.get("unsure", 0)
            pct = agree / total * 100 if total else 0.0
            out.append((t, agree, disagree, unsure, total, pct))
        return out

    # --- variant overlap ----------------------------------------------------
    overlap = Counter()
    for r in rows:
        a = r.get("psrm_v3a") is not None
        b = r.get("psrm_v3b") is not None
        overlap[("a" if a else "", "b" if b else "")] += 1

    # --- cross-tab vs v2 trust ---------------------------------------------
    bucket_agree = {
        "a": defaultdict(Counter),
        "b": defaultdict(Counter),
    }
    for r in rows:
        bucket = v2.get(r["subject_id"], "unknown")
        agree = r.get("verdict_agree", "unsure")
        if r.get("psrm_v3a"):
            bucket_agree["a"][bucket][agree] += 1
        if r.get("psrm_v3b"):
            bucket_agree["b"][bucket][agree] += 1

    # --- severity for agreed flags -----------------------------------------
    severity = Counter()
    for r in rows:
        if r.get("verdict_agree") == "agree":
            sev = r.get("verdict_severity") or "unspecified"
            severity[sev] += 1

    # --- write summary ------------------------------------------------------
    md = ["# v3 per-flag agreement summary\n"]
    md.append(f"Total non-deterministic PSRM flags graded: **{len(rows)}** "
              f"(missing verdicts: {len(missing)}).\n")

    md.append("\n## Per-type, per-variant precision (Claude agreement %)\n")
    for variant, d in [("v3a (Platt-cal τ=0.40)", per_type_v3a),
                       ("v3b (raw-rank percentile)", per_type_v3b)]:
        md.append(f"\n### {variant}")
        md.append("\n| error_type | agree | disagree | unsure | total | agree% |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for t, a, d_, u, tot, pct in precision_rows(d):
            md.append(f"| {t} | {a} | {d_} | {u} | {tot} | {pct:.1f}% |")

    md.append("\n## Variant overlap\n")
    md.append("| variants | n |\n|---|---:|")
    for (a, b), n in sorted(overlap.items()):
        label = " & ".join(filter(None, [a, b])) or "(neither)"
        md.append(f"| {label} | {n} |")

    md.append("\n## Cross-tab: v2 bucket × Claude agreement (per variant)\n")
    for variant, d in bucket_agree.items():
        md.append(f"\n### v3{variant}")
        md.append("\n| v2 bucket | agree | disagree | unsure | total |")
        md.append("|---|---:|---:|---:|---:|")
        for b in ["trust", "partial", "reject", "unknown"]:
            c = d.get(b, Counter())
            tot = sum(c.values())
            if tot == 0:
                continue
            md.append(f"| {b} | {c.get('agree',0)} | {c.get('disagree',0)} | {c.get('unsure',0)} | {tot} |")

    md.append("\n## Severity of agreed flags\n")
    md.append("| severity | n |\n|---|---:|")
    for sev in ["severe", "moderate", "minor", "unspecified"]:
        if severity.get(sev):
            md.append(f"| {sev} | {severity[sev]} |")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[agg] wrote summary -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
