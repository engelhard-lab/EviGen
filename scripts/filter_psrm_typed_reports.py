"""Stage A: deterministic format check for EviGen gpt-4o-mini typed reports.

For each row in outputs/psrm_typed/evigen_mini_psrm395.jsonl, verify the four
bracket-headed sections appear in order, that there are exactly 5 numbered
"Factor summary:" entries, and that each factor has a Supporting evidence
(quoted text + note_id) and a Reasoning line. Cross-reference PSRM n_flagged
from outputs/psrm_typed/mini_bon_results.json.

Writes outputs/psrm_typed/format_check.jsonl with per-patient verdicts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery")
DEFAULT_JSONL = ROOT / "outputs/psrm_typed/evigen_mini_psrm395.jsonl"
DEFAULT_BON = ROOT / "outputs/psrm_typed/mini_bon_results.json"
DEFAULT_OUT = ROOT / "outputs/psrm_typed/format_check.jsonl"

SECTIONS = [
    "[Prediction + Explanation]",
    "[Predictive Factors + Evidence-Based Reasoning]",
    "[Reasoning Lens]",
    "[Recommendations]",
]

# Match "1. Factor summary:" etc. anywhere a line begins (allow leading spaces).
FACTOR_HEAD_RE = re.compile(r"^\s*([1-5])\.\s*Factor summary:", re.MULTILINE)


def section_spans(report: str) -> tuple[list[tuple[str, int, int]] | None, list[str]]:
    """Return ordered (header, start, end) spans and missing-headers list.

    If any header is missing OR they don't occur in the canonical order,
    spans is None and missing lists the issues.
    """
    positions: list[tuple[str, int]] = []
    missing: list[str] = []
    for header in SECTIONS:
        idx = report.find(header)
        if idx < 0:
            missing.append(f"missing_section:{header}")
        else:
            positions.append((header, idx))
    if missing:
        return None, missing
    # check ordering
    last = -1
    for header, idx in positions:
        if idx < last:
            missing.append(f"section_out_of_order:{header}")
            last = idx
        else:
            last = idx
    if missing:
        return None, missing
    # build spans: each section runs until the next header start (or EOF).
    spans: list[tuple[str, int, int]] = []
    for i, (header, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(report)
        spans.append((header, start, end))
    return spans, []


def check_factors(factors_block: str) -> tuple[int, list[str]]:
    """Return (n_factors_found, issues) for the Predictive Factors block."""
    issues: list[str] = []
    matches = list(FACTOR_HEAD_RE.finditer(factors_block))
    nums = [m.group(1) for m in matches]
    if nums != ["1", "2", "3", "4", "5"]:
        issues.append(f"factor_numbering:{','.join(nums) if nums else 'none'}")
    if len(matches) != 5:
        return len(matches), issues
    # Split each factor body and check Supporting evidence + Reasoning + note_id.
    quote_re = re.compile(r'"[^"]+"', re.DOTALL)
    # Accept either a note_id: or an ICD code: citation (the system uses both
    # notes and codes as evidence; the format spec named note_id as the example
    # but ICD code citations are equivalent here).
    note_id_re = re.compile(
        r"(?:note_id|ICD\s*code)\s*:\s*[A-Za-z0-9\-\_\.]+", re.IGNORECASE
    )
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(factors_block)
        body = factors_block[start:end]
        if "Supporting evidence:" not in body:
            issues.append(f"factor{i+1}:missing_supporting_evidence")
            continue
        evid_start = body.find("Supporting evidence:")
        # Supporting evidence ends at first Reasoning:/Attribution score:/next factor — take rest of body.
        evid_segment = body[evid_start:]
        # Restrict the segment to before "Reasoning:" if present.
        reason_idx = evid_segment.find("Reasoning:")
        evid_only = evid_segment if reason_idx < 0 else evid_segment[:reason_idx]
        if not quote_re.search(evid_only):
            issues.append(f"factor{i+1}:no_quoted_evidence")
        if not note_id_re.search(evid_only):
            issues.append(f"factor{i+1}:no_note_id")
        if "Reasoning:" not in body:
            issues.append(f"factor{i+1}:missing_reasoning")
    return len(matches), issues


def evaluate(report: str) -> tuple[bool, list[str], int]:
    if not isinstance(report, str) or not report.strip():
        return False, ["empty_report"], 0
    spans, missing = section_spans(report)
    if spans is None:
        return False, missing, 0
    # factors block = between [Predictive Factors...] and [Reasoning Lens]
    factor_span = next(s for s in spans if s[0] == SECTIONS[1])
    factors_block = report[factor_span[1]:factor_span[2]]
    n_factors, issues = check_factors(factors_block)
    passed = (n_factors == 5) and not issues
    return passed, issues, n_factors


def load_n_flagged(bon_path: Path) -> dict[str, int]:
    with bon_path.open() as f:
        data = json.load(f)
    out: dict[str, int] = {}
    for sid, p in data.get("patients", {}).items():
        cands = p.get("candidates") or []
        if not cands:
            continue
        n = cands[0].get("n_flagged")
        if isinstance(n, int):
            out[str(sid)] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--bon", type=Path, default=DEFAULT_BON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    n_flagged_map = load_n_flagged(args.bon)
    print(f"Loaded n_flagged for {len(n_flagged_map)} patients from {args.bon}")

    counters = {
        "total": 0,
        "parse_ok_true": 0,
        "sections_pass": 0,
        "five_factors": 0,
        "evidence_complete": 0,
        "format_pass": 0,
        "psrm_le_3": 0,
        "stageA_pass": 0,  # format_pass AND n_flagged<=3
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open() as f, args.out.open("w") as out_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("subject_id")
            parse_ok = bool(rec.get("parse_ok"))
            report = rec.get("parsed_report") or ""
            n_flagged = n_flagged_map.get(str(sid))

            counters["total"] += 1
            if parse_ok:
                counters["parse_ok_true"] += 1

            sections_ok = section_spans(report)[0] is not None if isinstance(report, str) else False
            if sections_ok:
                counters["sections_pass"] += 1

            passed, issues, n_factors = evaluate(report)
            if n_factors == 5:
                counters["five_factors"] += 1
            ev_issues = [i for i in issues if "no_quoted_evidence" in i or "no_note_id" in i]
            if not ev_issues:
                counters["evidence_complete"] += 1

            format_pass = parse_ok and passed
            if format_pass:
                counters["format_pass"] += 1

            psrm_clean = (n_flagged is not None) and (n_flagged <= 3)
            if psrm_clean:
                counters["psrm_le_3"] += 1
            if format_pass and psrm_clean:
                counters["stageA_pass"] += 1

            missing = list(issues)
            if not parse_ok:
                missing.insert(0, "parse_ok_false")

            out_f.write(json.dumps({
                "subject_id": sid,
                "format_pass": format_pass,
                "missing_features": missing,
                "n_factors": n_factors,
                "n_flagged": n_flagged,
                "parse_ok": parse_ok,
            }) + "\n")

    print("\nStage A — format check counts")
    print(f"  total                : {counters['total']}")
    print(f"  parse_ok=True        : {counters['parse_ok_true']}")
    print(f"  4 sections present   : {counters['sections_pass']}")
    print(f"  exactly 5 factors    : {counters['five_factors']}")
    print(f"  evidence complete    : {counters['evidence_complete']}")
    print(f"  FORMAT PASS (all)    : {counters['format_pass']}")
    print(f"  PSRM n_flagged <= 3  : {counters['psrm_le_3']}")
    print(f"  Stage A pass (both)  : {counters['stageA_pass']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
