"""Stage A v2: split mini_typed_report.txt into per-patient typed reports and
check both (a) the EviGen 4-section structure AND (b) the PSRM annotation
format matches `IRIS-PSRM-3/output/2000p_multiclass/demo_report_typed_gold.txt`:

- Patient block delimited by lines of '=' (length 72) and 'EVIDENCE ALIGNMENT
  REPORT: Patient <sid>'.
- 'Mortality prediction: <float>' line.
- Four bracket sections in order.
- 5 numbered factors each with Factor summary / Supporting evidence (quote +
  note_id OR ICD code) / Reasoning / Attribution score.
- Per-factor flag lines (if present): '   >> Possible <error_type> — ...'.
- Footer: '-'×72 / 'Verification: X/Y steps passed.' / optional
  '  Flagged: Factor <n> — <error_type>.' lines / '=' x 72.

Emits:
- outputs/psrm_typed/typed_format_check.jsonl (per-patient pass/fail + reasons)
- outputs/psrm_typed/typed_reports_per_patient.json (sid -> full typed text)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATIENT_HDR = re.compile(r"^=+\nEVIDENCE ALIGNMENT REPORT: Patient (\d+)\n=+$", re.MULTILINE)
SECTION_HEADERS = [
    "[Prediction + Explanation]",
    "[Predictive Factors + Evidence-Based Reasoning]",
    "[Reasoning Lens]",
    "[Recommendations]",
]
FACTOR_HDR = re.compile(r"^([1-5])\.\s+Factor summary:", re.MULTILINE)
SUPPORTING = re.compile(r"^\s*Supporting evidence: \"[^\"\n]*\"\s*\((note_id|ICD code|note_id\(s\))[:\s][^)]+\)", re.MULTILINE)
REASONING = re.compile(r"^\s*Reasoning:", re.MULTILINE)
ATTRIB = re.compile(r"^\s*Attribution score:\s*[+\-]?\d+\.\d+", re.MULTILINE)
FLAG_LINE = re.compile(r"^\s*>> Possible (provenance error|hallucination|factual_inaccuracy|attribution_distortion|unhelpfulness|generic_negative|[a-z_ ]+(?: error)?) — ", re.MULTILINE)
VERIF = re.compile(r"^Verification: (\d+)/(\d+) steps passed\.$", re.MULTILINE)
FLAGGED_FACTOR = re.compile(r"^\s+Flagged: Factor (\d+) — ([a-z_ ]+(?: error)?)\.$", re.MULTILINE)


def split_patients(text: str) -> dict[str, str]:
    """Split the concatenated typed-report file into {sid: full_block}."""
    out: dict[str, str] = {}
    matches = list(re.finditer(r"=+\nEVIDENCE ALIGNMENT REPORT: Patient (\d+)\n=+", text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[start:end].rstrip() + "\n"
    return out


def check_one(sid: str, block: str) -> dict:
    failures: list[str] = []
    info: dict = {}

    # 1. Header
    if not re.search(rf"^=+\nEVIDENCE ALIGNMENT REPORT: Patient {sid}\n=+", block):
        failures.append("missing_patient_header")
    # 2. Mortality prediction line
    if not re.search(r"^Mortality prediction: [01]\.\d+$", block, re.MULTILINE):
        failures.append("missing_mortality_line")
    # 3. Four sections in order
    last_idx = -1
    for sh in SECTION_HEADERS:
        i = block.find(sh)
        if i < 0:
            failures.append(f"missing_section:{sh}")
            last_idx = 10**9
        elif i < last_idx:
            failures.append(f"out_of_order_section:{sh}")
        else:
            last_idx = i
    # 4. Five factors
    factor_hdrs = FACTOR_HDR.findall(block)
    if len(factor_hdrs) != 5 or factor_hdrs != ["1", "2", "3", "4", "5"]:
        failures.append(f"wrong_factor_count_or_order:{factor_hdrs}")
    n_supporting = len(SUPPORTING.findall(block))
    n_reasoning = len(REASONING.findall(block))
    n_attrib = len(ATTRIB.findall(block))
    if n_supporting < 5:
        failures.append(f"too_few_supporting_evidence:{n_supporting}")
    if n_reasoning < 5:
        failures.append(f"too_few_reasoning:{n_reasoning}")
    if n_attrib < 5:
        failures.append(f"too_few_attribution:{n_attrib}")

    # 5. Flag lines well-formed (if present)
    flag_lines = FLAG_LINE.findall(block)
    info["n_flag_lines_inline"] = len(flag_lines)

    # 6. Verification footer
    v = VERIF.search(block)
    if not v:
        failures.append("missing_verification_line")
    else:
        passed, total = int(v.group(1)), int(v.group(2))
        info["verif_passed"] = passed
        info["verif_total"] = total
        if passed > total:
            failures.append("verification_passed_gt_total")

    # 7. Flagged-factor footer lines (if any)
    flagged_factors = FLAGGED_FACTOR.findall(block)
    info["n_flagged_factors_footer"] = len(flagged_factors)
    for fn, et in flagged_factors:
        if int(fn) not in range(1, 6):
            failures.append(f"flagged_factor_out_of_range:{fn}")

    # 8. Closing '=' line
    if not block.rstrip().endswith("=" * 72):
        failures.append("missing_closing_separator")

    info["sid"] = int(sid)
    info["format_pass"] = len(failures) == 0
    info["failures"] = failures
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--typed-txt",
                    default="outputs/psrm_typed/mini_typed_report.txt")
    ap.add_argument("--out-check",
                    default="outputs/psrm_typed/typed_format_check.jsonl")
    ap.add_argument("--out-per-pat",
                    default="outputs/psrm_typed/typed_reports_per_patient.json")
    args = ap.parse_args()

    text = Path(args.typed_txt).read_text()
    per_pat = split_patients(text)
    print(f"parsed {len(per_pat)} patient blocks from {args.typed_txt}")

    Path(args.out_per_pat).write_text(json.dumps(per_pat, indent=2))
    print(f"wrote {args.out_per_pat}")

    rows = []
    for sid, block in per_pat.items():
        rows.append(check_one(sid, block))
    rows.sort(key=lambda r: r["sid"])
    with open(args.out_check, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_pass = sum(1 for r in rows if r["format_pass"])
    print(f"format_pass: {n_pass}/{len(rows)}")

    # Top failure modes
    from collections import Counter
    fc = Counter()
    for r in rows:
        for f in r["failures"]:
            # Bucket failures with values (e.g. "wrong_factor_count_or_order:[1,2]")
            key = f.split(":", 1)[0]
            fc[key] += 1
    print("top failure buckets:")
    for k, c in fc.most_common():
        print(f"  {k}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
