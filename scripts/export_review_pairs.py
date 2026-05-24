"""Pick 120 of the 129 coherent-set patients (deterministic seed) and export,
for each, two clinical reports side-by-side:

  clinical_reports_for_review/<sid>_evigen.txt           — EviGen mini T=0.0 + PSRM-typed annotations
  clinical_reports_for_review/<sid>_full_context_factors.txt — full-history full-context mini parsed_report
                                                            (factor-by-factor / 4-section format)

Also runs the same 4-section / 5-factor format check on the full_context reports
to confirm format consistency.

The two addendum baselines (<sid>_full_context_holistic.txt and <sid>_iris_raw.txt)
are emitted by scripts/export_review_pairs_addendum.py.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


FACTOR_HDR = re.compile(r"^([1-5])\.\s+Factor summary:", re.MULTILINE)
SUPPORTING = re.compile(
    r"^\s*Supporting evidence: \"[^\"\n]*\"\s*\((note_id|ICD code|note_id\(s\))[:\s][^)]+\)",
    re.MULTILINE,
)
REASONING = re.compile(r"^\s*Reasoning:", re.MULTILINE)
ATTRIB = re.compile(r"^\s*Attribution score:\s*[+\-]?\d+\.\d+", re.MULTILINE)
SECTION_HEADERS = [
    "[Prediction + Explanation]",
    "[Predictive Factors + Evidence-Based Reasoning]",
    "[Reasoning Lens]",
    "[Recommendations]",
]


def check_format(text: str, require_attribution: bool = True) -> tuple[bool, list[str]]:
    """4-section + 5-factor anatomy check. Attribution score lines are
    EviGen-only; pass require_attribution=False for full-context reports."""
    failures: list[str] = []
    last_idx = -1
    for sh in SECTION_HEADERS:
        i = text.find(sh)
        if i < 0:
            failures.append(f"missing_section:{sh}")
            last_idx = 10**9
        elif i < last_idx:
            failures.append(f"out_of_order_section:{sh}")
        else:
            last_idx = i
    factor_hdrs = FACTOR_HDR.findall(text)
    if factor_hdrs != ["1", "2", "3", "4", "5"]:
        failures.append(f"wrong_factor_count_or_order:{factor_hdrs}")
    if len(SUPPORTING.findall(text)) < 5:
        failures.append(f"too_few_supporting_evidence:{len(SUPPORTING.findall(text))}")
    if len(REASONING.findall(text)) < 5:
        failures.append(f"too_few_reasoning:{len(REASONING.findall(text))}")
    if require_attribution and len(ATTRIB.findall(text)) < 5:
        failures.append(f"too_few_attribution:{len(ATTRIB.findall(text))}")
    return len(failures) == 0, failures


def main() -> int:
    repo = Path("/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery")
    coherent_path = repo / "outputs/psrm_typed/selected_subject_ids_coherent.json"
    typed_per_pat_path = repo / "outputs/psrm_typed/typed_reports_per_patient.json"
    full_context_path = repo / "outputs/full_context/predictions_mini_full.jsonl"
    out_dir = repo / "clinical_reports_for_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    coherent = json.loads(coherent_path.read_text())["subject_ids"]
    typed_by_sid = json.loads(typed_per_pat_path.read_text())
    full_context_by_sid: dict[int, dict] = {}
    for line in full_context_path.open():
        r = json.loads(line)
        full_context_by_sid[int(r["subject_id"])] = r

    # Pre-filter: only keep coherent sids whose full_context ALSO passes format.
    eligible: list[int] = []
    excluded_for_format: list[tuple[int, list[str]]] = []
    for sid in coherent:
        ok, f = check_format(full_context_by_sid[sid]["parsed_report"],
                             require_attribution=False)
        if ok:
            eligible.append(sid)
        else:
            excluded_for_format.append((sid, f))
    print(f"coherent total: {len(coherent)}; "
          f"full_context format-pass eligible: {len(eligible)}; "
          f"excluded for full_context format: {len(excluded_for_format)}")

    # Clear stale files from previous runs before writing the fresh sample.
    for old in out_dir.glob("*.txt"):
        old.unlink()
    for old in out_dir.glob("*.json"):
        old.unlink()

    assert len(eligible) >= 120, f"only {len(eligible)} eligible sids, need 120"
    rng = random.Random(42)
    sample = sorted(rng.sample(eligible, 120))
    print(f"sampled {len(sample)} sids (seed=42)")

    n_pass_zs = 0
    failures_zs: list[tuple[int, list[str]]] = []
    for sid in sample:
        ev = typed_by_sid[str(sid)]
        zs_row = full_context_by_sid[sid]
        zs = zs_row["parsed_report"]
        ev_path = out_dir / f"{sid}_evigen.txt"
        zs_path = out_dir / f"{sid}_full_context_factors.txt"
        ev_path.write_text(ev)
        # Add a small header so the full_context file is identifiable.
        zs_path.write_text(
            f"# Full-context gpt-4o-mini, full history (notes+codes), "
            f"factor-by-factor 4-section / 5-factor format\n"
            f"# subject_id: {sid}    label: {zs_row['label']}    "
            f"parsed_probability: {zs_row['parsed_probability']}\n\n"
            f"{zs}\n"
        )
        ok, fails = check_format(zs, require_attribution=False)
        if ok:
            n_pass_zs += 1
        else:
            failures_zs.append((sid, fails))

    print(f"wrote {2 * len(sample)} files to {out_dir}")
    print(f"full_context format pass: {n_pass_zs}/{len(sample)}")
    if failures_zs:
        print("full_context failures:")
        for sid, fails in failures_zs:
            print(f"  {sid}: {fails}")
    # Manifest
    manifest = {
        "source_coherent_set": str(coherent_path),
        "n_coherent_total": len(coherent),
        "n_full_context_format_pass_in_coherent": len(eligible),
        "n_excluded_for_full_context_format": len(excluded_for_format),
        "excluded_for_full_context_format": [
            {"sid": s, "failures": f} for s, f in excluded_for_format
        ],
        "seed": 42,
        "n_sampled": len(sample),
        "subject_ids": sample,
        "full_context_format_pass_in_sample": n_pass_zs,
        "full_context_failures_in_sample": [{"sid": s, "failures": f}
                                       for s, f in failures_zs],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir/'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
