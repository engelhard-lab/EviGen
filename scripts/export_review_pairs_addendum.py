"""Emit the two new addendum baseline reports for the same 120 clinician-
review patients that ``scripts/export_review_pairs.py`` produced.

For each subject_id in ``clinical_reports_for_review/`` (derived from the
existing ``<sid>_evigen.txt`` files), writes:

* ``<sid>_full_context_holistic.txt`` — gpt-4o-mini, full history, holistic
  3-section narrative (no factor-by-factor, no citations). Source:
  ``outputs/full_context/predictions_mini_holistic.jsonl``.

* ``<sid>_iris_raw.txt`` — deterministic extraction of the EviGen
  predicted probability + top-5 attributed retrieved items (full chunk text,
  citation, attribution score). NO LLM-generated reasoning. Source:
  ``outputs/test_explanations.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evigen_dynamicquery.generate_gold_outputs import strip_chunk_prefix


REVIEW_DIR = REPO / "clinical_reports_for_review"
HOLISTIC_PREDS = REPO / "outputs/full_context/predictions_mini_holistic.jsonl"
EVIGEN_EXPLANATIONS = REPO / "outputs/test_explanations.json"


def collect_review_sids() -> list[int]:
    """The 120 sids are exactly the ones with an existing _evigen.txt."""
    sids = sorted(
        int(p.name.split("_", 1)[0])
        for p in REVIEW_DIR.glob("*_evigen.txt")
    )
    if len(sids) != 120:
        raise SystemExit(f"expected 120 review sids, found {len(sids)}")
    return sids


def load_holistic_by_sid() -> dict[int, dict]:
    """One row per patient from the holistic prediction JSONL."""
    by_sid: dict[int, dict] = {}
    with HOLISTIC_PREDS.open() as f:
        for line in f:
            r = json.loads(line)
            by_sid[int(r["subject_id"])] = r
    return by_sid


def load_evigen_by_sid() -> dict[int, dict]:
    """One record per patient from the IG explanation JSON."""
    return {int(r["subject_id"]): r for r in json.load(EVIGEN_EXPLANATIONS.open())}


def render_holistic(rec: dict) -> str:
    sid = int(rec["subject_id"])
    label = int(rec["label"])
    prob = rec.get("parsed_probability")
    prob_str = "None" if prob is None else f"{float(prob):.4f}"
    body = rec.get("parsed_report") or ""
    header = (
        "# Holistic gpt-4o-mini, full history (notes+codes), "
        "no factor-by-factor structure\n"
        f"# subject_id: {sid}    label: {label}    "
        f"parsed_probability: {prob_str}\n\n"
    )
    return header + body.rstrip() + "\n"


def render_iris_raw(rec: dict) -> str:
    sid = int(rec["subject_id"])
    prob = float(rec["probability"])
    items = sorted(
        rec.get("retrieved_items", []),
        key=lambda it: abs(it.get("attribution_score") or 0.0),
        reverse=True,
    )[:5]

    lines = [
        "=" * 72,
        f"RAW IRIS OUTPUT: Patient {sid}",
        "=" * 72,
        f"Predicted mortality probability (1-year, post-discharge): {prob:.4f}",
        "",
        "Top 5 attributed retrieved items (no LLM reasoning):",
        "",
    ]
    for i, it in enumerate(items, 1):
        attr = float(it.get("attribution_score") or 0.0)
        if it.get("query_type") == "ICD":
            handle = it.get("icd_code_ver") or ""
            citation = f"(ICD code: {handle})" if handle else "(ICD code: unknown)"
            text = " ".join(str(it.get("long_title") or "").split())
        else:
            handle = it.get("note_id") or ""
            citation = f"(note_id: {handle})" if handle else "(note_id: unknown)"
            text = " ".join(strip_chunk_prefix(str(it.get("chunk_text") or "")).split())

        lines.append(f"{i}. Supporting Evidence: \"{text}\"")
        lines.append(f"   Citation: {citation}")
        lines.append(f"   Attribution Score: {attr:+.4f}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def main() -> int:
    sids = collect_review_sids()
    print(f"review-set size: {len(sids)}")

    holistic_by_sid = load_holistic_by_sid()
    evigen_by_sid = load_evigen_by_sid()

    missing_h = [s for s in sids if s not in holistic_by_sid]
    missing_e = [s for s in sids if s not in evigen_by_sid]
    if missing_h:
        raise SystemExit(f"missing in holistic preds: {missing_h[:5]} (n={len(missing_h)})")
    if missing_e:
        raise SystemExit(f"missing in EviGen explanations: {missing_e[:5]} (n={len(missing_e)})")

    n_h = n_i = 0
    for sid in sids:
        (REVIEW_DIR / f"{sid}_full_context_holistic.txt").write_text(
            render_holistic(holistic_by_sid[sid])
        )
        n_h += 1
        (REVIEW_DIR / f"{sid}_iris_raw.txt").write_text(
            render_iris_raw(evigen_by_sid[sid])
        )
        n_i += 1

    print(f"wrote {n_h} _full_context_holistic.txt and {n_i} _iris_raw.txt files "
          f"to {REVIEW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
