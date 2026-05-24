"""Faithfulness evaluation for LLM-generated clinical reports.

Step 1 (implemented + executed): verify each factor's quoted supporting
evidence actually appears in the cited source (clinical note `text` or
ICD `long_title`), using rapidfuzz partial_ratio at multiple thresholds.

Step 2 (implemented + gated behind --step2, not run this iteration):
send each report alone (no medical history) to an OpenAI judge model
and score reasoning coherence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Normalization & parsing helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_QUOTE_STRIP = '"\'`“”‘’ \t'

# Strip the synthetic "Patient age: X years / Clinical Note:" prefix that the
# chunked-notes parquet adds at embedding time. This decoration is NOT in
# `data/dataset4death_note.csv` (the source-of-truth that the eval indexes
# against), so when a baseline's quote/content carries the prefix the fuzzy
# match drops even when the underlying evidence is verbatim. The prefix is
# metadata about the chunk, not clinical evidence — stripping it is correct.
#
# We also accept LLM-elided variants:
#   "Patient age: 53 years... [actual quote]"     ← LLM truncated middle
#   "Patient age: 70 years Clinical Note: ..."    ← canonical form
# and we strip leading/trailing "..." elision markers (these are LLM
# editing marks, not part of the source text). Affects all RAG and EviGen
# baselines (whose inputs come from the chunk parquet); zeroshot is
# unaffected because it pulls full notes from the CSV.
_CHUNK_PREFIX_RE = re.compile(
    r"^patient age:\s*\d+\s*years\s*(?:\.{2,}\s*)?(?:clinical note:\s*)?(?:\.{2,}\s*)?",
)
_ELLIPSIS_EDGE_RE = re.compile(r"^\s*\.{2,}\s*|\s*\.{2,}\s*$")
# Mid-string ellipsis marker (2+ literal dots OR the unicode horizontal-ellipsis
# code point). Used to split a quote into contiguous spans for fuzzy matching:
# a quote like `"A ... B"` is two spans, A and B, each of which must match the
# source on its own. NFKC normalization (in `normalize`) does NOT decompose
# `\u2026` to "...", so we accept both forms here.
_MID_ELLIPSIS_RE = re.compile(r"\.{2,}|\u2026")


def partial_ratio_multi_span(query: str, source: str) -> float:
    """`rapidfuzz.fuzz.partial_ratio` with multi-span (ellipsis-split) support.

    A quote like `"Wound Care: ... Physical Therapy: ..."` is two contiguous
    spans separated by a mid-string ellipsis (an elision marker). The default
    `partial_ratio` looks for one contiguous window in the source matching the
    whole query, so multi-span quotes score artificially low even when every
    elided segment is verbatim. We split the query on `\\.{2,}` / `\\u2026`,
    score each non-empty segment against the source, and return the MIN — i.e.
    "every quoted segment must be grounded" semantics. For single-span quotes
    (no interior ellipsis), this is identical to `fuzz.partial_ratio(query,
    source)` and Llama / gpt-4o-mini scores are unaffected.
    """
    if not query:
        return 0.0
    segments = [s.strip() for s in _MID_ELLIPSIS_RE.split(query)]
    # Drop empty / too-short segments (an isolated ellipsis between quote marks
    # produces empty fragments; <3 chars cannot be matched meaningfully).
    segments = [s for s in segments if len(s) >= 3]
    if not segments:
        return float(fuzz.partial_ratio(query, source))
    if len(segments) == 1:
        return float(fuzz.partial_ratio(segments[0], source))
    return float(min(fuzz.partial_ratio(seg, source) for seg in segments))


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text)
    text = text.strip().lower()
    text = _CHUNK_PREFIX_RE.sub("", text).strip()
    # Strip leading/trailing "..." elision markers; these are LLM editing
    # marks (e.g. "...complaint: low back pain s/p fall...") that do not
    # appear in the source CSV. They inflate the query length and degrade
    # the fuzzy alignment without contributing real signal.
    text = _ELLIPSIS_EDGE_RE.sub("", text).strip()
    return text


_PREDICTIVE_HDR = re.compile(
    r"^\s*(?:\*\*|\[)?\s*Predictive\s+Factors[^\n]*\n",
    re.MULTILINE | re.IGNORECASE,
)
# End the factors section at whichever of the next-section headers appears
# first: Reasoning Lens / Recommendations / a generic bold next-section.
_END_OF_FACTORS_HDR = re.compile(
    r"^\s*(?:\*\*|\[)?\s*(?:Reasoning\s+Lens|Recommendations)\b",
    re.MULTILINE | re.IGNORECASE,
)
# Factor start is a numbered list item, optionally with bold markers or a
# `Factor summary:` / `Factor:` prefix.
_FACTOR_START = re.compile(
    r"^\s*(\d+)\.\s+(?:\*\*)?(?:Factor\s+summary|Factor)?\s*(?:\*\*)?\s*:?\s*",
    re.MULTILINE | re.IGNORECASE,
)
# Accept straight and curly double quotes. Require >=3 chars inside.
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{3,}?)[\"”]")
_HANDLE_RE = re.compile(
    r"\(\s*(note_id|ICD\s*code)\s*:\s*([^)\s,;]+)", re.IGNORECASE
)
_CITATION_SPAN_RE = re.compile(
    r"\(\s*(?:note_id|ICD\s*code)\s*:[^)]*\)", re.IGNORECASE
)
_QUOTE_MARK_RE = re.compile(r'["“”]')


def _unescape_json_string(raw: str) -> str:
    """Undo the JSON string escapes (\\\", \\n, \\t, \\r, \\\\) that may be
    present when we've bypassed json.loads via manual extraction."""
    # Two-pass with placeholder to avoid double-replacing backslashes.
    return (
        raw.replace("\\\\", "\x00")
        .replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\x00", "\\")
    )


def _extract_report_value(text: str) -> str | None:
    """Extract the value of the `report` field from a JSON-like wrapper whose
    inner string value may contain unescaped control chars (raw newlines) —
    which is exactly what the two verbalizer files contain, so json.loads
    always raises on them. We scan for the terminating `"` manually, tracking
    backslash escapes and requiring the close quote to be followed by
    whitespace + `,` / `}` / end-of-string.
    """
    m = re.search(r'"report"\s*:\s*"', text)
    if not m:
        return None
    start = m.end()
    i = start
    close = None
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j == n or text[j] in ",}":
                close = i
                break
        i += 1
    if close is None:
        return None
    return _unescape_json_string(text[start:close])


def unwrap_report(parsed_report: Any) -> str | None:
    """Handle the 4 format variants observed across the 8 prediction files.

    - Plain text (zeroshot/rag *_70b_full): return as-is.
    - JSON-wrapped string (verbalizer variants; contains unescaped raw
      newlines inside the "report" string → json.loads fails): scan manually.
    - Dict: take .report.
    - None/empty: return None.
    """
    if parsed_report is None:
        return None
    if isinstance(parsed_report, dict):
        rep = parsed_report.get("report")
        return rep if isinstance(rep, str) and rep.strip() else None
    if not isinstance(parsed_report, str) or not parsed_report.strip():
        return None
    # Strip markdown code fences: `````json\n{...}\n`````  (common in 8B outputs).
    fenced = re.match(
        r"^\s*```(?:json|JSON)?\s*\n(.*?)\n\s*```\s*$",
        parsed_report,
        re.DOTALL,
    )
    if fenced:
        parsed_report = fenced.group(1)
    s = parsed_report.lstrip()
    if s.startswith("{"):
        try:
            obj = json.loads(parsed_report)
            rep = obj.get("report") if isinstance(obj, dict) else None
            if isinstance(rep, str) and rep.strip():
                return rep
        except json.JSONDecodeError:
            pass
        rep = _extract_report_value(parsed_report)
        if rep and rep.strip():
            return rep
        # Fallback: some verbalizer outputs are malformed — e.g. start with a
        # stray `{\n` followed directly by the report body (no `"report":`
        # field), or have an unterminated `"report": "..."` that ends with `}`
        # rather than `"`. Strip a leading `{` and trailing `}`/`"` and try
        # the remainder as-is if it contains any recognizable section header.
        stripped = s.lstrip("{").lstrip()
        if stripped.endswith("}"):
            stripped = stripped[:-1].rstrip()
        if stripped.endswith('"'):
            stripped = stripped[:-1].rstrip()
        # Also strip a leading `"report": "` if present (truncated close).
        stripped = re.sub(r'^"report"\s*:\s*"', "", stripped)
        if stripped and re.search(
            r"(?:\[|\*\*)?\s*(?:Prediction|Predictive\s+Factors|Reasoning\s+Lens|Recommendations)",
            stripped,
            re.IGNORECASE,
        ):
            return stripped
        return None
    return parsed_report


def extract_factors(report_text: str) -> list[dict[str, Any]]:
    """Return factor dicts {idx, body} found after a Predictive Factors header
    and before the next section (Reasoning Lens / Recommendations)."""
    m = _PREDICTIVE_HDR.search(report_text)
    if not m:
        return []
    body = report_text[m.end():]
    m2 = _END_OF_FACTORS_HDR.search(body)
    if m2:
        body = body[: m2.start()]

    starts = list(_FACTOR_START.finditer(body))
    if not starts:
        return []
    factors: list[dict[str, Any]] = []
    for i, s in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        idx = int(s.group(1))
        factors.append({"idx": idx, "body": body[s.start():end]})
    return factors


def extract_supporting_evidence_line(factor_body: str) -> str:
    """Return the 'Supporting evidence:' content — may span multiple lines
    if the model put `Age: ...; Date: ...` on a continuation line. We stop
    at the next in-factor subheader (Reasoning: / Consideration:) or a line
    that starts a new numbered item or bracketed section.

    Fallback: if no `Supporting evidence:` label is present (common in the
    verbalizer variants), return the entire factor body so downstream
    quote/citation extraction can still pick up evidence anywhere it lives.
    """
    m = re.search(r"Supporting\s+evidence\s*:\s*", factor_body, re.IGNORECASE)
    if not m:
        return factor_body.strip()
    tail = factor_body[m.end():]
    stop = re.search(
        r"\n\s*(?:Reasoning\s*[:\-]|Consideration\s*[:\-]|\d+\.\s+(?:Factor|Consideration)|\[)",
        tail,
        flags=re.IGNORECASE,
    )
    if stop:
        tail = tail[: stop.start()]
    return tail.strip()


def parse_evidence(ev_line: str) -> tuple[str | None, str | None, str | None]:
    """From a Supporting-evidence line, return (quote, handle_type, handle_id).

    quote: longest "..." substring of length >=3, or None.
    handle_type: 'note' | 'icd' | None.
    handle_id: e.g. '17490921-DS-20' or 'M4805-10', or None.
    """
    if not ev_line:
        return None, None, None
    quotes = _QUOTE_RE.findall(ev_line)
    quote = max(quotes, key=len).strip(_QUOTE_STRIP) if quotes else None
    if quote == "":
        quote = None

    handle_type = handle_id = None
    hm = _HANDLE_RE.search(ev_line)
    if hm:
        kind = hm.group(1).lower().replace(" ", "")  # "note_id" or "icdcode"
        handle_type = "note" if "note" in kind else "icd"
        handle_id = hm.group(2).strip().rstrip(".,;:)")
        # ICD index keys are digit/letter only (no dots): strip dots so that
        # "I10.9-10" still resolves to "I109-10".
        if handle_type == "icd" and "." in handle_id:
            handle_id = handle_id.replace(".", "")
    return quote, handle_type, handle_id


# ---------------------------------------------------------------------------
# Source index
# ---------------------------------------------------------------------------


@dataclass
class SourceIndex:
    notes_by_id: dict[str, str] = field(default_factory=dict)  # normalized text
    codes_by_ver: dict[str, str] = field(default_factory=dict)  # normalized long_title
    subject_notes: dict[int, set[str]] = field(default_factory=dict)
    subject_codes: dict[int, set[str]] = field(default_factory=dict)


def build_source_index(notes_csv: Path, codes_csv: Path) -> SourceIndex:
    print(f"[index] loading notes from {notes_csv}", flush=True)
    # Read only what we need
    notes_df = pd.read_csv(notes_csv, usecols=["note_id", "subject_id", "text"])
    print(f"[index] loaded {len(notes_df):,} note rows", flush=True)

    notes_by_id: dict[str, str] = {}
    subject_notes: dict[int, set[str]] = {}
    for note_id, subject_id, text in zip(
        notes_df["note_id"].values,
        notes_df["subject_id"].values,
        notes_df["text"].values,
    ):
        if pd.isna(note_id) or pd.isna(subject_id):
            continue
        notes_by_id[str(note_id)] = normalize(text if isinstance(text, str) else "")
        subject_notes.setdefault(int(subject_id), set()).add(str(note_id))

    del notes_df

    print(f"[index] loading codes from {codes_csv}", flush=True)
    codes_df = pd.read_csv(codes_csv, usecols=["subject_id", "icd_code_ver", "long_title"])
    print(f"[index] loaded {len(codes_df):,} code rows", flush=True)

    codes_by_ver: dict[str, str] = {}
    subject_codes: dict[int, set[str]] = {}
    for subj, ver, title in zip(
        codes_df["subject_id"].values,
        codes_df["icd_code_ver"].values,
        codes_df["long_title"].values,
    ):
        if pd.isna(ver) or pd.isna(subj):
            continue
        codes_by_ver[str(ver)] = normalize(title if isinstance(title, str) else "")
        subject_codes.setdefault(int(subj), set()).add(str(ver))

    return SourceIndex(notes_by_id, codes_by_ver, subject_notes, subject_codes)


# ---------------------------------------------------------------------------
# Step 1 evaluation
# ---------------------------------------------------------------------------


def _strip_citation_and_quotes(ev_line: str) -> str:
    """Remove citation parentheticals and quote marks from the evidence line,
    leaving just the asserted evidence text to compare against source."""
    s = _CITATION_SPAN_RE.sub(" ", ev_line)
    s = _QUOTE_MARK_RE.sub("", s)
    return s


def evaluate_factor(
    ev_line: str,
    quote: str | None,
    handle_type: str | None,
    handle_id: str | None,
    subject_id: int,
    notes_kept: set[str] | None,
    idx: SourceIndex,
    thresholds: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "has_quote": quote is not None,
        "quote_text": quote,
        "handle_type": handle_type,
        "handle_id": handle_id,
        "handle_valid_for_subject": False,
        "cited_dropped_note": False,
        "source_found": False,
        "quote_match_score": None,
        "content_match_score": None,
    }
    for t in thresholds:
        out[f"match_at_{t}"] = False
        out[f"faithful_at_{t}"] = False
        out[f"content_match_at_{t}"] = False
        out[f"content_faithful_at_{t}"] = False

    if handle_type is None or handle_id is None:
        return out

    if handle_type == "note":
        owned = idx.subject_notes.get(subject_id, set())
        out["handle_valid_for_subject"] = handle_id in owned
        if notes_kept is not None and handle_id in owned and handle_id not in notes_kept:
            out["cited_dropped_note"] = True
        source = idx.notes_by_id.get(handle_id)
    else:  # icd
        owned = idx.subject_codes.get(subject_id, set())
        out["handle_valid_for_subject"] = handle_id in owned
        source = idx.codes_by_ver.get(handle_id)

    out["source_found"] = source is not None

    # Quote-based faithfulness (strict: requires quote marks around evidence).
    if quote and source:
        score = partial_ratio_multi_span(normalize(quote), source)
        out["quote_match_score"] = round(float(score), 2)
        for t in thresholds:
            out[f"match_at_{t}"] = score >= t
            out[f"faithful_at_{t}"] = (
                out["has_quote"]
                and out["handle_valid_for_subject"]
                and not out["cited_dropped_note"]
                and out[f"match_at_{t}"]
            )

    # Content-based faithfulness: best fidelity score among the asserted-
    # evidence candidates. We compute partial_ratio over (a) the quote text
    # if present, and (b) the whole evidence line minus citation+quote-mark
    # decorations, and take the max. Reason: when the LLM emits a verbatim
    # quote that matches the source, surrounding labels like
    # "**Factor Summary:** ... **Supporting Evidence:** \"<quote>\""
    # would otherwise drag the line-level score down even though the quote
    # itself is a perfect match. This made the prior content_match
    # score conflate quote fidelity with model formatting, especially on
    # baselines whose factor blocks include extra structural labels.
    if source:
        candidate_scores: list[float] = []
        if quote:
            qscore = partial_ratio_multi_span(normalize(quote), source)
            candidate_scores.append(float(qscore))
        content = normalize(_strip_citation_and_quotes(ev_line))
        if len(content) >= 3:
            lscore = partial_ratio_multi_span(content, source)
            candidate_scores.append(float(lscore))
        if candidate_scores:
            cscore = max(candidate_scores)
            out["content_match_score"] = round(cscore, 2)
            for t in thresholds:
                out[f"content_match_at_{t}"] = cscore >= t
                out[f"content_faithful_at_{t}"] = (
                    out["handle_valid_for_subject"]
                    and not out["cited_dropped_note"]
                    and out[f"content_match_at_{t}"]
                )
    return out


def process_prediction_file(
    pred_path: Path,
    idx: SourceIndex,
    thresholds: list[int],
    out_dir: Path,
    max_records: int | None = None,
) -> dict[str, Any]:
    baseline = baseline_name_from_path(pred_path)
    factors_out = out_dir / f"{baseline}_factors.jsonl"
    records_out = out_dir / f"{baseline}_records.jsonl"

    n_total = n_parsable = 0
    n_truncated = 0
    n_factors = 0
    n_has_quote = n_handle_valid = n_source_found = n_cited_dropped = 0
    n_match = {t: 0 for t in thresholds}
    n_faithful = {t: 0 for t in thresholds}
    n_record_fully_faithful = {t: 0 for t in thresholds}
    n_content_match = {t: 0 for t in thresholds}
    n_content_faithful = {t: 0 for t in thresholds}
    n_record_content_fully_faithful = {t: 0 for t in thresholds}

    with pred_path.open() as f_in, \
         factors_out.open("w") as f_factors, \
         records_out.open("w") as f_records:

        for i, line in enumerate(f_in):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            subject_id = int(rec["subject_id"])
            parse_ok = bool(rec.get("parse_ok", True))
            truncated = bool(rec.get("truncated", False))
            if truncated:
                n_truncated += 1
            notes_kept = rec.get("notes_kept")
            notes_kept_set = (
                {str(x) for x in notes_kept}
                if isinstance(notes_kept, list)
                else None
            )

            rec_out: dict[str, Any] = {
                "subject_id": subject_id,
                "baseline": baseline,
                "parse_ok": parse_ok,
                "truncated": truncated,
                "factors_parsed": 0,
                "unevaluable": False,
            }
            for t in thresholds:
                rec_out[f"n_faithful_at_{t}"] = 0
                rec_out[f"record_fully_faithful_at_{t}"] = False
                rec_out[f"n_content_faithful_at_{t}"] = 0
                rec_out[f"record_content_fully_faithful_at_{t}"] = False

            # Try to unwrap regardless of parse_ok — upstream `parse_ok=False`
            # typically just means the raw completion failed strict JSON parse,
            # but the report content is often still present and recoverable.
            report_text = unwrap_report(rec.get("parsed_report"))
            if not report_text:
                rec_out["unevaluable"] = True
                f_records.write(json.dumps(rec_out) + "\n")
                continue

            n_parsable += 1
            factors = extract_factors(report_text)
            rec_out["factors_parsed"] = len(factors)

            per_record_faithful = {t: 0 for t in thresholds}
            per_record_content = {t: 0 for t in thresholds}
            for factor in factors:
                ev_line = extract_supporting_evidence_line(factor["body"])
                quote, htype, hid = parse_evidence(ev_line)
                eval_out = evaluate_factor(
                    ev_line, quote, htype, hid, subject_id, notes_kept_set, idx, thresholds
                )
                factor_row = {
                    "subject_id": subject_id,
                    "baseline": baseline,
                    "factor_idx": factor["idx"],
                    "evidence_line": ev_line[:500],
                    **eval_out,
                }
                f_factors.write(json.dumps(factor_row) + "\n")

                n_factors += 1
                if eval_out["has_quote"]:
                    n_has_quote += 1
                if eval_out["handle_valid_for_subject"]:
                    n_handle_valid += 1
                if eval_out["source_found"]:
                    n_source_found += 1
                if eval_out["cited_dropped_note"]:
                    n_cited_dropped += 1
                for t in thresholds:
                    if eval_out[f"match_at_{t}"]:
                        n_match[t] += 1
                    if eval_out[f"faithful_at_{t}"]:
                        n_faithful[t] += 1
                        per_record_faithful[t] += 1
                    if eval_out[f"content_match_at_{t}"]:
                        n_content_match[t] += 1
                    if eval_out[f"content_faithful_at_{t}"]:
                        n_content_faithful[t] += 1
                        per_record_content[t] += 1

            # Fully faithful = all 5 factors faithful (requires exactly 5 parsed)
            for t in thresholds:
                rec_out[f"n_faithful_at_{t}"] = per_record_faithful[t]
                if rec_out["factors_parsed"] == 5 and per_record_faithful[t] == 5:
                    rec_out[f"record_fully_faithful_at_{t}"] = True
                    n_record_fully_faithful[t] += 1
                rec_out[f"n_content_faithful_at_{t}"] = per_record_content[t]
                if rec_out["factors_parsed"] == 5 and per_record_content[t] == 5:
                    rec_out[f"record_content_fully_faithful_at_{t}"] = True
                    n_record_content_fully_faithful[t] += 1
            f_records.write(json.dumps(rec_out) + "\n")

    def pct(num: int, denom: int) -> float:
        return round(100.0 * num / denom, 2) if denom else 0.0

    summary = {
        "baseline": baseline,
        "n_records": n_total,
        "n_parsable": n_parsable,
        "n_truncated": n_truncated,
        "n_factors_parsed": n_factors,
        "parsable_pct": pct(n_parsable, n_total),
        "avg_factors_per_parsable_record": round(n_factors / n_parsable, 2) if n_parsable else 0.0,
        "has_quote_pct": pct(n_has_quote, n_factors),
        "handle_valid_pct": pct(n_handle_valid, n_factors),
        "source_found_pct": pct(n_source_found, n_factors),
        "cited_dropped_note_pct": pct(n_cited_dropped, n_factors),
    }
    for t in thresholds:
        summary[f"match_pct_at_{t}"] = pct(n_match[t], n_factors)
        summary[f"faithful_pct_at_{t}"] = pct(n_faithful[t], n_factors)
        summary[f"fully_faithful_record_pct_at_{t}"] = pct(n_record_fully_faithful[t], n_parsable)
        summary[f"content_match_pct_at_{t}"] = pct(n_content_match[t], n_factors)
        summary[f"content_faithful_pct_at_{t}"] = pct(n_content_faithful[t], n_factors)
        summary[f"content_fully_faithful_record_pct_at_{t}"] = pct(
            n_record_content_fully_faithful[t], n_parsable
        )
    return summary


def baseline_name_from_path(p: Path) -> str:
    """outputs/zeroshot/predictions_70b_full.jsonl -> zeroshot_70b_full"""
    parent = p.parent.name  # zeroshot | rag
    stem = p.stem  # predictions_70b_full
    stem = stem.removeprefix("predictions_")
    return f"{parent}_{stem}"


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_summary(summaries: list[dict[str, Any]], path: Path, thresholds: list[int]) -> None:
    base_cols = [
        "baseline", "n_records", "n_parsable", "parsable_pct",
        "avg_factors_per_parsable_record",
        "has_quote_pct", "handle_valid_pct", "source_found_pct",
        "cited_dropped_note_pct",
    ]

    lines = ["# Faithfulness (Step 1) — per-baseline summary\n"]

    # Table 1 — content_faithful (quote-agnostic, primary metric)
    cols_c = list(base_cols)
    for t in thresholds:
        cols_c += [
            f"content_match_pct_at_{t}",
            f"content_faithful_pct_at_{t}",
            f"content_fully_faithful_record_pct_at_{t}",
        ]
    lines.append("## Content-faithfulness (quote-agnostic — primary metric)\n")
    lines.append("| " + " | ".join(cols_c) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_c)) + " |")
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(c, "")) for c in cols_c) + " |")
    lines.append("")

    # Table 2 — quote-required faithful (strict format compliance)
    cols_q = list(base_cols)
    for t in thresholds:
        cols_q += [f"match_pct_at_{t}", f"faithful_pct_at_{t}", f"fully_faithful_record_pct_at_{t}"]
    lines.append("## Quote-required faithfulness (strict — requires `\"...\"` marks)\n")
    lines.append("| " + " | ".join(cols_q) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_q)) + " |")
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(c, "")) for c in cols_q) + " |")
    lines.append("")

    lines.append("## Definitions")
    lines.append("- **has_quote**: factor's Supporting evidence line contains a `\"...\"` quoted span (≥3 chars).")
    lines.append("- **handle_valid**: cited `note_id`/`ICD code` actually belongs to this subject.")
    lines.append("- **cited_dropped_note**: citation is a note that was truncated out of the LLM's input (not shown to the model).")
    lines.append(f"- **match@T**: `rapidfuzz.partial_ratio(quote, source) >= T` (thresholds: {thresholds}). Only defined when `has_quote`.")
    lines.append("- **faithful@T**: has_quote AND handle_valid AND NOT cited_dropped_note AND match@T. (Strict — penalizes models that drop quote marks even when the text is verbatim.)")
    lines.append("- **fully_faithful_record@T**: record parsed all 5 factors and all 5 are faithful@T.")
    lines.append("- **content_match@T**: `rapidfuzz.partial_ratio(evidence_line_minus_citation_and_quote_marks, source) >= T`. Quote-agnostic.")
    lines.append("- **content_faithful@T**: handle_valid AND NOT cited_dropped_note AND content_match@T. (Preferred: actual content fidelity, regardless of whether the model used `\"...\"`.)")
    lines.append("- **content_fully_faithful_record@T**: record parsed all 5 factors and all 5 are content_faithful@T.")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Step 2 — LLM-as-judge coherence eval (gated behind --step2)
# ---------------------------------------------------------------------------

STEP2_SYSTEM = (
    "You are auditing a clinical reasoning report for internal coherence. "
    "You will see ONE report. The report contains 5 numbered factors; each "
    "factor has a Supporting evidence line and a Reasoning sentence. The "
    "report ends with a Reasoning Lens that synthesizes the 5 factors. "
    "Do NOT judge medical correctness or whether the prognosis is right. "
    "Judge only whether the reasoning the report itself states follows from "
    "the evidence the report itself stated. Return strict JSON."
)

STEP2_SCHEMA_INSTR = (
    "Return a JSON object with this exact shape (no extra keys):\n"
    "{\n"
    '  "per_factor": [\n'
    "    {\n"
    '      "idx": <int 1-5>,\n'
    '      "reasoning_follows_from_evidence": <bool>,\n'
    '      "reasoning_uses_only_stated_evidence": <bool>,\n'
    '      "rationale": "<=1 short sentence"\n'
    "    }\n"
    "  ],\n"
    '  "overall_reasoning_follows_from_factors": <bool>,\n'
    '  "overall_uses_only_stated_factors": <bool>,\n'
    '  "overall_rationale": "<=2 sentences"\n'
    "}\n"
    "Definitions:\n"
    "- reasoning_follows_from_evidence: does the factor's Reasoning sentence "
    "logically follow from its Supporting evidence?\n"
    "- reasoning_uses_only_stated_evidence: does it avoid introducing facts "
    "not present in the Supporting evidence (no smuggled diagnoses, dates, "
    "severity, etc.)?\n"
    "- overall_reasoning_follows_from_factors: does the Reasoning Lens "
    "follow from the 5 factors?\n"
    "- overall_uses_only_stated_factors: does the Reasoning Lens avoid "
    "introducing factors or evidence not listed above?"
)


def _step2_key(model: str, subject_id: int, report: str) -> str:
    return hashlib.sha1(
        f"{model}|{subject_id}|{report}".encode("utf-8")
    ).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["key"]] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def collect_step2_records(
    predictions: list[Path],
    output_dir: Path,
    sample_per_baseline: int | None = None,
    step1_threshold: int = 90,
) -> list[tuple[str, int, str]]:
    """Return [(baseline, subject_id, report_text)] for every record that
    passed Step 1 (record_content_fully_faithful_at_<step1_threshold> == True)."""
    field = f"record_content_fully_faithful_at_{step1_threshold}"
    out: list[tuple[str, int, str]] = []
    for pred_path in predictions:
        if not pred_path.exists():
            continue
        baseline = baseline_name_from_path(pred_path)
        records_path = output_dir / f"{baseline}_records.jsonl"
        if not records_path.exists():
            print(f"[step2] skipping {baseline}: no records JSONL — run Step 1 first.",
                  file=sys.stderr)
            continue
        eligible: set[int] = set()
        for line in records_path.open():
            r = json.loads(line)
            if r.get(field):
                eligible.add(int(r["subject_id"]))
        if not eligible:
            continue
        # Walk predictions JSONL to fetch the report text for eligible subjects.
        added = 0
        for line in pred_path.open():
            rec = json.loads(line)
            sid = int(rec["subject_id"])
            if sid not in eligible:
                continue
            report = unwrap_report(rec.get("parsed_report"))
            if not report:
                continue
            out.append((baseline, sid, report))
            added += 1
            if sample_per_baseline is not None and added >= sample_per_baseline:
                break
        print(f"[step2] {baseline}: {added} records queued (eligible={len(eligible)})",
              flush=True)
    return out


async def step2_run_async(
    records_in: list[tuple[str, int, str]],  # (baseline, subject_id, report_text)
    model: str,
    concurrency: int,
    cache_path: Path,
) -> None:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        print("Step 2 requires `openai`. Install with: pip install openai", file=sys.stderr)
        return

    client = AsyncOpenAI()
    cache = _load_cache(cache_path)

    sem = asyncio.Semaphore(concurrency)
    cache_f = cache_path.open("a")

    todo = [
        (b, s, r) for (b, s, r) in records_in
        if _step2_key(model, s, r) not in cache
    ]
    print(f"[step2] {len(records_in)} requested, {len(records_in)-len(todo)} cached, "
          f"{len(todo)} to fetch", flush=True)

    progress = {"done": 0, "fail": 0}
    total = len(todo)

    async def one(baseline: str, subject_id: int, report: str) -> None:
        key = _step2_key(model, subject_id, report)
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": STEP2_SYSTEM},
                            {"role": "user", "content": report + "\n\n" + STEP2_SCHEMA_INSTR},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    content = resp.choices[0].message.content or "{}"
                    verdict = json.loads(content)
                    usage = getattr(resp, "usage", None)
                    row = {
                        "key": key,
                        "baseline": baseline,
                        "subject_id": subject_id,
                        "model": model,
                        "verdict": verdict,
                        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    }
                    cache_f.write(json.dumps(row) + "\n")
                    cache_f.flush()
                    cache[key] = row
                    progress["done"] += 1
                    if progress["done"] % 100 == 0:
                        print(f"[step2] {progress['done']}/{total} done "
                              f"({progress['fail']} failed)", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        progress["fail"] += 1
                        print(f"[step2] FAILED subj={subject_id} baseline={baseline}: {exc}",
                              file=sys.stderr)
                        return
                    await asyncio.sleep(2 ** attempt)

    await asyncio.gather(*(one(b, s, r) for b, s, r in todo))
    cache_f.close()
    print(f"[step2] complete: {progress['done']} succeeded, {progress['fail']} failed",
          flush=True)


def write_step2_summary(
    cache_path: Path,
    summary_md: Path,
    summary_json: Path,
    model_filter: str | None = None,
) -> None:
    """Aggregate per-baseline pass rates from the cache."""
    cache = _load_cache(cache_path)
    rows = [r for r in cache.values()
            if model_filter is None or r.get("model") == model_filter]

    by_baseline: dict[str, dict[str, Any]] = {}
    for r in rows:
        b = r["baseline"]
        v = r.get("verdict") or {}
        bucket = by_baseline.setdefault(b, {
            "n": 0,
            "n_factors_judged": 0,
            "n_per_factor_follows": 0,
            "n_per_factor_uses_only": 0,
            "n_per_factor_both": 0,
            "n_overall_follows": 0,
            "n_overall_uses_only": 0,
            "n_overall_both": 0,
            "n_record_all_factors_pass": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        })
        bucket["n"] += 1
        if r.get("input_tokens"):
            bucket["input_tokens"] += r["input_tokens"]
        if r.get("output_tokens"):
            bucket["output_tokens"] += r["output_tokens"]

        per = v.get("per_factor") or []
        all_pass = bool(per)  # require non-empty
        for f in per:
            bucket["n_factors_judged"] += 1
            ff = bool(f.get("reasoning_follows_from_evidence"))
            uo = bool(f.get("reasoning_uses_only_stated_evidence"))
            if ff: bucket["n_per_factor_follows"] += 1
            if uo: bucket["n_per_factor_uses_only"] += 1
            if ff and uo: bucket["n_per_factor_both"] += 1
            if not (ff and uo):
                all_pass = False
        if all_pass:
            bucket["n_record_all_factors_pass"] += 1

        of = bool(v.get("overall_reasoning_follows_from_factors"))
        ou = bool(v.get("overall_uses_only_stated_factors"))
        if of: bucket["n_overall_follows"] += 1
        if ou: bucket["n_overall_uses_only"] += 1
        if of and ou: bucket["n_overall_both"] += 1

    def pct(num: int, denom: int) -> float:
        return round(100.0 * num / denom, 2) if denom else 0.0

    summaries = []
    for b in sorted(by_baseline):
        x = by_baseline[b]
        summaries.append({
            "baseline": b,
            "model": model_filter,
            "n_judged": x["n"],
            "n_factors_judged": x["n_factors_judged"],
            "per_factor_follows_pct": pct(x["n_per_factor_follows"], x["n_factors_judged"]),
            "per_factor_uses_only_pct": pct(x["n_per_factor_uses_only"], x["n_factors_judged"]),
            "per_factor_both_pct": pct(x["n_per_factor_both"], x["n_factors_judged"]),
            "record_all_factors_pass_pct": pct(x["n_record_all_factors_pass"], x["n"]),
            "overall_follows_pct": pct(x["n_overall_follows"], x["n"]),
            "overall_uses_only_pct": pct(x["n_overall_uses_only"], x["n"]),
            "overall_both_pct": pct(x["n_overall_both"], x["n"]),
            "input_tokens": x["input_tokens"],
            "output_tokens": x["output_tokens"],
        })

    summary_json.write_text(json.dumps(summaries, indent=2))

    cols = [
        "baseline", "model", "n_judged", "n_factors_judged",
        "per_factor_follows_pct", "per_factor_uses_only_pct", "per_factor_both_pct",
        "record_all_factors_pass_pct",
        "overall_follows_pct", "overall_uses_only_pct", "overall_both_pct",
        "input_tokens", "output_tokens",
    ]
    lines = [f"# Step 2 — LLM-as-judge coherence summary (model={model_filter or 'ALL'})\n"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for s in summaries:
        lines.append("| " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- **per_factor_follows_pct**: % of all judged factors where the per-factor reasoning sentence follows from its Supporting evidence.")
    lines.append("- **per_factor_uses_only_pct**: % of factors whose reasoning avoids introducing facts not in the Supporting evidence (no smuggled diagnoses / severity / dates).")
    lines.append("- **per_factor_both_pct**: % satisfying both above (the strict per-factor pass rate).")
    lines.append("- **record_all_factors_pass_pct**: % of judged records where all 5 factors satisfy both per-factor checks.")
    lines.append("- **overall_follows_pct / overall_uses_only_pct / overall_both_pct**: same checks applied to the [Reasoning Lens] vs. the 5 stated factors.")
    summary_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True, type=Path,
                    help="Paths to predictions_*.jsonl files.")
    ap.add_argument("--notes-csv", type=Path, default=None,
                    help="Notes CSV (required unless --step2-only).")
    ap.add_argument("--codes-csv", type=Path, default=None,
                    help="Codes CSV (required unless --step2-only).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=[85, 90, 95])
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap records per file (smoke test for Step 1).")
    ap.add_argument("--step1-only", action="store_true",
                    help="Run only Step 1.")
    ap.add_argument("--step2", action="store_true",
                    help="Run Step 2 LLM-judge after Step 1.")
    ap.add_argument("--step2-only", action="store_true",
                    help="Skip Step 1 (do not load CSVs); run only Step 2 from "
                    "existing {baseline}_records.jsonl files.")
    ap.add_argument("--step2-sample-per-baseline", type=int, default=None,
                    help="Cap eligible records per baseline for Step 2 (smoke).")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--step2-concurrency", type=int, default=20)
    ap.add_argument("--step2-step1-threshold", type=int, default=90, choices=[85, 90, 95],
                    help="cf@T threshold a record must clear to be sent to the judge "
                         "(default 90; cf@85 is more inclusive).")
    ap.add_argument("--summary-suffix", type=str, default="",
                    help="Optional suffix appended to summary{,.json}.md filenames "
                         "(e.g. '_notes_only') so a side-by-side ablation run does "
                         "not clobber the main summary.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.step2_only:
        if args.notes_csv is None or args.codes_csv is None:
            ap.error("--notes-csv and --codes-csv are required unless --step2-only")
        idx = build_source_index(args.notes_csv, args.codes_csv)

        summaries = []
        for pred_path in args.predictions:
            if not pred_path.exists():
                print(f"[warn] missing: {pred_path}", file=sys.stderr)
                continue
            print(f"[eval] {pred_path}", flush=True)
            s = process_prediction_file(
                pred_path, idx, args.thresholds, args.output_dir, args.max_records
            )
            summaries.append(s)
            print(f"  -> parsable={s['n_parsable']}/{s['n_records']} "
                  f"faithful@90={s['faithful_pct_at_90']}% "
                  f"content@90={s['content_faithful_pct_at_90']}% "
                  f"content_fully@90={s['content_fully_faithful_record_pct_at_90']}%",
                  flush=True)

        sfx = args.summary_suffix
        summary_json = args.output_dir / f"summary{sfx}.json"
        summary_md = args.output_dir / f"summary{sfx}.md"
        summary_json.write_text(json.dumps(summaries, indent=2))
        write_summary(summaries, summary_md, args.thresholds)
        print(f"[done] wrote {summary_json} and {summary_md.name}", flush=True)

    if args.step1_only:
        return 0

    if args.step2 or args.step2_only:
        cache_path = args.output_dir / "step2_cache.jsonl"
        recs = collect_step2_records(
            args.predictions,
            args.output_dir,
            sample_per_baseline=args.step2_sample_per_baseline,
            step1_threshold=args.step2_step1_threshold,
        )
        print(f"[step2] dispatching {len(recs)} records to {args.judge_model}",
              flush=True)
        asyncio.run(step2_run_async(recs, args.judge_model,
                                    args.step2_concurrency, cache_path))
        # Per-model summary
        suffix = args.judge_model.replace("/", "_").replace("-", "_")
        write_step2_summary(
            cache_path,
            args.output_dir / f"step2_summary_{suffix}.md",
            args.output_dir / f"step2_summary_{suffix}.json",
            model_filter=args.judge_model,
        )
        print(f"[step2] wrote step2_summary_{suffix}.md / .json", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
