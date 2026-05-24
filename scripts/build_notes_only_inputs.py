"""Strip ICD code evidence from all baseline inputs to enable a handle-mix-
controlled faithfulness ablation.

Produces three new files:

    data/zeroshot_test_patients_notes_only.jsonl
    data/rag_test_patients_notes_only.jsonl
    outputs/test_explanations_notes_only.json

The first two are produced by post-processing the existing JSONLs with a
regex that removes the ``===== DIAGNOSTIC CODES =====`` section (and
everything after it). The third filters the IG-explanation file's
``retrieved_items`` to ``query_type=='Note'`` only; the downstream
``generate_gold_outputs.py`` already takes top-5 by ``abs(attribution)``,
so feeding the filtered file gives top-5 notes per patient.

After this script runs, all 10 baseline inputs are note-only and the
existing baseline runners and gold-gen module need no code changes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

ZS_CODES_RE = re.compile(
    r"\n\n===== DIAGNOSTIC CODES \(oldest -> newest, N=\d+\) =====\n.*?\Z",
    flags=re.DOTALL,
)
# IMPORTANT: leave a stub `===== DIAGNOSTIC CODES (oldest -> newest, N=0) =====`
# header in place. baselines/zeroshot/run_zeroshot.py:_parse_patient_text bails
# out (returning empty notes_blocks, disabling truncation) if either the
# CLINICAL NOTES or DIAGNOSTIC CODES delimiter is missing. Without a stub,
# long-history patients produce prompts >131K tokens and vLLM errors with
# "VLLMValidationError: maximum context length is 131072 tokens". The stub
# is purely structural — the LLM sees zero codes either way.
ZS_STUB = "\n\n===== DIAGNOSTIC CODES (oldest -> newest, N=0) =====\n"

RAG_CODES_RE = re.compile(
    r"\n\n===== DIAGNOSTIC CODES \(retrieved, N=\d+\) =====\n.*?\Z",
    flags=re.DOTALL,
)
# RAG doesn't have a notes-truncation parser (its prompts are short by
# construction), so we strip cleanly without a stub.


def strip_zeroshot(in_path: Path, out_path: Path) -> dict:
    n_total = n_with_codes = 0
    pre_tokens = post_tokens = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            text = r["patient_text"]
            # Replace the codes section with a stub header (no body) so that
            # run_zeroshot.py's note-truncation parser still finds both the
            # CLINICAL NOTES and DIAGNOSTIC CODES delimiters. Without the
            # stub the parser bails out and disables truncation, producing
            # >131K-token prompts on long-history patients.
            stripped = ZS_CODES_RE.sub(ZS_STUB.rstrip("\n"), text)
            stripped = stripped.rstrip() + "\n"
            had_codes = stripped != text
            if had_codes:
                n_with_codes += 1
            r["patient_text"] = stripped
            r["num_codes"] = 0
            pre_tokens += int(r.get("approx_tokens", 0))
            r["approx_tokens"] = int(len(stripped.split()) * 1.3)
            post_tokens += r["approx_tokens"]
            # The stub header still contains "DIAGNOSTIC CODES" but with N=0
            # — verify there's no [Code] line below it.
            assert "[Code]" not in stripped, "regex left code rows in place"
            fout.write(json.dumps(r) + "\n")
            n_total += 1
    return {
        "name": "zeroshot",
        "in": str(in_path), "out": str(out_path),
        "records": n_total,
        "records_with_codes_stripped": n_with_codes,
        "mean_approx_tokens_pre": pre_tokens / n_total if n_total else 0,
        "mean_approx_tokens_post": post_tokens / n_total if n_total else 0,
    }


def strip_rag(in_path: Path, out_path: Path) -> dict:
    n_total = n_with_codes = 0
    pre_tokens = post_tokens = 0
    pre_codes_total = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            text = r["patient_text"]
            stripped = RAG_CODES_RE.sub("", text).rstrip() + "\n"
            had_codes = stripped != text
            if had_codes:
                n_with_codes += 1
            r["patient_text"] = stripped
            pre_codes_total += int(r.get("num_codes", 0))
            r["num_codes"] = 0
            pre_tokens += int(r.get("approx_tokens", 0))
            r["approx_tokens"] = int(len(stripped.split()) * 1.3)
            post_tokens += r["approx_tokens"]
            assert "DIAGNOSTIC CODES" not in stripped, "regex failed to strip codes section"
            fout.write(json.dumps(r) + "\n")
            n_total += 1
    return {
        "name": "rag",
        "in": str(in_path), "out": str(out_path),
        "records": n_total,
        "records_with_codes_stripped": n_with_codes,
        "mean_codes_per_record_pre": pre_codes_total / n_total if n_total else 0,
        "mean_approx_tokens_pre": pre_tokens / n_total if n_total else 0,
        "mean_approx_tokens_post": post_tokens / n_total if n_total else 0,
    }


def filter_evigen(in_path: Path, out_path: Path) -> dict:
    with in_path.open() as f:
        data = json.load(f)
    note_counts_pre = []
    icd_counts_pre = []
    note_counts_post = []
    for r in data:
        items_pre = r["retrieved_items"]
        notes = [it for it in items_pre if it.get("query_type") == "Note"]
        icds = [it for it in items_pre if it.get("query_type") == "ICD"]
        note_counts_pre.append(len(notes))
        icd_counts_pre.append(len(icds))
        r["retrieved_items"] = notes
        note_counts_post.append(len(r["retrieved_items"]))
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    return {
        "name": "evigen",
        "in": str(in_path), "out": str(out_path),
        "records": len(data),
        "min_notes_per_patient": min(note_counts_post),
        "max_notes_per_patient": max(note_counts_post),
        "median_notes_per_patient": sorted(note_counts_post)[len(note_counts_post) // 2],
        "median_icds_dropped_per_patient": sorted(icd_counts_pre)[len(icd_counts_pre) // 2],
    }


def main():
    inputs = [
        (
            PROJECT_DIR / "data" / "zeroshot_test_patients.jsonl",
            PROJECT_DIR / "data" / "zeroshot_test_patients_notes_only.jsonl",
            strip_zeroshot,
        ),
        (
            PROJECT_DIR / "data" / "rag_test_patients.jsonl",
            PROJECT_DIR / "data" / "rag_test_patients_notes_only.jsonl",
            strip_rag,
        ),
        (
            PROJECT_DIR / "outputs" / "test_explanations.json",
            PROJECT_DIR / "outputs" / "test_explanations_notes_only.json",
            filter_evigen,
        ),
    ]

    print("=" * 72)
    print("Build note-only baseline inputs")
    print("=" * 72)

    for in_path, out_path, fn in inputs:
        if not in_path.exists():
            print(f"[skip] {in_path} not found", file=sys.stderr)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[{fn.__name__}]")
        stats = fn(in_path, out_path)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v}")

    # Sanity: every EviGen patient must have >=5 note items (we take top-5).
    evigen_path = PROJECT_DIR / "outputs" / "test_explanations_notes_only.json"
    with evigen_path.open() as f:
        d = json.load(f)
    short = [r["subject_id"] for r in d if len(r["retrieved_items"]) < 5]
    if short:
        print(f"\nWARNING: {len(short)} EviGen patients have <5 note items: {short[:10]}",
              file=sys.stderr)
    else:
        print(f"\n[ok] all {len(d)} EviGen patients have >=5 note items.")

    print("\nDone.")


if __name__ == "__main__":
    main()
