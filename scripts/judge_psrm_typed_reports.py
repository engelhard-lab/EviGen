"""Stage B: GPT-4o LLM-judge for clinical trustworthiness of EviGen typed reports.

Reads outputs/psrm_typed/format_check.jsonl (from Stage A) and runs gpt-4o on
every patient where format_pass==True AND n_flagged<=3. The judge returns a
JSON verdict on plausibility, evidence support, hedging, hallucination, and an
overall trust/partial/reject label.

Writes outputs/psrm_typed/clinical_judge.jsonl (one row per patient).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery")
DEFAULT_JSONL = ROOT / "outputs/psrm_typed/evigen_mini_psrm395.jsonl"
DEFAULT_FORMAT = ROOT / "outputs/psrm_typed/format_check.jsonl"
DEFAULT_OUT = ROOT / "outputs/psrm_typed/clinical_judge.jsonl"

SYSTEM_PROMPT = (
    "You are a senior internist reviewing AI-generated 1-year mortality risk "
    "explanations. You will see ONE report. Judge its clinical "
    "trustworthiness, not whether the prognosis is exactly right. Return "
    "strict JSON in the schema you are given."
)

SCHEMA_INSTR = (
    "Return a JSON object with this exact shape (no extra keys):\n"
    "{\n"
    '  "factors_clinically_plausible": <bool>,\n'
    '  "evidence_supports_factor": <bool>,\n'
    '  "language_appropriately_hedged": <bool>,\n'
    '  "no_obvious_hallucination": <bool>,\n'
    '  "overall_trust": "<trust|partial|reject>",\n'
    '  "notes": "<one short sentence>"\n'
    "}\n"
    "Definitions:\n"
    "- factors_clinically_plausible: would a clinician find all 5 "
    "factor->outcome chains medically reasonable for a 1-year mortality "
    "prediction? (advanced malignancy, end-stage organ failure, hospice "
    "transition, etc. are plausible.)\n"
    "- evidence_supports_factor: does each cited verbatim quote actually "
    "support the factor it is attached to? (i.e., the quote and the factor "
    "summary are about the same thing.)\n"
    "- language_appropriately_hedged: language uses 'may', 'associated "
    "with', 'increases risk of' — no over-confident claims like 'X causes "
    "death within Y years' or absolute prognoses.\n"
    "- no_obvious_hallucination: quotes look like plausible clinical-note "
    "language (vitals, diagnoses, plans, discharge summaries) — not invented "
    "or impossibly precise.\n"
    "- overall_trust: set to 'trust' ONLY if all four booleans above are "
    "true. Use 'partial' if 1-2 are false. Use 'reject' if 3+ are false or "
    "if there is a glaring problem."
)


def _key(model: str, subject_id: int, report: str) -> str:
    return hashlib.sha1(
        f"{model}|{subject_id}|{report}".encode("utf-8")
    ).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
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


def load_stageA_passers(format_path: Path, max_flagged: int) -> set[int]:
    out: set[int] = set()
    with format_path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("format_pass"):
                continue
            n = r.get("n_flagged")
            if n is None or n > max_flagged:
                continue
            out.add(int(r["subject_id"]))
    return out


def load_reports(jsonl_path: Path, eligible: set[int]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            sid = int(rec["subject_id"])
            if sid not in eligible:
                continue
            report = rec.get("parsed_report")
            if not report:
                continue
            out.append((sid, report))
    return out


async def run_judge(
    records: list[tuple[int, str]],
    model: str,
    concurrency: int,
    cache_path: Path,
) -> None:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        print("Requires `openai`. Install with: pip install openai", file=sys.stderr)
        sys.exit(2)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY env var is required.", file=sys.stderr)
        sys.exit(2)

    client = AsyncOpenAI()
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    cache_f = cache_path.open("a")
    lock = asyncio.Lock()

    todo = [(s, r) for (s, r) in records if _key(model, s, r) not in cache]
    print(
        f"[judge] {len(records)} requested, "
        f"{len(records) - len(todo)} cached, {len(todo)} to fetch",
        flush=True,
    )

    progress = {"done": 0, "fail": 0}
    total = len(todo)

    async def one(subject_id: int, report: str) -> None:
        key = _key(model, subject_id, report)
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": report + "\n\n" + SCHEMA_INSTR,
                            },
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    content = resp.choices[0].message.content or "{}"
                    verdict = json.loads(content)
                    usage = getattr(resp, "usage", None)
                    row = {
                        "key": key,
                        "subject_id": subject_id,
                        "model": model,
                        "verdict": verdict,
                        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    }
                    async with lock:
                        cache_f.write(json.dumps(row) + "\n")
                        cache_f.flush()
                        cache[key] = row
                        progress["done"] += 1
                        if progress["done"] % 25 == 0:
                            print(
                                f"[judge] {progress['done']}/{total} "
                                f"done ({progress['fail']} failed)",
                                flush=True,
                            )
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        async with lock:
                            progress["fail"] += 1
                        print(
                            f"[judge] FAILED subj={subject_id}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return
                    await asyncio.sleep(2 ** attempt)

    await asyncio.gather(*(one(s, r) for s, r in todo))
    cache_f.close()
    print(
        f"[judge] complete: {progress['done']} succeeded, "
        f"{progress['fail']} failed",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--format-check", type=Path, default=DEFAULT_FORMAT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-flagged", type=int, default=3)
    args = ap.parse_args()

    eligible = load_stageA_passers(args.format_check, args.max_flagged)
    print(f"Stage A passers (format_pass & n_flagged<={args.max_flagged}): "
          f"{len(eligible)}")

    records = load_reports(args.jsonl, eligible)
    print(f"Loaded {len(records)} reports to judge.")

    asyncio.run(run_judge(records, args.model, args.concurrency, args.out))
    print(f"Wrote verdicts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
