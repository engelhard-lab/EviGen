"""Slice T=1.0 N=10 rollouts into 10 per-rank JSONLs using PSRM's calibrated score.

Inputs:
  outputs/bon_psrm/70b_t10_psrm_results.json   (from 08_inference_bon.py)
  outputs/notes_only/evigen_70b_bon10_t10_rollouts.jsonl

Outputs:
  outputs/notes_only/evigen_70b_bon10_t10_rank{1..10}.jsonl

PSRM emits patients[sid].candidates pre-sorted descending by
sample_score_calibrated (08_inference_bon.py L468-L469). Each candidate's `idx`
field is the original rollout_idx PSRM received (preserved through
run_psrm_bon_select.py to_json mode, which sorts by rollout_idx). We look the
original row up by (sid, candidate.idx).

Patients with fewer than k valid candidates are simply absent from rank-k JSONL
(no padding). The aggregator reports n_patients per rank from the file row
count, so the missing patients are visible.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psrm-results", type=Path,
                    default=Path("outputs/bon_psrm/70b_t10_psrm_results.json"))
    ap.add_argument("--rollouts", type=Path,
                    default=Path("outputs/notes_only/evigen_70b_bon10_t10_rollouts.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/notes_only"))
    ap.add_argument("--out-prefix", type=str,
                    default="evigen_70b_bon10_t10_rank")
    ap.add_argument("--max-rank", type=int, default=10)
    args = ap.parse_args()

    print(f"[rank-slicer] loading rollouts from {args.rollouts}", flush=True)
    by_key: dict[tuple[int, int], dict] = {}
    n_rows = 0
    with args.rollouts.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = int(r["subject_id"])
            ridx = int(r.get("rollout_idx", 0))
            by_key[(sid, ridx)] = r
            n_rows += 1
    print(f"[rank-slicer] indexed {n_rows} rollout rows ({len(by_key)} unique (sid,ridx))",
          flush=True)

    print(f"[rank-slicer] loading PSRM results from {args.psrm_results}", flush=True)
    psrm_data = json.loads(args.psrm_results.read_text())
    patients = psrm_data.get("patients", {})
    print(f"[rank-slicer] PSRM scored {len(patients)} patients", flush=True)

    # Open output files for each rank.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_files = {}
    for k in range(1, args.max_rank + 1):
        out_path = args.out_dir / f"{args.out_prefix}{k}.jsonl"
        out_files[k] = (out_path, out_path.open("w"))

    n_written_per_rank = defaultdict(int)
    n_missing_match = 0
    n_psrm_error_skipped = 0
    candidates_per_patient: list[int] = []

    for sid_str, pinfo in patients.items():
        sid = int(sid_str)
        if "error" in pinfo and not pinfo.get("candidates"):
            n_psrm_error_skipped += 1
            continue
        cands = pinfo.get("candidates", [])
        candidates_per_patient.append(len(cands))
        # PSRM emits candidates sorted desc by sample_score_calibrated (or
        # sample_score if no calibration). We trust that order; rank-1 is
        # candidates[0], rank-2 is candidates[1], etc.
        for rank_idx, cand in enumerate(cands, start=1):
            if rank_idx > args.max_rank:
                break
            ridx = int(cand.get("idx", -1))
            row = by_key.get((sid, ridx))
            if row is None:
                n_missing_match += 1
                continue
            out_row = dict(row)
            out_row["psrm_rank"] = rank_idx
            out_row["psrm_score"] = float(cand.get("sample_score", 0.0))
            out_row["psrm_score_calibrated"] = float(
                cand.get("sample_score_calibrated", cand.get("sample_score", 0.0)))
            out_row["psrm_n_flagged"] = int(cand.get("n_flagged", 0))
            out_row["psrm_n_steps"] = int(cand.get("n_steps", 0))
            out_row["psrm_n_content_steps"] = int(cand.get("n_content_steps", 0))
            out_row["n_candidates_valid_for_patient"] = int(
                pinfo.get("n_candidates_valid",
                          pinfo.get("n_candidates_scored", len(cands))))
            _, fh = out_files[rank_idx]
            fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_written_per_rank[rank_idx] += 1

    for k, (path, fh) in out_files.items():
        fh.close()
        print(f"[rank-slicer] rank {k:>2}: wrote {n_written_per_rank[k]} rows -> {path.name}")

    if n_missing_match:
        print(f"[rank-slicer] WARN: {n_missing_match} (sid,ridx) pairs in PSRM output "
              f"could not be matched to the rollouts JSONL", flush=True)
    if n_psrm_error_skipped:
        print(f"[rank-slicer] WARN: {n_psrm_error_skipped} patients skipped (PSRM error)",
              flush=True)
    if candidates_per_patient:
        cpp = sorted(candidates_per_patient)
        print(f"[rank-slicer] candidates/patient: "
              f"min={cpp[0]} median={cpp[len(cpp)//2]} max={cpp[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
