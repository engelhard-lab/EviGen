"""Build the zeroshot-baseline per-patient input dataset for the test split.

For every test subject (same split as training: random_state=42, stratified on
label, 80/10/10), this script pulls the patient's full note history and ICD
code history from the raw CSVs in data/, sorts them into chronological order,
formats them into a single LLM-ready text block, and writes one JSON line per
patient to data/zeroshot_test_patients.jsonl.

Runs CPU-only (safe for biostat-gpu sbatch). Uses DuckDB for streaming
CSV/parquet reads so peak RAM stays well under 32GB.
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evigen_dynamicquery.config import load_config_from_yaml  # noqa: E402
from evigen_dynamicquery.data import create_subject_splits  # noqa: E402


def get_split_subjects(note_parquet: str, cfg, split: str):
    """Reproduce the canonical EviGen split for ``split`` in {train,val,test}.

    Critical: must match the row ordering used by
    `data.load_embeddings_and_mappings` so the seeded `train_test_split`
    inside `create_subject_splits` produces the same partition. That loader
    reads via `pyarrow.iter_batches` (subject_id+label only here) and applies
    `pd.concat(...).drop_duplicates()` which preserves first-occurrence order.
    DuckDB `SELECT DISTINCT` does NOT preserve order and produced a divergent
    split previously.
    """
    pf = pq.ParquetFile(note_parquet)
    parts = []
    for batch in pf.iter_batches(batch_size=50_000, columns=["subject_id", "label"]):
        parts.append(batch.to_pandas())
    note_labels_df = pd.concat(parts, ignore_index=True).drop_duplicates()
    train_df, val_df, test_df = create_subject_splits(note_labels_df, cfg)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    if split not in splits:
        raise ValueError(f"Unknown split={split!r}; must be one of {list(splits)}")
    df = splits[split].reset_index(drop=True)
    print(
        f"{split} label distribution: {df['label'].value_counts().to_dict()}",
        flush=True,
    )
    return df


def get_test_subjects(note_parquet: str, cfg):
    """Back-compat wrapper kept in case external callers imported it."""
    return get_split_subjects(note_parquet, cfg, "test")


def load_notes_for_subjects(note_csv: Path, subject_ids):
    """Return a DataFrame of relevant note rows for the given subjects."""
    id_list = ",".join(str(int(s)) for s in subject_ids)
    query = f"""
        SELECT subject_id, note_id, note_type, charttime, age_at_note, text
        FROM read_csv_auto('{note_csv}')
        WHERE subject_id IN ({id_list})
    """
    print(f"Loading notes for {len(subject_ids)} subjects from {note_csv} ...", flush=True)
    df = duckdb.query(query).df()
    print(f"  loaded {len(df):,} note rows", flush=True)
    return df


def load_codes_for_subjects(code_csv: Path, subject_ids):
    """Return a DataFrame of relevant ICD code rows for the given subjects."""
    id_list = ",".join(str(int(s)) for s in subject_ids)
    query = f"""
        SELECT subject_id, icd_code, icd_version, age_at_code,
               DaysToLastNotes, long_title
        FROM read_csv_auto('{code_csv}')
        WHERE subject_id IN ({id_list})
    """
    print(f"Loading codes for {len(subject_ids)} subjects from {code_csv} ...", flush=True)
    df = duckdb.query(query).df()
    print(f"  loaded {len(df):,} code rows", flush=True)
    return df


def format_patient_text(subject_id: int, notes: pd.DataFrame, codes: pd.DataFrame) -> dict:
    """Format one patient's history into a single text block."""
    lines = [f"Patient ID: {subject_id}"]

    # --- Notes (oldest -> newest by charttime) ---
    notes = notes.sort_values("charttime", na_position="last", kind="mergesort")
    n_notes = len(notes)
    lines.append("")
    lines.append(f"===== CLINICAL NOTES (oldest -> newest, N={n_notes}) =====")
    for i, row in enumerate(notes.itertuples(index=False), start=1):
        age = getattr(row, "age_at_note", None)
        note_type = getattr(row, "note_type", "") or ""
        charttime = getattr(row, "charttime", "") or ""
        note_id = getattr(row, "note_id", "") or ""
        text = (getattr(row, "text", "") or "").strip()
        header = (
            f"[Note {i}] (note_id: {note_id}) Age: {age}; Date: {charttime}; Type: {note_type}"
        )
        lines.append(header)
        lines.append(text)
        lines.append("")

    # --- Codes (oldest -> newest by DaysToLastNotes descending) ---
    codes = codes.sort_values("DaysToLastNotes", ascending=False, na_position="last", kind="mergesort")
    # Deduplicate in display order on (icd_code, icd_version, age_at_code)
    codes = codes.drop_duplicates(subset=["icd_code", "icd_version", "age_at_code"], keep="first")
    n_codes = len(codes)
    lines.append(f"===== DIAGNOSTIC CODES (oldest -> newest, N={n_codes}) =====")
    for row in codes.itertuples(index=False):
        age = getattr(row, "age_at_code", None)
        icd = getattr(row, "icd_code", "") or ""
        ver = getattr(row, "icd_version", "") or ""
        title = (getattr(row, "long_title", "") or "").strip()
        lines.append(f"[Code] (ICD code: {icd}-{ver}) {title}  (Age: {age})")

    patient_text = "\n".join(lines)
    approx_tokens = int(len(patient_text.split()) * 1.3)

    return {
        "subject_id": int(subject_id),
        "patient_text": patient_text,
        "num_notes": int(n_notes),
        "num_codes": int(n_codes),
        "approx_tokens": approx_tokens,
    }


def build_split(split: str, cfg, note_csv: Path, code_csv: Path, out_path: Path):
    subj_df = get_split_subjects(note_parquet=cfg.note_parquet, cfg=cfg, split=split)
    subject_ids = subj_df["subject_id"].tolist()
    label_by_sid = dict(
        zip(subj_df["subject_id"].astype(int), subj_df["label"].astype(float))
    )

    notes_df = load_notes_for_subjects(note_csv, subject_ids)
    codes_df = load_codes_for_subjects(code_csv, subject_ids)

    notes_by_sid = dict(tuple(notes_df.groupby("subject_id")))
    codes_by_sid = dict(tuple(codes_df.groupby("subject_id")))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    empty_cols_notes = notes_df.iloc[0:0]
    empty_cols_codes = codes_df.iloc[0:0]

    n_written = 0
    n_no_notes = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sid in subject_ids:
            sid_int = int(sid)
            notes = notes_by_sid.get(sid_int, empty_cols_notes)
            codes = codes_by_sid.get(sid_int, empty_cols_codes)
            if len(notes) == 0:
                n_no_notes += 1
            record = format_patient_text(sid_int, notes, codes)
            record["label"] = label_by_sid[sid_int]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[{split}] wrote {n_written} patients to {out_path}", flush=True)
    if n_no_notes:
        print(f"[{split}] WARNING: {n_no_notes} patients had zero notes", flush=True)

    token_counts = []
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            token_counts.append(json.loads(line)["approx_tokens"])
    s = pd.Series(token_counts)
    print(
        f"[{split}] approx_tokens stats: min={s.min()} median={int(s.median())} "
        f"p95={int(s.quantile(0.95))} max={s.max()}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_DIR / "configs" / "config.yaml"),
        help="Path to YAML config for note_parquet + split seeds",
    )
    parser.add_argument(
        "--note-csv",
        type=str,
        default=str(PROJECT_DIR / "data" / "dataset4death_note.csv"),
    )
    parser.add_argument(
        "--code-csv",
        type=str,
        default=str(PROJECT_DIR / "data" / "dataset4death_code.csv"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to build. 'all' -> train + val + test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSONL path. Ignored when --split all (files are derived "
            "from --output-dir). Defaults to data/zeroshot_{split}_patients.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "data"),
        help="Directory for per-split output files when --split all.",
    )
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    print(
        f"Using note_parquet={cfg.note_parquet} "
        f"train_set_ratio={cfg.train_set_ratio} random_state={cfg.random_state}",
        flush=True,
    )

    note_csv = Path(args.note_csv)
    code_csv = Path(args.code_csv)

    if args.split == "all":
        out_dir = Path(args.output_dir)
        for split in ("train", "val", "test"):
            build_split(
                split,
                cfg,
                note_csv,
                code_csv,
                out_dir / f"zeroshot_{split}_patients.jsonl",
            )
    else:
        out_path = (
            Path(args.output)
            if args.output
            else Path(args.output_dir) / f"zeroshot_{args.split}_patients.jsonl"
        )
        build_split(args.split, cfg, note_csv, code_csv, out_path)


if __name__ == "__main__":
    main()
