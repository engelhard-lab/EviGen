"""Build the RAG baseline per-patient input dataset for the test split.

For every test subject (same canonical split as full_context), this script:
1. Embeds 8 hand-crafted mortality-risk queries with Qwen3-Embedding-8B
2. Preloads all note/code embeddings + metadata from LanceDB into memory
3. For each patient, computes cosine similarity to find top-K per query
4. Deduplicates and formats retrieved evidence into patient_text
5. Writes data/rag_test_patients.jsonl

Requires 1x GPU for Qwen3-Embedding-8B (~16.5 GB fp16).
Peak RAM ~12 GB for preloaded embeddings.
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    import lancedb
    import pandas as pd
    import pyarrow as pa
except ImportError:
    lancedb = None
    pd = None

# ---------------------------------------------------------------------------
# Hand-crafted queries for mortality risk retrieval
# ---------------------------------------------------------------------------

NOTE_QUERIES = [
    (
        "End-of-life care, palliative care, hospice referral, comfort measures, "
        "do not resuscitate, DNR, DNI, advance directive, goals of care discussion, "
        "code status change"
    ),
    (
        "Organ failure, critical illness, mechanical ventilation, vasopressors, "
        "intensive care unit admission, cardiac arrest, respiratory failure, "
        "renal failure requiring dialysis, multiorgan system failure"
    ),
    (
        "Multiple chronic conditions, comorbidity burden, diabetes with complications, "
        "heart failure, chronic kidney disease, chronic obstructive pulmonary disease, "
        "liver cirrhosis, end-stage disease"
    ),
    (
        "Functional decline, debility, failure to thrive, weight loss, malnutrition, "
        "bedbound, decreased mobility, cognitive decline, dementia, poor prognosis"
    ),
]

CODE_QUERIES = [
    (
        "Cardiac arrest, acute myocardial infarction, stroke, pulmonary embolism, "
        "acute respiratory distress syndrome, aortic dissection, sudden cardiac death"
    ),
    (
        "End-stage renal disease, chronic heart failure, hepatic failure, "
        "chronic respiratory failure, dialysis dependence, organ transplant"
    ),
    (
        "Malignant neoplasm, metastatic disease, secondary malignancy, "
        "advanced cancer, oncology, carcinoma"
    ),
    (
        "Sepsis, severe sepsis, septic shock, bacteremia, systemic infection, "
        "infectious disease complication"
    ),
]


# ---------------------------------------------------------------------------
# Qwen model loading (same pattern as scripts/vectordb4notes.py)
# ---------------------------------------------------------------------------

def load_embedding_model():
    print("Loading Qwen3-Embedding-8B ...", flush=True)
    cache_dir = "/hpc/home/fl105/engelhardlab/fl105/cache"

    # Try flash_attention_2 first; fall back to sdpa if flash-attn is not installed
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-8B",
                cache_folder=cache_dir,
                model_kwargs={
                    "attn_implementation": attn_impl,
                    "dtype": torch.float16,
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
            print(f"  Using attn_implementation={attn_impl}", flush=True)
            break
        except (ImportError, ValueError) as e:
            print(f"  {attn_impl} not available: {e}", flush=True)
            if attn_impl == "eager":
                raise
            continue

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Disable KV cache (not needed for embedding, causes OOM otherwise)
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "use_cache"):
            mod.config.use_cache = False
            break

    print(f"Model loaded on {device}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Preload embeddings from LanceDB into memory
# ---------------------------------------------------------------------------

def preload_note_data(db_dir, table_name, test_sids):
    """Preload note embeddings and metadata for test subjects.

    Returns:
        note_by_sid: dict[int, dict] with keys:
            vectors: np.ndarray (N, 4096) L2-normalized
            chunks: list[str]
            days_to_prediction: list[int]
            note_ids: list[str]
    """
    print(f"Preloading note data from {table_name} ...", flush=True)
    t0 = time.time()

    db = lancedb.connect(db_dir)
    tbl = db.open_table(table_name)

    # Read only test subjects' data using a WHERE filter on all test sids
    sid_set = set(int(s) for s in test_sids)
    # Read full table in batches via lance, filter in python (faster than WHERE with large IN list)
    lance_ds = tbl.to_lance()

    note_by_sid = {}
    n_rows = 0

    for batch in lance_ds.to_batches(
        columns=["subject_id", "vector", "chunk", "days_to_prediction", "note_id"],
        batch_size=50000,
    ):
        df = batch.to_pandas()
        for sid, group in df.groupby("subject_id"):
            sid = int(sid)
            if sid not in sid_set:
                continue
            vecs = np.vstack(group["vector"].values).astype("float32")
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vecs = vecs / norms

            if sid in note_by_sid:
                # Append to existing
                prev = note_by_sid[sid]
                prev["vectors"] = np.vstack([prev["vectors"], vecs])
                prev["chunks"].extend(group["chunk"].tolist())
                prev["days_to_prediction"].extend(group["days_to_prediction"].tolist())
                prev["note_ids"].extend(group["note_id"].tolist())
            else:
                note_by_sid[sid] = {
                    "vectors": vecs,
                    "chunks": group["chunk"].tolist(),
                    "days_to_prediction": group["days_to_prediction"].tolist(),
                    "note_ids": group["note_id"].tolist(),
                }
            n_rows += len(group)

    elapsed = time.time() - t0
    print(f"  Loaded {n_rows:,} note rows for {len(note_by_sid)} subjects in {elapsed:.1f}s",
          flush=True)
    return note_by_sid


def preload_code_data(db_dir, table_name, test_sids):
    """Preload code embeddings and metadata for test subjects.

    Returns:
        code_by_sid: dict[int, dict] with keys:
            vectors: np.ndarray (N, 4096) L2-normalized
            long_titles: list[str]
            icd_code_vers: list[str]
    """
    print(f"Preloading code data from {table_name} ...", flush=True)
    t0 = time.time()

    db = lancedb.connect(db_dir)
    tbl = db.open_table(table_name)
    lance_ds = tbl.to_lance()

    sid_set = set(int(s) for s in test_sids)
    code_by_sid = {}
    n_rows = 0

    for batch in lance_ds.to_batches(
        columns=["subject_id", "vector", "long_title", "icd_code_ver"],
        batch_size=50000,
    ):
        df = batch.to_pandas()
        for sid, group in df.groupby("subject_id"):
            sid = int(sid)
            if sid not in sid_set:
                continue
            vecs = np.vstack(group["vector"].values).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vecs = vecs / norms

            if sid in code_by_sid:
                prev = code_by_sid[sid]
                prev["vectors"] = np.vstack([prev["vectors"], vecs])
                prev["long_titles"].extend(group["long_title"].tolist())
                prev["icd_code_vers"].extend(group["icd_code_ver"].tolist())
            else:
                code_by_sid[sid] = {
                    "vectors": vecs,
                    "long_titles": group["long_title"].tolist(),
                    "icd_code_vers": group["icd_code_ver"].tolist(),
                }
            n_rows += len(group)

    elapsed = time.time() - t0
    print(f"  Loaded {n_rows:,} code rows for {len(code_by_sid)} subjects in {elapsed:.1f}s",
          flush=True)
    return code_by_sid


# ---------------------------------------------------------------------------
# Fast in-memory retrieval
# ---------------------------------------------------------------------------

def retrieve_notes_inmemory(note_data, query_vectors, k):
    """Retrieve top-K note chunks per query using preloaded data.

    query_vectors: np.ndarray (Q, d), already L2-normalized
    note_data: dict with 'vectors' (N, d) normalized, 'chunks', 'days_to_prediction', 'note_ids'

    Returns deduplicated list of dicts sorted by days_to_prediction desc (oldest first).
    """
    if note_data is None:
        return []

    patient_vecs = note_data["vectors"]  # (N, d), already normalized
    # Cosine similarity = dot product of normalized vectors
    # sim: (Q, N)
    sim = query_vectors @ patient_vecs.T

    seen_indices = set()
    results = []

    for q_idx in range(sim.shape[0]):
        scores = sim[q_idx]
        # Top-K indices for this query
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        for idx in top_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            results.append({
                "chunk": note_data["chunks"][idx],
                "days_to_prediction": int(note_data["days_to_prediction"][idx]),
                "note_id": note_data["note_ids"][idx],
            })

    # Sort oldest first (highest days_to_prediction)
    results.sort(key=lambda x: x["days_to_prediction"], reverse=True)
    return results


def retrieve_codes_inmemory(code_data, query_vectors, k):
    """Retrieve top-K ICD codes per query using preloaded data.

    Returns deduplicated list of dicts.
    """
    if code_data is None:
        return []

    patient_vecs = code_data["vectors"]  # (N, d), already normalized
    sim = query_vectors @ patient_vecs.T  # (Q, N)

    seen_indices = set()
    results = []

    for q_idx in range(sim.shape[0]):
        scores = sim[q_idx]
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        for idx in top_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            results.append({
                "long_title": code_data["long_titles"][idx],
                "icd_code_ver": code_data["icd_code_vers"][idx],
            })

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_patient_text(subject_id, note_results, code_results):
    """Format retrieved evidence into a patient_text string."""
    lines = [f"Patient ID: {subject_id}"]

    # Notes
    n_notes = len(note_results)
    lines.append("")
    lines.append(f"===== CLINICAL NOTES (retrieved, N={n_notes}) =====")
    for i, note in enumerate(note_results, start=1):
        lines.append(
            f"[Note {i}] (note_id: {note['note_id']}) "
            f"days_to_prediction: {note['days_to_prediction']}"
        )
        lines.append(note["chunk"])
        lines.append("")

    # Codes
    n_codes = len(code_results)
    lines.append(f"===== DIAGNOSTIC CODES (retrieved, N={n_codes}) =====")
    for code in code_results:
        lines.append(
            f"[Code] (ICD code: {code['icd_code_ver']}) {code['long_title']}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build RAG baseline dataset: embed queries, retrieve from LanceDB"
    )
    parser.add_argument("--db-dir", type=str, default="/hpc/dctrl/fl105/EviGen",
                        help="LanceDB directory")
    parser.add_argument("--note-table", type=str, default="death_age_chunksize200_qwen")
    parser.add_argument("--code-table", type=str, default="death_age_codes_qwen")
    parser.add_argument("--full_context-jsonl", type=str,
                        default="data/full_context_test_patients.jsonl",
                        help="Full-context JSONL to get subject_ids, labels, and approx_tokens "
                             "(ignored when --splits is given)")
    parser.add_argument("--output", type=str, default="data/rag_test_patients.jsonl",
                        help="Output JSONL path (ignored when --splits is given)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated list of splits (train,val,test) to build "
                             "in one shot, reusing a single LanceDB preload. "
                             "Inputs derived from data/full_context_{split}_patients.jsonl, "
                             "outputs written to data/rag_{split}_patients.jsonl. "
                             "Overrides --full_context-jsonl/--output when given.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory to look for full_context JSONLs and write RAG JSONLs "
                             "when --splits is given")
    parser.add_argument("--k-per-query", type=int, default=4,
                        help="Top-K chunks to retrieve per query (default: 4)")
    args = parser.parse_args()

    # Assemble list of (split_name, full_context_jsonl, output_path) jobs
    if args.splits:
        split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
        valid = {"train", "val", "test"}
        bad = [s for s in split_list if s not in valid]
        if bad:
            raise ValueError(f"Invalid split names in --splits: {bad}; must be in {valid}")
        data_dir = Path(args.data_dir)
        split_jobs = [
            (s,
             data_dir / f"full_context_{s}_patients.jsonl",
             data_dir / f"rag_{s}_patients.jsonl")
            for s in split_list
        ]
    else:
        split_jobs = [(None, Path(args.full_context_jsonl), Path(args.output))]

    # Ensure all output directories exist early
    for _, _, out_path in split_jobs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load patient rosters for every requested split
    split_patients = []  # list of (split_name, patients_list, out_path)
    all_sids = set()
    for split_name, zs_path, out_path in split_jobs:
        tag = split_name or "default"
        print(f"Loading subjects from {zs_path} (split={tag}) ...", flush=True)
        patients = []
        with open(zs_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                patients.append({
                    "subject_id": int(rec["subject_id"]),
                    "label": float(rec["label"]),
                    "full_context_approx_tokens": int(rec.get("approx_tokens", 0)),
                })
        sids = [p["subject_id"] for p in patients]
        all_sids.update(sids)
        split_patients.append((split_name, patients, out_path))
        print(f"  {len(patients)} subjects loaded for split={tag}", flush=True)

    test_sids = sorted(all_sids)  # union for preload
    print(f"Total unique subjects across splits: {len(test_sids)}", flush=True)

    # 2. Load embedding model and embed queries
    model = load_embedding_model()

    all_queries = NOTE_QUERIES + CODE_QUERIES
    print(f"Embedding {len(all_queries)} queries ...", flush=True)
    query_embeddings = model.encode(all_queries, show_progress_bar=False)
    # L2 normalize query embeddings
    qnorms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    qnorms = np.maximum(qnorms, 1e-8)
    query_embeddings = query_embeddings / qnorms

    note_query_vecs = query_embeddings[: len(NOTE_QUERIES)]  # (4, 4096)
    code_query_vecs = query_embeddings[len(NOTE_QUERIES) :]  # (4, 4096)
    print(f"  Query embeddings shape: {query_embeddings.shape}", flush=True)

    # Free GPU memory — we don't need the model anymore
    del model
    torch.cuda.empty_cache()
    print("Embedding model freed from GPU", flush=True)

    # 3. Preload all note and code data for test subjects into memory
    note_by_sid = preload_note_data(args.db_dir, args.note_table, test_sids)
    code_by_sid = preload_code_data(args.db_dir, args.code_table, test_sids)

    # 4. Retrieve for each patient, per requested split
    k = args.k_per_query

    for split_name, patients, out_path in split_patients:
        tag = split_name or "default"
        print(f"\nRetrieving top-{k} per query for "
              f"{len(patients)} patients (split={tag}) -> {out_path}",
              flush=True)
        t0 = time.time()

        n_written = 0
        total_note_chunks = 0
        total_code_items = 0
        n_no_notes = 0
        n_no_codes = 0

        with out_path.open("w", encoding="utf-8") as f:
            for i, pat in enumerate(patients):
                sid = pat["subject_id"]

                note_data = note_by_sid.get(sid)
                code_data = code_by_sid.get(sid)

                if note_data is None:
                    n_no_notes += 1
                if code_data is None:
                    n_no_codes += 1

                note_results = retrieve_notes_inmemory(note_data, note_query_vecs, k)
                code_results = retrieve_codes_inmemory(code_data, code_query_vecs, k)

                patient_text = format_patient_text(sid, note_results, code_results)
                approx_tokens = int(len(patient_text.split()) * 1.3)

                record = {
                    "subject_id": sid,
                    "patient_text": patient_text,
                    "label": pat["label"],
                    "num_notes": len(note_results),
                    "num_codes": len(code_results),
                    "approx_tokens": approx_tokens,
                    "full_context_approx_tokens": pat["full_context_approx_tokens"],
                    "k_per_query": k,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                total_note_chunks += len(note_results)
                total_code_items += len(code_results)

                if (i + 1) % 500 == 0 or (i + 1) == len(patients):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(
                        f"  [{tag}][{i+1}/{len(patients)}] {elapsed:.1f}s "
                        f"({rate:.1f} patients/s)",
                        flush=True,
                    )

        elapsed_total = time.time() - t0
        avg_notes = total_note_chunks / max(1, n_written)
        avg_codes = total_code_items / max(1, n_written)

        print(f"[{tag}] Done in {elapsed_total:.1f}s", flush=True)
        print(f"[{tag}] Wrote {n_written} patients to {out_path}", flush=True)
        print(f"[{tag}] Avg note chunks per patient: {avg_notes:.1f}", flush=True)
        print(f"[{tag}] Avg codes per patient: {avg_codes:.1f}", flush=True)
        if n_no_notes > 0:
            print(f"[{tag}] WARNING: {n_no_notes} patients had no notes in LanceDB",
                  flush=True)
        if n_no_codes > 0:
            print(f"[{tag}] WARNING: {n_no_codes} patients had no codes in LanceDB",
                  flush=True)

        # Sanity stats
        token_counts = []
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                token_counts.append(json.loads(line)["approx_tokens"])
        s = pd.Series(token_counts)
        print(
            f"[{tag}] approx_tokens stats: min={s.min()} median={int(s.median())} "
            f"p95={int(s.quantile(0.95))} max={s.max()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
