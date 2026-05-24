# data.py
"""
Data loading module for EviGen-DynamicQuery.
Loads embeddings from Parquet files for memory-efficient training.
"""
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


class SubjectDataset(Dataset):
    """
    Dataset of (subject_id, label).
    Embeddings are stored globally on CPU and looked up in the collate_fn
    to build per-patient note/code sequences.
    """
    def __init__(self, subject_ids, labels):
        self.subject_ids = [int(s) for s in subject_ids]
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx: int):
        return int(self.subject_ids[idx]), float(self.labels[idx])


def collate_notes_codes_batch(
    note_chunk_list: List[torch.Tensor],
    code_chunk_list: List[torch.Tensor],
    labels_list: List[float],
):
    """
    Pad note and code sequences for each patient to the maximum length in the batch.

    note_chunk_list: list of [L_n_i, d]
    code_chunk_list: list of [L_c_i, d]

    Returns:
      note_chunks: [B, L_n_max, d]
      note_mask:   [B, L_n_max] bool
      code_chunks: [B, L_c_max, d]
      code_mask:   [B, L_c_max] bool
      labels:      [B]
    """
    batch_size = len(note_chunk_list)
    assert batch_size == len(code_chunk_list) == len(labels_list)

    emb_dim = note_chunk_list[0].size(1)

    # Notes
    max_len_notes = max(x.size(0) for x in note_chunk_list)
    note_chunks = torch.zeros(batch_size, max_len_notes, emb_dim, dtype=torch.float32)
    note_mask = torch.zeros(batch_size, max_len_notes, dtype=torch.bool)

    for i, x in enumerate(note_chunk_list):
        L_n = x.size(0)
        note_chunks[i, :L_n, :] = x
        note_mask[i, :L_n] = True

    # Codes
    max_len_codes = max(x.size(0) for x in code_chunk_list)
    code_chunks = torch.zeros(batch_size, max_len_codes, emb_dim, dtype=torch.float32)
    code_mask = torch.zeros(batch_size, max_len_codes, dtype=torch.bool)

    for i, x in enumerate(code_chunk_list):
        L_c = x.size(0)
        code_chunks[i, :L_c, :] = x
        code_mask[i, :L_c] = True

    labels = torch.tensor(labels_list, dtype=torch.float32)
    return note_chunks, note_mask, code_chunks, code_mask, labels


def make_collate_fn(
    icd_vectors_cpu: torch.Tensor,
    note_vectors_cpu: torch.Tensor,
    icd_subject_to_indices: Dict[int, List[int]],
    note_subject_to_indices: Dict[int, List[int]],
    emb_dim: int,
):
    """
    Build a collate_fn that:
      - takes a batch of (subject_id, label)
      - looks up ICD + note embeddings for each subject
      - builds separate note chunks and code chunks
      - pads to form:
          note_chunks: [B, L_n_max, d], note_mask: [B, L_n_max]
          code_chunks: [B, L_c_max, d], code_mask: [B, L_c_max]
    """
    def collate_fn(batch: List[Tuple[int, float]]):
        subject_ids, labels = zip(*batch)
        note_chunk_list: List[torch.Tensor] = []
        code_chunk_list: List[torch.Tensor] = []

        for sid in subject_ids:
            sid = int(sid)

            note_idxs = note_subject_to_indices.get(sid, [])
            icd_idxs = icd_subject_to_indices.get(sid, [])

            # Notes
            if len(note_idxs) > 0:
                note_chunks = note_vectors_cpu[note_idxs]
            else:
                # Preserve an empty sequence so downstream retrieval can treat this
                # subject as having no valid chunks instead of a fake all-zero chunk.
                note_chunks = torch.zeros(0, emb_dim, dtype=torch.float32)

            # Codes
            if len(icd_idxs) > 0:
                code_chunks = icd_vectors_cpu[icd_idxs]
            else:
                code_chunks = torch.zeros(0, emb_dim, dtype=torch.float32)

            note_chunk_list.append(note_chunks)
            code_chunk_list.append(code_chunks)

        return collate_notes_codes_batch(
            note_chunk_list, code_chunk_list, list(labels)
        )

    return collate_fn


def build_subject_to_indices(subject_ids: np.ndarray) -> Dict[int, List[int]]:
    """
    Build a mapping from subject_id -> list of row indices in the embedding array.
    """
    mapping: Dict[int, List[int]] = {}
    for row_idx, sid in enumerate(subject_ids):
        sid_int = int(sid)
        if sid_int not in mapping:
            mapping[sid_int] = []
        mapping[sid_int].append(row_idx)
    return mapping


def load_embeddings_and_mappings(cfg):
    """
    Load note + code embeddings from Parquet files into CPU tensors.

    This function loads embeddings in a memory-efficient way by:
    1. Reading parquet files directly into pandas
    2. Extracting vectors into numpy arrays
    3. Immediately deleting DataFrames to free memory
    4. Building subject_id -> row indices mappings for batch assembly

    Args:
        cfg: Config object with note_parquet, code_parquet, emb_dim fields

    Returns:
        note_labels_df: DataFrame with subject_id and label columns
        icd_vectors_cpu: Tensor of code embeddings [N_codes, emb_dim]
        note_vectors_cpu: Tensor of note embeddings [N_notes, emb_dim]
        icd_subject_to_indices: Dict mapping subject_id -> list of code indices
        note_subject_to_indices: Dict mapping subject_id -> list of note indices
        icd_keep_indices: np.ndarray of original parquet row indices kept after
            deduplicating (subject_id, icd_code_ver). Used by explain.py to align
            code_meta_df with the (deduped) icd_vectors_cpu tensor. If no code
            identifier column is found, this is np.arange(N_original).
    """
    import pyarrow.parquet as pq

    print("Loading embeddings from Parquet files to CPU memory...")
    print("=" * 80)

    # --- ICD code embeddings ---
    print(f"Loading ICD codes from: {cfg.code_parquet}")
    # Read in batches to avoid PyArrow int32 list offset overflow
    # (vector column with 4096 floats × many rows can exceed 2^31 elements)
    pf = pq.ParquetFile(cfg.code_parquet)
    print(f"Code table columns: {pf.schema.names}")

    # Detect a code-identifier column for deduplication
    code_id_col = next(
        (c for c in ["icd_code_ver", "code_ver", "icd_code"] if c in pf.schema.names),
        None,
    )
    read_cols = ["subject_id", "vector"] + ([code_id_col] if code_id_col else [])

    icd_sid_parts, icd_vec_parts, icd_ver_parts = [], [], []
    for batch in pf.iter_batches(batch_size=50_000, columns=read_cols):
        batch_df = batch.to_pandas()
        icd_sid_parts.append(batch_df["subject_id"].values)
        icd_vec_parts.append(np.vstack(batch_df["vector"].values).astype("float32"))
        if code_id_col:
            icd_ver_parts.append(batch_df[code_id_col].astype(str).values)
        del batch_df
    icd_subject_ids = np.concatenate(icd_sid_parts)
    icd_vectors = np.vstack(icd_vec_parts)
    icd_code_vers = np.concatenate(icd_ver_parts) if code_id_col else None
    del icd_sid_parts, icd_vec_parts, icd_ver_parts

    print(f"ICD codes: {len(icd_subject_ids)} embeddings, shape: {icd_vectors.shape}")

    # Deduplicate: keep first occurrence of each (subject_id, icd_code_ver) pair.
    # Rationale: the code parquet has one row per diagnosis event, so a patient
    # with the same ICD code recorded on multiple encounters has duplicate rows
    # with identical embeddings, which inflates retrieval and produces duplicate
    # items in explanations.
    if code_id_col is not None:
        n_before = len(icd_subject_ids)
        dedup_df = pd.DataFrame({"sid": icd_subject_ids, "ver": icd_code_vers})
        keep_mask = ~dedup_df.duplicated(subset=["sid", "ver"], keep="first")
        icd_keep_indices = np.where(keep_mask.values)[0]
        del dedup_df, keep_mask, icd_code_vers

        icd_subject_ids = icd_subject_ids[icd_keep_indices]
        icd_vectors = icd_vectors[icd_keep_indices]
        n_after = len(icd_subject_ids)
        print(
            f"Deduplicated ICD rows by (subject_id, {code_id_col}): "
            f"{n_before} -> {n_after} ({n_before - n_after} duplicates removed)"
        )
    else:
        icd_keep_indices = np.arange(len(icd_subject_ids))
        print(
            "No ICD code identifier column found (looked for icd_code_ver / "
            "code_ver / icd_code); skipping deduplication."
        )

    import gc; gc.collect()

    # Check for NaN in ICD vectors
    icd_has_nan = np.isnan(icd_vectors).any()
    if icd_has_nan:
        icd_nan_count = np.isnan(icd_vectors).sum()
        icd_nan_rows = np.where(np.isnan(icd_vectors).any(axis=1))[0]
        print(f"⚠️ WARNING: ICD vectors contain {icd_nan_count} NaN values!")
        print(f"   Rows with NaN: {len(icd_nan_rows)}/{len(icd_vectors)}")
        if len(icd_nan_rows) > 0:
            print(f"   Example row indices with NaN: {icd_nan_rows[:10]}")
        # Replace NaN with zeros
        icd_vectors = np.nan_to_num(icd_vectors, nan=0.0)
        print("   ✓ Replaced all NaN values in ICD vectors with 0.0")
    else:
        print("✓ ICD vectors: No NaN values")

    # --- Note embeddings ---
    print(f"\nLoading notes from: {cfg.note_parquet}")
    pf = pq.ParquetFile(cfg.note_parquet)
    print(f"Note table columns: {pf.schema.names}")
    note_sid_parts, note_vec_parts, label_parts = [], [], []
    for batch in pf.iter_batches(batch_size=50_000, columns=["subject_id", "vector", "label"]):
        batch_df = batch.to_pandas()
        note_sid_parts.append(batch_df["subject_id"].values)
        note_vec_parts.append(np.vstack(batch_df["vector"].values).astype("float32"))
        label_parts.append(batch_df[["subject_id", "label"]])
        del batch_df
    note_subject_ids = np.concatenate(note_sid_parts)
    note_vectors = np.vstack(note_vec_parts)
    note_labels_df = pd.concat(label_parts, ignore_index=True).drop_duplicates()
    del note_sid_parts, note_vec_parts, label_parts

    print(f"\nNotes: {len(note_subject_ids)} embeddings, shape: {note_vectors.shape}")

    gc.collect()

    note_has_nan = np.isnan(note_vectors).any()
    if note_has_nan:
        note_nan_count = np.isnan(note_vectors).sum()
        note_nan_rows = np.where(np.isnan(note_vectors).any(axis=1))[0]
        print(f"⚠️ WARNING: Note vectors contain {note_nan_count} NaN values!")
        print(f"   Rows with NaN: {len(note_nan_rows)}/{len(note_vectors)}")
        if len(note_nan_rows) > 0:
            print(f"   Example row indices with NaN: {note_nan_rows[:10]}")
        note_vectors = np.nan_to_num(note_vectors, nan=0.0)
        print("   ✓ Replaced all NaN values in note vectors with 0.0")
    else:
        print("✓ Note vectors: No NaN values")

    # Convert to CPU tensors
    icd_vectors_cpu = torch.from_numpy(icd_vectors)
    note_vectors_cpu = torch.from_numpy(note_vectors)

    print(f"\nCPU Memory usage estimate:")
    icd_gb = icd_vectors_cpu.element_size() * icd_vectors_cpu.nelement() / 1024 ** 3
    note_gb = note_vectors_cpu.element_size() * note_vectors_cpu.nelement() / 1024 ** 3
    print(f"  ICD vectors:  {icd_gb:.2f} GB")
    print(f"  Note vectors: {note_gb:.2f} GB")

    print("\nFinal NaN verification:")
    print(f"  ICD tensors have NaN: {torch.isnan(icd_vectors_cpu).any().item()}")
    print(f"  Note tensors have NaN: {torch.isnan(note_vectors_cpu).any().item()}")

    # Build subject_id -> indices mappings
    icd_subject_to_indices = build_subject_to_indices(icd_subject_ids)
    note_subject_to_indices = build_subject_to_indices(note_subject_ids)

    print("=" * 80)
    print("✓ All embeddings loaded to CPU memory!")
    print("  (Will be transferred to GPU batch-by-batch during training)")

    return (
        note_labels_df,
        icd_vectors_cpu,
        note_vectors_cpu,
        icd_subject_to_indices,
        note_subject_to_indices,
        icd_keep_indices,
    )


def create_subject_splits(note_labels_df, cfg):
    """
    Create train/val/test splits of subject_ids and labels.

    Args:
        note_labels_df: DataFrame with columns ['subject_id', 'label']
        cfg: Config object with train_set_ratio and random_state
    """
    patient_data = note_labels_df.drop_duplicates()

    train_df, temp_df = train_test_split(
        patient_data,
        test_size=1 - cfg.train_set_ratio,
        random_state=cfg.random_state,
        stratify=patient_data["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=cfg.random_state,
        stratify=temp_df["label"],
    )

    print(f"Size of training set:   {len(train_df)}")
    print(f"Size of validation set: {len(val_df)}")
    print(f"Size of testing set:    {len(test_df)}")

    return train_df, val_df, test_df
