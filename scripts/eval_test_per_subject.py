"""Run a EviGen checkpoint over the test split and emit per-subject
probabilities to JSONL, so we can join with the zeroshot predictions and
plot per-bin AUC/Acc.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evigen_dynamicquery.config import load_config_from_yaml, set_seed
from evigen_dynamicquery.data import (
    create_subject_splits,
    load_embeddings_and_mappings,
    make_collate_fn,
    SubjectDataset,
)
from evigen_dynamicquery.model_evigen import EviGen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--subjects-jsonl", default=None,
                    help="If set, evaluate only subjects (and use labels) from this JSONL "
                         "(expects keys subject_id and label). Otherwise uses the canonical "
                         "create_subject_splits test set.")
    args = ap.parse_args()

    cfg = load_config_from_yaml(args.config)
    set_seed(cfg.seed)

    (
        note_labels_df,
        icd_vectors_cpu,
        note_vectors_cpu,
        icd_subject_to_indices,
        note_subject_to_indices,
        _,
    ) = load_embeddings_and_mappings(cfg)

    if args.subjects_jsonl:
        rows = [json.loads(l) for l in open(args.subjects_jsonl)]
        test_subjects = np.array([int(r["subject_id"]) for r in rows])
        test_labels = np.array([float(r["label"]) for r in rows])
        print(f"Test set (from {args.subjects_jsonl}): {len(test_subjects)} subjects",
              flush=True)
    else:
        _, _, test_df = create_subject_splits(note_labels_df, cfg)
        test_subjects = test_df["subject_id"].values
        test_labels = test_df["label"].values
        print(f"Test set: {len(test_subjects)} subjects", flush=True)

    test_ds = SubjectDataset(test_subjects, test_labels)
    collate_fn = make_collate_fn(
        icd_vectors_cpu=icd_vectors_cpu,
        note_vectors_cpu=note_vectors_cpu,
        icd_subject_to_indices=icd_subject_to_indices,
        note_subject_to_indices=note_subject_to_indices,
        emb_dim=cfg.emb_dim,
    )
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    model = EviGen(cfg)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(cfg.device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for note_chunks, note_mask, code_chunks, code_mask, labels in loader:
            note_chunks = note_chunks.to(cfg.device)
            note_mask = note_mask.to(cfg.device)
            code_chunks = code_chunks.to(cfg.device)
            code_mask = code_mask.to(cfg.device)
            logits, *_ = model(
                note_chunks, note_mask, code_chunks, code_mask,
                temperature=cfg.final_temperature, use_sampling_retrieval=False,
            )
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits))
    assert len(probs) == len(test_subjects)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for sid, lab, p in zip(test_subjects, labels, probs):
            f.write(json.dumps({
                "subject_id": int(sid),
                "label": int(lab),
                "probability": float(p),
            }) + "\n")
    print(f"Wrote {len(probs)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
