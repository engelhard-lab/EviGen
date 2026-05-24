"""Compute AUC and accuracy for the zeroshot baseline predictions.

Reads outputs/zeroshot/predictions*.jsonl, extracts parsed probabilities
against true labels, and writes outputs/zeroshot/metrics*.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Predictions JSONL from run_zeroshot.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write metrics JSON")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fallback-prob", type=float, default=0.5,
                        help="Probability to assign when parsing fails; keeps patient "
                             "in the evaluation instead of dropping it.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    n_total = len(rows)
    n_parsed = sum(1 for r in rows if r.get("parsed_probability") is not None)
    n_parse_ok_json = sum(1 for r in rows if r.get("parse_ok"))
    n_truncated = sum(1 for r in rows if r.get("truncated"))

    labels = np.array([float(r["label"]) for r in rows])
    probs_parsed_mask = np.array(
        [r.get("parsed_probability") is not None for r in rows]
    )
    probs = np.array([
        float(r["parsed_probability"]) if r.get("parsed_probability") is not None
        else args.fallback_prob
        for r in rows
    ])

    # Metrics on parsed-only subset
    if probs_parsed_mask.sum() > 0 and len(np.unique(labels[probs_parsed_mask])) >= 2:
        auc_parsed = float(roc_auc_score(labels[probs_parsed_mask], probs[probs_parsed_mask]))
    else:
        auc_parsed = None

    # Metrics on full set (with fallback for unparsed)
    if len(np.unique(labels)) >= 2:
        auc_all = float(roc_auc_score(labels, probs))
    else:
        auc_all = None

    preds = (probs >= args.threshold).astype(int)
    acc_all = float(accuracy_score(labels.astype(int), preds))
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels.astype(int), preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels.astype(int), preds, labels=[0, 1]).tolist()

    metrics = {
        "input": str(in_path),
        "n_total": n_total,
        "n_parsed": int(n_parsed),
        "n_parse_ok_json": int(n_parse_ok_json),
        "n_truncated": int(n_truncated),
        "parse_failure_rate": (n_total - n_parsed) / n_total if n_total else None,
        "threshold": args.threshold,
        "fallback_prob": args.fallback_prob,
        "label_distribution": {
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
        },
        "auc_parsed_only": auc_parsed,
        "auc_all_with_fallback": auc_all,
        "accuracy_all": acc_all,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix_labels_0_1": cm,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
