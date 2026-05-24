"""Paired bootstrap CIs on AUC/accuracy across all methods + precision/recall.

Loads per-subject predictions for the five methods reported in WORKLOG
(4 full_context variants + EviGen), aligns them on subject_id over the
canonical 1,340-patient test split, and computes:

- Point estimates per method: AUC (parsed-only and with fallback), accuracy,
  precision, recall, F1, confusion matrix.
- Paired bootstrap 95% CIs on AUC, accuracy, precision, recall, and F1
  (N=1000, seed=42). Paired means the same resampled patient indices are used
  for every method in each iteration, so CIs and pairwise differences are
  directly comparable.
- Pairwise P(EviGen_AUC > Method_AUC) across bootstrap iterations and a
  95% CI on the AUC gap — the empirical one-sided significance test for the
  headline comparison in the paper.

For full_context methods, unparsed predictions (only 8B free-form has any) are
assigned fallback_prob=0.5 so all methods share the same 1,340 paired
subjects. This matches compute_metrics.py's auc_all_with_fallback.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

METHODS = [
    ("full_context_8b_freeform",
     "outputs/full_context/predictions_full.jsonl",
     "parsed_probability"),
    ("full_context_70b_freeform",
     "outputs/full_context/predictions_70b_full.jsonl",
     "parsed_probability"),
    ("full_context_8b_verbalizer",
     "outputs/full_context/predictions_verbalizer_full.jsonl",
     "parsed_probability"),
    ("full_context_70b_verbalizer",
     "outputs/full_context/predictions_verbalizer_70b_full.jsonl",
     "parsed_probability"),
    ("rag_8b_freeform",
     "outputs/rag/predictions_full.jsonl",
     "parsed_probability"),
    ("rag_70b_freeform",
     "outputs/rag/predictions_70b_full.jsonl",
     "parsed_probability"),
    ("rag_8b_verbalizer",
     "outputs/rag/predictions_verbalizer_full.jsonl",
     "parsed_probability"),
    ("rag_70b_verbalizer",
     "outputs/rag/predictions_verbalizer_70b_full.jsonl",
     "parsed_probability"),
    # gpt-4o-mini variants (May 2026)
    ("full_context_mini_freeform",
     "outputs/full_context/predictions_mini_full.jsonl",
     "parsed_probability"),
    ("full_context_mini_verbalizer",
     "outputs/full_context/predictions_verbalizer_mini_full.jsonl",
     "parsed_probability"),
    ("rag_mini_freeform",
     "outputs/rag/predictions_mini_full.jsonl",
     "parsed_probability"),
    ("rag_mini_verbalizer",
     "outputs/rag/predictions_verbalizer_mini_full.jsonl",
     "parsed_probability"),
    # Qwen3-32B variants (free-form runs in thinking mode; verbalizer
    # runs with thinking disabled so position-0 = Yes/No).
    ("full_context_qwen3_32b_freeform",
     "outputs/full_context/predictions_qwen3_32b_full.jsonl",
     "parsed_probability"),
    ("full_context_qwen3_32b_verbalizer",
     "outputs/full_context/predictions_verbalizer_qwen3_32b_full.jsonl",
     "parsed_probability"),
    ("rag_qwen3_32b_freeform",
     "outputs/rag/predictions_qwen3_32b_full.jsonl",
     "parsed_probability"),
    ("rag_qwen3_32b_verbalizer",
     "outputs/rag/predictions_verbalizer_qwen3_32b_full.jsonl",
     "parsed_probability"),
    ("evigen",
     "outputs/full_context/evigen_test_predictions.jsonl",
     "probability"),
    # New learned baselines (April 2026)
    ("static_iris",
     "outputs/baselines/static_iris/test_predictions.jsonl",
     "probability"),
    ("meanpool",
     "outputs/baselines/meanpool/test_predictions.jsonl",
     "probability"),
    ("mil_abmil",
     "outputs/baselines/mil_abmil/test_predictions.jsonl",
     "probability"),
    ("modernbert_base",
     "outputs/baselines/modernbert_base/test_predictions.jsonl",
     "probability"),
    ("longformer_base",
     "outputs/baselines/longformer_base/test_predictions.jsonl",
     "probability"),
    # QLoRA-verbalizer fine-tunes (April 2026)
    ("qlora_rag_8b",
     "outputs/baselines/qlora_rag_8b/test_predictions.jsonl",
     "parsed_probability"),
    ("qlora_full_8b",
     "outputs/baselines/qlora_full_8b/test_predictions.jsonl",
     "parsed_probability"),
]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str,
        default="/hpc/group/engelhardlab/fl105/IRIS-extended/EviGen-DynamicQuery",
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/full_context/bootstrap_metrics_with_rag.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-prob", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root)

    # ---- Load all methods ----
    method_probs_by_subj = {}   # name -> {sid: prob or None}
    label_by_subject = {}       # sid -> int label (filled from first method that has it)
    for name, rel_path, prob_key in METHODS:
        rows = load_jsonl(root / rel_path)
        probs = {}
        for r in rows:
            sid = int(r["subject_id"])
            p = r.get(prob_key)
            probs[sid] = p
            lbl = int(r["label"])
            if sid in label_by_subject:
                if label_by_subject[sid] != lbl:
                    raise RuntimeError(
                        f"Label mismatch for subject {sid} in {name}: "
                        f"{lbl} vs existing {label_by_subject[sid]}"
                    )
            else:
                label_by_subject[sid] = lbl
        method_probs_by_subj[name] = probs
        print(f"Loaded {len(rows)} rows from {name} ({rel_path})", flush=True)

    # ---- Align on common subject_ids ----
    method_names = [m[0] for m in METHODS]
    common = set(label_by_subject.keys())
    for name in method_names:
        common &= set(method_probs_by_subj[name].keys())
    subject_ids = sorted(common)
    n = len(subject_ids)
    for name in method_names:
        ni = len(method_probs_by_subj[name])
        if ni != n:
            print(
                f"WARNING: {name} has {ni} subjects, common set is {n}",
                flush=True,
            )
    print(f"Common test subjects: {n}", flush=True)

    labels = np.array([label_by_subject[s] for s in subject_ids], dtype=int)

    # Build (n, n_methods) probability matrix + parsed mask (False where fallback was used)
    n_methods = len(method_names)
    probs_mat = np.zeros((n, n_methods), dtype=float)
    parsed_mask = np.ones((n, n_methods), dtype=bool)
    for j, name in enumerate(method_names):
        pdict = method_probs_by_subj[name]
        for i, sid in enumerate(subject_ids):
            p = pdict[sid]
            if p is None:
                probs_mat[i, j] = args.fallback_prob
                parsed_mask[i, j] = False
            else:
                probs_mat[i, j] = float(p)

    # ---- Point estimates per method ----
    point = {}
    for j, name in enumerate(method_names):
        probs_j = probs_mat[:, j]
        parsed_j = parsed_mask[:, j]

        if parsed_j.sum() >= 2 and len(np.unique(labels[parsed_j])) >= 2:
            auc_parsed = float(roc_auc_score(labels[parsed_j], probs_j[parsed_j]))
        else:
            auc_parsed = None
        auc_all = float(roc_auc_score(labels, probs_j))

        preds_j = (probs_j >= args.threshold).astype(int)
        acc = float(accuracy_score(labels, preds_j))
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds_j, average="binary", zero_division=0,
        )
        cm = confusion_matrix(labels, preds_j, labels=[0, 1]).tolist()

        point[name] = {
            "auc_parsed_only": auc_parsed,
            "auc_all_with_fallback": auc_all,
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix_0_1": cm,
            "n_parsed": int(parsed_j.sum()),
            "n_total": n,
        }

    # ---- Paired bootstrap ----
    rng = np.random.default_rng(args.seed)
    n_boot = args.n_bootstrap
    auc_boot = np.full((n_boot, n_methods), np.nan, dtype=float)
    acc_boot = np.full((n_boot, n_methods), np.nan, dtype=float)
    prec_boot = np.full((n_boot, n_methods), np.nan, dtype=float)
    rec_boot = np.full((n_boot, n_methods), np.nan, dtype=float)
    f1_boot = np.full((n_boot, n_methods), np.nan, dtype=float)

    n_skipped = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            n_skipped += 1
            continue
        preds_all = (probs_mat[idx] >= args.threshold).astype(int)
        for j in range(n_methods):
            auc_boot[b, j] = roc_auc_score(y, probs_mat[idx, j])
            acc_boot[b, j] = accuracy_score(y, preds_all[:, j])
            p, r, f, _ = precision_recall_fscore_support(
                y, preds_all[:, j], average="binary", zero_division=0,
            )
            prec_boot[b, j] = p
            rec_boot[b, j] = r
            f1_boot[b, j] = f

    if n_skipped:
        print(
            f"Skipped {n_skipped} bootstrap iterations (single-class resample)",
            flush=True,
        )

    def _ci(arr_j):
        return {
            "mean": float(arr_j.mean()),
            "ci95_lo": float(np.percentile(arr_j, 2.5)),
            "ci95_hi": float(np.percentile(arr_j, 97.5)),
        }

    ci = {}
    for j, name in enumerate(method_names):
        mask = ~np.isnan(auc_boot[:, j])
        ci[name] = {
            "auc": _ci(auc_boot[mask, j]),
            "accuracy": _ci(acc_boot[mask, j]),
            "precision": _ci(prec_boot[mask, j]),
            "recall": _ci(rec_boot[mask, j]),
            "f1": _ci(f1_boot[mask, j]),
            "n_valid_boot": int(mask.sum()),
        }

    # ---- Pairwise significance: EviGen vs each other method ----
    di_idx = method_names.index("evigen")
    di_vs_others = {}
    for j, name in enumerate(method_names):
        if j == di_idx:
            continue
        valid = ~np.isnan(auc_boot[:, di_idx]) & ~np.isnan(auc_boot[:, j])
        if valid.sum() == 0:
            di_vs_others[name] = {
                "p_di_gt": None,
                "auc_gap_mean": None,
                "auc_gap_ci95_lo": None,
                "auc_gap_ci95_hi": None,
                "n_valid": 0,
            }
            continue
        diff = auc_boot[valid, di_idx] - auc_boot[valid, j]
        di_vs_others[name] = {
            "p_di_gt": float((diff > 0).mean()),
            "auc_gap_mean": float(diff.mean()),
            "auc_gap_ci95_lo": float(np.percentile(diff, 2.5)),
            "auc_gap_ci95_hi": float(np.percentile(diff, 97.5)),
            "n_valid": int(valid.sum()),
        }

    out = {
        "n_subjects": n,
        "label_distribution": {
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
        },
        "n_bootstrap": n_boot,
        "n_bootstrap_skipped": n_skipped,
        "seed": args.seed,
        "fallback_prob": args.fallback_prob,
        "threshold": args.threshold,
        "methods": method_names,
        "point_estimates": point,
        "bootstrap_ci95": ci,
        "di_vs_others": di_vs_others,
    }

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
