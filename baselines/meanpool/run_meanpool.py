"""Mean-pool baseline for one-year mortality prediction.

Per-patient pooling of Qwen-8B note/code embeddings:
  - mean(note chunks with mask) -> [B, d]
  - mean(code embeddings with mask) -> [B, d]
  - concat -> 2-layer MLP -> binary logit

Serves as a floor baseline: does the attention/retrieval machinery in
EviGen help beyond simply averaging Qwen embeddings?
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evigen_dynamicquery.config import load_config_from_yaml, set_seed
from evigen_dynamicquery.data import (
    SubjectDataset,
    create_subject_splits,
    load_embeddings_and_mappings,
    make_collate_fn,
)


class MeanPoolClassifier(nn.Module):
    def __init__(self, emb_dim: int = 4096, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, note_chunks, note_mask, code_chunks, code_mask):
        # note_chunks: [B, L_n, d], note_mask: [B, L_n] (bool)
        # code_chunks: [B, L_c, d], code_mask: [B, L_c] (bool)
        nm = note_mask.float().unsqueeze(-1)
        cm = code_mask.float().unsqueeze(-1)
        nlen = nm.sum(dim=1).clamp(min=1.0)
        clen = cm.sum(dim=1).clamp(min=1.0)
        note_mean = (note_chunks * nm).sum(dim=1) / nlen
        code_mean = (code_chunks * cm).sum(dim=1) / clen
        z = torch.cat([note_mean, code_mean], dim=-1)
        return self.head(z).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_sids = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for note_chunks, note_mask, code_chunks, code_mask, labels in loader:
            note_chunks = note_chunks.to(device)
            note_mask = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask = code_mask.to(device)
            labels = labels.to(device)

            logits = model(note_chunks, note_mask, code_chunks, code_mask)
            total_loss += criterion(logits, labels).item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(float)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {
        "acc": acc,
        "auc": auc,
        "loss": total_loss / max(1, n_batches),
        "probs": probs,
        "labels": labels,
    }


def predict_per_subject(model, ds, collate_fn, device, batch_size):
    """Return list of {subject_id, label, probability} aligned with ds.subject_ids."""
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    model.eval()
    all_probs = []
    with torch.no_grad():
        for note_chunks, note_mask, code_chunks, code_mask, _ in loader:
            note_chunks = note_chunks.to(device)
            note_mask = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask = code_mask.to(device)
            logits = model(note_chunks, note_mask, code_chunks, code_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs)
    assert len(all_probs) == len(ds.subject_ids) == len(ds.labels)
    return [
        {
            "subject_id": int(sid),
            "label": int(lab),
            "probability": float(p),
        }
        for sid, lab, p in zip(ds.subject_ids, ds.labels, all_probs)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Mean-pool baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_DIR / "configs" / "config.yaml"),
        help="YAML config (parquet paths + splits). Optimization fields overridden by CLI.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="meanpool")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config_from_yaml(args.config)
    set_seed(cfg.seed)
    device = cfg.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[meanpool] device={device}  output_dir={out_dir}", flush=True)

    # ── Data ──
    (
        note_labels_df,
        icd_vectors_cpu,
        note_vectors_cpu,
        icd_subject_to_indices,
        note_subject_to_indices,
        _icd_keep_indices,
    ) = load_embeddings_and_mappings(cfg)

    train_df, val_df, test_df = create_subject_splits(note_labels_df, cfg)
    train_ds = SubjectDataset(train_df["subject_id"].values, train_df["label"].values)
    val_ds = SubjectDataset(val_df["subject_id"].values, val_df["label"].values)
    test_ds = SubjectDataset(test_df["subject_id"].values, test_df["label"].values)

    collate_fn = make_collate_fn(
        icd_vectors_cpu=icd_vectors_cpu,
        note_vectors_cpu=note_vectors_cpu,
        icd_subject_to_indices=icd_subject_to_indices,
        note_subject_to_indices=note_subject_to_indices,
        emb_dim=cfg.emb_dim,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Model ──
    model = MeanPoolClassifier(
        emb_dim=cfg.emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[meanpool] trainable params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ── Train ──
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    best_auc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())  # fallback so we always save
    best_epoch = 0
    no_improve = 0
    training_start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss, n_batches = 0.0, 0
        for note_chunks, note_mask, code_chunks, code_mask, labels in train_loader:
            note_chunks = note_chunks.to(device)
            note_mask = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask = code_mask.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            logits = model(note_chunks, note_mask, code_chunks, code_mask)
            loss = criterion(logits, labels)
            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["acc"])

        improved = val_metrics["auc"] > best_auc
        if improved:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[meanpool] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"{'*' if improved else ' '} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"[meanpool] early stopping after epoch {epoch}", flush=True)
            break

    training_seconds = time.time() - training_start_time
    print(
        f"[meanpool] best val AUC {best_auc:.4f} at epoch {best_epoch} | "
        f"total training wall-clock: {training_seconds:.1f}s "
        f"({training_seconds / 60:.2f} min) over {len(history['train_loss'])} epochs",
        flush=True,
    )
    model.load_state_dict(best_state)

    # ── Save checkpoint + test predictions ──
    ckpt_path = out_dir / f"{args.run_name}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(cfg),
            "args": vars(args),
            "history": history,
            "best_val_auc": best_auc,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    print(f"[meanpool] saved checkpoint -> {ckpt_path}", flush=True)

    preds = predict_per_subject(model, test_ds, collate_fn, device, args.batch_size)
    test_pred_path = out_dir / "test_predictions.jsonl"
    with open(test_pred_path, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    print(f"[meanpool] wrote {len(preds)} test predictions -> {test_pred_path}", flush=True)

    # Compute + print + save test metrics
    probs = np.array([r["probability"] for r in preds])
    labels = np.array([r["label"] for r in preds])
    test_acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    test_auc = roc_auc_score(labels, probs)
    print(
        f"[meanpool] Test accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}",
        flush=True,
    )

    metrics = {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "n_test": int(len(preds)),
        "training_seconds": float(training_seconds),
        "training_epochs": int(len(history["train_loss"])),
        "args": vars(args),
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Also drop a minimal config YAML for reproducibility
    with open(out_dir / f"{args.run_name}.config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config_yaml": args.config, "run_args": vars(args)}, f)


if __name__ == "__main__":
    main()
