"""Fine-tune a long-context text encoder for one-year mortality prediction.

Supported backbones (select via --model):
  - modernbert:  answerdotai/ModernBERT-base  (8192 tokens)
  - longformer:  allenai/longformer-base-4096 (4096 tokens, global attn on CLS)

Loads per-split JSONL produced by scripts/build_full_context_dataset.py (fields:
subject_id, label, patient_text), full fine-tunes the backbone + 1-logit
classification head, tracks val AUC with early stopping, writes
test_predictions.jsonl in the standard per-subject schema.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[2]


MODEL_SPECS = {
    "modernbert": {
        "hf_id": "answerdotai/ModernBERT-base",
        "max_length": 8192,
    },
    "longformer": {
        "hf_id": "allenai/longformer-base-4096",
        "max_length": 4096,
    },
}


class PatientTextDataset(Dataset):
    """Holds (subject_id, label, patient_text) rows; tokenization happens in the collator."""

    def __init__(self, jsonl_path: Path, max_patients: int = -1):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.rows.append(
                    {
                        "subject_id": int(r["subject_id"]),
                        "label": float(r["label"]),
                        "patient_text": r["patient_text"],
                    }
                )
                if 0 < max_patients <= len(self.rows):
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(tokenizer, max_length: int, is_longformer: bool):
    def collate(batch):
        texts = [b["patient_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        subject_ids = [b["subject_id"] for b in batch]

        enc = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "subject_ids": subject_ids,
            "num_input_tokens": enc["attention_mask"].sum(dim=1).tolist(),
        }
        if is_longformer:
            # Put global attention on the [CLS] token only.
            gam = torch.zeros_like(enc["input_ids"])
            gam[:, 0] = 1
            out["global_attention_mask"] = gam
        return out

    return collate


def run_forward(model, batch, device, use_global_attn: bool):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    # Do NOT pass labels into the HF model: with problem_type="regression" it
    # computes MSE internally, which is unused and misleading. We always
    # compute BCE manually outside this helper.
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if use_global_attn:
        kwargs["global_attention_mask"] = batch["global_attention_mask"].to(
            device, non_blocking=True
        )
    out = model(**kwargs)
    # num_labels=1 -> out.logits is [B, 1]; flatten to [B].
    logits = out.logits.squeeze(-1)
    return logits, labels


def evaluate(model, loader, device, use_global_attn: bool, use_amp: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits, labels = run_forward(model, batch, device, use_global_attn)
            loss = bce(logits.float(), labels.float())
            total_loss += float(loss.item())
            n_batches += 1
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(float)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {
        "acc": accuracy_score(labels, preds),
        "auc": auc,
        "loss": total_loss / max(1, n_batches),
    }


def predict(model, loader, device, use_global_attn: bool, use_amp: bool):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits, labels = run_forward(model, batch, device, use_global_attn)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for sid, lab, p, ntok in zip(
                batch["subject_ids"], labels.cpu().numpy(), probs, batch["num_input_tokens"]
            ):
                records.append(
                    {
                        "subject_id": int(sid),
                        "label": int(lab),
                        "probability": float(p),
                        "num_input_tokens": int(ntok),
                    }
                )
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Text-encoder fine-tune baseline")
    parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Override; defaults to 4 (modernbert) / 1 (longformer).")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Override; defaults chosen so effective BS = 16.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--max-train-patients", type=int, default=-1,
                        help="Cap training set (for smoke tests).")
    parser.add_argument("--max-eval-patients", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--truncate-direction", choices=["recent", "early"], default="recent",
                        help="recent = left-truncate (keep most recent tokens); "
                             "early = right-truncate (keep earliest tokens).")
    return parser.parse_args()


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_cosine_schedule_with_warmup

    args = parse_args()
    set_all_seeds(args.seed)

    spec = MODEL_SPECS[args.model]
    hf_id = spec["hf_id"]
    max_length = spec["max_length"]
    is_longformer = args.model == "longformer"

    # Defaults tuned per model (overridable via CLI).
    if args.micro_batch_size is None:
        args.micro_batch_size = 1 if is_longformer else 4
    if args.grad_accum_steps is None:
        args.grad_accum_steps = max(1, 16 // args.micro_batch_size)

    run_name = args.run_name or f"{args.model}_base"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(
        f"[{run_name}] hf_id={hf_id} max_length={max_length} device={device} "
        f"amp={use_amp} micro_bs={args.micro_batch_size} accum={args.grad_accum_steps}",
        flush=True,
    )
    print(f"[{run_name}] output_dir={out_dir}", flush=True)

    # ── Tokenizer + model ──
    print(f"[{run_name}] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    # truncate_direction=recent → left-truncate (keep RIGHT/most-recent tokens);
    # truncate_direction=early  → right-truncate (keep LEFT/earliest tokens).
    tokenizer.truncation_side = "left" if args.truncate_direction == "recent" else "right"
    print(f"[{run_name}] truncate_direction={args.truncate_direction} "
          f"(tokenizer.truncation_side={tokenizer.truncation_side})", flush=True)
    # Use the pad token configured in the checkpoint; fall back to EOS if missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{run_name}] loading model", flush=True)
    model_kwargs = {
        "num_labels": 1,
        "problem_type": "regression",  # keeps HF head as a single Linear(d, 1);
        # we NEVER consume out.loss — BCE is computed manually from logits.
    }
    # ModernBERT supports FA2 natively (~30–50% speedup at 8k context).
    # Longformer uses its own sliding-window kernel that ignores
    # attn_implementation, so leave it unset for longformer.
    # Keep master weights in fp32: torch.amp.autocast(..., bf16) below still
    # runs compute (and FA2) in bf16. Loading weights in bf16 quantizes the
    # optimizer's master copy, which makes lr=2e-5 AdamW updates round away
    # — stalled this run at val AUC ~0.79 last time.
    if args.model == "modernbert":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[{run_name}] using attn_implementation=flash_attention_2 "
              f"(fp32 weights, bf16 compute via autocast)", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, **model_kwargs)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # use_cache must be False for gradient checkpointing to work cleanly,
        # otherwise HF raises / warns every forward.
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print(f"[{run_name}] gradient checkpointing enabled", flush=True)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run_name}] trainable params: {n_params:,}", flush=True)

    # ── Data ──
    train_ds = PatientTextDataset(Path(args.train_jsonl), args.max_train_patients)
    val_ds = PatientTextDataset(Path(args.val_jsonl), args.max_eval_patients)
    test_ds = PatientTextDataset(Path(args.test_jsonl), args.max_eval_patients)
    print(
        f"[{run_name}] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}",
        flush=True,
    )

    collate = make_collate(tokenizer, max_length=max_length, is_longformer=is_longformer)
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Optimizer + schedule ──
    # Classify by module type, not substring match: ModernBERT's norms are
    # attn_norm / mlp_norm / final_norm, which a name-based list like
    # ("LayerNorm.weight", "layer_norm.weight") silently misses — every LN
    # scale ended up in the decay bucket and got pulled toward zero.
    norm_param_names: set[str] = set()
    norm_types: tuple = (nn.LayerNorm,)
    if hasattr(nn, "RMSNorm"):
        norm_types = norm_types + (nn.RMSNorm,)
    for mod_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for p_name, _ in module.named_parameters(recurse=False):
                norm_param_names.add(f"{mod_name}.{p_name}" if mod_name else p_name)

    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in norm_param_names or n.endswith(".bias"):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    print(
        f"[{run_name}] optimizer groups — decay: {len(decay_params)}  "
        f"no-decay (norms+biases): {len(nodecay_params)}",
        flush=True,
    )
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    # Steps-per-epoch counts FULL accumulation groups + one extra optim.step()
    # we issue at end-of-epoch if a partial group remains (see training loop).
    n_micro = len(train_loader)
    full_groups = n_micro // args.grad_accum_steps
    has_partial = (n_micro % args.grad_accum_steps) != 0
    steps_per_epoch = full_groups + (1 if has_partial else 0)
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(
        f"[{run_name}] total_steps={total_steps}  warmup={warmup_steps}  "
        f"steps_per_epoch={steps_per_epoch}",
        flush=True,
    )

    bce = nn.BCEWithLogitsLoss()

    # ── Train ──
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    best_auc = -float("inf")
    # Initialize best_state to the pre-training state so smoke tests with
    # degenerate val sets (all-one-class, NaN AUC) still save a checkpoint.
    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    no_improve = 0
    training_start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        optim.zero_grad()
        pending_grads = False

        n_micro = len(train_loader)
        for step, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits, labels = run_forward(model, batch, device, is_longformer)
                loss = bce(logits.float(), labels.float())
                loss_to_backprop = loss / args.grad_accum_steps
            loss_to_backprop.backward()
            pending_grads = True

            is_last_micro_in_epoch = (step + 1) == n_micro
            if (step + 1) % args.grad_accum_steps == 0 or is_last_micro_in_epoch:
                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                pending_grads = False

            running_loss += float(loss.item())
            n_batches += 1

        # Safety: no partial grads should linger past end-of-epoch.
        assert not pending_grads

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device, is_longformer, use_amp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["acc"])

        improved = val_metrics["auc"] > best_auc
        if improved:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[{run_name}] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"{'*' if improved else ' '} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"[{run_name}] early stopping after epoch {epoch}", flush=True)
            break

    training_seconds = time.time() - training_start_time
    if best_epoch == 0:
        print(
            f"[{run_name}] WARNING: no val AUC improvement over any epoch "
            "— saving pre-training state (degenerate run?)",
            flush=True,
        )
    print(
        f"[{run_name}] best val AUC {best_auc:.4f} at epoch {best_epoch} | "
        f"total training wall-clock: {training_seconds:.1f}s "
        f"({training_seconds / 60:.2f} min) over {len(history['train_loss'])} epochs",
        flush=True,
    )
    model.load_state_dict(best_state)

    # ── Save checkpoint + test predictions ──
    ckpt_path = out_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "hf_id": hf_id,
            "history": history,
            "best_val_auc": best_auc,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    print(f"[{run_name}] saved checkpoint -> {ckpt_path}", flush=True)

    preds = predict(model, test_loader, device, is_longformer, use_amp)
    assert len(preds) == len(test_ds), (
        f"predictions {len(preds)} != test_ds {len(test_ds)}"
    )
    pred_path = out_dir / "test_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    print(f"[{run_name}] wrote {len(preds)} predictions -> {pred_path}", flush=True)

    probs = np.array([r["probability"] for r in preds])
    labels = np.array([r["label"] for r in preds])
    test_acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    test_auc = roc_auc_score(labels, probs)
    print(
        f"[{run_name}] Test accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}",
        flush=True,
    )

    metrics = {
        "hf_id": hf_id,
        "max_length": max_length,
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


if __name__ == "__main__":
    main()
