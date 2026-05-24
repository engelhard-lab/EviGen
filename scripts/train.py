import argparse
import copy
import json
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evigen_dynamicquery.data import (
    load_embeddings_and_mappings,
    create_subject_splits,
    SubjectDataset,
    make_collate_fn,
)
from evigen_dynamicquery.config import Config, set_seed, load_config_from_yaml
from evigen_dynamicquery.model_evigen import EviGen, TemperatureScheduler


def train_model(
    model: EviGen,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    checkpoint_path: str = "outputs/checkpoints/run.pt",
):
    device = cfg.device
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # LR warmup (constant after warmup)
    total_train_steps = cfg.num_epochs * max(1, len(train_loader))
    warmup_steps = cfg.warmup_steps

    def lr_lambda(step: int):
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Temperature annealing: anneal over temp_anneal_epochs worth of steps
    temp_anneal_steps = cfg.temp_anneal_epochs * max(1, len(train_loader))
    temp_scheduler = TemperatureScheduler(
        cfg.init_temperature, cfg.final_temperature, temp_anneal_steps
    )

    global_step = 0
    training_start_time = time.time()

    # Track training history
    history = {
        'train_loss': [],
        'train_aux_loss': [],
        'train_diversity': [],
        'train_simplicity': [],
        'train_dyn_active': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_dyn_active': [],
        'temperature': []
    }

    # Best model tracking and early stopping
    early_stopping_patience = 3
    best_val_acc = -float('inf')
    best_val_auc = -float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_scheduler_state = None
    best_global_step = 0
    best_epoch = 0
    epochs_without_auc_improvement = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        aux_loss_accum = 0.0
        diversity_accum = 0.0
        simplicity_accum = 0.0
        dyn_active_accum = 0.0
        temp_accum = 0.0
        n_batches = 0

        for note_chunks, note_mask, code_chunks, code_mask, labels in train_loader:
            # Update temperature per batch
            if global_step < temp_anneal_steps and cfg.use_sampling_retrieval:
                use_sampling_this_batch = True
                current_temp = temp_scheduler.get(global_step)
            else:
                use_sampling_this_batch = False
                current_temp = cfg.final_temperature

            note_chunks = note_chunks.to(device)
            note_mask = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask = code_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, gating_info, summary, context = model(
                note_chunks,
                note_mask,
                code_chunks,
                code_mask,
                temperature=current_temp,
                use_sampling_retrieval=use_sampling_this_batch,
            )

            main_loss = criterion(logits, labels)
            aux_loss, diversity, simplicity = model.moe_aux_loss()
            loss = main_loss + cfg.aux_loss_weight * aux_loss

            loss.backward()
            if cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            optimizer.step()
            scheduler.step()
            global_step += 1

            # Training statistics
            epoch_loss += loss.item()
            aux_loss_accum += aux_loss.item()
            diversity_accum += diversity.item()
            simplicity_accum += simplicity.item()
            temp_accum += current_temp
            mask_q = gating_info["mask"]  # [B, N]
            dyn_active = mask_q[:, 1:].sum(dim=1).float().mean().item()
            dyn_active_accum += dyn_active
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_aux = aux_loss_accum / max(1, n_batches)
        avg_diversity = diversity_accum / max(1, n_batches)
        avg_simplicity = simplicity_accum / max(1, n_batches)
        avg_dyn_active = dyn_active_accum / max(1, n_batches)
        avg_temp = temp_accum / max(1, n_batches)

        val_metrics = evaluate_model(
            model, val_loader, cfg, use_sampling=False
        )

        # Record history
        history['train_loss'].append(avg_loss)
        history['train_aux_loss'].append(avg_aux)
        history['train_diversity'].append(avg_diversity)
        history['train_simplicity'].append(avg_simplicity)
        history['train_dyn_active'].append(avg_dyn_active)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_dyn_active'].append(val_metrics['avg_active_dynamic'])
        history['temperature'].append(avg_temp)

        # Save best model based on validation AUC
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['acc']
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = copy.deepcopy(scheduler.state_dict())
            best_global_step = global_step
            best_epoch = epoch + 1
            epochs_without_auc_improvement = 0
        else:
            epochs_without_auc_improvement += 1

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Diversity: {avg_diversity:.4f} | "
            f"Simplicity: {avg_simplicity:.4f} | "
            f"Val acc: {val_metrics['acc']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if epochs_without_auc_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs: "
                f"validation AUC did not improve for {early_stopping_patience} consecutive epochs."
            )
            break

    # Load best model
    training_seconds = time.time() - training_start_time
    print(
        f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch} "
        f"(accuracy: {best_val_acc:.4f})"
    )
    print(
        f"Total training wall-clock: {training_seconds:.1f}s "
        f"({training_seconds / 60:.2f} min) over {len(history['train_loss'])} epochs"
    )
    model.load_state_dict(best_model_state)

    # Save best checkpoint plus its hyperparameters to disk
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_name = checkpoint_path.stem

    config_dict = asdict(cfg)
    checkpoint = {
        'epoch': best_epoch,
        'global_step': best_global_step,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state,
        'scheduler_state_dict': best_scheduler_state,
        'val_auc': best_val_auc,
        'val_acc': best_val_acc,
        'train_loss': history['train_loss'][best_epoch - 1],
        'training_seconds': float(training_seconds),
        'training_epochs': int(len(history['train_loss'])),
        'config': config_dict,
        'history': history,
        'checkpoint_path': str(checkpoint_path),
        'run_name': run_name,
    }
    torch.save(checkpoint, checkpoint_path)

    config_path = checkpoint_path.with_suffix('.config.yaml')
    metadata_path = checkpoint_path.with_suffix('.metadata.json')

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'config_path': str(config_path),
        'run_name': run_name,
        'epoch': best_epoch,
        'global_step': best_global_step,
        'val_auc': best_val_auc,
        'val_acc': best_val_acc,
        'train_loss': history['train_loss'][best_epoch - 1],
        'training_seconds': float(training_seconds),
        'training_epochs': int(len(history['train_loss'])),
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Checkpoint config saved to: {config_path}")
    print(f"Checkpoint metadata saved to: {metadata_path}")

    return model, history


def evaluate_model(
    model: EviGen,
    data_loader: DataLoader,
    cfg: Config,
    use_sampling: bool = False,
):
    device = cfg.device
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    total_dyn_active = 0.0
    total_note_active = 0.0
    total_code_active = 0.0
    total_samples = 0
    total_loss = 0.0
    n_batches = 0

    # For evaluation, use the final (low) temperature
    temperature = cfg.final_temperature
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for note_chunks, note_mask, code_chunks, code_mask, labels in data_loader:
            note_chunks = note_chunks.to(device)
            note_mask = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask = code_mask.to(device)
            labels = labels.to(device)

            logits, gating_info, summary, context = model(
                note_chunks,
                note_mask,
                code_chunks,
                code_mask,
                temperature=temperature,
                use_sampling_retrieval=use_sampling,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            mask_q = gating_info["mask"]  # [B, N], hard
            dyn_active = mask_q[:, 1:].sum(dim=1).float()  # [B]
            total_dyn_active += dyn_active.sum().item()
            note_active = mask_q[:, :cfg.num_note_queries].sum(dim=1).float()
            code_active = mask_q[:, cfg.num_note_queries:].sum(dim=1).float()
            total_note_active += note_active.sum().item()
            total_code_active += code_active.sum().item()
            total_samples += labels.size(0)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(float)

    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")

    avg_active_dynamic = total_dyn_active / max(1, total_samples)
    avg_active_note = total_note_active / max(1, total_samples)
    avg_active_code = total_code_active / max(1, total_samples)
    avg_loss = total_loss / max(1, n_batches)

    return {
        "acc": acc,
        "auc": auc,
        "avg_active_dynamic": avg_active_dynamic,
        "avg_active_note": avg_active_note,
        "avg_active_code": avg_active_code,
        "loss": avg_loss,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train EviGen with separate note/code retrieval")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name for this run (used in checkpoint filename; blank uses a timestamp)",
    )
    return parser.parse_args()


def make_run_stem(run_name: str) -> str:
    run_name = run_name.strip()
    if not run_name:
        run_name = datetime.now().strftime('evigen_%Y%m%d_%H%M%S')
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in run_name)


def plot_training_curves(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Train/Val Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    best_epoch_idx = np.argmax(history['val_acc'])
    axes[1].scatter(
        [best_epoch_idx + 1], [history['val_acc'][best_epoch_idx]],
        color='red', s=100, zorder=5,
        label=f'Best Val Acc (Epoch {best_epoch_idx + 1})',
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    args = parse_args()

    RUN_STEM = make_run_stem(args.run_name)
    CHECKPOINT_DIR = PROJECT_DIR / 'outputs' / 'checkpoints'
    CHECKPOINT_PATH = CHECKPOINT_DIR / f'{RUN_STEM}.pt'
    config_file = args.config if Path(args.config).is_absolute() else str(PROJECT_DIR / args.config)

    # 1) Load configuration from YAML
    cfg = load_config_from_yaml(config_file)
    print(f"Loaded config from {config_file}")
    print(cfg)
    print(f"Run name: {RUN_STEM}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    set_seed(cfg.seed)

    # 2) Load embeddings + mappings from Parquet (memory efficient!)
    (
        note_labels_df,
        icd_vectors_cpu,
        note_vectors_cpu,
        icd_subject_to_indices,
        note_subject_to_indices,
        _icd_keep_indices,
    ) = load_embeddings_and_mappings(cfg)

    emb_dim = cfg.emb_dim
    if icd_vectors_cpu.size(1) != emb_dim or note_vectors_cpu.size(1) != emb_dim:
        raise ValueError(
            f"Embedding dimension mismatch: cfg.emb_dim={emb_dim}, "
            f"ICD dim={icd_vectors_cpu.size(1)}, note dim={note_vectors_cpu.size(1)}"
        )

    # 3) Create train/val/test subject splits
    train_df, val_df, test_df = create_subject_splits(note_labels_df, cfg)

    # 4) Build Dataset objects
    train_ds = SubjectDataset(train_df["subject_id"].values, train_df["label"].values)
    val_ds   = SubjectDataset(val_df["subject_id"].values,   val_df["label"].values)
    test_ds  = SubjectDataset(test_df["subject_id"].values,  test_df["label"].values)

    # 5) Build DataLoaders with collate_fn that assembles per-patient note/code embeddings
    collate_fn = make_collate_fn(
        icd_vectors_cpu=icd_vectors_cpu,
        note_vectors_cpu=note_vectors_cpu,
        icd_subject_to_indices=icd_subject_to_indices,
        note_subject_to_indices=note_subject_to_indices,
        emb_dim=emb_dim,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 6) Initialize model
    model = EviGen(cfg)
    print(f"Model initialized on device: {cfg.device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 7) Train model
    print("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        checkpoint_path=str(CHECKPOINT_PATH),
    )

    # Plot and save training curves
    plot_path = CHECKPOINT_PATH.with_suffix('.training_curves.png')
    plot_training_curves(history, plot_path)

    # 8) Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, cfg, use_sampling=False)
    print(
        f"Test accuracy: {test_metrics['acc']:.4f} | "
        f"Test AUC: {test_metrics['auc']:.4f} | "
        f"Avg active dynamic queries per patient: {test_metrics['avg_active_dynamic']:.2f} | "
        f"Avg active note queries: {test_metrics['avg_active_note']:.2f} | "
        f"Avg active code queries: {test_metrics['avg_active_code']:.2f}"
    )
