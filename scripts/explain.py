import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

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
from evigen_dynamicquery.model_evigen import EviGen
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients explainability for EviGen"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: outputs/checkpoints/generating_200_samples_for_prm.pt)",
    )
    parser.add_argument("--num-patients", type=int, default=200)
    parser.add_argument(
        "--baseline", type=str, default="zero", choices=["zero", "mean"]
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--output", type=str, default="outputs/death_prediction_explanations.json"
    )
    parser.add_argument(
        "--exclude-subjects-from",
        type=str,
        default=None,
        help="Path to a JSON file of prior explanation entries; subject_ids in those "
             "entries are skipped when selecting patients.",
    )
    parser.add_argument(
        "--include-subjects-from",
        type=str,
        default=None,
        help="Path to a JSON file of prior explanation entries; ONLY subject_ids in "
             "those entries are considered for selection. Combine with --splits all to "
             "search across train+val+test.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        choices=["test", "all"],
        help="Which subjects to scan for correctly-predicted deaths: 'test' (default) "
             "or 'all' (train+val+test combined).",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Process every subject in the selected split regardless of label / "
             "prediction. Default behavior keeps the legacy filter "
             "(label==1 AND predicted==1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def build_patient_tensors(subject_id, cfg, device,
                          note_subject_to_indices, icd_subject_to_indices,
                          note_vectors_cpu, icd_vectors_cpu):
    note_idxs = note_subject_to_indices.get(int(subject_id), [])
    icd_idxs  = icd_subject_to_indices.get(int(subject_id), [])

    note_embs = note_vectors_cpu[note_idxs].to(device) if note_idxs else \
                torch.zeros(0, cfg.emb_dim, device=device)
    code_embs = icd_vectors_cpu[icd_idxs].to(device) if icd_idxs else \
                torch.zeros(0, cfg.emb_dim, device=device)

    note_chunks = note_embs.unsqueeze(0)  # [1, L_n, d]
    note_mask   = torch.ones(1, note_embs.size(0), dtype=torch.bool, device=device)
    code_chunks = code_embs.unsqueeze(0)  # [1, L_c, d]
    code_mask   = torch.ones(1, code_embs.size(0), dtype=torch.bool, device=device)

    return note_chunks, note_mask, code_chunks, code_mask, note_idxs, icd_idxs


def get_topk_retrieved_indices(model, cfg, note_chunks, note_mask, code_chunks, code_mask):
    """
    Returns:
      gating_mask : [N] numpy (1 = active, 0 = inactive)
      note_topk   : list[Qn] of int arrays of length K  (local indices into patient note array)
      note_sims   : list[Qn] of float arrays (cosine similarity per retrieved chunk)
      code_topk   : list[Qc] of int arrays
      code_sims   : list[Qc] of float arrays
    """
    model.eval()
    device = note_chunks.device

    with torch.no_grad():
        summary_note = model.summarize_modality(note_chunks, note_mask)
        summary_code = model.summarize_modality(code_chunks, code_mask) \
                       if model.num_code_queries > 0 else None
        mask_q, _, _ = model.hybrid_gating(summary_note, summary_code)
        gating_mask  = mask_q[0].cpu().numpy()  # [N]

        Qn = model.num_note_queries
        q_notes_norm = F.normalize(model.query_vectors[:Qn], dim=-1)  # [Qn, d]
        note_sim     = torch.einsum("bld,qd->bql", note_chunks, q_notes_norm)[0]  # [Qn, L_n]
        mask_n = note_mask[0]
        K = min(cfg.k_retrieval, int(mask_n.sum().item()))

        note_topk, note_sims = [], []
        for q in range(Qn):
            sim_q = note_sim[q].clone()
            sim_q[~mask_n] = float('-1e9')
            topk = torch.topk(sim_q, k=K)
            note_topk.append(topk.indices.cpu().numpy())
            note_sims.append(note_sim[q][topk.indices].cpu().numpy())

        code_topk, code_sims = [], []
        Qc = model.num_code_queries
        if Qc > 0:
            q_codes_norm = F.normalize(model.query_vectors[Qn:], dim=-1)
            code_sim = torch.einsum("bld,qd->bql", code_chunks, q_codes_norm)[0]
            mask_c   = code_mask[0]
            Kc = min(cfg.k_retrieval, int(mask_c.sum().item()))
            for q in range(Qc):
                sim_q = code_sim[q].clone()
                sim_q[~mask_c] = float('-1e9')
                topk = torch.topk(sim_q, k=Kc)
                code_topk.append(topk.indices.cpu().numpy())
                code_sims.append(code_sim[q][topk.indices].cpu().numpy())

    return gating_mask, note_topk, note_sims, code_topk, code_sims


def post_retrieval_forward_pass(model, cfg, selected_notes, selected_codes, mask_q_hard):
    """
    selected_notes : [1, Qn, K, d]  (may require grad on specific entries)
    selected_codes : [1, Qc, K, d]
    mask_q_hard    : [1, N] float tensor  (fixed binary gating)
    Returns: scalar probability
    """
    Qn = model.num_note_queries
    Qc = model.num_code_queries
    D  = cfg.emb_dim

    context_notes = model.linear_attention(selected_notes, model.query_vectors[:Qn])  # [1, Qn, d]
    if Qc > 0:
        context_codes = model.linear_attention(selected_codes, model.query_vectors[Qn:])  # [1, Qc, d]
    else:
        context_codes = selected_notes.new_zeros(1, 0, D)

    context = torch.cat([context_notes, context_codes], dim=1)  # [1, N, d]

    expert_outputs = []
    for i in range(cfg.num_queries):
        v     = context[:, i, :]
        z     = model.expert_norms[i](v)
        delta = model.experts[i](z)
        expert_outputs.append(v + delta)
    expert_outputs = torch.stack(expert_outputs, dim=1)  # [1, N, d]

    active_counts = mask_q_hard.sum(dim=1, keepdim=True).clamp(min=1.0)
    mixed = (expert_outputs * mask_q_hard.unsqueeze(-1)).sum(dim=1) / active_counts
    mixed = model.final_ln(mixed)
    prob  = torch.sigmoid(model.classifier(mixed).squeeze(-1)).flatten()[0]
    return prob


def build_ig_baseline(note_sel_base, code_sel_base, baseline_type='zero'):
    D = note_sel_base.shape[-1]
    if baseline_type == 'zero':
        return torch.zeros(D, device=note_sel_base.device)
    if baseline_type == 'mean':
        tensors = []
        if note_sel_base.numel() > 0:
            tensors.append(note_sel_base.reshape(-1, D))
        if code_sel_base.numel() > 0:
            tensors.append(code_sel_base.reshape(-1, D))
        if not tensors:
            return torch.zeros(D, device=note_sel_base.device)
        return torch.cat(tensors, dim=0).mean(dim=0)
    raise ValueError(f"Unknown baseline_type: {baseline_type}")


def compute_ig_for_patient_dynamic(
    subject_id, model, cfg, device,
    note_subject_to_indices, icd_subject_to_indices,
    note_vectors_cpu, icd_vectors_cpu,
    note_meta_df, code_meta_df, note_chunk_col, has_code_meta,
    code_ver_col=None,
    baseline_type='zero', num_steps=50,
):
    model.eval()

    note_chunks, note_mask, code_chunks, code_mask, note_idxs, icd_idxs = \
        build_patient_tensors(
            subject_id, cfg, device,
            note_subject_to_indices, icd_subject_to_indices,
            note_vectors_cpu, icd_vectors_cpu,
        )

    gating_mask, note_topk, note_sims, code_topk, code_sims = \
        get_topk_retrieved_indices(model, cfg, note_chunks, note_mask, code_chunks, code_mask)

    Qn = model.num_note_queries
    Qc = model.num_code_queries
    D  = cfg.emb_dim
    K  = note_topk[0].shape[0] if note_topk else 0
    Kc = code_topk[0].shape[0] if code_topk else 0

    # Pre-build selected embedding tensors (fixed)
    note_sel_base = torch.zeros(1, Qn, K, D, device=device)
    for q in range(Qn):
        for k, li in enumerate(note_topk[q]):
            note_sel_base[0, q, k] = note_chunks[0, li]

    if Qc > 0 and Kc > 0:
        code_sel_base = torch.zeros(1, Qc, Kc, D, device=device)
        for q in range(Qc):
            for k, li in enumerate(code_topk[q]):
                code_sel_base[0, q, k] = code_chunks[0, li]
    else:
        code_sel_base = torch.zeros(1, 0, max(Kc, 1), D, device=device)

    mask_q_tensor = torch.tensor(gating_mask, dtype=torch.float32, device=device).unsqueeze(0)
    baseline = build_ig_baseline(note_sel_base, code_sel_base, baseline_type=baseline_type)

    baseline_note = baseline.view(1, 1, 1, D).expand_as(note_sel_base)
    baseline_code = baseline.view(1, 1, 1, D).expand_as(code_sel_base)
    note_delta = note_sel_base - baseline_note
    code_delta = code_sel_base - baseline_code

    alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
    note_attr_accum = torch.zeros_like(note_sel_base, dtype=torch.float64)
    code_attr_accum = torch.zeros_like(code_sel_base, dtype=torch.float64)

    for s, alpha in enumerate(alphas):
        note_interp = (baseline_note + alpha * note_delta).detach().requires_grad_(True)
        code_interp = (baseline_code + alpha * code_delta).detach().requires_grad_(True)

        prob = post_retrieval_forward_pass(model, cfg, note_interp, code_interp, mask_q_tensor)
        grad_inputs = [note_interp]
        if code_interp.numel() > 0:
            grad_inputs.append(code_interp)
        grads = torch.autograd.grad(prob, grad_inputs, retain_graph=False)

        weight = 0.5 if (s == 0 or s == num_steps) else 1.0
        note_attr_accum += weight * (grads[0].double() * note_delta.double())
        if code_interp.numel() > 0:
            code_attr_accum += weight * (grads[1].double() * code_delta.double())

    note_ig = note_attr_accum / num_steps
    code_ig = code_attr_accum / num_steps

    with torch.no_grad():
        prob_actual   = float(post_retrieval_forward_pass(model, cfg, note_sel_base, code_sel_base, mask_q_tensor))
        prob_baseline = float(post_retrieval_forward_pass(model, cfg, baseline_note, baseline_code, mask_q_tensor))
    total_attr = float(note_ig.sum().item() + code_ig.sum().item())
    print(
        f"  Completeness check: total attribution={total_attr:.6f}, "
        f"prob delta={prob_actual - prob_baseline:.6f}"
    )

    results = []

    for q in range(Qn):
        if gating_mask[q] < 0.5:
            continue
        for rank, local_idx in enumerate(note_topk[q]):
            global_idx = int(note_idxs[local_idx]) if note_idxs else -1
            item = {
                'query_type':        'Note',
                'query_idx':         q,
                'rank_in_query':     rank,
                'global_index':      global_idx,
                'similarity_score':  float(note_sims[q][rank]),
                'attribution_score': float(note_ig[0, q, rank].sum().item()),
            }
            if note_chunk_col and global_idx >= 0:
                item['chunk_text'] = str(note_meta_df.iloc[global_idx][note_chunk_col])
            if 'note_id' in note_meta_df.columns and global_idx >= 0:
                item['note_id'] = str(note_meta_df.iloc[global_idx]['note_id'])
            if 'label' in note_meta_df.columns and global_idx >= 0:
                item['label'] = int(note_meta_df.iloc[global_idx]['label'])
            results.append(item)

    for q in range(Qc):
        q_global = Qn + q
        if gating_mask[q_global] < 0.5:
            continue
        for rank, local_idx in enumerate(code_topk[q]):
            global_idx = int(icd_idxs[local_idx]) if icd_idxs else -1
            item = {
                'query_type':        'ICD',
                'query_idx':         q_global,
                'rank_in_query':     rank,
                'global_index':      global_idx,
                'similarity_score':  float(code_sims[q][rank]),
                'attribution_score': float(code_ig[0, q, rank].sum().item()),
            }
            if has_code_meta and global_idx >= 0:
                code_ver = str(code_meta_df.iloc[global_idx][code_ver_col])
                item['icd_code_ver'] = code_ver
                if 'long_title' in code_meta_df.columns:
                    item['long_title'] = str(code_meta_df.iloc[global_idx]['long_title'])
                else:
                    item['long_title'] = 'Unknown'
            results.append(item)

    return results


# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------

def find_correctly_predicted_death_patients(
    model, cfg, test_loader, test_df, num_patients=10,
    exclude_subject_ids=None, include_subject_ids=None,
    include_incorrect=False,
):
    model.eval()
    device = cfg.device
    selected = []
    patient_offset = 0
    test_subject_ids = test_df['subject_id'].values
    exclude = set(int(s) for s in (exclude_subject_ids or []))
    include = set(int(s) for s in (include_subject_ids or [])) if include_subject_ids else None

    with torch.no_grad():
        for note_chunks, note_mask, code_chunks, code_mask, labels in test_loader:
            note_chunks = note_chunks.to(device)
            note_mask   = note_mask.to(device)
            code_chunks = code_chunks.to(device)
            code_mask   = code_mask.to(device)

            logits, _, _, _ = model(
                note_chunks, note_mask, code_chunks, code_mask,
                temperature=cfg.final_temperature,
                use_sampling_retrieval=False,
            )
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(labels.size(0)):
                sid = int(test_subject_ids[patient_offset + i])
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                if include_incorrect:
                    keep = (
                        sid not in exclude
                        and (include is None or sid in include)
                    )
                else:
                    keep = (
                        labels[i].item() == 1.0
                        and preds[i].item() == 1.0
                        and sid not in exclude
                        and (include is None or sid in include)
                    )
                if keep:
                    selected.append({
                        'patient_id':       sid,
                        'true_label':       true_label,
                        'predicted_label':  pred_label,
                        'probability':      round(probs[i].item(), 4),
                        'logit':            round(logits[i].item(), 4),
                    })
                if len(selected) >= num_patients:
                    return selected
            patient_offset += labels.size(0)

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_patients_with_explanations(
    model, cfg, device,
    test_loader, test_df,
    note_subject_to_indices, icd_subject_to_indices,
    note_vectors_cpu, icd_vectors_cpu,
    note_meta_df, code_meta_df, note_chunk_col, has_code_meta,
    code_ver_col=None,
    num_patients: int = 10,
    baseline_type: str = 'zero',
    num_steps: int = 50,
    output_file: str = 'outputs/death_prediction_explanations.json',
    exclude_subject_ids=None,
    include_subject_ids=None,
    include_incorrect: bool = False,
):
    print('=' * 80)
    print('DEATH PREDICTION EXPLAINABILITY ANALYSIS (EviGen)')
    print('=' * 80)

    if include_incorrect:
        print(f'\nStep 1: Selecting up to {num_patients} subjects (no label/pred filter)...')
    else:
        print(f'\nStep 1: Finding {num_patients} correctly predicted death patients...')
    selected_patients = find_correctly_predicted_death_patients(
        model, cfg, test_loader, test_df,
        num_patients=num_patients,
        exclude_subject_ids=exclude_subject_ids,
        include_subject_ids=include_subject_ids,
        include_incorrect=include_incorrect,
    )
    print(f'  Found {len(selected_patients)} patients.')

    all_results = []

    for i, patient_info in enumerate(selected_patients):
        subject_id = patient_info['patient_id']
        print(f'\nProcessing patient {i+1}/{len(selected_patients)} (ID: {subject_id})...')

        try:
            ig_results = compute_ig_for_patient_dynamic(
                subject_id=subject_id,
                model=model,
                cfg=cfg,
                device=device,
                note_subject_to_indices=note_subject_to_indices,
                icd_subject_to_indices=icd_subject_to_indices,
                note_vectors_cpu=note_vectors_cpu,
                icd_vectors_cpu=icd_vectors_cpu,
                note_meta_df=note_meta_df,
                code_meta_df=code_meta_df,
                note_chunk_col=note_chunk_col,
                has_code_meta=has_code_meta,
                code_ver_col=code_ver_col,
                baseline_type=baseline_type,
                num_steps=num_steps,
            )

            if not ig_results:
                print(f'  No retrieved data for patient {subject_id}, skipping.')
                continue

            patient_result = {
                'subject_id':      subject_id,
                'label':           patient_info['true_label'],
                'logit':           patient_info['logit'],
                'probability':     patient_info['probability'],
                'predicted_label': patient_info.get('predicted_label', 1),
                'retrieved_items': [],
            }

            for item in ig_results:
                entry = {
                    'query_type':        item['query_type'],
                    'query_idx':         item['query_idx'],
                    'rank_in_query':     item['rank_in_query'],
                    'global_index':      item['global_index'],
                    'similarity_score':  round(item['similarity_score'], 4),
                    'attribution_score': round(item['attribution_score'], 4),
                }
                if item['query_type'] == 'ICD':
                    entry['icd_code_ver'] = item.get('icd_code_ver', '')
                    entry['long_title']   = item.get('long_title', '')
                else:
                    entry['note_id']    = item.get('note_id', '')
                    entry['chunk_text'] = item.get('chunk_text', '')
                    entry['label']      = item.get('label', -1)
                patient_result['retrieved_items'].append(entry)

            all_results.append(patient_result)

            n_note    = sum(1 for x in ig_results if x['query_type'] == 'Note')
            n_icd     = sum(1 for x in ig_results if x['query_type'] == 'ICD')
            mean_attr = np.mean([x['attribution_score'] for x in ig_results])
            print(
                f'  Note items: {n_note}, ICD items: {n_icd}, '
                f'mean attribution: {mean_attr:.4f}'
            )

        except Exception as e:
            print(f'  Error for patient {subject_id}: {e}')
            traceback.print_exc()
            continue

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj

    print(f'\nSaving {len(all_results)} patient results to {output_file}...')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=to_serializable)

    print(f'Done. Processed {len(all_results)}/{len(selected_patients)} patients.')
    print('=' * 80)
    return all_results


if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint_path = args.checkpoint or str(
        PROJECT_DIR / 'outputs' / 'checkpoints' / 'generating_200_samples_for_prm.pt'
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_file = args.config if Path(args.config).is_absolute() else str(PROJECT_DIR / args.config)
    if isinstance(ckpt.get('config'), dict):
        cfg = Config(**ckpt['config'])
        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("[explain] CUDA not available, fallback to CPU.")
            cfg.device = "cpu"
        print(f"Loaded config from checkpoint: {checkpoint_path}")
    else:
        cfg = load_config_from_yaml(config_file)
        print(f"Loaded config from {config_file}")
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = EviGen(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  epoch: {ckpt.get('epoch', 'n/a')}  best_val_acc: {ckpt.get('val_acc', 'n/a')}")

    # Load embeddings
    (
        note_labels_df,
        icd_vectors_cpu,
        note_vectors_cpu,
        icd_subject_to_indices,
        note_subject_to_indices,
        _icd_keep_indices,
    ) = load_embeddings_and_mappings(cfg)

    # Load parquet metadata columns (skip 'vector' for memory efficiency)
    print("Loading parquet metadata columns...")
    note_pq_cols = pq.read_schema(cfg.note_parquet).names
    code_pq_cols = pq.read_schema(cfg.code_parquet).names

    note_meta_df = pd.read_parquet(cfg.note_parquet, columns=[c for c in note_pq_cols if c != 'vector'])
    code_meta_df = pd.read_parquet(cfg.code_parquet, columns=[c for c in code_pq_cols if c != 'vector'])

    note_chunk_col = next((c for c in ['chunk', 'chunk_text', 'text'] if c in note_meta_df.columns), None)
    code_ver_col   = next((c for c in ['icd_code_ver', 'code_ver', 'icd_code'] if c in code_meta_df.columns), None)
    has_code_meta  = code_ver_col is not None

    print(f"Note parquet columns: {note_meta_df.columns.tolist()}")
    print(f"Code parquet columns: {code_meta_df.columns.tolist()}")
    print(f"Note text column: {note_chunk_col}")
    print(f"Code ver column: {code_ver_col}")
    print(f"Code ICD metadata available: {has_code_meta}")
    print(f"Code long_title available: {'long_title' in code_meta_df.columns}")

    # Build DataLoader over the requested split(s)
    emb_dim = cfg.emb_dim
    train_df, val_df, test_df = create_subject_splits(note_labels_df, cfg)
    if args.splits == "all":
        eval_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(
            f"Using combined train+val+test: {len(eval_df)} subjects "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
        )
    else:
        eval_df = test_df
        print(f"Using test split: {len(eval_df)} subjects")
    test_ds = SubjectDataset(eval_df["subject_id"].values, eval_df["label"].values)
    collate_fn = make_collate_fn(
        icd_vectors_cpu=icd_vectors_cpu,
        note_vectors_cpu=note_vectors_cpu,
        icd_subject_to_indices=icd_subject_to_indices,
        note_subject_to_indices=note_subject_to_indices,
        emb_dim=emb_dim,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    output_file = args.output if Path(args.output).is_absolute() else str(PROJECT_DIR / args.output)

    # Optional: load subject_ids to skip from a prior explanations JSON
    exclude_subject_ids = set()
    if args.exclude_subjects_from:
        exclude_path = (
            args.exclude_subjects_from
            if Path(args.exclude_subjects_from).is_absolute()
            else str(PROJECT_DIR / args.exclude_subjects_from)
        )
        with open(exclude_path) as f:
            prior = json.load(f)
        exclude_subject_ids = {int(p["subject_id"]) for p in prior}
        print(
            f"Excluding {len(exclude_subject_ids)} subject_ids loaded from "
            f"{exclude_path}"
        )

    # Optional: restrict to subject_ids from a prior explanations JSON
    include_subject_ids = None
    if args.include_subjects_from:
        include_path = (
            args.include_subjects_from
            if Path(args.include_subjects_from).is_absolute()
            else str(PROJECT_DIR / args.include_subjects_from)
        )
        with open(include_path) as f:
            prior = json.load(f)
        include_subject_ids = {int(p["subject_id"]) for p in prior}
        print(
            f"Restricting to {len(include_subject_ids)} subject_ids loaded from "
            f"{include_path}"
        )

    # Run explainability analysis
    process_patients_with_explanations(
        model=model,
        cfg=cfg,
        device=device,
        test_loader=test_loader,
        test_df=eval_df,
        note_subject_to_indices=note_subject_to_indices,
        icd_subject_to_indices=icd_subject_to_indices,
        note_vectors_cpu=note_vectors_cpu,
        icd_vectors_cpu=icd_vectors_cpu,
        note_meta_df=note_meta_df,
        code_meta_df=code_meta_df,
        note_chunk_col=note_chunk_col,
        has_code_meta=has_code_meta,
        code_ver_col=code_ver_col,
        num_patients=args.num_patients,
        baseline_type=args.baseline,
        num_steps=args.num_steps,
        output_file=output_file,
        exclude_subject_ids=exclude_subject_ids,
        include_subject_ids=include_subject_ids,
        include_incorrect=args.include_incorrect,
    )
