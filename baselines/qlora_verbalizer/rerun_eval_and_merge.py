"""Post-hoc eval + re-merge using the FIXED logit-position read.

Use this to recover from training runs where the eval picked the wrong adapter
as "best" due to reading logits[-1] (after-target) instead of logits[-2]
(target-prediction). Loads each candidate adapter, runs val-set AUC with the
correct position, picks the best, and re-writes <out>/merged/.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "baselines" / "qlora_verbalizer"))
from train_qlora import (  # noqa: E402
    VerbalizerDataset, left_pad_collate, load_variant_module,
)


@torch.no_grad()
def eval_adapter(base_model_id, adapter_path, variant, val_jsonl, max_seq_len,
                 eval_batch_size, device, verbalizer_style="yesno"):
    """Load base+adapter, run val AUC with correct logit position (-2)."""
    verb_mod, free_mod = load_variant_module(variant, verbalizer_style)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pos_str, neg_str = verb_mod.resolve_verbalizer(verbalizer_style)
    yes_id = tokenizer.encode(pos_str, add_special_tokens=False)[0]
    no_id = tokenizer.encode(neg_str, add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    val_ds = VerbalizerDataset(val_jsonl, tokenizer, verb_mod, free_mod,
                                max_seq_len, yes_id, no_id)

    loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: left_pad_collate(b, pad_id),
    )
    all_probs, all_labels, all_sids = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits[:, -2, :]
        lp_yes = logits[:, yes_id]
        lp_no = logits[:, no_id]
        m = torch.maximum(lp_yes, lp_no)
        ey = torch.exp(lp_yes - m)
        en = torch.exp(lp_no - m)
        prob_yes = (ey / (ey + en)).float().cpu().numpy()
        all_probs.extend(prob_yes.tolist())
        all_labels.extend(batch["patient_label"].tolist())
        all_sids.extend(batch["subject_id"].tolist())
    auc = float(roc_auc_score(all_labels, all_probs))

    # Free memory before the next adapter
    del model, base
    torch.cuda.empty_cache()
    return auc, all_sids, all_labels, all_probs


def merge_and_save(base_model_id, adapter_path, out_dir):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    peft = PeftModel.from_pretrained(base, str(adapter_path))
    peft = peft.merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft.save_pretrained(str(out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model_id)
    tok.save_pretrained(str(out_dir))
    del peft, base
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["rag", "full"], required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Same dir the training script wrote to "
                             "(e.g. outputs/baselines/qlora_rag_8b)")
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--base-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=str,
                        default="adapters_best,adapters_last",
                        help="Comma-separated adapter subdirs to evaluate")
    parser.add_argument("--verbalizer-style",
                        choices=["yesno", "plusminus", "reserved"],
                        default="yesno",
                        help="Must match the style used at training time.")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    out_dir = Path(args.output_dir)
    results = {}
    for cand in args.candidates.split(","):
        cand = cand.strip()
        adapter_dir = out_dir / cand
        if not adapter_dir.exists():
            print(f"[skip] {adapter_dir} does not exist", flush=True)
            continue
        print(f"\n=== Evaluating {cand} ({adapter_dir}) ===", flush=True)
        auc, sids, labels, probs = eval_adapter(
            args.base_model, adapter_dir, args.variant,
            args.val_jsonl, args.max_seq_len,
            args.eval_batch_size, device,
            verbalizer_style=args.verbalizer_style,
        )
        print(f"  val AUC = {auc:.4f}", flush=True)
        results[cand] = {"auc": auc, "n": len(sids)}

    # Pick winner
    if not results:
        raise RuntimeError("No adapters evaluated.")
    winner = max(results.keys(), key=lambda k: results[k]["auc"])
    winner_dir = out_dir / winner
    print(f"\nWinner: {winner} with val AUC {results[winner]['auc']:.4f}",
          flush=True)

    # Write rerun summary
    summary_path = out_dir / "rerun_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_jsonl": args.val_jsonl,
            "max_seq_len": args.max_seq_len,
            "results": results,
            "winner": winner,
            "note": "Post-hoc eval with corrected logit position (-2).",
        }, f, indent=2)
    print(f"Wrote {summary_path}", flush=True)

    # Back up any existing merged dir (from buggy run), then re-merge winner
    merged_dir = out_dir / "merged"
    if merged_dir.exists():
        backup = out_dir / "merged_original_buggy"
        if not backup.exists():
            print(f"Backing up existing merged -> {backup}", flush=True)
            merged_dir.rename(backup)
        else:
            # Already backed up; assume we're re-running recovery
            import shutil
            shutil.rmtree(merged_dir)

    print(f"\nRe-merging {winner} into {merged_dir} ...", flush=True)
    merge_and_save(args.base_model, winner_dir, merged_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
