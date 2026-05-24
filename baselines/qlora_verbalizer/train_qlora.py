"""QLoRA fine-tuning for Llama-3.1-8B verbalizer mortality baseline.

Supervises only the first answer token (" Yes" or " No") after the assistant
prefill "Answer: ". All preceding positions are masked with -100.

Two variants sharing this script:
  --variant rag  : uses baselines.rag.run_rag_verbalizer (short context ~3K)
  --variant full : uses baselines.zeroshot.run_zeroshot_verbalizer (full history,
                   note-aware left-truncation to --max-seq-len, recency-biased)

Designed for torchrun --nproc_per_node=2 on a single node (DDP). Rank 0 owns
all I/O (logs, checkpoints, eval). Per-epoch checkpoint + resume is the only
checkpointing granularity: on restart the script looks for
<out>/trainer_state.json and continues at the next unfinished epoch.
"""
import argparse
import datetime
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)


def load_variant_module(variant: str, verbalizer_style: str = "yesno"):
    """Load and return the verbalizer module for the chosen variant.

    Monkey-patches the freeform script's build_chat_prompt with the verbalizer
    version (closing over the chosen verbalizer style) so that
    fit_prompt_to_budget uses the verbalizer prompt for truncation (same trick
    as at inference time).
    """
    if variant == "rag":
        subdir = REPO_ROOT / "baselines" / "rag"
        verb_name = "run_rag_verbalizer"
        free_name = "run_rag"
    elif variant == "full":
        subdir = REPO_ROOT / "baselines" / "zeroshot"
        verb_name = "run_zeroshot_verbalizer"
        free_name = "run_zeroshot"
    else:
        raise ValueError(f"Unknown variant={variant!r}")

    if str(subdir) not in sys.path:
        sys.path.insert(0, str(subdir))

    free_mod = importlib.import_module(free_name)
    verb_mod = importlib.import_module(verb_name)

    pos_str, neg_str = verb_mod.resolve_verbalizer(verbalizer_style)

    def _patched_build(tokenizer, patient_text):
        return verb_mod.build_chat_prompt(tokenizer, patient_text, pos_str, neg_str)

    free_mod.build_chat_prompt = _patched_build
    return verb_mod, free_mod


class DistributedLengthGroupedSampler(Sampler):
    """Length-bucketed distributed sampler.

    Each epoch: shuffle, slice into mega-batches of `mega_batch_mult * world_size`,
    sort each mega-batch by descending length, then assign the k-th sample of every
    world_size-block to rank k%world_size. Within each global step the
    `world_size` ranks see samples of similar length, so DDP no longer pays the
    max-of-N idle tax under heavy-tailed length distributions.
    """

    def __init__(self, lengths, world_size, rank, seed=0, mega_batch_mult=50):
        self.lengths = list(lengths)
        self.world_size = max(1, world_size)
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.mega_batch_size = max(self.world_size, mega_batch_mult * self.world_size)
        self.num_samples_per_rank = len(self.lengths) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        n_global = self.num_samples_per_rank * self.world_size
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        idx = idx[:n_global]
        out = []
        for start in range(0, n_global, self.mega_batch_size):
            mb = idx[start:start + self.mega_batch_size]
            mb.sort(key=lambda i: -self.lengths[i])
            out.extend(mb)
        return iter(out[self.rank::self.world_size])

    def __len__(self):
        return self.num_samples_per_rank


class VerbalizerDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, verb_mod, free_mod,
                 max_seq_len, yes_id, no_id, truncate_direction: str = "recent"):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.verb = verb_mod
        self.free = free_mod
        self.max_seq_len = max_seq_len
        self.yes_id = yes_id
        self.no_id = no_id
        self.pad_id = tokenizer.pad_token_id
        if truncate_direction not in ("recent", "early"):
            raise ValueError(f"Unknown truncate_direction: {truncate_direction!r}")
        self.truncate_direction = truncate_direction

    def __len__(self):
        return len(self.rows)

    def get_lengths(self):
        # Pre-tokenization length proxy from the JSONL `approx_tokens` field,
        # capped at max_seq_len. Used by DistributedLengthGroupedSampler to
        # bucket similar-length samples into the same global step (eliminates
        # max-of-N DDP waste under heavy-tailed length distributions).
        return [min(int(r.get("approx_tokens", self.max_seq_len)), self.max_seq_len)
                for r in self.rows]

    def __getitem__(self, idx):
        r = self.rows[idx]
        label = int(r["label"])
        patient_text = r["patient_text"]

        # Reserve 1 token for the answer target; 2 tokens safety slack.
        # NB: fit_prompt_to_budget uses the monkey-patched verbalizer
        # build_chat_prompt, which already appends ASSISTANT_PREFILL.
        budget = max(64, self.max_seq_len - 2)
        prompt, _, _, _, _ = self.free.fit_prompt_to_budget(
            self.tokenizer, patient_text, budget,
            direction=self.truncate_direction,
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.max_seq_len - 1:
            # Hard safety net at token level. Mirror direction:
            #   recent -> drop leftmost (oldest) tokens, keep tail
            #   early  -> drop rightmost (newest) tokens, keep head
            if self.truncate_direction == "recent":
                prompt_ids = prompt_ids[-(self.max_seq_len - 1):]
            else:
                prompt_ids = prompt_ids[: self.max_seq_len - 1]

        target = self.yes_id if label == 1 else self.no_id
        input_ids = prompt_ids + [target]
        labels = [-100] * len(prompt_ids) + [target]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "subject_id": int(r["subject_id"]),
            "patient_label": label,
        }


def left_pad_collate(batch, pad_id):
    """Left-pad a variable-length batch; labels pad with -100.

    Left padding keeps the answer token always at position -1, which makes
    the eval-time logit extraction trivially `logits[:, -1, :]`.
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    ids, labs, mask = [], [], []
    for b in batch:
        L = b["input_ids"].size(0)
        pad = max_len - L
        ids.append(F.pad(b["input_ids"], (pad, 0), value=pad_id))
        labs.append(F.pad(b["labels"], (pad, 0), value=-100))
        mask.append(F.pad(b["attention_mask"], (pad, 0), value=0))
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labs),
        "attention_mask": torch.stack(mask),
        "patient_label": torch.tensor([b["patient_label"] for b in batch],
                                       dtype=torch.long),
        "subject_id": torch.tensor([b["subject_id"] for b in batch],
                                     dtype=torch.long),
    }


@torch.no_grad()
def evaluate_val(model, tokenizer, val_ds, yes_id, no_id,
                 eval_batch_size, device, pad_id,
                 world_size=1, rank=0):
    """Distributed eval: each rank processes its shard, results all-gathered.

    Returns (val_auc, sids, labels, probs). On non-rank-0 ranks the val_auc
    is still computed (so every rank returns the same value); the caller
    typically only logs/saves on rank 0.
    """
    model.eval()
    if world_size > 1:
        sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
    else:
        sampler = None
    loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        collate_fn=lambda b: left_pad_collate(b, pad_id),
    )
    my_probs, my_labels, my_sids = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask, use_cache=False,
                        logits_to_keep=2)
        # input_ids ends with [..., prompt_last, target]. logits[t] predicts
        # input[t+1], so the target-prediction logits live at position -2.
        # With logits_to_keep=2, out.logits is (B, 2, V); -2 indexes the first slot
        # (= position L-2 of the full sequence).
        logits_last = out.logits[:, -2, :]  # (B, V)
        lp_yes = logits_last[:, yes_id]
        lp_no = logits_last[:, no_id]
        m = torch.maximum(lp_yes, lp_no)
        exp_yes = torch.exp(lp_yes - m)
        exp_no = torch.exp(lp_no - m)
        prob_yes = (exp_yes / (exp_yes + exp_no)).float().cpu().numpy()
        my_probs.extend(prob_yes.tolist())
        my_labels.extend(batch["patient_label"].tolist())
        my_sids.extend(batch["subject_id"].tolist())

    if world_size > 1:
        # Gather python lists from every rank; DistributedSampler pads the
        # val set so every rank sees the same count — dedupe by subject_id
        # before computing AUC.
        gathered_probs = [None] * world_size
        gathered_labels = [None] * world_size
        gathered_sids = [None] * world_size
        dist.all_gather_object(gathered_probs, my_probs)
        dist.all_gather_object(gathered_labels, my_labels)
        dist.all_gather_object(gathered_sids, my_sids)
        all_probs, all_labels, all_sids = [], [], []
        seen = set()
        for probs_r, labels_r, sids_r in zip(gathered_probs, gathered_labels,
                                              gathered_sids):
            for p, l, s in zip(probs_r, labels_r, sids_r):
                if s in seen:
                    continue
                seen.add(s)
                all_probs.append(p)
                all_labels.append(l)
                all_sids.append(s)
    else:
        all_probs, all_labels, all_sids = my_probs, my_labels, my_sids

    auc = float(roc_auc_score(all_labels, all_probs))
    return auc, all_sids, all_labels, all_probs


def save_adapter(model_unwrapped, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_unwrapped.save_pretrained(str(out_dir))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["rag", "full"], required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip post-training adapter merge step")
    parser.add_argument("--subset-train", type=int, default=-1,
                        help="If >0, keep only first N train rows (for smoke tests)")
    parser.add_argument("--verbalizer-style",
                        choices=["yesno", "plusminus", "reserved"],
                        default="yesno",
                        help="Target token pair: yesno (' Yes'/' No'), "
                             "plusminus ('+'/'-'), or reserved "
                             "(<|reserved_special_token_0|>/<|reserved_special_token_1|>)")
    parser.add_argument("--train-embed-and-head", action="store_true",
                        help="Add embed_tokens + lm_head to PEFT modules_to_save "
                             "(trained in full bf16 precision, bypassing NF4). "
                             "Required when using --verbalizer-style reserved, "
                             "since reserved-token lm_head rows are otherwise frozen "
                             "at random init and the loss cannot move "
                             "(see WORKLOG 2026-05-03 reserved-token negative result).")
    parser.add_argument("--truncate-direction", choices=["recent", "early"],
                        default="recent",
                        help="When patient_text exceeds the budget, which end of "
                             "the chronological note list to keep. 'recent' (default, "
                             "backward-compatible) keeps the most recent notes; "
                             "'early' keeps the earliest notes (first-encounter, "
                             "baseline labs). Threaded through to "
                             "fit_prompt_to_budget and the token-level safety net.")
    parser.add_argument("--length-bucketed", action="store_true",
                        help="Use DistributedLengthGroupedSampler for the train "
                             "loader: groups similar-length samples into the same "
                             "global step so DDP no longer pays the max-of-N idle "
                             "tax under heavy-tailed length distributions. "
                             "Off by default to preserve existing run reproducibility.")
    parser.add_argument("--early-stop-on-val-drop", action="store_true",
                        help="Stop training as soon as val AUC drops below the "
                             "previous epoch's val AUC. The merge step still uses "
                             "adapters_best (highest val AUC seen). Designed for "
                             "the val curve we observe here: monotonic rise that "
                             "eventually plateaus or starts overfitting.")
    args = parser.parse_args()

    # ---- Distributed init ----
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # 2h NCCL timeout so long single-rank operations (e.g. eval on val set,
        # adapter save, merge) don't blow up the other rank sitting at a barrier.
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed(args.seed + rank)
    out_dir = Path(args.output_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    rank_print(f"=== QLoRA verbalizer training ({args.variant}) ===")
    rank_print(f"world_size={world_size}  rank={rank}  device={device}")
    rank_print(f"args: {vars(args)}")

    # ---- Load verbalizer prompt module + tokenizer ----
    verb_mod, free_mod = load_variant_module(args.variant, args.verbalizer_style)
    rank_print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pos_str, neg_str = verb_mod.resolve_verbalizer(args.verbalizer_style)
    pos_ids = tokenizer.encode(pos_str, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_str, add_special_tokens=False)
    assert len(pos_ids) == 1, (
        f"Verbalizer pos token {pos_str!r} did not tokenize to 1 token: {pos_ids}"
    )
    assert len(neg_ids) == 1, (
        f"Verbalizer neg token {neg_str!r} did not tokenize to 1 token: {neg_ids}"
    )
    yes_id = pos_ids[0]
    no_id = neg_ids[0]
    rank_print(
        f"Verbalizer style: {args.verbalizer_style}  "
        f"Token ids — pos {pos_str!r}={yes_id}  neg {neg_str!r}={no_id}  "
        f"pad={tokenizer.pad_token_id}"
    )

    # ---- Datasets ----
    train_ds = VerbalizerDataset(args.train_jsonl, tokenizer, verb_mod, free_mod,
                                 args.max_seq_len, yes_id, no_id,
                                 truncate_direction=args.truncate_direction)
    val_ds = VerbalizerDataset(args.val_jsonl, tokenizer, verb_mod, free_mod,
                               args.max_seq_len, yes_id, no_id,
                               truncate_direction=args.truncate_direction)
    if args.subset_train > 0:
        train_ds.rows = train_ds.rows[: args.subset_train]
    rank_print(f"Train rows: {len(train_ds)}  Val rows: {len(val_ds)}")

    if world_size > 1:
        if args.length_bucketed:
            train_sampler = DistributedLengthGroupedSampler(
                lengths=train_ds.get_lengths(),
                world_size=world_size,
                rank=rank,
                seed=args.seed,
            )
            rank_print(f"Using length-bucketed sampler "
                       f"(mega-batch={train_sampler.mega_batch_size}, "
                       f"samples_per_rank={len(train_sampler)})")
        else:
            train_sampler = DistributedSampler(train_ds, shuffle=True,
                                               seed=args.seed, drop_last=False)
    else:
        train_sampler = None

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: left_pad_collate(b, pad_id),
        drop_last=False,
    )

    # ---- Model (4-bit QLoRA) ----
    rank_print("Loading base model (4-bit NF4) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    if args.train_embed_and_head:
        lora_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    if is_main_process():
        model.print_trainable_parameters()

    # ---- Resume support ----
    state_path = out_dir / "trainer_state.json"
    completed_epochs = 0
    best_val_auc = -1.0
    best_epoch = -1
    history = []
    # Last epoch's val AUC, tracked on every rank for the early-stop check.
    # Initialized to -1 (so the first epoch can never trigger early stop).
    prev_val_auc = -1.0
    if state_path.exists():
        with state_path.open("r") as fh:
            state = json.load(fh)
        completed_epochs = int(state.get("completed_epochs", 0))
        best_val_auc = float(state.get("best_val_auc", -1.0))
        best_epoch = int(state.get("best_epoch", -1))
        history = state.get("history", [])
        if history:
            prev_val_auc = float(history[-1].get("val_auc", -1.0))
        last_adapter = out_dir / "adapters_last"
        if completed_epochs > 0 and last_adapter.exists():
            rank_print(f"Resuming from {last_adapter} (completed_epochs={completed_epochs}, "
                       f"best_val_auc={best_val_auc:.4f})")
            # Reload adapter weights into the PEFT model
            adapter_state = torch.load(last_adapter / "adapter_model.safetensors"
                                        if (last_adapter / "adapter_model.safetensors").exists()
                                        else last_adapter / "adapter_model.bin",
                                        map_location="cpu", weights_only=False) \
                if (last_adapter / "adapter_model.bin").exists() else None
            # Simpler: reload via PeftModel helper
            from peft.utils.save_and_load import set_peft_model_state_dict
            from safetensors.torch import load_file as safe_load
            sfile = last_adapter / "adapter_model.safetensors"
            if sfile.exists():
                sd = safe_load(str(sfile))
            else:
                sd = torch.load(last_adapter / "adapter_model.bin", map_location="cpu")
            set_peft_model_state_dict(model, sd)
            rank_print("Adapter state loaded.")

    if completed_epochs >= args.epochs:
        rank_print("All epochs already completed; nothing to do. Proceeding to merge step.")

    # ---- Wrap in DDP ----
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- Optimizer + scheduler ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=args.lr)
        rank_print("Using 8-bit AdamW.")
    except Exception as e:
        rank_print(f"8-bit AdamW unavailable ({e}); falling back to torch AdamW.")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # ---- Training loop ----
    global_step = 0
    if completed_epochs > 0:
        # Fast-forward the scheduler to the current position.
        for _ in range(completed_epochs * steps_per_epoch):
            scheduler.step()
            global_step += 1

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(completed_epochs, args.epochs):
        rank_print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch + args.seed)
        model.train()

        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        accum_loss = 0.0
        accum_n = 0
        epoch_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False,
                            logits_to_keep=2)
            # logits_to_keep=2 => out.logits is (B, 2, V) covering positions L-2, L-1.
            # Only position L-2 predicts a non-(-100) label (the answer at L-1);
            # all earlier label positions are -100 by construction.
            loss_logits = out.logits[:, -2, :]  # (B, V), position L-2
            loss_target = labels[:, -1]         # (B,), answer token id
            loss = loss_fct(loss_logits, loss_target)

            loss = loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * args.grad_accum_steps
            accum_n += 1

            is_step_boundary = ((step_idx + 1) % args.grad_accum_steps == 0) or \
                               (step_idx + 1 == len(train_loader))
            if is_step_boundary:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                cur_loss = accum_loss / max(1, accum_n)
                epoch_loss_sum += cur_loss
                epoch_loss_n += 1
                accum_loss = 0.0
                accum_n = 0

                if global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    rank_print(f"[ep {epoch+1} step {global_step}] "
                               f"loss={cur_loss:.4f}  lr={lr_now:.2e}  "
                               f"seq_len={ids.size(1)}")

        epoch_time = time.time() - epoch_t0
        mean_epoch_loss = epoch_loss_sum / max(1, epoch_loss_n)
        rank_print(f"Epoch {epoch+1} train loss: {mean_epoch_loss:.4f}  "
                   f"time: {epoch_time:.1f}s")

        # ---- Sync + save-before-eval + distributed eval ----
        if world_size > 1:
            dist.barrier()

        # Save adapters_last FIRST so a subsequent crash during eval still
        # leaves us with the epoch's weights on disk.
        peft_model = model.module if isinstance(model, DDP) else model
        if is_main_process():
            save_adapter(peft_model, out_dir / "adapters_last")
        if world_size > 1:
            dist.barrier()

        # Distributed eval — every rank forwards its shard, all-gather results.
        # Using the DDP-wrapped model so attention mask handling is consistent.
        val_auc, _sids, _lbls, _probs = evaluate_val(
            model, tokenizer, val_ds, yes_id, no_id,
            args.eval_batch_size, device, pad_id,
            world_size=world_size, rank=rank,
        )

        if is_main_process():
            rank_print(f"Epoch {epoch+1} val AUC: {val_auc:.4f}")

            history.append({"epoch": epoch + 1,
                            "train_loss": mean_epoch_loss,
                            "val_auc": val_auc,
                            "epoch_time_s": epoch_time})

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                save_adapter(peft_model, out_dir / "adapters_best")
                rank_print(f"  -> new best val AUC {best_val_auc:.4f} @ epoch {best_epoch}")

            with state_path.open("w") as fh:
                json.dump({
                    "completed_epochs": epoch + 1,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                    "history": history,
                    "args": vars(args),
                }, fh, indent=2)

            with (out_dir / "training_metrics.json").open("w") as fh:
                json.dump({
                    "history": history,
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                }, fh, indent=2)
        if world_size > 1:
            dist.barrier()

        # ---- Early stop ----
        # All ranks share val_auc (evaluate_val all-gathers + computes locally),
        # so each rank takes the same decision without any extra collectives.
        if args.early_stop_on_val_drop and prev_val_auc >= 0 and val_auc < prev_val_auc:
            rank_print(
                f"  Early stop: val AUC {val_auc:.4f} < previous epoch's "
                f"{prev_val_auc:.4f}. Stopping training. "
                f"adapters_best (val AUC {best_val_auc:.4f} @ ep {best_epoch}) "
                f"will be used for the merge."
            )
            break
        prev_val_auc = val_auc

    # ---- End of training: merge adapters ----
    if is_main_process() and not args.skip_merge:
        rank_print("\nMerging best adapter into base weights ...")
        # Free DDP + 4-bit model memory
        if isinstance(model, DDP):
            model = model.module
        del model
        torch.cuda.empty_cache()

        # Re-load base in bf16 and merge
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank},
        )
        adapter_dir = out_dir / "adapters_best"
        if not adapter_dir.exists():
            adapter_dir = out_dir / "adapters_last"
        rank_print(f"Loading adapter from {adapter_dir}")
        peft = PeftModel.from_pretrained(base, str(adapter_dir))
        peft = peft.merge_and_unload()
        merged_dir = out_dir / "merged"
        peft.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        rank_print(f"Merged model saved to {merged_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
