"""Binary verbalizer RAG baseline for one-year mortality prediction.

Same verbalizer protocol as baselines/full_context/run_full_context_verbalizer.py —
assistant-turn prefill with "Answer: " so the first generated token is at the
Yes/No decision position, then extract softmax-normalized P(Yes) from the
position-0 token logprobs.

Input: data/rag_test_patients.jsonl (built by scripts/build_rag_dataset.py).
Prompts reference "retrieved clinical evidence" rather than "complete clinical
history". Truncation is a near no-op because retrieved context is small, but
the code path is preserved for safety.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make `run_rag` importable regardless of CWD (sbatch may cd to repo root)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_rag import fit_prompt_to_budget, _iter_balanced_json_objects  # noqa: E402


def resolve_verbalizer(style: str):
    """Map verbalizer style to (pos_str, neg_str) target token strings.

    `pos_str` is the target for label=1 (death), `neg_str` for label=0.
    Both must encode to a single token under the base-model tokenizer.
    """
    if style == "yesno":
        return " Yes", " No"
    if style == "plusminus":
        return "+", "-"
    if style == "reserved":
        return "<|reserved_special_token_0|>", "<|reserved_special_token_1|>"
    raise ValueError(f"Unknown verbalizer style: {style!r}")


def build_system_prompt(pos_str: str, neg_str: str) -> str:
    pos = pos_str.strip()
    neg = neg_str.strip()
    return (
        "You are a clinical interpretability assistant. Given retrieved clinical "
        "evidence from a patient's medical record (selected clinical note "
        "excerpts and ICD diagnostic codes most relevant to mortality risk), you "
        f"(1) answer a binary ({pos}/{neg}) question about whether the patient "
        "will die within 1 year of their last hospital discharge, and (2) produce "
        "a conservative, evidence-grounded clinical report identifying the top 5 "
        "predictive factors drawn ONLY from the provided retrieved evidence.\n\n"
        "Hard rules:\n"
        "- Do NOT infer diagnoses, severity, or mechanisms beyond what is explicitly "
        "stated in the evidence.\n"
        "- Keep physiologic reasoning general and non-specific.\n"
        "- Provide only high-level care considerations, not specific treatment plans.\n"
        "- Cite EVERY supporting-evidence line using one of these EXACT formats, "
        "copying the identifier verbatim from the retrieved evidence:\n"
        "    (note_id: XXXXX)       for clinical-note evidence\n"
        "    (ICD code: XXXXX-X)    for ICD-based evidence\n"
        "- Do not use any other citation format. Omit attribution scores.\n"
        "- In the report, do NOT state a numeric probability. Instead, phrase the "
        "prediction as \"the patient is predicted to die within 1 year after "
        f"hospital discharge\" (if Answer is {pos}) or \"the patient is predicted "
        f"to survive at least 1 year after hospital discharge\" (if Answer is {neg})."
    )


def build_user_instruction(pos_str: str, neg_str: str) -> str:
    pos = pos_str.strip()
    neg = neg_str.strip()
    return (
        "Below is retrieved clinical evidence from a single patient's medical "
        "record. These are the note excerpts and diagnostic codes most relevant "
        "to mortality risk assessment. Each note header includes its note_id and "
        "each code line includes its ICD code-version identifier.\n\n"
        "Question: Will this patient die within one year after their last "
        "discharge?\n\n"
        f"Respond in EXACTLY this format, with the literal token {pos} or {neg} "
        "on the Answer line and a single JSON object on the next line (no markdown "
        "fences, no other text):\n"
        f"Answer: <{pos} or {neg}>\n"
        '{"report": "<structured clinical report as described below, with \\n line breaks>"}\n\n'
        "The report string must follow EXACTLY this structure (use \\n for line "
        "breaks inside the JSON string):\n\n"
        "[Prediction + Explanation]\n"
        "- State plainly that the patient is predicted to die within 1 year after "
        f"hospital discharge (if you answered {pos}) or predicted to survive at "
        f"least 1 year after hospital discharge (if you answered {neg}). Do NOT "
        "state a numeric probability.\n"
        "- Summarize in plain, non-technical language the main clinical patterns or "
        "types of information suggested by the retrieved evidence.\n"
        "- Mention \"chronic\", \"acute\", or \"end-stage\" only if clearly supported "
        "by the plain text.\n"
        "- You may group related factors under a brief, clinically intuitive heading "
        "(e.g., \"Chronic medical conditions\") only if the grouping is clearly "
        "supported by the evidence.\n\n"
        "[Predictive Factors + Evidence-Based Reasoning]\n"
        "For each of the top 5 factors (numbered 1 through 5):\n"
        "  1. Factor summary: a strictly descriptive paraphrase of the supporting "
        "evidence or ICD label. Do NOT add diagnoses, mechanisms, or severity "
        "descriptors not explicitly present in the evidence.\n"
        "  2. Supporting evidence: a short direct quote (for notes) or the ICD "
        "description. End EVERY supporting-evidence line with a citation in one of "
        "these EXACT formats:\n"
        "       (note_id: XXXXX)       for clinical-note evidence\n"
        "       (ICD code: XXXXX-X)    for ICD-based evidence\n"
        "  3. Reasoning (general and conservative, 1 sentence): (evidence as stated) "
        "\u2192 (broad physiologic or prognostic implication, e.g. \"may reflect advanced "
        "disease\" or \"may worsen overall organ function\") \u2192 (impact on 1-year "
        "mortality risk).\n\n"
        "[Reasoning Lens]\n"
        "In 3-4 sentences, synthesize the 5 predictive factors into a coherent "
        "narrative explaining why the patient is predicted to die or survive "
        "within 1 year. Do NOT introduce new diagnoses or events. Stay strictly "
        "within what is supported by the predictive factors.\n\n"
        "[Recommendations]\n"
        "- General: 1-2 brief considerations related to prognosis communication, "
        "alignment of goals-of-care, and symptom-focused support. Use tentative "
        "language (\"may\", \"could be relevant\", \"may warrant attention\").\n"
        "- Per Predictive Factor (one per factor, numbered to match): a brief care "
        "consideration linked directly to that factor's evidence. Tentative "
        "language only. Do NOT recommend specific diagnostics, medications, or "
        "procedures.\n\n"
        "Retrieved clinical evidence:\n"
    )


# Backward-compat module-level constants (default Yes/No verbalizer)
SYSTEM_PROMPT = build_system_prompt(" Yes", " No")
USER_INSTRUCTION = build_user_instruction(" Yes", " No")

# Prefill: the assistant turn starts with "Answer: " so position 0 is the
# verbalizer token (e.g. " Yes" or "+").
ASSISTANT_PREFILL = "Answer: "


def parse_report_from_completion(raw_text: str) -> str:
    """Extract the 'report' field from a verbalizer completion.

    The expected output after "Answer: " prefill is:
        Yes
        {"report": "..."}
    We scan for the first balanced {...} block with a 'report' key.
    """
    for candidate in _iter_balanced_json_objects(raw_text):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and ("report" in obj or "summary" in obj):
            return str(obj.get("report", obj.get("summary", "")))
    lines = raw_text.split("\n", 1)
    if len(lines) > 1:
        return lines[1].strip()
    return ""


def build_chat_prompt(tokenizer, patient_text: str,
                      pos_str: str = " Yes", neg_str: str = " No",
                      enable_thinking=None) -> str:
    """Build prompt with assistant prefill so first generated token is the
    verbalizer (Yes/No or +/- depending on pos_str/neg_str)."""
    messages = [
        {"role": "system", "content": build_system_prompt(pos_str, neg_str)},
        {"role": "user", "content": build_user_instruction(pos_str, neg_str) + patient_text},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    return prompt + ASSISTANT_PREFILL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to rag_test_patients.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions JSONL")
    parser.add_argument("--model-id", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-patients", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-input-tokens", type=int, default=122880)
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens for the Yes/No answer plus structured report")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--verbalizer-style",
                        choices=["yesno", "plusminus", "reserved"],
                        default="yesno",
                        help="Target token pair: yesno (' Yes'/' No'), "
                             "plusminus ('+'/'-'), or reserved "
                             "(<|reserved_special_token_0|>/<|reserved_special_token_1|>)")
    parser.add_argument("--rope-scaling-json", type=str, default=None,
                        help="Optional JSON for vLLM hf_overrides={'rope_scaling': ...}.")
    parser.add_argument("--enable-thinking", choices=["auto", "true", "false"],
                        default="auto",
                        help="Pass enable_thinking=<bool> to apply_chat_template; "
                             "'auto' (default) does not pass the kwarg. For Qwen3 "
                             "verbalizer runs use 'false'.")
    args = parser.parse_args()

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load patients
    patients = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            patients.append(json.loads(line))
    if args.max_patients > 0:
        patients = patients[: args.max_patients]
    print(f"Loaded {len(patients)} patients from {in_path}", flush=True)

    # Load tokenizer and find verbalizer token IDs
    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    pos_str, neg_str = resolve_verbalizer(args.verbalizer_style)
    pos_ids = tokenizer.encode(pos_str, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_str, add_special_tokens=False)
    assert len(pos_ids) == 1, (
        f"Verbalizer pos token {pos_str!r} did not tokenize to 1 token: {pos_ids}"
    )
    assert len(neg_ids) == 1, (
        f"Verbalizer neg token {neg_str!r} did not tokenize to 1 token: {neg_ids}"
    )
    yes_token_id = pos_ids[0]
    no_token_id = neg_ids[0]
    print(f"Verbalizer style: {args.verbalizer_style}", flush=True)
    print(f"Token IDs — pos {pos_str!r}: {yes_token_id}, "
          f"neg {neg_str!r}: {no_token_id}", flush=True)

    # Monkey-patch run_rag.build_chat_prompt so fit_prompt_to_budget uses our
    # verbalizer prompt (with prefill) during truncation, not the free-form JSON one.
    import run_rag
    _orig_build = run_rag.build_chat_prompt
    run_rag.build_chat_prompt = lambda tok, pt, enable_thinking=None: build_chat_prompt(
        tok, pt, pos_str, neg_str, enable_thinking=enable_thinking
    )

    # Build prompts with truncation
    print("Building prompts ...", flush=True)
    prompts = []
    meta = []
    for p in patients:
        prompt, ntok, truncated, kept, total_notes = fit_prompt_to_budget(
            tokenizer, p["patient_text"], args.max_input_tokens,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)
        meta.append({
            "subject_id": p["subject_id"],
            "label": p["label"],
            "num_notes": p.get("num_notes"),
            "num_codes": p.get("num_codes"),
            "num_input_tokens": ntok,
            "truncated": truncated,
            "notes_kept": kept,
            "notes_total": total_notes,
        })

    # Restore original
    run_rag.build_chat_prompt = _orig_build

    tok_counts = [m["num_input_tokens"] for m in meta]
    n_trunc = sum(1 for m in meta if m["truncated"])
    print(
        f"Prompt token stats: min={min(tok_counts)} "
        f"median={sorted(tok_counts)[len(tok_counts)//2]} "
        f"max={max(tok_counts)} truncated={n_trunc}/{len(meta)}",
        flush=True,
    )

    # Load vLLM
    print(f"Loading vLLM model: {args.model_id}", flush=True)
    llm_kwargs = dict(
        model=args.model_id,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
    )
    if args.rope_scaling_json:
        rope_scaling = json.loads(args.rope_scaling_json)
        llm_kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
        print(f"vLLM hf_overrides rope_scaling: {rope_scaling}", flush=True)
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"vLLM engine ready in {time.time() - t0:.1f}s", flush=True)

    # Full generation with logprobs — position 0 is the verbalizer token.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    # Run inference
    n_done = 0
    n_yes = 0
    n_no = 0
    n_other = 0
    probs_list = []

    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            batch_meta = meta[batch_start: batch_start + args.batch_size]
            t_batch = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - t_batch

            for m, out in zip(batch_meta, outputs):
                raw_text = out.outputs[0].text if out.outputs else ""
                logprobs_list = out.outputs[0].logprobs if out.outputs else []

                first_token_logprobs = logprobs_list[0] if logprobs_list else {}

                # What token was actually generated at position 0?
                first_token = ""
                if logprobs_list:
                    for tid, lp_obj in first_token_logprobs.items():
                        if hasattr(lp_obj, 'rank') and lp_obj.rank == 1:
                            first_token = lp_obj.decoded_token
                            break
                    if not first_token:
                        first_token = raw_text.split()[0] if raw_text.strip() else ""

                if first_token.strip() == pos_str.strip():
                    n_yes += 1
                elif first_token.strip() == neg_str.strip():
                    n_no += 1
                else:
                    n_other += 1

                # Extract log-probabilities for Yes and No tokens
                logp_yes = None
                logp_no = None
                for token_id, logprob_obj in first_token_logprobs.items():
                    if token_id == yes_token_id:
                        logp_yes = logprob_obj.logprob
                    elif token_id == no_token_id:
                        logp_no = logprob_obj.logprob

                # Compute normalized P(Yes)
                if logp_yes is not None and logp_no is not None:
                    max_logp = max(logp_yes, logp_no)
                    prob_yes = math.exp(logp_yes - max_logp) / (
                        math.exp(logp_yes - max_logp) + math.exp(logp_no - max_logp)
                    )
                elif logp_yes is not None:
                    prob_yes = 1.0
                elif logp_no is not None:
                    prob_yes = 0.0
                else:
                    prob_yes = 0.5  # fallback

                probs_list.append(prob_yes)

                # Parse the structured report JSON from the text after Yes/No
                report = parse_report_from_completion(raw_text)

                record = dict(m)
                record["raw_completion"] = ASSISTANT_PREFILL + raw_text
                record["generated_token"] = first_token.strip()
                record["logp_yes"] = logp_yes
                record["logp_no"] = logp_no
                record["parsed_probability"] = prob_yes
                record["parsed_report"] = report
                record["parse_ok"] = logp_yes is not None or logp_no is not None
                record["verbalizer_style"] = args.verbalizer_style
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            n_done += len(batch_prompts)
            print(
                f"[{n_done}/{len(prompts)}] batch {len(batch_prompts)} prompts in "
                f"{elapsed:.1f}s",
                flush=True,
            )

    probs_arr = np.array(probs_list)
    print(f"\nWrote {n_done} predictions to {out_path}", flush=True)
    print(f"Position-0 token distribution: {pos_str.strip()}={n_yes}  "
          f"{neg_str.strip()}={n_no}  Other={n_other}", flush=True)
    print(f"P(Yes) stats: min={probs_arr.min():.4f}  median={np.median(probs_arr):.4f}  "
          f"max={probs_arr.max():.4f}  mean={probs_arr.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
