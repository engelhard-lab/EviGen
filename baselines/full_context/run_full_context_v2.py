"""Full-context LLM baseline for one-year mortality prediction.

Consumes data/full_context_test_patients.jsonl (produced by
scripts/build_full_context_dataset.py) and, for each patient, asks
Llama-3.1-8B-Instruct to return a STRICT JSON object with a risk probability
and an evidence summary. Uses vLLM for batched GPU inference.

Truncation policy (when the tokenized prompt exceeds --max-input-tokens):
    drop clinical notes from the OLDEST end until the prompt fits. The ICD
    code list is always preserved in full and the most-recent note is kept.

Writes outputs/full_context/predictions[_suffix].jsonl with one row per patient.
"""

import argparse
import json
import re
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a clinical interpretability assistant. Given a patient's complete "
    "clinical history (clinical notes and ICD diagnostic codes in chronological "
    "order), you (1) estimate the probability of all-cause mortality within 1 "
    "year of the last hospital discharge, and (2) produce a conservative, "
    "evidence-grounded clinical report identifying the top 5 predictive factors "
    "drawn ONLY from the provided history.\n\n"
    "Hard rules:\n"
    "- Do NOT infer diagnoses, severity, or mechanisms beyond what is explicitly "
    "stated in the evidence.\n"
    "- Keep physiologic reasoning general and non-specific.\n"
    "- Provide only high-level care considerations, not specific treatment plans.\n"
    "- Cite EVERY supporting-evidence line using one of these EXACT formats, "
    "copying the identifier verbatim from the patient history:\n"
    "    (note_id: XXXXX)       for clinical-note evidence\n"
    "    (ICD code: XXXXX-X)    for ICD-based evidence\n"
    "- Do not use any other citation format. Omit attribution scores.\n\n"
    "Respond with STRICT JSON only."
)

USER_INSTRUCTION = (
    "Below is a single patient's complete clinical history, ordered from oldest "
    "to newest. Each note header includes its note_id and each code line "
    "includes its ICD code-version identifier.\n\n"
    "Step 1. Estimate the probability (float in [0.0, 1.0]) that this patient "
    "will die within one year after their last discharge.\n\n"
    "Step 2. Write a clinical report following EXACTLY this structure. Output "
    "it as the value of the \"report\" field (use \\n for line breaks).\n\n"
    "[Prediction + Explanation]\n"
    "- Clearly state that this is a prediction of all-cause mortality within 1 "
    "year after hospital discharge, and report the probability you estimated.\n"
    "- Summarize in plain, non-technical language the main clinical patterns or "
    "types of information suggested by the evidence.\n"
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
    "description. For note-based evidence, explicitly include the note's "
    "documented Date and Age as shown in the history. End EVERY supporting-"
    "evidence line with a citation in one of these EXACT formats:\n"
    "       (note_id: XXXXX)       for clinical-note evidence\n"
    "       (ICD code: XXXXX-X)    for ICD-based evidence\n"
    "  3. Reasoning:\n"
    "       - Write one or two plain sentences connecting the evidence to "
    "the patient's mortality risk.\n"
    "       - Start by restating the specific clinical finding from the "
    "evidence, then explain a broad physiologic or prognostic implication "
    "(e.g., may reflect advanced disease, may worsen overall organ "
    "function), then connect to 1-year mortality risk.\n"
    "       - Keep reasoning general and conservative; avoid naming "
    "complications or mechanisms not mentioned in the evidence.\n"
    "       - Do NOT use placeholder text, arrow notation, or template-style "
    "formatting. Write in natural clinical prose.\n"
    "       - Some evidence reflects healthcare-process documentation "
    "(e.g., teach-back attestations, registry merges, template-language "
    "quotations, routine exam templates, absence-of-documentation lines, "
    "counseling check-boxes) rather than a patient-specific clinical "
    "finding. When this is the case, acknowledge it as a healthcare-"
    "engagement proxy (e.g., \"may reflect healthcare engagement, which "
    "correlates with diagnosis rates\") \u2014 do not reframe it as a direct "
    "clinical observation. The reasoning must stop at the engagement-"
    "correlation framing; do not construct a secondary pathway from "
    "engagement to clinical risk.\n\n"
    "[Reasoning Lens]\n"
    "In 3-4 sentences, synthesize the 5 predictive factors into a coherent "
    "narrative explaining why the estimated 1-year mortality risk is high or "
    "low. Do NOT introduce new diagnoses or events. Stay strictly within what "
    "is supported by the predictive factors.\n\n"
    "[Recommendations]\n"
    "- General: 1-2 brief considerations related to prognosis communication, "
    "alignment of goals-of-care, and symptom-focused support. Use tentative "
    "language (\"may\", \"could be relevant\", \"may warrant attention\").\n"
    "- Per Predictive Factor (one per factor, numbered to match): a brief care "
    "consideration linked directly to that factor's evidence. Tentative "
    "language only. Do NOT recommend specific diagnostics, medications, or "
    "procedures.\n\n"
    "Output format: respond with ONLY a single JSON object, no prose outside "
    "the object, no markdown fences, matching exactly this schema:\n"
    '{"probability": <float in [0.0, 1.0]>, '
    '"report": "<structured clinical report as described above, with \\n line breaks>"}\n\n'
    "Patient history:\n"
)


NOTE_BLOCK_RE = re.compile(r"(?m)^\[Note \d+\] ")
NOTES_HEADER_RE = re.compile(r"^===== CLINICAL NOTES.*$", re.MULTILINE)
CODES_HEADER_RE = re.compile(r"^===== DIAGNOSTIC CODES.*$", re.MULTILINE)


def _parse_patient_text(patient_text: str):
    """Split a formatted patient_text into (preamble, notes_header, notes_blocks, codes_section).

    notes_blocks is a list of strings in display order (oldest -> newest). Each
    block starts with its '[Note N] ' header and ends at a blank line before
    the next '[Note N] ' marker or the codes section.
    """
    notes_header_match = NOTES_HEADER_RE.search(patient_text)
    codes_header_match = CODES_HEADER_RE.search(patient_text)
    if not notes_header_match or not codes_header_match:
        # Fall back: no truncation possible
        return patient_text, "", [], ""

    preamble = patient_text[: notes_header_match.start()]
    notes_header_line = patient_text[notes_header_match.start(): notes_header_match.end()]

    notes_body = patient_text[notes_header_match.end() : codes_header_match.start()]
    codes_section = patient_text[codes_header_match.start():]

    # Split notes_body into blocks on '[Note N] '
    notes_blocks = []
    positions = [m.start() for m in NOTE_BLOCK_RE.finditer(notes_body)]
    positions.append(len(notes_body))
    for i in range(len(positions) - 1):
        block = notes_body[positions[i]: positions[i + 1]].rstrip() + "\n"
        notes_blocks.append(block)

    return preamble, notes_header_line, notes_blocks, codes_section


def _reassemble(preamble, notes_header_line, kept_note_blocks, codes_section,
                n_kept, n_total, direction: str = "recent"):
    notes_section = notes_header_line + "\n"
    kept_text = "".join(block + "\n" for block in kept_note_blocks)
    if n_kept < n_total:
        omitted = n_total - n_kept
        if direction == "recent":
            marker = f"[Older {omitted} notes omitted due to context length]\n\n"
            notes_section += marker + kept_text
        elif direction == "early":
            marker = f"[Newer {omitted} notes omitted due to context length]\n\n"
            notes_section += kept_text + marker
        else:
            raise ValueError(f"Unknown truncate direction: {direction!r}")
    else:
        notes_section += kept_text
    return preamble + notes_section + codes_section


def build_chat_prompt(tokenizer, patient_text: str,
                      enable_thinking=None,
                      extra_system_text: str = "") -> str:
    sys_text = SYSTEM_PROMPT + (("\n\n" + extra_system_text) if extra_system_text else "")
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": USER_INSTRUCTION + patient_text},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def fit_prompt_to_budget(tokenizer, patient_text: str, max_input_tokens: int,
                         direction: str = "recent",
                         enable_thinking=None,
                         extra_system_text: str = ""):
    """Return (prompt_str, num_tokens, truncated_bool, notes_kept, notes_total).

    direction="recent" (default): keep the largest tail `notes_blocks[-mid:]`.
    direction="early": keep the largest head `notes_blocks[:mid]`.
    """
    if direction not in ("recent", "early"):
        raise ValueError(f"Unknown truncate direction: {direction!r}")
    full_prompt = build_chat_prompt(tokenizer, patient_text,
                                    enable_thinking=enable_thinking,
                                    extra_system_text=extra_system_text)
    num_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))

    preamble, notes_header_line, notes_blocks, codes_section = _parse_patient_text(patient_text)
    notes_total = len(notes_blocks)

    if num_tokens <= max_input_tokens or notes_total <= 1:
        return full_prompt, num_tokens, False, notes_total, notes_total

    # Binary search the largest number of notes (from the chosen end) that fit.
    lo, hi = 1, notes_total
    best_kept = 1
    best_prompt = None
    best_tokens = None
    while lo <= hi:
        mid = (lo + hi) // 2
        kept_blocks = notes_blocks[-mid:] if direction == "recent" else notes_blocks[:mid]
        candidate_text = _reassemble(
            preamble, notes_header_line, kept_blocks, codes_section, mid, notes_total, direction
        )
        candidate_prompt = build_chat_prompt(tokenizer, candidate_text,
                                             enable_thinking=enable_thinking,
                                             extra_system_text=extra_system_text)
        candidate_tokens = len(tokenizer.encode(candidate_prompt, add_special_tokens=False))
        if candidate_tokens <= max_input_tokens:
            best_kept = mid
            best_prompt = candidate_prompt
            best_tokens = candidate_tokens
            lo = mid + 1
        else:
            hi = mid - 1

    if best_prompt is None:
        # Could not fit even a single note — still return it, let vLLM handle.
        kept_blocks = notes_blocks[-1:] if direction == "recent" else notes_blocks[:1]
        candidate_text = _reassemble(
            preamble, notes_header_line, kept_blocks, codes_section, 1, notes_total, direction
        )
        best_prompt = build_chat_prompt(tokenizer, candidate_text,
                                        enable_thinking=enable_thinking,
                                        extra_system_text=extra_system_text)
        best_tokens = len(tokenizer.encode(best_prompt, add_special_tokens=False))
        best_kept = 1

    return best_prompt, best_tokens, True, best_kept, notes_total


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

PROB_REGEX = re.compile(
    r'"probability"\s*:\s*([01](?:\.\d+)?|\.\d+)',
    re.IGNORECASE,
)


def _iter_balanced_json_objects(text: str):
    """Yield bracket-balanced {...} substrings that handle strings and escapes.

    The structured report can contain unescaped '{' or '}' inside quoted text,
    so a naive non-greedy regex finds the wrong boundary. This walker tracks
    string state and returns full outermost objects.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def parse_completion(text: str):
    """Extract (probability, report, parse_ok)."""
    text = text.strip()

    # Try strict JSON on the whole text first
    try:
        obj = json.loads(text)
        prob = float(obj["probability"])
        report = str(obj.get("report", obj.get("summary", "")))
        if 0.0 <= prob <= 1.0:
            return prob, report, True
    except Exception:
        pass

    # Try to find a balanced {...} block (tolerates pre/post text and braces in strings)
    for candidate in _iter_balanced_json_objects(text):
        try:
            obj = json.loads(candidate)
            prob = float(obj["probability"])
            report = str(obj.get("report", obj.get("summary", "")))
            if 0.0 <= prob <= 1.0:
                return prob, report, True
        except Exception:
            continue

    # Regex fallback for probability
    m = PROB_REGEX.search(text)
    if m:
        try:
            prob = float(m.group(1))
            if 0.0 <= prob <= 1.0:
                return prob, text, False
        except Exception:
            pass

    return None, text, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to full_context_test_patients.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions JSONL")
    parser.add_argument("--model-id", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-patients", type=int, default=-1,
                        help="Process only the first N patients (<=0 means all)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of prompts per vLLM.generate call")
    parser.add_argument("--max-model-len", type=int, default=131072,
                        help="Max context length to configure vLLM with")
    parser.add_argument("--max-input-tokens", type=int, default=122880,
                        help="Prompt token budget after which we truncate oldest notes "
                             "(default 120K, leaving ~8K for response)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens for the structured clinical report response")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size (set to N to shard across N GPUs)")
    parser.add_argument("--rope-scaling-json", type=str, default=None,
                        help="Optional JSON string passed to vLLM as "
                             "hf_overrides={'rope_scaling': <json>}. Used to enable "
                             "YaRN for Qwen3-32B at 128K context, e.g. "
                             "'{\"rope_type\":\"yarn\",\"factor\":4.0,"
                             "\"original_max_position_embeddings\":32768}'.")
    parser.add_argument("--enable-thinking", choices=["auto", "true", "false"],
                        default="auto",
                        help="Pass enable_thinking=<bool> to apply_chat_template. "
                             "'auto' (default) does not pass the kwarg (Llama "
                             "behavior). For Qwen3 free-form runs use 'true'; "
                             "for Qwen3 verbalizer runs use 'false'.")
    parser.add_argument("--extra-system-text", type=str, default="",
                        help="Optional text appended to SYSTEM_PROMPT (separated by "
                             "two newlines). Use to inject reasoning-model-specific "
                             "verbatim/handle constraints without touching the "
                             "canonical Llama prompt.")
    parser.add_argument("--extra-system-text-file", type=str, default=None,
                        help="Path to a text file whose contents are used as "
                             "--extra-system-text (overrides --extra-system-text "
                             "if both given).")
    args = parser.parse_args()

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]

    extra_sys = args.extra_system_text
    if args.extra_system_text_file:
        extra_sys = Path(args.extra_system_text_file).read_text(encoding="utf-8")
    if extra_sys:
        print(f"[prompt] extra_system_text ({len(extra_sys)} chars) appended to SYSTEM_PROMPT",
              flush=True)

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

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build prompts (with truncation as needed)
    print("Building prompts (and applying oldest-first truncation where needed) ...",
          flush=True)
    prompts = []
    meta = []  # parallel list of dicts with record metadata
    for p in patients:
        prompt, ntok, truncated, kept, total_notes = fit_prompt_to_budget(
            tokenizer, p["patient_text"], args.max_input_tokens,
            enable_thinking=enable_thinking,
            extra_system_text=extra_sys,
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

    tok_counts = [m["num_input_tokens"] for m in meta]
    n_trunc = sum(1 for m in meta if m["truncated"])
    print(
        f"Prompt token stats: min={min(tok_counts)} "
        f"median={sorted(tok_counts)[len(tok_counts)//2]} "
        f"max={max(tok_counts)} truncated={n_trunc}/{len(meta)}",
        flush=True,
    )

    # Load vLLM engine
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
        # Override max_position_embeddings alongside rope_scaling so vLLM
        # builds the rotary embedding table at the extended context size.
        # Setting only rope_scaling without max_position_embeddings in 0.19
        # leaves positions >native producing NaN (device-side CUDA assert).
        llm_kwargs["hf_overrides"] = {
            "rope_scaling": rope_scaling,
            "max_position_embeddings": args.max_model_len,
        }
        print(f"vLLM hf_overrides: rope_scaling={rope_scaling}, "
              f"max_position_embeddings={args.max_model_len}", flush=True)
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"vLLM engine ready in {time.time() - t0:.1f}s", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    # Run inference in batches, write results incrementally
    n_done = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            batch_meta = meta[batch_start: batch_start + args.batch_size]
            t_batch = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - t_batch

            # vLLM returns RequestOutputs in the same order as prompts
            for m, out in zip(batch_meta, outputs):
                raw = out.outputs[0].text if out.outputs else ""
                prob, report, parse_ok = parse_completion(raw)
                record = dict(m)
                record["raw_completion"] = raw
                record["parsed_probability"] = prob
                record["parsed_report"] = report
                record["parse_ok"] = parse_ok
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch_prompts)
            print(
                f"[{n_done}/{len(prompts)}] batch {len(batch_prompts)} prompts in "
                f"{elapsed:.1f}s",
                flush=True,
            )

    print(f"Wrote {n_done} predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
