"""Generate clinical reports for the EviGen pipeline.

Reads an IG explanation JSON (one record per patient with top-k attributed
retrieved items, produced by ``scripts/explain.py``) and prompts a local
Llama-3.1-Instruct model via vLLM to produce a structured 4-section clinical
report. Writes a predictions JSONL whose schema matches
``outputs/zeroshot/predictions_*.jsonl`` so ``scripts/eval_faithfulness.py``
consumes it without changes.

Ported from /hpc/group/engelhardlab/hsb26/IRIS-PSRM-3/scripts/01_generate_gold_outputs.py
(API-based) and from baselines/zeroshot/run_zeroshot.py (vLLM batching).

Usage:
    python -u src/evigen_dynamicquery/generate_gold_outputs.py \\
        --input outputs/test_explanations.json \\
        --output outputs/evigen/predictions_8b.jsonl \\
        --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --tensor-parallel-size 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PROJECT_DIR = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You are a conservative, evidence-grounded clinical interpretability assistant."
)

# The chunked-notes parquet (and therefore the IG explanation file's
# `chunk_text` field) prefixes each chunk with a synthetic header:
#   "Patient age: <N> years\n \nClinical Note:\n<actual note text>"
# This prefix is added at embedding time and is NOT present in
# `data/dataset4death_note.csv`. If the LLM quotes the prefix, the
# faithfulness eval's fuzzy match against the CSV `text` column will fail.
# Strip the prefix here so the LLM only sees the underlying note content.
_AGE_PREFIX_RE = re.compile(
    r"^\s*Patient\s+age:\s*\d+\s*years\s*\n*\s*\n*\s*Clinical\s+Note:\s*\n*",
    re.IGNORECASE,
)


def strip_chunk_prefix(text: str) -> str:
    """Strip the synthetic 'Patient age: X years / Clinical Note:' prefix."""
    return _AGE_PREFIX_RE.sub("", text, count=1).lstrip()


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def format_patient_context(patient_rec: dict, k: int = 5) -> str:
    """Format the patient-context section of the user prompt.

    Picks the top-k items by absolute attribution, formatting clinical-note
    chunks (with their `note_id`) and ICD entries (with their
    `icd_code_ver`) side by side. The rest of the prompt — the task
    instructions — is appended afterwards from the prompt template.
    """
    items = sorted(
        patient_rec["retrieved_items"],
        key=lambda x: abs(x.get("attribution_score", 0.0)),
        reverse=True,
    )[:k]

    lines = []
    lines.append("Model name: iris")
    lines.append(f"Patient ID: {patient_rec['subject_id']}")
    lines.append(f"Predicted probability: {patient_rec['probability']:.6f}")
    lines.append("")
    lines.append(
        f"Top {k} chunks (sorted by absolute attribution, most important first):"
    )

    for i, chunk in enumerate(items, 1):
        attr = chunk.get("attribution_score")
        sim = chunk.get("similarity_score")
        meta_bits = []
        if attr is not None:
            meta_bits.append(f"attribution_score: {attr:+.4f}")
        if sim is not None:
            meta_bits.append(f"similarity_score: {sim:.4f}")

        if chunk.get("query_type") == "ICD":
            handle = chunk.get("icd_code_ver", "")
            heading_cite = f"(ICD code: {handle})" if handle else ""
            text = str(chunk.get("long_title", "")).strip()
        else:
            handle = chunk.get("note_id") or chunk.get("global_index", "")
            heading_cite = f"(note_id: {handle})" if handle != "" else ""
            text = strip_chunk_prefix(str(chunk.get("chunk_text", "")))
        text = " ".join(text.split())  # whitespace collapse, no length cap

        # Header line carries the citation handle and metadata; the chunk
        # body follows on the next line under a `text:` anchor. Mirrors
        # RAG's `[Note N] (note_id: ...)` layout — the heading-cite gives
        # the LLM a short attention hop from chunk body to chunk-id for
        # citation accuracy — and the `text:` label keeps the LLM
        # treating the chunk body as quotable material (without it,
        # smaller models latched onto the heading-cite as a shortcut and
        # stopped including verbatim quotes in their `Supporting
        # evidence:` line).
        header = f"{i}. {heading_cite}".rstrip()
        if meta_bits:
            header = f"{header}  " + "  ".join(meta_bits) if heading_cite else f"{i}. " + "  ".join(meta_bits)
        lines.append(header)
        lines.append(f"   text: {text}")

    return "\n".join(lines)


def build_user_prompt(patient_rec: dict, template: str, k: int = 5) -> str:
    ctx = format_patient_context(patient_rec, k=k)
    return f"{ctx}\n\nTask instructions:\n{template.format(k=k)}"


def build_chat_prompt(tokenizer, user_text: str,
                      enable_thinking=None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


_SUBFIELD_LABELS = (
    "Factor summary",
    "Supporting evidence",
    "Reasoning",
    "Attribution score",
    "Cited source",
    "Provenance",
)
_LABEL_GROUP = "|".join(re.escape(s) for s in _SUBFIELD_LABELS)
_SUBFIELD_BULLET_RE = re.compile(
    rf"^(\s*)[-*•]\s+((?:{_LABEL_GROUP})\s*:)",
    re.MULTILINE,
)
# Strip ** bold around sub-field labels: `**Supporting evidence**:` → `Supporting evidence:`
_SUBFIELD_BOLD_RE = re.compile(
    rf"\*\*\s*({_LABEL_GROUP})\s*\*\*\s*:",
)
# Strip ** bold around the inline label after a numbered heading prefix:
# `1. **Factor summary**: text` → `1. Factor summary: text`
_NUMBERED_BOLD_LABEL_RE = re.compile(
    rf"^(\s*\d+\.\s+)\*\*\s*({_LABEL_GROUP})\s*\*\*\s*(:?)",
    re.MULTILINE,
)


# `**1. Title**` (whole numbered heading wrapped in bold) → `1. Title`
_BOLD_NUMBERED_HEADING_RE = re.compile(
    r"^(\s*)\*\*\s*(\d+)\.\s+([^\n]+?)\s*\*\*",
    re.MULTILINE,
)
# Section markers — used to scope factor renumbering to the correct block.
_SECTION_HDR_RE = re.compile(
    r"^\s*(?:\*\*|\[)?\s*(Prediction\b[^\n]*|Predictive\s+Factors\b[^\n]*|"
    r"Reasoning\s+Lens\b[^\n]*|Recommendation[^\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_FACTOR_SUMMARY_RE = re.compile(r"^(\s*)(Factor\s+summary\s*:)", re.MULTILINE | re.IGNORECASE)
_NUMBERED_FACTOR_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


def _renumber_factor_summaries(report: str) -> str:
    """Inside the [Predictive Factors] section, prefix bare ``Factor summary:``
    lines with sequential ``N. `` numbering when the existing factor block has
    no numeric prefix (e.g. when the model groups factors under category
    headings instead of numbering them).
    """
    headers = list(_SECTION_HDR_RE.finditer(report))
    pf_idx = None
    for i, h in enumerate(headers):
        if "predictive factors" in h.group(1).lower():
            pf_idx = i
            break
    if pf_idx is None:
        return report

    block_start = headers[pf_idx].end()
    block_end = headers[pf_idx + 1].start() if pf_idx + 1 < len(headers) else len(report)
    block = report[block_start:block_end]

    summary_positions = [m.start() for m in _FACTOR_SUMMARY_RE.finditer(block)]
    if not summary_positions:
        return report
    # Find numbered factor positions
    numbered_positions = [m.start() for m in _NUMBERED_FACTOR_RE.finditer(block)]

    def is_already_numbered(summary_pos: int) -> bool:
        # If the previous numbered marker is on the same line OR the line just
        # before, treat the summary as already numbered.
        line_start = block.rfind("\n", 0, summary_pos) + 1
        # Same line: the line begins with `N.`
        if _NUMBERED_FACTOR_RE.match(block[line_start: summary_pos + 1]):
            return True
        # Previous line ends with `N. **Title**` already (heading-then-summary)
        prev_line_end = line_start - 1
        prev_line_start = block.rfind("\n", 0, prev_line_end) + 1
        prev_line = block[prev_line_start:prev_line_end]
        if _NUMBERED_FACTOR_RE.match(prev_line):
            return True
        return False

    # Renumber bare summaries
    out_chunks = [report[:block_start]]
    cursor = 0
    counter = 0
    for pos in summary_positions:
        if is_already_numbered(pos):
            continue
        counter += 1
        out_chunks.append(block[cursor:pos])
        # Insert "N. " before the indent of the Factor summary line
        m = _FACTOR_SUMMARY_RE.match(block[pos:])
        if m is None:
            continue
        indent = m.group(1)
        out_chunks.append(f"{counter}. ")
        # Skip over the indent we just replicated above
        out_chunks.append(block[pos + len(indent): pos + len(m.group(0))])
        cursor = pos + len(m.group(0))
    out_chunks.append(block[cursor:])
    out_chunks.append(report[block_end:])
    return "".join(out_chunks)


def clean_response(text: str) -> str:
    """Normalize the model's report into a flat-indented, label-explicit form.

    Models we run (Llama-3.1-8B / 70B) format the PSRM clinical-report template
    in slightly different ways. To get the eval's per-factor parser to see all
    five factors, we pre-normalize the response:

    - Strip markdown ATX headers (``###`` etc).
    - Drop leading bullet markers from indented sub-field lines
      (``   - Supporting evidence: ...`` → ``   Supporting evidence: ...``).
    - Strip ``**bold**`` around sub-field labels and around whole numbered
      headings (``**1. Title**`` → ``1. Title``).
    - Auto-prefix bare ``Factor summary:`` lines with ``N. `` when the model
      grouped factors under category headings instead of numbering them.
    """
    text = re.sub(r"^#{1,4}\s*", "", text, flags=re.MULTILINE)
    text = _SUBFIELD_BULLET_RE.sub(r"\1\2", text)
    text = _BOLD_NUMBERED_HEADING_RE.sub(r"\1\2. \3", text)
    text = _NUMBERED_BOLD_LABEL_RE.sub(r"\1\2\3", text)
    text = _SUBFIELD_BOLD_RE.sub(r"\1:", text)
    text = _renumber_factor_summaries(text)
    return text


_SECTION_PATTERNS = {
    "prediction":   re.compile(r"(?im)^\s*(?:\*\*|\[)?\s*Prediction\b"),
    "factors":      re.compile(r"(?im)^\s*(?:\*\*|\[)?\s*Predictive\s+Factors\b"),
    "reasoning":    re.compile(r"(?im)^\s*(?:\*\*|\[)?\s*Reasoning\s+Lens\b"),
    "recommend":    re.compile(r"(?im)^\s*(?:\*\*|\[)?\s*Recommendation"),
}


def validate_output(text: str) -> bool:
    """All four required section headers present and the report is non-trivial.

    Accepts ``[Section]``, ``**Section**``, or bare ``Section`` heading styles
    — matching the permissiveness of ``scripts/eval_faithfulness.py`` parsing.
    """
    if len(text) <= 500:
        return False
    return all(p.search(text) is not None for p in _SECTION_PATTERNS.values())


def collect_note_ids(patient_rec: dict) -> list[str]:
    seen, ordered = set(), []
    for it in patient_rec.get("retrieved_items", []):
        if it.get("query_type") != "Note":
            continue
        nid = it.get("note_id")
        if not nid:
            continue
        nid = str(nid)
        if nid in seen:
            continue
        seen.add(nid)
        ordered.append(nid)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="IG explanation JSON (output of scripts/explain.py).")
    ap.add_argument("--output", type=str, required=True,
                    help="Predictions JSONL output path.")
    ap.add_argument("--template", type=str, default="templates/tuned_prompt.txt")
    ap.add_argument("--model-id", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n-rollouts", type=int, default=1,
                    help="Number of samples per patient. >1 emits one row per rollout "
                         "with rollout_idx/temperature/model fields added.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed (only meaningful when temperature > 0).")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable vLLM CUDA-graph + flashinfer fusion paths "
                         "(needed when nvcc is absent on the compute node, e.g. "
                         "70B TP=2 on h200-hp).")
    ap.add_argument("--disable-custom-all-reduce", action="store_true",
                    help="Skip vLLM's custom allreduce path (TP>1 only).")
    ap.add_argument("--max-patients", type=int, default=-1)
    ap.add_argument("--rope-scaling-json", type=str, default=None,
                    help="Optional JSON for vLLM hf_overrides={'rope_scaling': ...}.")
    ap.add_argument("--enable-thinking", choices=["auto", "true", "false"],
                    default="auto",
                    help="Pass enable_thinking=<bool> to apply_chat_template; "
                         "'auto' (default) does not pass the kwarg. For Qwen3 "
                         "EviGen runs use 'true'.")
    args = ap.parse_args()

    enable_thinking = {"auto": None, "true": True, "false": False}[args.enable_thinking]

    tpl_path = Path(args.template)
    if not tpl_path.is_absolute():
        tpl_path = PROJECT_DIR / args.template
    template = load_template(tpl_path)

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_DIR / args.input
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        patients = json.load(f)
    if args.max_patients > 0:
        patients = patients[: args.max_patients]
    print(f"Loaded {len(patients)} patients from {in_path}", flush=True)

    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Building prompts (k={args.top_k}) ...", flush=True)
    prompts, meta = [], []
    for p in patients:
        user_text = build_user_prompt(p, template, k=args.top_k)
        prompt = build_chat_prompt(tokenizer, user_text,
                                   enable_thinking=enable_thinking)
        ntok = len(tokenizer.encode(prompt, add_special_tokens=False))
        kept = collect_note_ids(p)
        prompts.append(prompt)
        meta.append({
            "subject_id": int(p["subject_id"]),
            "label": int(p.get("label", -1)),
            "probability": float(p.get("probability", 0.0)),
            "predicted_label": int(p.get("predicted_label", -1)),
            "num_input_tokens": ntok,
            "truncated": False,
            "notes_kept": kept,
            "notes_total": len(kept),
        })

    tok_counts = [m["num_input_tokens"] for m in meta]
    print(
        "Prompt token stats: "
        f"min={min(tok_counts)} median={sorted(tok_counts)[len(tok_counts)//2]} "
        f"max={max(tok_counts)}",
        flush=True,
    )
    safety = args.max_model_len - args.max_new_tokens - 64
    if max(tok_counts) >= safety:
        print(
            f"WARNING: max prompt tokens ({max(tok_counts)}) is close to "
            f"max_model_len - max_new_tokens ({safety}). Consider raising "
            "--max-model-len.",
            file=sys.stderr,
        )

    print(f"Loading vLLM model: {args.model_id}", flush=True)
    t0 = time.time()
    llm_kwargs = dict(
        model=args.model_id,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    if args.rope_scaling_json:
        rope_scaling = json.loads(args.rope_scaling_json)
        llm_kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
        print(f"vLLM hf_overrides rope_scaling: {rope_scaling}", flush=True)
    llm = LLM(**llm_kwargs)
    print(f"vLLM engine ready in {time.time() - t0:.1f}s", flush=True)

    sp_kwargs = dict(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=args.n_rollouts,
    )
    if args.seed is not None:
        sp_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sp_kwargs)

    n_done = 0
    n_parse_ok = 0
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            batch_meta = meta[batch_start: batch_start + args.batch_size]
            t_batch = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - t_batch

            for m, out in zip(batch_meta, outputs):
                # vLLM returns one CompletionOutput per sample; iterate them.
                cands = list(out.outputs) if out.outputs else []
                for r_idx, cand in enumerate(cands):
                    raw = cand.text
                    report = clean_response(raw)
                    parse_ok = validate_output(report)
                    record = dict(m)
                    if args.n_rollouts > 1:
                        record["rollout_idx"] = r_idx
                        record["temperature"] = float(args.temperature)
                        record["model"] = args.model_id
                    record["raw_completion"] = raw
                    record["parsed_probability"] = m["probability"]
                    record["parsed_report"] = report
                    record["parse_ok"] = parse_ok
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_rows += 1
                    if parse_ok:
                        n_parse_ok += 1
            f_out.flush()
            n_done += len(batch_prompts)
            print(
                f"[{n_done}/{len(prompts)}] {len(batch_prompts)} prompts in "
                f"{elapsed:.1f}s | rows={n_rows} parse_ok={n_parse_ok}/{n_rows}",
                flush=True,
            )

    print(f"Wrote {n_rows} rows ({n_done} patients × {args.n_rollouts}) to {out_path}", flush=True)
    print(
        f"Final parse_ok rate: {n_parse_ok}/{n_rows} "
        f"({100*n_parse_ok/max(n_rows,1):.2f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
