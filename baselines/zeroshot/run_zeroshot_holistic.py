"""Holistic zero-shot prompt variant for one-year mortality prediction.

Same input as ``run_zeroshot.py`` (full patient history: notes + ICD codes),
same JSON output schema, but the model is asked for a single prose
explanation + recommendation instead of a numbered top-5 factor list. Used
to isolate the contribution of factor-by-factor structure: identical model,
identical history, only the report shape differs.

The parser is re-used verbatim from ``run_zeroshot`` since the output
envelope is still ``{"probability": <float>, "report": "<text>"}``.
"""

from __future__ import annotations

# Reuse the JSON+regex extraction logic from the canonical zeroshot module
# so the two variants cannot drift on parsing edge cases.
from baselines.zeroshot.run_zeroshot import parse_completion  # noqa: F401


SYSTEM_PROMPT = (
    "You are a clinical interpretability assistant. Given a patient's complete "
    "clinical history (clinical notes and ICD diagnostic codes in chronological "
    "order), you (1) estimate the probability of all-cause mortality within 1 "
    "year of the last hospital discharge, and (2) write a short, conservative, "
    "evidence-grounded clinical narrative explaining the estimate and offering "
    "high-level care considerations.\n\n"
    "Hard rules:\n"
    "- Do NOT infer diagnoses, severity, or mechanisms beyond what is explicitly "
    "stated in the history.\n"
    "- Keep physiologic reasoning general and non-specific.\n"
    "- Provide only high-level care considerations, not specific treatment plans.\n"
    "- Do NOT produce a numbered list of factors and do NOT cite individual "
    "note_ids or ICD codes. The report is intended to read as a holistic "
    "narrative, not a factor-by-factor breakdown.\n\n"
    "Respond with STRICT JSON only."
)

USER_INSTRUCTION = (
    "Below is a single patient's complete clinical history, ordered from oldest "
    "to newest. Each note header includes its note_id and each code line "
    "includes its ICD code-version identifier.\n\n"
    "Step 1. Estimate the probability (float in [0.0, 1.0]) that this patient "
    "will die within one year after their last discharge.\n\n"
    "Step 2. Write a holistic clinical narrative with EXACTLY these three "
    "sections (in this order, with these exact bracketed headers). Output it "
    "as the value of the \"report\" field (use \\n for line breaks).\n\n"
    "[Prediction]\n"
    "- One sentence stating this is a prediction of all-cause mortality within "
    "1 year after hospital discharge, and reporting the probability you "
    "estimated.\n\n"
    "[Explanation]\n"
    "- One free-form paragraph (3-5 sentences) synthesizing the patient's "
    "history into a coherent narrative explaining why the estimated 1-year "
    "mortality risk is high or low.\n"
    "- Do NOT enumerate or number distinct factors.\n"
    "- Do NOT include direct quotes, note_ids, or ICD codes.\n"
    "- Mention \"chronic\", \"acute\", or \"end-stage\" only if clearly supported "
    "by the history.\n\n"
    "[Recommendation]\n"
    "- One short paragraph (2-3 sentences) of high-level, tentative care "
    "considerations: prognosis communication, alignment of goals-of-care, and "
    "symptom-focused support. Use tentative language (\"may\", \"could be "
    "relevant\", \"may warrant attention\").\n"
    "- Do NOT recommend specific diagnostics, medications, or procedures, and "
    "do NOT break the recommendation into per-factor bullets.\n\n"
    "Output format: respond with ONLY a single JSON object, no prose outside "
    "the object, no markdown fences, matching exactly this schema:\n"
    '{"probability": <float in [0.0, 1.0]>, '
    '"report": "<structured clinical narrative as described above, with \\n line breaks>"}\n\n'
    "Patient history:\n"
)
