# EviGen

Evidence-aware mortality prediction on MIMIC-IV. EviGen learns a small set of
clinically grounded queries over per-patient note and ICD code embeddings,
retrieves the top-k most relevant chunks per query, and aggregates them through
a lightweight transformer head for 1-year all-cause mortality prediction.
Integrated Gradients over the retrieval scores produces per-patient evidence
that an LLM renders into a structured clinical report.

## Repository layout

```
src/evigen_dynamicquery/   Core package
    config.py              Config dataclass + YAML loader
    data.py                Parquet loader for note/code embeddings; subject splits
    model_evigen.py        EviGen model (addition fusion: note + code experts averaged)
    model_evigen_concat.py EviGenConcat variant (concatenation fusion)
    generate_gold_outputs.py  vLLM report generation from IG explanations

scripts/                   Runnable entrypoints
    train.py / train_concat.py    Training (addition / concat variants)
    explain.py                    Integrated Gradients per patient
    eval_test_per_subject.py      Per-subject test predictions
    eval_faithfulness.py          Faithfulness eval of generated reports
    vectordb4notes.py             Build LanceDB note vector index
    vectordb4ICD_byPatients.py    Build LanceDB ICD code vector index
    convert_lancedb_to_parquet.py
    build_zeroshot_dataset.py     Build zeroshot LLM baseline input
    build_rag_dataset.py          Build RAG baseline input (8 mortality-risk queries)
    build_notes_only_inputs.py    Notes-only ablation
    compute_bootstrap_metrics.py  Bootstrap CIs + pairwise significance
    (PSRM typed-report eval + selection scripts)

baselines/
    zeroshot/         vLLM zeroshot LLM (Llama-3.1) + verbalizer variants
    rag/              RAG baseline (top-k retrieval over 8 risk queries)
    text_encoder/     Text-only LLM baseline (no embeddings)
    qlora_verbalizer/ QLoRA fine-tuned verbalizer
    meanpool/         Mean-pool over Qwen embeddings
    mil/              Single-head MIL baseline

configs/    YAML configs (default + chunksize / fusion / sweep ablations)
templates/  Prompt templates for clinical report generation
```

## Install

```bash
pip install -r requirements.txt
```

GPU required for training, IG, and any vLLM/QLoRA baseline. CPU is fine for the
dataset / vector DB build scripts.

## Data

Not included. Point the configs at your own MIMIC-IV-derived parquets:

- `note_parquet` — chunked clinical notes with per-chunk Qwen embeddings.
  Built from a `dataset4death_note.csv` via `scripts/vectordb4notes.py` followed
  by `scripts/convert_lancedb_to_parquet.py`. Chunks carry an age prefix
  (`"Patient age: ... years"`) before the note text.
- `code_parquet` — patient ICD codes deduplicated on `(subject_id, icd_code_ver)`,
  embedded from `long_title` (no age prefix). Built via
  `scripts/vectordb4ICD_byPatients.py`.

Embeddings are produced with **Qwen3-Embedding-8B** (output dim = 4096).

## Workflow

```bash
# 1. Build vector DBs from raw notes / codes CSVs
python scripts/vectordb4notes.py
python scripts/vectordb4ICD_byPatients.py
python scripts/convert_lancedb_to_parquet.py

# 2. Train
python scripts/train.py --config configs/config.yaml --run-name my_run

# 3. Per-subject test predictions
python scripts/eval_test_per_subject.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/<run>.pt \
    --output outputs/<run>_test.jsonl

# 4. Integrated Gradients explanations
python scripts/explain.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/<run>.pt

# 5. Generate clinical reports from IG output (vLLM + Llama-3.1)
python -m evigen_dynamicquery.generate_gold_outputs \
    --explanations outputs/<run>_explanations.json \
    --template templates/tuned_prompt.txt

# 6. Faithfulness eval of the reports
python scripts/eval_faithfulness.py --reports <reports.json>
```

## Baselines

All baselines share the same train/val/test subject split as EviGen. Build
their inputs first:

```bash
python scripts/build_zeroshot_dataset.py   # zeroshot + text_encoder
python scripts/build_rag_dataset.py        # rag
```

Then run any baseline directly, e.g.:

```bash
python baselines/zeroshot/run_zeroshot.py
python baselines/rag/run_rag.py
python baselines/qlora_verbalizer/train_qlora.py
```

## Reproducibility note

The canonical train/val/test split lives in `data.create_subject_splits` and
depends on the row order of the notes DataFrame loaded via
`pyarrow.iter_batches` + `pd.concat(...).drop_duplicates()` (first-occurrence
order). Any new dataset-construction script that needs the same test subjects
must reproduce that exact ordering before calling `create_subject_splits` —
`DuckDB SELECT DISTINCT` does **not** preserve row order and will produce a
divergent split despite the same seed.
