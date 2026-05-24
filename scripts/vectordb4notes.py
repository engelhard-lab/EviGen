import argparse
from pathlib import Path
import lancedb
import pyarrow as pa
import pandas as pd
import duckdb
from sentence_transformers import SentenceTransformer
import time
from typing import List
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DB_DIR = Path("/hpc/dctrl/fl105/EviGen")
DEFAULT_CSV_NAME = "dataset4death_note.csv"
DEFAULT_CHUNK_LENGTH = 300
DEFAULT_BATCH_SIZE = 1000
DEFAULT_ENCODE_BATCH_SIZE = 16
VECTOR_DIM = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Build note LanceDB table from local CSV data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--csv-file", type=str, default=DEFAULT_CSV_NAME)
    parser.add_argument("--chunk-length", type=int, default=DEFAULT_CHUNK_LENGTH)
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in words. Defaults to 20%% of --chunk-length.",
    )
    parser.add_argument("--table-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encode-batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE)
    return parser.parse_args()


def load_model():
    print("loading model ...")
    cache_dir = "/hpc/home/fl105/engelhardlab/fl105/cache"
    # Omit device_map so no dispatch wrapper is added — model is 16.5GB fp16, fits trivially
    # on a single GPU. Without device_map the post-load config patch works cleanly.
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        cache_folder=cache_dir,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "dtype": torch.float16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Disable KV cache: not needed for embedding-only inference. With use_cache=True (default),
    # all 36 layers accumulate KV tensors (~3.7GB/layer at B=5120, S=223), causing OOM
    # around layer 15 even when the first-layer FFN allocation fits.
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "use_cache"):
            mod.config.use_cache = False
            break
    print(f"use_cache = {model._first_module().auto_model.config.use_cache}")  # must print False

    print("model loaded")
    return model

def format_note(text: str, age_years: int) -> str:
    age_line = f"Patient age: {age_years} years"
    return f"{age_line}\n \n Clinical Note:\n{text.strip()}"
    
# chunks each document into segments
def create_chunks(text: str, chunk_length, overlap) -> List[str]:
    words = text.split()
    
    if len(words) <= chunk_length:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(words):
        # Create current chunk
        end_idx = min(i + chunk_length, len(words))
        chunk = ' '.join(words[i:end_idx])
        chunks.append(chunk)
        
        # If we've reached the end, break
        if end_idx == len(words):
            break
            
        # Calculate next starting position
        next_start = i + chunk_length - overlap
        
        # If the remaining text is small enough to fit in the overlap,
        # just extend the current chunk instead of creating a tiny new chunk
        remaining_words = len(words) - next_start
        if remaining_words <= overlap:
            # Remove the last chunk and create a longer one
            chunks.pop()
            chunk = ' '.join(words[i:])
            chunks.append(chunk)
            break
        
        i = next_start
    
    return chunks

def make_batches_from_csv(csv_file, model, chunk_size, overlap, batch_size, encode_batch_size):
    prev_time = time.time()
    for num_batch, chunk in enumerate(pd.read_csv(csv_file, chunksize=batch_size), start=1):
        if num_batch > 1:
            batch_time = time.time() - prev_time
            print(f"batch {num_batch-1} processed in {batch_time:.3f}s")
        prev_time = time.time()

        # Collect all chunks and metadata first
        all_chunks = []
        chunk_metadata = []
        
        for row in chunk.itertuples(index=False):
            review_chunks = create_chunks(row.text, chunk_size, overlap)
            for review_chunk in review_chunks:
                review_chunk_formatted = format_note(review_chunk, row.age_at_note) # append age 
                all_chunks.append(review_chunk_formatted)
                chunk_metadata.append({
                    "note_id": row.note_id,
                    "subject_id": row.subject_id,
                    "days_to_prediction": row.DaysToLastNotes,
                    "label": row.label
                })
        
        # Batch encode all chunks at once
        embeddings = model.encode(all_chunks, batch_size=encode_batch_size, show_progress_bar=False)
        
        # Create rows with batch-encoded embeddings
        rows = []
        for i, embedding in enumerate(embeddings):
            rows.append({
                "note_id": chunk_metadata[i]["note_id"],
                "subject_id": chunk_metadata[i]["subject_id"],
                "chunk": all_chunks[i],
                "days_to_prediction": chunk_metadata[i]["days_to_prediction"],
                "vector": embedding.tolist(),
                "label": chunk_metadata[i]["label"],
            })
        
        yield pd.DataFrame(rows)

def main():
    args = parse_args()
    csv_file = Path(args.csv_file)
    if not csv_file.is_absolute():
        csv_file = args.data_dir / csv_file
    db_path = args.db_dir
    table_name = args.table_name or f"death_age_chunksize{args.chunk_length}_qwen"
    chunk_overlap = args.chunk_overlap
    if chunk_overlap is None:
        chunk_overlap = max(1, int(round(args.chunk_length * 0.2)))

    model = load_model()
    db = lancedb.connect(str(db_path))

    schema = pa.schema([
        pa.field("note_id", pa.utf8()),
        pa.field("subject_id", pa.int64()),
        pa.field("days_to_prediction", pa.int64()),
        pa.field("chunk", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("label", pa.int8()),
    ])

    print(f"csv_file={csv_file}")
    print(f"db_path={db_path}")
    print(f"table_name={table_name}")
    print(f"chunk_length={args.chunk_length}")
    print(f"chunk_overlap={chunk_overlap}")

    print("start encoding ...")
    db.create_table(
        table_name,
        make_batches_from_csv(
            csv_file=csv_file,
            model=model,
            chunk_size=args.chunk_length,
            overlap=chunk_overlap,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
        ),
        schema=schema,
        mode="overwrite",
    )

    # display the first 10 rows
    tbl = db.open_table(table_name)
    lance_tbl = tbl.to_lance()
    df = duckdb.sql("SELECT note_id, chunk FROM lance_tbl LIMIT 10").to_df()
    print(df)

    # create search index
    # choosing hyperparameters:
    # num_partitions: keeping each partition 1K-4K rows
    # num_sub_vectors: dimension / num_sub_vectors should be a multiple of 8
    # for optimum SIMD efficiency
    # default is the square root of number of rows
    # comment this line for exact search
    # tbl.create_index(num_partitions=1, num_sub_vectors=256, accelerator="cuda")


if __name__ == "__main__":
    main()
